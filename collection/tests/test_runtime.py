"""Host-daemon routing and early rejection of unusable task runtimes."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from tokenmoe_collect import runtime, serving
from tokenmoe_collect.minisweagent_adapter import TracedEnvironment
from tokenmoe_collect.records import RecordStore


@pytest.fixture
def config_path(tmp_path):
    (tmp_path / "items.json").write_text(
        json.dumps([{"instance_id": "test__repo-1", "image_name": "swebench/test@sha256:abc"}])
    )
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "runtime": "docker",
                "runtime_socket": "/run/host docker.sock",
                "dataset_path": "items.json",
                "instances": ["test__repo-1"],
                "cpu_quota": 2,
                "memory_limit": "4g",
            }
        )
    )
    return path


def test_pull_start_exec_and_cleanup_use_the_same_explicit_daemon(
    config_path, tmp_path, monkeypatch
):
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(subprocess, "run", execute)
    # An ambient remote default must not change which daemon owns the tasks.
    monkeypatch.setenv("DOCKER_HOST", "tcp://other-host:2376")
    serving.pull_images(config_path)
    with RecordStore(tmp_path / "records") as store:
        env = TracedEnvironment(
            store=store,
            session_id="ses_test",
            runtime="docker",
            runtime_socket="/run/host docker.sock",
            image="swebench/test@sha256:abc",
            cwd="/testbed",
            create_workdir=True,
            run_args=runtime.container_run_args(yaml.safe_load(config_path.read_text())),
        )
        container = env.container_id
        tool_argv = env.executor.command("printf 'hello world'", cwd="/testbed/subdir")
        env.cleanup()
    commands = [call.args[0] for call in execute.call_args_list]
    prefix = ["docker", "--host", "unix:///run/host docker.sock"]
    assert [cmd[3] for cmd in commands] == ["pull", "run", "exec", "rm"]
    assert all(cmd[:3] == prefix for cmd in [*commands, tool_argv])
    assert tool_argv[3:] == [
        "exec",
        "-w",
        "/testbed/subdir",
        container,
        "bash",
        "-lc",
        "printf 'hello world'",
    ]
    assert commands[-1][3:] == ["rm", "-f", container]
    assert "--cpus" in commands[1] and "--memory" in commands[1]
    assert not any(arg in commands[1] for arg in ["--privileged", "--volume", "--mount"])


def test_preflight_starts_cached_image_with_limits_and_checks_same_host(config_path, monkeypatch):
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text()
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=boot_id, stderr=""))
    monkeypatch.setattr(subprocess, "run", execute)
    report = runtime.check_runtime(config_path)
    assert report["passed"] and report["boot_id"] == boot_id.strip()
    command = execute.call_args.args[0]
    assert command[:4] == ["docker", "--host", "unix:///run/host docker.sock", "run"]
    assert command[command.index("--cpus") + 1] == "2"
    assert command[command.index("--memory") + 1] == "4g"
    assert "--pull=never" in command and "--rm" in command
    assert command[-3:] == ["swebench/test@sha256:abc", "-c", "cat /proc/sys/kernel/random/boot_id"]
    execute.return_value.stdout = "different-host-boot-id\n"
    with pytest.raises(ValueError, match="same Linux host"):
        runtime.check_runtime(config_path)


@pytest.mark.parametrize("failure", ["permission", "mount", "timeout"])
def test_preflight_reports_failure_and_cleans_only_its_own_probe(config_path, monkeypatch, failure):
    ok = subprocess.CompletedProcess([], 0, stdout="", stderr="")
    if failure == "permission":
        error = subprocess.CalledProcessError(1, ["docker", "info"], stderr="permission denied")
        responses = [error]
    elif failure == "mount":
        error = subprocess.CalledProcessError(
            1, ["docker", "run"], stderr="mount proc: Operation not permitted"
        )
        responses = [ok, error, ok]
    else:
        error = subprocess.TimeoutExpired(["docker", "run"], 300)
        responses = [ok, error, ok]
    execute = Mock(side_effect=responses)
    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="socket"):
        runtime.check_runtime(config_path)
    if failure == "permission":
        assert execute.call_count == 1
    else:
        start = execute.call_args_list[1].args[0]
        cleanup = execute.call_args_list[2].args[0]
        assert cleanup[-1] == start[start.index("--name") + 1]
        assert cleanup[-1].startswith("tokenmoe-preflight-")
        assert cleanup[3:5] == ["rm", "-f"]


def test_pilot_rejects_runtime_failure_before_starting_model(config_path, tmp_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    config.update(model_config="model.yaml", output_dir=str(tmp_path / "pilot"))
    config_path.write_text(yaml.safe_dump(config))
    gate = tmp_path / "gate.json"
    gate.write_text('{"passed":true,"num_prompts":50}')
    check = Mock(side_effect=RuntimeError("runtime unavailable"))
    start = Mock()
    monkeypatch.setattr(serving, "check_runtime", check)
    monkeypatch.setattr(subprocess, "Popen", start)
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        serving.pilot(config_path, equivalence_record=gate)
    start.assert_not_called()
    assert not (tmp_path / "pilot").exists()


@pytest.mark.parametrize(
    ("backend", "socket"),
    [("podman", "/run/docker.sock"), ("local", "/run/docker.sock"), ("docker", "relative.sock")],
)
def test_invalid_transport_is_rejected(backend, socket):
    with pytest.raises(ValueError):
        runtime.runtime_command(backend, socket)


def test_native_runtimes_and_local_tests_remain_supported(tmp_path):
    assert runtime.runtime_command("podman") == ["podman"]
    assert runtime.runtime_command("docker") == ["docker"]
    config = tmp_path / "local.yaml"
    config.write_text("runtime: local\n")
    assert runtime.check_runtime(config)["scope"] == "CPU tests only"


def test_setup_failure_is_not_hidden_by_cleanup_error(tmp_path, monkeypatch):
    start_error = subprocess.CalledProcessError(1, ["docker", "run"], stderr="original mount error")
    cleanup_error = subprocess.CalledProcessError(1, ["docker", "rm"], stderr="no such container")
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=[start_error, cleanup_error]))
    with RecordStore(tmp_path / "records") as store:
        with pytest.raises(RuntimeError, match="original mount error"):
            TracedEnvironment(store=store, session_id="ses_test", runtime="docker", image="test")
