"""Host-daemon routing and early rejection of unusable task runtimes."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from tokenmoe_collect import container_probe, runtime, serving
from tokenmoe_collect.minisweagent_adapter import TracedEnvironment
from tokenmoe_collect.records import RecordStore


@pytest.fixture
def kernel_probe():
    return {
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cpu_quota_us": 200000,
        "cpu_period_us": 100000,
        "memory_limit_bytes": 4294967296,
    }


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


def test_preflight_starts_cached_image_with_limits_and_checks_same_host(
    config_path, kernel_probe, monkeypatch
):
    execute = Mock(
        return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(kernel_probe), stderr="")
    )
    monkeypatch.setattr(subprocess, "run", execute)
    report = runtime.check_runtime(config_path)
    assert report["passed"] and report["boot_id"] == kernel_probe["boot_id"]
    assert report["kernel_limits"]["memory_limit_bytes"] == 4294967296
    command = execute.call_args.args[0]
    assert command[:4] == ["docker", "--host", "unix:///run/host docker.sock", "run"]
    assert command[command.index("--cpus") + 1] == "2"
    assert command[command.index("--memory") + 1] == "4g"
    assert "--pull=never" in command and "--rm" in command
    assert command[command.index("/bin/sh") + 1] == "swebench/test@sha256:abc"
    assert command[-1] == Path(container_probe.__file__).read_text()
    execute.return_value.stdout = json.dumps(kernel_probe | {"boot_id": "different-host-boot-id"})
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
    with pytest.raises(RuntimeError, match="Podman"):
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
    assert runtime.runtime_command("podman") == ["podman", "--remote=false"]
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


def test_private_podman_entry_is_used_for_the_whole_lifecycle(config_path, tmp_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    config.update(runtime="podman", runtime_executable="/task/private runtime/bin/podman")
    config.pop("runtime_socket")
    config_path.write_text(yaml.safe_dump(config))
    execute = Mock(return_value=subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    monkeypatch.setattr(subprocess, "run", execute)
    monkeypatch.setenv("CONTAINER_HOST", "ssh://unavailable-platform-host")
    serving.pull_images(config_path)
    with RecordStore(tmp_path / "records") as store:
        env = TracedEnvironment(
            store=store,
            session_id="ses_private",
            runtime="podman",
            runtime_executable=config["runtime_executable"],
            image="test",
            create_workdir=True,
        )
        tool = env.executor.command("printf hello")
        env.cleanup()
    commands = [call.args[0] for call in execute.call_args_list]
    prefix = [config["runtime_executable"], "--remote=false"]
    assert all(command[:2] == prefix for command in [*commands, tool])
    assert [command[2] for command in commands] == ["pull", "run", "exec", "rm"]


def test_local_podman_denial_never_falls_back_to_host_docker(config_path, monkeypatch):
    config = yaml.safe_load(config_path.read_text())
    config.update(runtime="podman")
    config.pop("runtime_socket")
    config_path.write_text(yaml.safe_dump(config))
    execute = Mock(
        side_effect=subprocess.CalledProcessError(
            125,
            ["podman", "info"],
            stderr="cannot clone: Operation not permitted",
        )
    )
    monkeypatch.setattr(subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="cannot clone") as caught:
        runtime.check_runtime(config_path)
    assert "delegated cgroup" in str(caught.value)
    execute.assert_called_once()
    assert execute.call_args.args[0] == ["podman", "--remote=false", "info"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("cpu_quota_us", None),
        ("cpu_quota_us", -1),
        ("cpu_quota_us", 400000),
        ("memory_limit_bytes", None),
        ("memory_limit_bytes", 8589934592),
    ],
)
def test_preflight_rejects_ignored_or_incorrect_kernel_limits(
    config_path, kernel_probe, monkeypatch, field, value
):
    monkeypatch.setattr(
        subprocess,
        "run",
        Mock(
            return_value=subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps(kernel_probe | {field: value}),
                stderr="ignored resource flags",
            )
        ),
    )
    with pytest.raises(ValueError, match="cgroup"):
        runtime.check_runtime(config_path)


@pytest.mark.parametrize(
    "root,group", [("/outer", "/outer/task"), ("/outer", "/task"), ("/", "/task")]
)
def test_probe_resolves_private_and_outer_cgroup_v2_views(tmp_path, root, group):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot")
    for name in ["uid_map", "gid_map"]:
        (proc / "self" / name).write_text("0 100000 65536\n")
    mount = tmp_path / "cgroup mount"
    directory = mount / "task"
    directory.mkdir(parents=True)
    (directory / "cpu.max").write_text("200000 100000\n")
    (directory / "memory.max").write_text("4294967296\n")
    escaped = str(mount).replace(" ", "\\040")
    (proc / "self/mountinfo").write_text(f"1 0 0:1 {root} {escaped} rw - cgroup2 cgroup rw\n")
    (proc / "self/cgroup").write_text(f"0::{group}\n")
    probe = container_probe.snapshot(proc)
    runtime.validate_limits({"cpu_quota": 2, "memory_limit": "4g"}, probe)
    assert probe["cpu_cgroup"] == str(directory)
    assert probe["uid_map"] == "0 100000 65536"


def test_probe_resolves_split_cgroup_v1_controllers(tmp_path):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("boot")
    for name in ["uid_map", "gid_map"]:
        (proc / "self" / name).write_text("0 0 65536\n")
    mounts = []
    for controller, values in [
        ("cpu", {"cpu.cfs_quota_us": "200000", "cpu.cfs_period_us": "100000"}),
        ("memory", {"memory.limit_in_bytes": "4294967296"}),
    ]:
        mount = tmp_path / controller
        directory = mount / "task"
        directory.mkdir(parents=True)
        for name, value in values.items():
            (directory / name).write_text(value)
        mounts.append(f"1 0 0:1 /outer {mount} rw - cgroup cgroup rw,{controller}\n")
    (proc / "self/mountinfo").write_text("".join(mounts))
    (proc / "self/cgroup").write_text("2:cpu,cpuacct:/outer/task\n3:memory:/outer/task\n")
    probe = container_probe.snapshot(proc)
    runtime.validate_limits({"cpu_quota": 2, "memory_limit": "4g"}, probe)
    assert probe["memory_cgroup_type"] == "cgroup"
