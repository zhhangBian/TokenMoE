"""Portable handoff scripts: no downloads, containers or GPUs during tests."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name):
    spec = importlib.util.spec_from_file_location(f"handoff_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def prepared(tmp_path):
    output = tmp_path / "configuration files"
    script("prepare_configs").main(
        [
            "--model-root",
            str(tmp_path / "weights"),
            "--dataset-dir",
            str(tmp_path / "benchmark"),
            "--output-dir",
            str(output),
            "--run-root",
            str(tmp_path / "runs"),
            "--runtime",
            "docker",
            "--concurrency",
            "2",
            "3",
            "--base-seed",
            "17",
            "--step-limit",
            "42",
            "--time-limit-seconds",
            "300",
            "--model-path",
            f"dots3-note={tmp_path / 'custom weights'}",
            "--tp",
            "dots3-note=4",
        ]
    )
    return output


def test_generated_configs_preserve_workloads_and_resolve_machine_parameters(prepared, tmp_path):
    assert len(list(prepared.rglob("*.yaml"))) == 9
    for path in (prepared / "runs").glob("*.yaml"):
        content = path.read_text()
        cfg = yaml.safe_load(content)
        assert "${" not in content and "/home/youwei" not in content
        assert Path(cfg["model_config"]).is_file()
        assert cfg["dataset_path"] == str(tmp_path / "benchmark")
        assert cfg["runtime"] == "docker" and cfg["concurrency_matrix"] == [2, 3]
        assert cfg["num_concurrent_sessions"] == 2 and cfg["base_seed"] == 17
        assert cfg["step_limit"] == 42 and cfg["time_limit_seconds"] == 300
        assert len(cfg["instances"]) == (500 if path.stem.endswith("-full") else 20)
        assert Path(cfg["output_dir"]) == tmp_path / "runs" / path.stem
    dots = yaml.safe_load((prepared / "models/dots3-note.yaml").read_text())
    assert dots["model_path"] == str(tmp_path / "custom weights")
    assert dots["serve"]["tensor_parallel_size"] == 4


def test_prepare_dry_run_does_not_write_and_refuses_existing_configuration(tmp_path, prepared):
    module = script("prepare_configs")
    args = [
        "--model-root",
        str(tmp_path / "weights"),
        "--dataset-dir",
        str(tmp_path / "data"),
        "--run-root",
        str(tmp_path / "run"),
        "--output-dir",
        str(tmp_path / "new"),
    ]
    assert module.main([*args, "--dry-run"]) == 0
    assert not (tmp_path / "new").exists()
    before = {p: p.read_bytes() for p in prepared.rglob("*.yaml")}
    with pytest.raises(SystemExit):
        module.main([*args[:-1], str(prepared)])
    assert before == {p: p.read_bytes() for p in prepared.rglob("*.yaml")}


def test_download_routes_revisions_endpoint_and_dataset_filters(tmp_path, monkeypatch):
    download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setenv("HF_ENDPOINT", "https://original.invalid")
    args = [
        "--models",
        "gpt-oss-120b",
        "--model-root",
        str(tmp_path / "models"),
        "--dataset-dir",
        str(tmp_path / "data"),
        "--model-revision",
        "gpt-oss-120b=abc123",
        "--dataset-revision",
        "def456",
        "--endpoint",
        "https://mirror.invalid",
        "--max-workers",
        "3",
    ]
    module = script("download_assets")
    assert module.main([*args, "--dry-run"]) == 0
    download.assert_not_called()
    assert not (tmp_path / "models").exists()
    assert module.main(args) == 0
    first, second = download.call_args_list
    assert first.kwargs["repo_id"] == "openai/gpt-oss-120b"
    assert first.kwargs["revision"] == "abc123"
    assert first.kwargs["local_dir"] == str(tmp_path / "models/openai/gpt-oss-120b")
    assert second.kwargs["repo_type"] == "dataset" and second.kwargs["revision"] == "def456"
    assert second.kwargs["allow_patterns"] == ["*.parquet"]
    assert (
        second.kwargs["endpoint"] == "https://mirror.invalid" and second.kwargs["max_workers"] == 3
    )


def test_full_runner_forwards_explicit_gpu_port_timeout_and_output(prepared, tmp_path, monkeypatch):
    from tokenmoe_collect import serving

    runner = script("run_collection")
    fork = tmp_path / "fork-venv"
    (fork / "bin").mkdir(parents=True)
    (fork / "bin/vllm").touch()
    gate_dir = tmp_path / "equiv/static/equivalence"
    gate_dir.mkdir(parents=True)
    gate = gate_dir / "engine.json"
    gate.write_text("{}")
    execute = Mock(return_value=[{"validation": {"passed": True}}])
    monkeypatch.setattr(serving, "pilot", execute)
    monkeypatch.setenv("TOKENMOE_FORK_VENV", "original")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    args = [
        "--stage",
        "full",
        "--config",
        str(prepared / "runs/gpt-oss-120b-full.yaml"),
        "--fork-venv",
        str(fork),
        "--gpus",
        "1,3",
        "--port",
        "9123",
        "--startup-timeout",
        "37",
        "--equivalence-record",
        str(tmp_path / "equiv"),
        "--output-dir",
        str(tmp_path / "override"),
    ]
    assert runner.main([*args, "--dry-run"]) == 0
    execute.assert_not_called()
    assert not (tmp_path / "override").exists()
    assert runner.main(args) == 0
    assert execute.call_args.kwargs == {
        "equivalence_record": gate,
        "port": 9123,
        "output_dir": tmp_path / "override",
        "startup_timeout": 37,
    }
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1,3"
    assert os.environ["TOKENMOE_FORK_VENV"] == str(fork)
    with pytest.raises(SystemExit):
        runner.main([*args, "--gpus", "0"])


def test_smoke_cleans_up_owned_server_when_startup_fails(prepared, tmp_path, monkeypatch):
    from tokenmoe_collect import equivalence

    runner = script("run_collection")
    fork = tmp_path / "venv"
    (fork / "bin").mkdir(parents=True)
    (fork / "bin/vllm").touch()
    process = object()
    monkeypatch.setattr(runner.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setenv("TOKENMOE_FORK_VENV", "original")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")
    wait = Mock(side_effect=TimeoutError("not ready"))
    stop = Mock()
    monkeypatch.setattr(equivalence, "wait_ready", wait)
    monkeypatch.setattr(equivalence, "stop_server", stop)
    with pytest.raises(TimeoutError):
        runner.main(
            [
                "--stage",
                "smoke",
                "--model-config",
                str(prepared / "models/gpt-oss-120b.yaml"),
                "--output-dir",
                str(tmp_path / "smoke"),
                "--fork-venv",
                str(fork),
                "--gpus",
                "0,1",
                "--startup-timeout",
                "31",
            ]
        )
    stop.assert_called_once_with(process)
    assert wait.call_args.kwargs["timeout"] == 31


def test_summary_checks_thresholds_without_claiming_manual_checks_passed(tmp_path, capsys):
    module = script("summarize_runs")
    runs = []
    for n in [1, 4]:
        root = tmp_path / f"N{n}"
        (root / "runtime").mkdir(parents=True)
        (root / "run_summary.json").write_text(
            json.dumps({"sessions": [{"status": "succeeded"}] * 9 + [{"status": "timeout"}]})
        )
        (root / "validation.json").write_text('{"passed":true}')
        (root / "alignment.json").write_text('{"location_rate":0.995}')
        runs.append({"run_dir": f"/original-host/N{n}"})
    (tmp_path / "pilot_summary.json").write_text(json.dumps(runs))
    assert module.main(["--run-dir", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["manual_checks_required"]
    assert len(report["reports"][0]["runs"]) == 2
    assert module.main(["--run-dir", str(tmp_path), "--min-success-rate", "0.95"]) == 1
    (tmp_path / "pilot_summary.json").unlink()
    assert module.main(["--run-dir", str(tmp_path)]) == 1


def test_image_pull_resolves_dataset_relative_to_its_config(tmp_path, monkeypatch):
    from tokenmoe_collect import serving, static

    config_path = tmp_path / "configs/run.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {"runtime": "podman", "dataset_path": "../dataset", "instances": ["test__repo-1"]}
        )
    )
    benchmark = Mock(return_value=[{"instance_id": "test__repo-1"}])
    execute = Mock()
    monkeypatch.setattr(static, "load_benchmark", benchmark)
    monkeypatch.setattr(serving.subprocess, "run", execute)
    serving.pull_images(config_path)
    benchmark.assert_called_once_with(tmp_path / "dataset", ["test__repo-1"])
    assert execute.call_args.args[0] == [
        "podman",
        "pull",
        "docker.io/swebench/sweb.eval.x86_64.test_1776_repo-1:latest",
    ]
