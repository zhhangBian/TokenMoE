"""Build and run explicit model-server commands from the model profile YAML."""

import json
import os
import shlex
import subprocess
from pathlib import Path

from .static import load_yaml


def command(model_config, *, port=8000, capture=True):
    config = load_yaml(model_config)
    repo = Path(__file__).resolve().parents[2]
    binary = Path(os.environ.get("TOKENMOE_FORK_VENV", str(repo / "vllm/.venv"))) / "bin/vllm"
    args = [
        str(binary),
        "serve",
        config["model_path"],
        "--served-model-name",
        config["name"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--enable-prefix-caching",
        "--no-async-scheduling",
        "--generation-config",
        "vllm",
        "--enable-auto-tool-choice",
    ]
    if capture:
        args.append("--enable-return-routed-experts")
    for key, value in config["serve"].items():
        if value is None or value is False:
            continue
        args.append("--" + key.replace("_", "-"))
        if value is not True:
            args.append(json.dumps(value) if isinstance(value, (dict, list)) else str(value))
    return args


def serve(model_config, *, trace_dir, port=8000, capture=True):
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "vllm")
    if capture:
        env["TOKENMOE_TRACE_DIR"] = str(Path(trace_dir).resolve())
    else:
        env.pop("TOKENMOE_TRACE_DIR", None)
        env.pop("TOKENMOE_TRACE_ENGINE_DIR", None)
    args = command(model_config, port=port, capture=capture)
    print(shlex.join(args), flush=True)
    os.execvpe(args[0], args, env)


def pull_images(run_config):
    from .launcher import image_name
    from .static import load_benchmark

    config_path = Path(run_config).expanduser().resolve()
    config = load_yaml(config_path)
    if config["runtime"] not in {"docker", "podman"}:
        raise ValueError("Image pulls require docker or podman")
    dataset_path = (config_path.parent / Path(config["dataset_path"]).expanduser()).resolve()
    for item in load_benchmark(dataset_path, config["instances"]):
        args = [config["runtime"], "pull", image_name(item)]
        print(shlex.join(args), flush=True)
        subprocess.run(args, check=True)


def pilot(run_config, *, equivalence_record, port=8000, output_dir=None, startup_timeout=1200):
    """One engine lifetime per concurrency setting; stop it before finalization."""
    import sys

    from .equivalence import stop_server, wait_ready
    from .finalize import finalize
    from .jsonl import write_json
    from .launcher import run

    config_path = Path(run_config).resolve()
    config = load_yaml(config_path)
    model_path = (config_path.parent / config["model_config"]).resolve()
    base = Path(output_dir or config["output_dir"]).expanduser().resolve()
    if base.exists():
        raise FileExistsError(f"Pilot output must be new: {base}")
    gate = json.loads(Path(equivalence_record).read_text())
    if not gate.get("passed") or gate.get("num_prompts") != 50:
        raise ValueError("Pilot requires a passing 50-prompt equivalence record")
    base.mkdir(parents=True)
    reports = []
    for concurrency in config.get("concurrency_matrix", [config["num_concurrent_sessions"]]):
        trace = base / f"engine-N{concurrency}"
        trace.mkdir()
        args = [
            sys.executable,
            "-m",
            "tokenmoe_collect",
            "serve",
            "--model-config",
            str(model_path),
            "--trace-dir",
            str(trace),
            "--port",
            str(port),
        ]
        print(shlex.join(args), flush=True)
        with (trace / "server.log").open("w") as log:
            process = subprocess.Popen(
                args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                wait_ready(f"http://127.0.0.1:{port}", process, timeout=startup_timeout)
                engine = next(trace.glob("*/engine_meta.json")).parent
                meta = json.loads((engine / "engine_meta.json").read_text())
                if meta["engine_config_id"] != gate["engine_config_id"]:
                    raise ValueError("Equivalence record does not match this engine configuration")
                current = config | {
                    "model_config": str(model_path),
                    "engine_dir": str(engine),
                    "num_concurrent_sessions": concurrency,
                    "output_dir": str(base / f"N{concurrency}"),
                    "server_url": f"http://127.0.0.1:{port}/v1",
                }
                current.pop("concurrency_matrix", None)
                if current.get("mini_config"):
                    current["mini_config"] = str(
                        (config_path.parent / current["mini_config"]).resolve()
                    )
                current["dataset_path"] = str(
                    (config_path.parent / current["dataset_path"]).resolve()
                )
                snapshot = base / f"run-N{concurrency}.json"
                write_json(snapshot, current)
                destination = run(snapshot)
            finally:
                stop_server(process)
        write_json(destination / "static/equivalence" / f"{gate['engine_config_id']}.json", gate)
        reports.append({"run_dir": str(destination), "validation": finalize(destination)})
    write_json(base / "pilot_summary.json", reports)
    return reports
