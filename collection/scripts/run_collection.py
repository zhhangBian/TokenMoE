#!/usr/bin/env python3
"""Run a collection stage with explicit configs, GPU selection and output paths."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from tokenmoe_collect.static import load_yaml


def gate_path(path):
    path = path.expanduser().resolve()
    if path.is_dir():
        matches = sorted((path / "static/equivalence").glob("*.json"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one equivalence JSON under {path}, found {len(matches)}"
            )
        return matches[0]
    if not path.is_file():
        raise FileNotFoundError(f"Equivalence record does not exist: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["smoke", "equiv", "pull-images", "pilot", "full"], required=True
    )
    parser.add_argument("--model-config", type=Path, help="Required for smoke/equiv")
    parser.add_argument("--config", type=Path, help="Run config for pull-images/pilot/full")
    parser.add_argument("--dataset-dir", type=Path, help="Required for equivalence")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Required for smoke/equiv; optional run-config override otherwise",
    )
    parser.add_argument(
        "--equivalence-record", type=Path, help="Passing JSON or the equiv stage's output directory"
    )
    parser.add_argument(
        "--fork-venv", type=Path, help="Fork virtual environment containing bin/vllm"
    )
    parser.add_argument(
        "--gpus", help="Comma-separated visible GPU indices or UUIDs, for example 0,1"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--startup-timeout", type=int, default=1200, help="Seconds allowed for each server startup"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan without starting services or writing data",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout must be positive")
    managed_run = args.stage in {"pilot", "full"}
    needs_run_config = managed_run or args.stage == "pull-images"
    config = None
    if needs_run_config:
        if args.model_config is not None or args.dataset_dir is not None:
            parser.error("For this stage, model and dataset paths come from --config")
        if args.config is None:
            parser.error(f"--config is required for {args.stage}")
        args.config = args.config.expanduser().resolve()
        config = load_yaml(args.config)
        model_config = (args.config.parent / config["model_config"]).resolve()
        dataset = (args.config.parent / config["dataset_path"]).resolve()
        output = (args.output_dir or Path(config["output_dir"])).expanduser().resolve()
    else:
        if args.config is not None:
            parser.error("--config is only used for pull-images/pilot/full")
        if args.model_config is None or args.output_dir is None:
            parser.error("--model-config and --output-dir are required for smoke/equiv")
        model_config = args.model_config.expanduser().resolve()
        output = args.output_dir.expanduser().resolve()
        dataset = args.dataset_dir.expanduser().resolve() if args.dataset_dir else None
    if args.stage == "equiv" and dataset is None:
        parser.error("--dataset-dir is required for equivalence")
    if managed_run and args.equivalence_record is None:
        parser.error("--equivalence-record is required for pilot/full")
    model = load_yaml(model_config)
    if args.stage != "pull-images":
        if not args.gpus or args.fork_venv is None:
            parser.error("--gpus and --fork-venv are required for GPU stages")
        devices = [part.strip() for part in args.gpus.split(",")]
        if any(not part for part in devices) or len(set(devices)) != len(devices):
            parser.error("--gpus must contain distinct, nonempty GPU identifiers")
        required = model["serve"]["tensor_parallel_size"]
        if len(devices) != required:
            parser.error(
                f"This model config needs {required} GPUs, got {len(devices)}; use matching --tp when preparing configs"
            )
        args.fork_venv = args.fork_venv.expanduser().resolve()
        if not args.dry_run and not (args.fork_venv / "bin/vllm").is_file():
            parser.error(f"Fork executable is missing: {args.fork_venv / 'bin/vllm'}")
        os.environ["TOKENMOE_FORK_VENV"] = str(args.fork_venv)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    plan = {
        "stage": args.stage,
        "model_config": str(model_config),
        "model_path": model["model_path"],
        "dataset_dir": str(dataset) if dataset else None,
        "output_dir": str(output),
        "gpus": args.gpus,
        "fork_venv": str(args.fork_venv) if args.fork_venv else None,
        "port": args.port,
        "startup_timeout": args.startup_timeout,
        "equivalence_record": str(args.equivalence_record) if args.equivalence_record else None,
        "dry_run": args.dry_run,
    }
    if config is not None:
        plan.update(
            instances=len(config["instances"]),
            runtime=config["runtime"],
            concurrency=config.get("concurrency_matrix", [config["num_concurrent_sessions"]]),
        )
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    if args.stage == "pull-images":
        from tokenmoe_collect.serving import pull_images

        pull_images(args.config)
        return 0
    if managed_run:
        from tokenmoe_collect.serving import pilot

        reports = pilot(
            args.config,
            equivalence_record=gate_path(args.equivalence_record),
            port=args.port,
            output_dir=output,
            startup_timeout=args.startup_timeout,
        )
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0 if all(report["validation"]["passed"] for report in reports) else 1
    if args.stage == "equiv":
        from tokenmoe_collect.equivalence import equivalence

        result = equivalence(
            model_config, dataset, output, port=args.port, startup_timeout=args.startup_timeout
        )
    else:
        from tokenmoe_collect.equivalence import stop_server, wait_ready
        from tokenmoe_collect.smoke import smoke

        output.mkdir(parents=True, exist_ok=False)
        trace = output / "engine"
        command = [
            sys.executable,
            "-m",
            "tokenmoe_collect",
            "serve",
            "--model-config",
            str(model_config),
            "--trace-dir",
            str(trace),
            "--port",
            str(args.port),
        ]
        print(shlex.join(command), flush=True)
        with (output / "server.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                wait_ready(f"http://127.0.0.1:{args.port}", process, timeout=args.startup_timeout)
                result = smoke(
                    model_config,
                    trace,
                    url=f"http://127.0.0.1:{args.port}",
                    timeout=args.startup_timeout,
                )
            finally:
                stop_server(process)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
