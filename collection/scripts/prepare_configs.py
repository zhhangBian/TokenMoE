#!/usr/bin/env python3
"""Generate explicit model and run YAMLs without machine-specific placeholders."""

import argparse
import copy
import json
from pathlib import Path

import yaml

MODEL_NAMES = ("gpt-oss-120b", "deepseek-v4-flash", "dots3-note")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def overrides(parser, settings, models, convert, option):
    result = {}
    for setting in settings:
        model, separator, value = setting.partition("=")
        if not separator or model not in models or not value or model in result:
            parser.error(f"Invalid or repeated {option}: {setting}")
        try:
            result[model] = convert(value)
        except (ValueError, argparse.ArgumentTypeError) as exc:
            parser.error(f"Invalid {option} {setting}: {exc}")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--stages", nargs="+", choices=["pilot", "full"], default=["pilot", "full"])
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for generated configurations"
    )
    parser.add_argument(
        "--run-root", type=Path, required=True, help="Parent for <model>-<stage> outputs"
    )
    parser.add_argument("--runtime", choices=["podman", "docker"], default="podman")
    parser.add_argument("--model-path", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--tp", action="append", default=[], metavar="MODEL=COUNT")
    parser.add_argument("--max-model-len", type=positive)
    parser.add_argument("--concurrency", nargs="+", type=positive)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--step-limit", type=positive)
    parser.add_argument("--time-limit-seconds", type=positive)
    parser.add_argument("--tool-timeout", type=positive)
    parser.add_argument("--request-timeout", type=positive)
    parser.add_argument("--cpu-quota", type=float)
    parser.add_argument("--memory-limit", help="Container memory limit, for example 4g")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_quota is not None and args.cpu_quota <= 0:
        parser.error("--cpu-quota must be positive")
    if args.concurrency and len(set(args.concurrency)) != len(args.concurrency):
        parser.error("--concurrency must not contain duplicates")
    model_paths = overrides(
        parser,
        args.model_path,
        args.models,
        lambda p: str(Path(p).expanduser().resolve()),
        "--model-path",
    )
    tensor_parallel = overrides(parser, args.tp, args.models, positive, "--tp")
    output = args.output_dir.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    dataset = args.dataset_dir.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Configuration output directory must be new or empty: {output}")
    generated = {}
    for model in dict.fromkeys(args.models):
        profile = yaml.safe_load((CONFIG_ROOT / "models" / f"{model}.yaml").read_text())
        profile["model_path"] = model_paths.get(
            model, str((args.model_root.expanduser().resolve() / profile["model_id"]).resolve())
        )
        if model in tensor_parallel:
            profile["serve"]["tensor_parallel_size"] = tensor_parallel[model]
        if args.max_model_len is not None:
            profile["serve"]["max_model_len"] = args.max_model_len
        model_file = output / "models" / f"{model}.yaml"
        generated[model_file] = profile
        for stage in dict.fromkeys(args.stages):
            cfg = copy.deepcopy(
                yaml.safe_load((CONFIG_ROOT / "runs" / f"{stage}.yaml").read_text())
            )
            cfg.update(
                model_config=str(model_file),
                dataset_path=str(dataset),
                runtime=args.runtime,
                output_dir=str(run_root / f"{model}-{stage}"),
            )
            # The managed runner supplies the actual engine directory and endpoint.
            cfg.pop("engine_dir", None)
            cfg.pop("server_url", None)
            if args.concurrency is not None:
                cfg["concurrency_matrix"] = args.concurrency
            cfg["num_concurrent_sessions"] = cfg["concurrency_matrix"][0]
            for key in [
                "base_seed",
                "step_limit",
                "time_limit_seconds",
                "tool_timeout",
                "request_timeout",
                "cpu_quota",
                "memory_limit",
            ]:
                value = getattr(args, key)
                if value is not None:
                    cfg[key] = value
            generated[output / "runs" / f"{model}-{stage}.yaml"] = cfg
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "configurations": {
                    str(path): {
                        key: value[key]
                        for key in [
                            "model_path",
                            "serve",
                            "model_config",
                            "dataset_path",
                            "output_dir",
                            "runtime",
                            "concurrency_matrix",
                            "base_seed",
                            "step_limit",
                            "time_limit_seconds",
                        ]
                        if key in value
                    }
                    | ({"instance_count": len(value["instances"])} if "instances" in value else {})
                    for path, value in generated.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.dry_run:
        for path, value in generated.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as destination:
                yaml.safe_dump(value, destination, sort_keys=False, allow_unicode=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
