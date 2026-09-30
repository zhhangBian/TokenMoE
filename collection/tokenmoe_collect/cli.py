"""Command-line entry points for collection and offline checks."""

import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tokenmoe-collect")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "pull-images"]:
        commands.add_parser(name).add_argument("--config", required=True)
    for name in ["finalize", "validate"]:
        commands.add_parser(name).add_argument("run_dir")
    estimate = commands.add_parser("estimate")
    estimate.add_argument("--model-config", required=True)
    estimate.add_argument("--run-dir")
    estimate.add_argument("--sessions", type=int, default=500)
    serve = commands.add_parser("serve")
    serve.add_argument("--model-config", required=True)
    serve.add_argument("--trace-dir", required=True)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--capture-off", action="store_true")
    equiv = commands.add_parser("equiv")
    equiv.add_argument("--model-config", required=True)
    equiv.add_argument("--dataset-path", required=True)
    equiv.add_argument("--output-dir", required=True)
    equiv.add_argument("--port", type=int, default=8001)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--model-config", required=True)
    smoke_parser.add_argument("--trace-dir", required=True)
    smoke_parser.add_argument("--url", default="http://127.0.0.1:8000")
    pilot_parser = commands.add_parser("pilot")
    pilot_parser.add_argument("--config", required=True)
    pilot_parser.add_argument("--equivalence-record", required=True)
    pilot_parser.add_argument("--port", type=int, default=8000)
    commands.add_parser("profile").add_argument("run_dir")
    args = parser.parse_args(argv)
    if args.command == "profile":
        from pathlib import Path

        print((Path(args.run_dir) / "static/model_profiles.jsonl").read_text(), end="")
        return 0
    if args.command == "pilot":
        from .serving import pilot

        reports = pilot(args.config, equivalence_record=args.equivalence_record, port=args.port)
        print(json.dumps(reports, indent=2))
        return 0 if all(r["validation"]["passed"] for r in reports) else 1
    if args.command == "smoke":
        from .smoke import smoke

        print(json.dumps(smoke(args.model_config, args.trace_dir, url=args.url), indent=2))
        return 0
    if args.command == "equiv":
        from .equivalence import equivalence

        result = equivalence(args.model_config, args.dataset_path, args.output_dir, port=args.port)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "run":
        from .launcher import run

        print(run(args.config))
        return 0
    if args.command == "pull-images":
        from .serving import pull_images

        pull_images(args.config)
        return 0
    if args.command == "serve":
        from .serving import serve

        serve(
            args.model_config,
            trace_dir=args.trace_dir,
            port=args.port,
            capture=not args.capture_off,
        )
    if args.command in {"finalize", "validate"}:
        if args.command == "finalize":
            from .finalize import finalize as operation
        else:
            from .validate import validate as operation
        result = operation(args.run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    if args.command == "estimate":
        from pathlib import Path

        from .estimate import formula, measured
        from .static import load_yaml

        model = load_yaml(args.model_config)
        hf_path = Path(model["model_path"]) / "config.json"
        hf = json.loads(hf_path.read_text()) if hf_path.exists() else model["facts"]
        facts = model.get("facts", {})
        kwargs = dict(
            layers=facts.get("moe_layers", hf.get("num_hidden_layers")),
            top_k=facts.get("top_k", hf.get("num_experts_per_tok")),
            experts=facts.get(
                "num_routed_experts", hf.get("num_experts", hf.get("n_routed_experts"))
            ),
        )
        result = (
            measured(args.run_dir, **kwargs)
            if args.run_dir
            else formula(**kwargs, sessions=args.sessions)
        )
        print(json.dumps(result, indent=2))
        return 0
    return 1
