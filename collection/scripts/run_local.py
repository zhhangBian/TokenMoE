#!/usr/bin/env python3
"""Run or resume the bounded local Qwen runtime-collection protocol."""

import argparse
import json
from pathlib import Path

from tokenmoe_collect.local import LocalCollection, health_snapshot, make_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--pilot-config", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--fork-venv", type=Path)
    parser.add_argument("--gpus", default="1,2")
    parser.add_argument("--port", type=int, default=18241)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=[
            "preflight",
            "smoke",
            "equiv",
            "tool-smoke",
            "pilot",
            "collect",
            "overhead",
            "report",
        ],
        default=[
            "preflight",
            "smoke",
            "equiv",
            "tool-smoke",
            "pilot",
            "collect",
            "overhead",
            "report",
        ],
    )
    parser.add_argument("--review-interval", type=int, default=1800)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only:
        print(json.dumps(health_snapshot(args.experiment_dir), indent=2))
        return 0
    for name in ["config", "pilot_config", "model_config", "fork_venv"]:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    devices = args.gpus.split(",")
    if (
        args.review_interval <= 0
        or not 1 <= args.port <= 65535
        or any(not d for d in devices)
        or len(set(devices)) != len(devices)
    ):
        parser.error("Use positive review intervals, a valid port, and distinct GPU identifiers")
    plan = make_plan(args.config, args.pilot_config, args.model_config)
    print(
        json.dumps(
            {
                "plan": plan,
                "stages": args.stages,
                "gpus": devices,
                "fork_venv": str(args.fork_venv.resolve()),
                "dry_run": args.dry_run,
            },
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0
    LocalCollection(
        args.experiment_dir,
        plan,
        fork_venv=args.fork_venv,
        devices=devices,
        port=args.port,
        review_interval=args.review_interval,
    ).run(args.stages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
