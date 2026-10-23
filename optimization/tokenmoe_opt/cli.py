"""CPU command entry points. Importing this module never imports torch or vLLM."""

import argparse
import sys
from pathlib import Path

from tokenmoe_collect.jsonl import read_jsonl

from .artifacts import Output, manifest, read_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tokenmoe-opt")
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze")
    analysis.add_argument("--run", type=Path, action="append", required=True)
    analysis.add_argument("--output", type=Path, required=True)
    analysis.add_argument(
        "--label-version",
        choices=["command-outcome-v1", "shell-command-v2"],
        default="shell-command-v2",
    )
    analysis.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 96, 128])
    analysis.add_argument("--tune", action="store_true")
    plan = commands.add_parser("plan")
    plan.add_argument("--inputs", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--demands", type=Path, required=True)
    replay_parser.add_argument("--cost-profile", type=Path, required=True)
    replay_parser.add_argument("--capacity", type=int, required=True)
    replay_parser.add_argument("--policy", choices=["lru", "lfu"], default="lru")
    replay_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    command = ["tokenmoe-opt", *(sys.argv[1:] if argv is None else argv)]
    if args.command == "analyze":
        from .analysis import analyze

        print(
            analyze(
                args.run,
                args.output,
                command=command,
                budgets=args.budgets,
                label_version=args.label_version,
                tune=args.tune,
            )
        )
        return
    if args.command == "plan":
        from .protocol import local_protocol

        inputs = read_manifest(args.inputs, "input")
        # Split artifact has the authoritative task groups, not paths/replica IDs.
        splits = read_manifest(args.inputs.parent / "splits.json", "split")
        if splits["identity"] != inputs["identity"]:
            raise ValueError("Split and input identities differ")
        out = Output(args.output, kind="experiment", identity=inputs["identity"], command=command)
        out.write("protocol.json", local_protocol(inputs["identity"], splits["groups"]))
        out.finish(protocol_status="preregistered-unexecuted")
    else:
        from .replay import CopyCost, replay

        profile = read_manifest(args.cost_profile, "cost-profile")
        cost = CopyCost(**profile["copy"])
        out = Output(args.output, kind="result", identity=profile["identity"], command=command)
        try:
            result = replay(
                read_jsonl(args.demands), capacity=args.capacity, cost=cost, policy=args.policy
            )
            out.write("metrics.json", manifest("result", profile["identity"], **result))
            out.finish(evidence=result["evidence"])
        except Exception as exc:
            out.fail(exc)
            raise
    print(out.path)
