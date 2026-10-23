"""CPU command entry points. Importing this module never imports torch or vLLM."""

import argparse
import json
import sys
from pathlib import Path

from tokenmoe_collect.assets import sha256_file
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
    plan.add_argument("--confirmation-split", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    arrivals = commands.add_parser("arrivals")
    arrivals.add_argument("--inputs", type=Path, required=True)
    arrivals.add_argument("--output", type=Path, required=True)
    resources = commands.add_parser("resources")
    resources.add_argument("--pid", type=int, required=True)
    resources.add_argument("--devices", type=int, nargs="+", required=True)
    resources.add_argument("--output", type=Path, required=True)
    costs = commands.add_parser("fit-costs")
    costs.add_argument("--native", type=Path, required=True)
    costs.add_argument("--cache", type=Path, required=True)
    costs.add_argument("--inputs", type=Path, required=True)
    costs.add_argument("--output", type=Path, required=True)
    control = commands.add_parser("control")
    for name in (
        "predictor",
        "inputs",
        "cost-profile",
        "agent-run",
        "observations",
        "workers",
        "output",
    ):
        control.add_argument("--" + name, type=Path, required=True)
    control.add_argument("--duration", type=float, default=60)
    control.add_argument(
        "--policy",
        choices=[
            "global",
            "stage",
            "previous",
            "ewma",
            "previous+global",
            "previous+stage",
            "previous+tool",
        ],
        default="previous+tool",
    )
    metrics = commands.add_parser("diagnostic-metrics")
    metrics.add_argument("--run", type=Path, required=True)
    metrics.add_argument("--output", type=Path, required=True)
    agent_metrics = commands.add_parser("agent-metrics")
    agent_metrics.add_argument("--run", type=Path, required=True)
    agent_metrics.add_argument("--output", type=Path, required=True)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("--demands", type=Path, required=True)
    replay_parser.add_argument("--cost-profile", type=Path, required=True)
    replay_parser.add_argument("--capacity", type=int, required=True)
    replay_parser.add_argument("--policy", choices=["lru", "lfu"], default="lru")
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.add_argument(
        "--timeline-kind",
        choices=["neutral", "contains-demand-stalls"],
        default="contains-demand-stalls",
    )
    simulation = commands.add_parser("simulate")
    simulation.add_argument("--graph", type=Path, required=True)
    simulation.add_argument("--cost-profile", type=Path, required=True)
    simulation.add_argument("--capacity", type=int, required=True)
    simulation.add_argument("--policy", choices=["lru", "lfu"], default="lru")
    simulation.add_argument("--output", type=Path, required=True)
    calibration = commands.add_parser("calibrate")
    for name in ("run", "cost-profile", "protocol", "output"):
        calibration.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    command = ["tokenmoe-opt", *(sys.argv[1:] if argv is None else argv)]
    if args.command == "calibrate":
        from .calibration import validate_run

        print(
            validate_run(args.run, args.cost_profile, args.protocol, args.output, command=command)
        )
        return
    if args.command == "simulate":
        from .replay import profile_costs
        from .timing import simulate

        graph = read_manifest(args.graph, "input")
        profile = read_manifest(args.cost_profile, "cost-profile")
        if graph["identity"] != profile["identity"] or not profile.get("controller", {}).get(
            "measured"
        ):
            raise ValueError("Timing diagnostics require matching measured cost-profile identity")
        out = Output(args.output, kind="result", identity=profile["identity"], command=command)
        try:
            out.write("graph.json", graph)
            out.write(
                "inputs.json",
                {
                    "sources": {
                        str(p.resolve()): sha256_file(p) for p in (args.graph, args.cost_profile)
                    },
                    "cost_profile": profile,
                },
            )
            out.write(
                "metrics.json",
                simulate(
                    graph["nodes"],
                    capacity=args.capacity,
                    costs=profile_costs(profile),
                    policy=args.policy,
                    native_service=profile.get("native_service"),
                    initial_snapshot=graph.get("initial_snapshot"),
                ),
            )
            out.finish(evidence="uncalibrated-causal-timing-diagnostic", calibrated_timing=False)
        except Exception as exc:
            out.fail(exc)
            raise
        print(out.path)
        return
    if args.command == "agent-metrics":
        from .agent_metrics import extract

        print(extract(args.run, args.output, command=command))
        return
    if args.command == "fit-costs":
        from .costs import fit

        print(fit(args.native, args.cache, args.inputs, args.output, command=command))
        return
    if args.command == "control":
        from .control_run import run

        print(
            run(
                args.predictor,
                args.inputs,
                args.cost_profile,
                args.agent_run,
                args.observations,
                args.workers,
                args.output,
                command=command,
                duration=args.duration,
                policy=args.policy,
            )
        )
        return
    if args.command == "diagnostic-metrics":
        from .metrics import extract_run

        record = json.loads((args.run / "result.json").read_text())
        out = Output(
            args.output,
            kind="result",
            identity={
                "scope": "qualification-plain-prompt",
                "model_path": record["config"]["model"],
                "template": None,
            },
            command=command,
        )
        try:
            out.write("metrics.json", extract_run(args.run))
            out.finish(
                evidence="live-diagnostic-not-controlled",
                source_run=str(args.run.resolve()),
                source_hash=sha256_file(args.run / "result.json"),
            )
        except Exception as exc:
            out.fail(exc)
            raise
        print(out.path)
        return
    if args.command == "resources":
        from .resources import monitor

        monitor(args.pid, args.devices, args.output)
        return
    if args.command == "arrivals":
        from .arrival_eval import evaluate

        print(evaluate(args.inputs, args.output, command=command))
        return
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
        from .protocol import local_protocol, training_sampling

        inputs = read_manifest(args.inputs, "input")
        # Split artifact has the authoritative task groups, not paths/replica IDs.
        splits = read_manifest(args.inputs.parent / "splits.json", "split")
        if splits["identity"] != inputs["identity"]:
            raise ValueError("Split and input identities differ")
        confirmation = read_manifest(args.confirmation_split, "split")
        if confirmation["identity"] != inputs["identity"]:
            raise ValueError("Confirmation and input identities differ")
        protocol = local_protocol(
            inputs["identity"],
            splits["groups"],
            confirmation["groups"],
            sampling=training_sampling(inputs),
        )
        protocol["confirmation_source"] = {
            "path": str(args.confirmation_split.resolve()),
            "sha256": sha256_file(args.confirmation_split),
            "selection": confirmation.get("selection"),
        }
        out = Output(args.output, kind="experiment", identity=inputs["identity"], command=command)
        out.write("protocol.json", protocol)
        out.finish(protocol_status="preregistered-unexecuted")
    else:
        from .replay import CopyCost, profile_costs, replay

        profile = read_manifest(args.cost_profile, "cost-profile")
        cost = CopyCost(**profile["copy"])
        out = Output(args.output, kind="result", identity=profile["identity"], command=command)
        try:
            out.write(
                "inputs.json",
                manifest(
                    "input",
                    profile["identity"],
                    sources={
                        str(path.resolve()): sha256_file(path)
                        for path in (args.demands, args.cost_profile)
                    },
                    cost_profile=profile,
                ),
            )
            result = replay(
                read_jsonl(args.demands),
                capacity=args.capacity,
                cost=cost,
                policy=args.policy,
                rank_costs=profile_costs(profile),
                timeline_kind=args.timeline_kind,
            )
            out.write("metrics.json", manifest("result", profile["identity"], **result))
            out.finish(evidence=result["evidence"])
        except Exception as exc:
            out.fail(exc)
            raise
    print(out.path)
