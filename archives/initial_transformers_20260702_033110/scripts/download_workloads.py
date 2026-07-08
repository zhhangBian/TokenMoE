#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tokenmoe.schema import write_workload_jsonl
from tokenmoe.workloads import WORKFLOW_TEMPLATES, generate_workloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized TokenMoE workloads.")
    parser.add_argument("--output", default="data/workloads/agent_workloads.jsonl")
    parser.add_argument("--per-workflow", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--workflow",
        action="append",
        choices=sorted(WORKFLOW_TEMPLATES),
        help="Workflow to include. May be repeated. Defaults to all workflows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = generate_workloads(
        per_workflow=args.per_workflow,
        seed=args.seed,
        workflows=args.workflow,
    )
    output = Path(args.output)
    write_workload_jsonl(records, output)
    print(f"wrote {len(records)} workload records to {output}")
    by_workflow: dict[str, int] = {}
    for record in records:
        by_workflow[record.workflow] = by_workflow.get(record.workflow, 0) + 1
    for workflow, count in sorted(by_workflow.items()):
        print(f"{workflow}: {count}")


if __name__ == "__main__":
    main()
