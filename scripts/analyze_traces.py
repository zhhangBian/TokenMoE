#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tokenmoe.reporting import write_reports
from tokenmoe.schema import read_workload_jsonl
from tokenmoe.trace import read_trace_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze TokenMoE traces and write reports.")
    parser.add_argument("--traces", default="data/traces/tokenmoe_traces.jsonl")
    parser.add_argument("--workload", default="data/workloads/agent_workloads.jsonl")
    parser.add_argument("--analysis-dir", default="analysis")
    parser.add_argument("--report-dir", default="reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traces = read_trace_jsonl(args.traces)
    workloads = read_workload_jsonl(args.workload)
    metrics = write_reports(
        records=traces,
        workloads=workloads,
        output_dir=args.analysis_dir,
        report_dir=args.report_dir,
    )
    print(f"records: {len(traces)}")
    best_routesig = max(
        (
            row
            for row in metrics["locality"]["top_m_hit_rate"]
            if row["predictor"] == "routesig"
        ),
        key=lambda row: row["hit_rate"],
    )
    print(f"best routesig hit_rate: {best_routesig['hit_rate']:.4f}")
    print(f"wrote {args.analysis_dir}/locality_report.md")
    print(f"wrote {args.report_dir}/tokenmoe_report.md")


if __name__ == "__main__":
    main()
