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
    prediction_rows = metrics["prediction"].get("results", [])
    routesig_2x = next(
        (
            row
            for row in prediction_rows
            if row["predictor"] == "routesig" and row["budget"] == "2x"
        ),
        None,
    )
    if routesig_2x:
        print(
            "routesig 2x expert_label_hit_rate: "
            f"{routesig_2x['expert_label_hit_rate']:.4f}"
        )
    print(f"wrote {args.analysis_dir}/locality_report.md")
    print(f"wrote {args.report_dir}/tokenmoe_report.md")


if __name__ == "__main__":
    main()
