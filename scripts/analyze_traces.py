#!/usr/bin/env python3
"""Evaluate TokenMoE predictors on one homogeneous vLLM trace file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenmoe.evaluation import BOOTSTRAP_RESAMPLES, evaluate_traces
from tokenmoe.trace import read_trace_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temporal-window", type=int, default=64)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_traces(
        read_trace_jsonl(args.traces),
        temporal_window=args.temporal_window,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote evaluation to {output}")


if __name__ == "__main__":
    main()
