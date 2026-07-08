#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tokenmoe.metrics import evaluate_predictors
from tokenmoe.routesig import RouteSigStore
from tokenmoe.trace import read_trace_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RouteSig fallback/confidence behavior.")
    parser.add_argument("--traces", default="data/traces/tokenmoe_traces.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_trace_jsonl(args.traces)
    if len(records) < 4:
        raise SystemExit("need at least 4 trace records for validation")
    store = RouteSigStore(top_m=4, min_samples=8)
    for record in records[: len(records) // 2]:
        store.update_trace(record)
    cold = records[-1].metadata
    sig = store.lookup(cold, 0)
    if sig is None:
        raise SystemExit("RouteSig lookup did not fall back to broader statistics")
    if not (0.0 <= sig.confidence <= 1.0):
        raise SystemExit(f"invalid confidence {sig.confidence}")
    results = evaluate_predictors(records)
    if not results:
        raise SystemExit("no predictor results")
    print("RouteSig validation passed")
    print(f"fallback signature key={sig.key} layer={sig.layer_id} confidence={sig.confidence:.3f}")


if __name__ == "__main__":
    main()
