"""Compare source-stable short backend costs without claiming agent outcomes."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from tokenmoe_collect.assets import sha256_file

from tokenmoe_opt.artifacts import Output
from tokenmoe_opt.metrics import extract_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [(path, json.loads((path / "result.json").read_text())) for path in args.run]
    reference = records[0][1]

    def devices(record):
        return {d["uuid"] for d in record["resources"]["metadata"]["devices"]}

    keys = (
        "model dtype tensor_parallel_size max_model_len max_num_batched_tokens "
        "max_num_seqs kv_cache_memory_bytes enable_prefix_caching async_scheduling "
        "kernel_config seed"
    ).split()
    rows = []
    for path, record in records:
        if not record.get("source_stable") or record.get("shutdown") != "complete":
            raise ValueError(f"Incomplete or source-changing execution: {path}")
        if any(record["config"][key] != reference["config"][key] for key in keys):
            raise ValueError(f"Unmatched KV/scheduler/workload configuration: {path}")
        if (
            record["source"]["fork"]["runtime_files"]
            != reference["source"]["fork"]["runtime_files"]
        ):
            raise ValueError(f"Fork runtime sources differ: {path}")
        if devices(record) != devices(reference):
            raise ValueError(f"Device group differs: {path}")
        if record["measurement"] != reference["measurement"]:
            raise ValueError(f"Warmup/repetition/generation protocol differs: {path}")
        trials = record["trials"]
        if len(trials) < 3 or any(t["tokens"] != reference["trials"][0]["tokens"] for t in trials):
            raise ValueError(f"Insufficient repetitions or greedy mismatch: {path}")
        metrics = extract_run(path)
        counters = []
        for rank in record["final_memory"]:
            total = {}
            for cache in rank["caches"]:
                for key, value in cache["counters"].items():
                    if key not in {"first_prefetch_started_at", "last_prefetch_started_at"}:
                        total[key] = total.get(key, 0) + value
            counters.append({"rank": rank["rank"], "counters": total})
        rows.append(
            {
                "run": str(path.resolve()),
                "result_sha256": sha256_file(path / "result.json"),
                "config": record["config"],
                "measurement": record["measurement"],
                "trial_seconds": [t["seconds"] for t in trials],
                "median_seconds": float(np.median([t["seconds"] for t in trials])),
                "resources": record["resources"],
                "whole_engine_counters_including_warmup": counters,
                "cache_observation": metrics["cache"],
                "greedy_equal": True,
            }
        )
    out = Output(
        args.output,
        kind="result",
        identity={"scope": "short-plain-prompt", "model_path": reference["config"]["model"]},
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    out.write("comparison.json", rows)
    shutil.copy2(__file__, out.path / "reproduce.py")
    out.finish(
        evidence="short-backend-cost-diagnostic",
        agent_policy_comparison_eligible=False,
        limitations=[
            "Three within-engine repetitions per mode; mode order was not rotated",
            "Whole-engine counters include profiling and warmup; no per-trial attribution",
            "Observer differences are confounded by host load and sequential mode order",
            "Sampled device peaks can miss instantaneous allocations",
            "No agent task outcomes or task-level uncertainty",
        ],
    )
    print(out.path)


if __name__ == "__main__":
    main()
