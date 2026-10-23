"""Apply the predeclared strict tensor/router gate to matching fixed-input runs."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tokenmoe_opt.artifacts import Output


def main():
    parser = argparse.ArgumentParser()
    for name in ("native", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    native, cached = [
        json.loads((p / "result.json").read_text()) for p in (args.native, args.cache)
    ]
    for key in (
        "model",
        "dtype",
        "tensor_parallel_size",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_memory_bytes",
        "kernel_config",
    ):
        if native["config"][key] != cached["config"][key]:
            raise ValueError(f"Unmatched execution configuration: {key}")
    if native["criteria"] != cached["criteria"] or native.get("prompt_source") != cached.get(
        "prompt_source"
    ):
        raise ValueError("Qualification criteria or prompt identities differ")
    if not all(r.get("source_stable") for r in (native, cached)) or (
        native["source"]["fork"]["runtime_files"] != cached["source"]["fork"]["runtime_files"]
    ):
        raise ValueError("Qualification requires matching stable fork sources")
    paths = [
        set(p.relative_to(root / "probes") for p in (root / "probes").rglob("*.npz"))
        for root in (args.native, args.cache)
    ]
    expected = sum(r["layers"] * r["frames"] for r in native["probes"])
    if paths[0] != paths[1] or len(paths[0]) != expected:
        raise ValueError("Missing or unmatched rank/layer/frame tensors")
    criteria = native["criteria"]
    if criteria["greedy_token_mismatches"] != 0:
        raise ValueError("Cache qualification requires zero token and router mismatches")
    rows = []
    for path in sorted(paths[0]):
        with (
            np.load(args.native / "probes" / path, allow_pickle=False) as left,
            np.load(args.cache / "probes" / path, allow_pickle=False) as right,
        ):
            row = {
                "file": str(path),
                "shape_equal": all(left[k].shape == right[k].shape for k in left.files),
                "padding_equal": bool(np.array_equal(left["padding"], right["padding"])),
            }
            for key in ("x", "coefficients", "output"):
                same = left[key].shape == right[key].shape
                finite = same and np.isfinite(left[key]).all() and np.isfinite(right[key]).all()
                row[key + "_max_abs_error"] = (
                    float(np.max(np.abs(left[key] - right[key]))) if finite else None
                )
                row[key + "_within_tolerance"] = bool(
                    finite
                    and np.allclose(
                        left[key],
                        right[key],
                        atol=criteria["tensor_atol"],
                        rtol=criteria["tensor_rtol"],
                    )
                )
            row["router_mismatches"] = (
                int(np.count_nonzero(left["ids"] != right["ids"])) if row["shape_equal"] else None
            )
            rows.append(row)
    greedy = [r["tokens"] for r in native["trials"]] == [r["tokens"] for r in cached["trials"]]
    passed = greedy and all(
        r["shape_equal"]
        and r["padding_equal"]
        and r["router_mismatches"] == 0
        and all(r[k + "_within_tolerance"] for k in ("x", "coefficients", "output"))
        for r in rows
    )
    out = Output(
        args.output,
        kind="result",
        identity={
            "scope": "fixed-input tensor qualification",
            "model_path": native["config"]["model"],
        },
        command=[sys.executable, __file__, *sys.argv[1:]],
    )
    out.write(
        "comparison.json",
        {
            "native": str(args.native.resolve()),
            "cache": str(args.cache.resolve()),
            "criteria": criteria,
            "tensor_files": len(rows),
            "greedy_tokens_equal": greedy,
            "passed": passed,
            "rows": rows,
        },
    )
    out.finish(
        passed=passed,
        evidence="fixed-input-tensor-and-router-gate",
        policy_comparison_qualified=False,
    )
    print(out.path)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
