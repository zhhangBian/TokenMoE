#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tokenmoe.schema import WorkloadRecord, read_workload_jsonl
from tokenmoe.vllm_sidecar import (
    infer_model_capability,
    metadata_sidecar_decision,
)


LATENCY_FIELDS = [
    "mode",
    "batch_size",
    "batch_id",
    "request_id",
    "workflow",
    "role",
    "phase",
    "latency_ms",
    "batch_latency_ms",
    "prompt_chars",
    "output_chars",
    "output_tokens",
    "sidecar_elapsed_us",
    "sidecar_fallback_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real vLLM/Qwen end-to-end TokenMoE fallback experiments."
    )
    parser.add_argument(
        "--model",
        default="/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct",
        help="Local Hugging Face model path served by vLLM.",
    )
    parser.add_argument(
        "--workload",
        default=str(REPO_ROOT / "data/workloads/agent_workloads.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data/vllm_qwen"),
    )
    parser.add_argument(
        "--analysis-dir",
        default=str(REPO_ROOT / "analysis/vllm_qwen"),
    )
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8",
        help="Comma-separated batch sizes. Each size is run for baseline and TokenMoE sidecar modes.",
    )
    parser.add_argument(
        "--modes",
        default="baseline,tokenmoe_sidecar",
        help="Comma-separated modes from: baseline, tokenmoe_sidecar.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append records to existing JSONL/CSV files. Used by sharded runs.",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Use vLLM eager mode. Useful for debugging; disabled by default for real serving path.",
    )
    return parser.parse_args()


def import_vllm() -> tuple[Any, Any, str, str]:
    """Import installed vLLM without resolving to this repository's vllm checkout."""

    removed_paths: list[str] = []
    repo = REPO_ROOT.resolve()
    for path in list(sys.path):
        if not path:
            continue
        try:
            if Path(path).resolve() == repo:
                sys.path.remove(path)
                removed_paths.append(path)
        except OSError:
            continue
    sys.modules.pop("vllm", None)
    from vllm import LLM, SamplingParams
    import vllm

    for path in removed_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
    return LLM, SamplingParams, str(getattr(vllm, "__version__", "unknown")), str(
        getattr(vllm, "__file__", "")
    )


def load_config(model: str) -> dict[str, Any]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return cfg.to_dict()


def batched(records: list[WorkloadRecord], batch_size: int) -> list[list[WorkloadRecord]]:
    return [
        records[idx : idx + batch_size] for idx in range(0, len(records), batch_size)
    ]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def summarize_latencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["mode"]), int(row["batch_size"])), []).append(row)
    for (mode, batch_size), group in sorted(grouped.items()):
        latencies = [float(row["latency_ms"]) for row in group]
        output_tokens = sum(int(row["output_tokens"]) for row in group)
        wall_s = sum(float(row["latency_ms"]) for row in group) / 1000.0
        summary_rows.append(
            {
                "mode": mode,
                "batch_size": batch_size,
                "requests": len(group),
                "mean_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": percentile(latencies, 0.95),
                "throughput_req_s": len(group)
                / max(wall_s / max(batch_size, 1), 1e-9),
                "throughput_output_tok_s": output_tokens
                / max(wall_s / max(batch_size, 1), 1e-9),
                "mean_sidecar_us": statistics.mean(
                    float(row["sidecar_elapsed_us"]) for row in group
                ),
            }
        )
    return {"latency_summary": summary_rows}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    all_records = read_workload_jsonl(args.workload)
    records = all_records[args.offset : args.offset + args.limit]
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown_modes = sorted(set(modes) - {"baseline", "tokenmoe_sidecar"})
    if unknown_modes:
        raise ValueError(f"unknown modes: {unknown_modes}")
    config = load_config(args.model)
    capability = infer_model_capability(config)

    LLM, SamplingParams, vllm_version, vllm_file = import_vllm()

    env = {
        "model": args.model,
        "workload": args.workload,
        "offset": args.offset,
        "limit": len(records),
        "workload_total": len(all_records),
        "batch_sizes": batch_sizes,
        "modes": modes,
        "max_tokens": args.max_tokens,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vllm_version": vllm_version,
        "vllm_file": vllm_file,
        "capability": capability.to_dict(),
        "time_start": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (analysis_dir / "model_capability.json").write_text(
        json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(env, indent=2, ensure_ascii=False), flush=True)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "trust_remote_code": True,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
    }
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
        ignore_eos=False,
    )

    generation_path = output_dir / "generations.jsonl"
    decision_path = output_dir / "sidecar_decisions.jsonl"
    latency_path = output_dir / "latency.csv"
    latency_rows: list[dict[str, Any]] = []
    file_mode = "a" if args.append else "w"
    write_latency_header = (
        not args.append or not latency_path.exists() or latency_path.stat().st_size == 0
    )

    with generation_path.open(file_mode, encoding="utf-8") as gen_f, decision_path.open(
        file_mode, encoding="utf-8"
    ) as dec_f, latency_path.open(file_mode, encoding="utf-8", newline="") as lat_f:
        latency_writer = csv.DictWriter(lat_f, fieldnames=LATENCY_FIELDS)
        if write_latency_header:
            latency_writer.writeheader()
        for batch_size in batch_sizes:
            for mode in modes:
                for batch_id, batch in enumerate(batched(records, batch_size)):
                    sidecar_decisions = []
                    sidecar_elapsed_total_us = 0.0
                    if mode == "tokenmoe_sidecar":
                        for record in batch:
                            decision = metadata_sidecar_decision(
                                record.meta, capability
                            )
                            sidecar_decisions.append(decision)
                            sidecar_elapsed_total_us += decision.elapsed_us
                            dec_f.write(
                                json.dumps(
                                    {
                                        "mode": mode,
                                        "batch_size": batch_size,
                                        "batch_id": batch_id,
                                        "request_id": record.request_id,
                                        "decision": decision.to_dict(),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                    prompts = [record.prompt for record in batch]
                    started = time.perf_counter()
                    outputs = llm.generate(prompts, sampling, use_tqdm=False)
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    per_request_ms = elapsed_ms / max(len(batch), 1)
                    output_by_index = list(outputs)
                    for idx, (record, output) in enumerate(zip(batch, output_by_index)):
                        text = output.outputs[0].text if output.outputs else ""
                        token_ids = output.outputs[0].token_ids if output.outputs else []
                        row = {
                            "mode": mode,
                            "batch_size": batch_size,
                            "batch_id": batch_id,
                            "request_id": record.request_id,
                            "workflow": record.workflow,
                            "role": record.meta.role,
                            "phase": record.meta.phase,
                            "latency_ms": per_request_ms,
                            "batch_latency_ms": elapsed_ms,
                            "prompt_chars": len(record.prompt),
                            "output_chars": len(text),
                            "output_tokens": len(token_ids),
                            "sidecar_elapsed_us": (
                                sidecar_decisions[idx].elapsed_us
                                if mode == "tokenmoe_sidecar"
                                else 0.0
                            ),
                            "sidecar_fallback_reason": (
                                sidecar_decisions[idx].fallback_reason
                                if mode == "tokenmoe_sidecar"
                                else ""
                            ),
                        }
                        latency_rows.append(row)
                        latency_writer.writerow(row)
                        gen_f.write(
                            json.dumps(
                                {
                                    **row,
                                    "prompt": record.prompt,
                                    "output_text": text,
                                    "meta": record.meta.to_dict(),
                                    "model": args.model,
                                    "vllm_version": vllm_version,
                                    "capability": capability.to_dict(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    print(
                        (
                            f"mode={mode} batch_size={batch_size} batch={batch_id} "
                            f"requests={len(batch)} latency_ms={elapsed_ms:.2f} "
                            f"sidecar_us={sidecar_elapsed_total_us:.2f}"
                        ),
                        flush=True,
                    )
                    gen_f.flush()
                    dec_f.flush()
                    lat_f.flush()
                    gc.collect()

    summary = {
        **env,
        **summarize_latencies(latency_rows),
        "records_written": {
            "generations": str(generation_path),
            "sidecar_decisions": str(decision_path),
            "latency_csv": str(latency_path),
        },
        "time_end": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if args.append:
        shard_dir = analysis_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        summary_path = (
            shard_dir
            / f"summary_offset_{args.offset}_limit_{len(records)}_"
            f"batches_{'-'.join(map(str, batch_sizes))}_"
            f"modes_{'-'.join(modes)}.json"
        )
    else:
        summary_path = analysis_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
