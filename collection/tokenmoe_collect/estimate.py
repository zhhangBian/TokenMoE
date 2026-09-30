"""Uncompressed planning estimates and hardlink-aware measured storage."""

from pathlib import Path

import numpy as np

from .jsonl import read_jsonl


def formula(
    *,
    layers,
    top_k,
    experts,
    sessions=500,
    steps=60,
    computed_tokens=78000,
    total_request_tokens=2000000,
    prompt_text_bytes=7000000,
    engine_raw_bytes=6000000,
    engine_parquet_bytes=1000000,
    tool_bytes=1000000,
    harness_bytes=1000000,
):
    if min(layers, top_k, experts, sessions) <= 0 or top_k > experts:
        raise ValueError("Positive model facts and session count are required; K must not exceed E")
    if (
        min(
            steps,
            computed_tokens,
            total_request_tokens,
            prompt_text_bytes,
            engine_raw_bytes,
            engine_parquet_bytes,
            tool_bytes,
            harness_bytes,
        )
        < 0
    ):
        raise ValueError("Storage assumptions must be non-negative")
    categories = {
        "expert_ids": computed_tokens * layers * top_k * (1 if experts <= 256 else 2),
        "token_ids": total_request_tokens * 4,
        "routing_step_indices": computed_tokens * 4,
        "routing_token_positions": computed_tokens * 4,
        "prompts": prompt_text_bytes,
        "engine_steps_raw": engine_raw_bytes,
        "engine_steps_parquet": engine_parquet_bytes,
        "tool_outputs": tool_bytes,
        "harness": harness_bytes,
    }
    per_session = sum(categories.values())
    return {
        "mode": "formula",
        "assumptions": {
            "layers": layers,
            "top_k": top_k,
            "experts": experts,
            "sessions": sessions,
            "steps_per_session": steps,
            "computed_tokens_per_session": computed_tokens,
            "total_request_tokens_per_session": total_request_tokens,
        },
        "bytes_per_category_per_session": categories,
        "bytes_per_session": per_session,
        "bytes_per_run": per_session * sessions,
    }


def measured(run_dir, *, layers, top_k, experts):
    root = Path(run_dir)
    sessions = list(read_jsonl(root / "runtime/sessions.jsonl"))
    requests = list(read_jsonl(root / "runtime/llm_requests.jsonl"))
    if not sessions:
        raise ValueError("Pilot has no finalized sessions")
    count = len(sessions)
    computed = 0
    for path in (root / "runtime/routing").rglob("*.npz"):
        with np.load(path, allow_pickle=False) as routing:
            computed += len(routing["token_positions"])
    tokens = sum(
        (r.get("num_prompt_tokens") or 0) + (r.get("num_output_tokens") or 0) for r in requests
    )
    seen = set()
    categories = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in seen:
            continue
        seen.add(inode)
        parts = path.relative_to(root).parts
        if "routing" in parts:
            category = "routing"
        elif "tool_outputs" in parts:
            category = "tool_outputs"
        elif "prompts" in parts:
            category = "prompts"
        elif path.name in {"steps.jsonl", "engine_steps.parquet"}:
            category = "engine_steps"
        elif "static" in parts:
            category = "static"
        else:
            category = "other_records"
        row = categories.setdefault(category, {"bytes": 0, "allocated_bytes": 0, "files": 0})
        row["bytes"] += stat.st_size
        row["allocated_bytes"] += stat.st_blocks * 512
        row["files"] += 1
    prediction = formula(
        layers=layers,
        top_k=top_k,
        experts=experts,
        sessions=count,
        steps=len(requests) / count,
        computed_tokens=computed / count,
        total_request_tokens=tokens / count,
    )
    return {
        "mode": "measured",
        "sessions": count,
        "requests": len(requests),
        "computed_tokens": computed,
        "total_request_tokens": tokens,
        "measured_categories": categories,
        "measured_bytes": sum(row["bytes"] for row in categories.values()),
        "prediction": prediction,
    }
