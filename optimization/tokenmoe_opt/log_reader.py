"""Causal prompt preparation and a preregistered local log-reader resource gate."""

import json
import math
import re
from pathlib import Path

from tokenmoe_collect.assets import inspect_model, sha256_file
from tokenmoe_collect.jsonl import read_jsonl

from .artifacts import Output, read_manifest


def parse_forecast(text):
    match = re.search(r"\{[^{}]*\}", text)
    if match is None:
        raise ValueError("Reader returned no flat JSON forecast")
    value = json.loads(match[0])
    keys = ("p10", "remaining_seconds", "p90")
    if any(
        type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) or value[k] < 0
        for k in keys
    ):
        raise ValueError("Reader forecast must contain finite nonnegative numeric bounds")
    if not value["p10"] <= value["remaining_seconds"] <= value["p90"]:
        raise ValueError("Reader forecast bounds are out of order")
    return {key: value[key] for key in keys}


def prepare(corpus, evaluation, model, output, *, command):
    corpus, evaluation, model = map(lambda p: Path(p).resolve(), (corpus, evaluation, model))
    data = read_manifest(corpus / "manifest.json", "result")
    measured = read_manifest(evaluation / "manifest.json", "result")
    if not data.get("coverage_passed") or measured.get("status") != "complete":
        raise ValueError(
            "Reader qualification requires validated coverage and complete baseline forecasts"
        )
    baseline_protocol = read_manifest(evaluation / "protocol.json", "experiment")
    if Path(baseline_protocol["corpus"]).resolve() != corpus:
        raise ValueError("Reader and baseline use different corpora")
    assets = inspect_model(model)
    if not assets["valid"]:
        raise ValueError("Local reader checkpoint assets are incomplete")
    predictions = {
        (r["job_id"], r["elapsed"]): r["prediction"]
        for r in json.loads((evaluation / "predictions.json").read_text())
        if r["method"] == "elapsed-history"
    }
    prompts = []
    for row in sorted(
        json.loads((corpus / "tools.json").read_text()), key=lambda r: (r["task"], r["job_id"])
    ):
        root = Path(row["case_root"])
        for relative, digest in row["source_hashes"].items():
            if sha256_file(root / relative) != digest:
                raise ValueError("Reader source observations changed")
        case = json.loads((root / "case.json").read_text())
        if case["kind"] != "focused":
            continue
        chunks = list(read_jsonl(root / case["producer"] / "tool_output_chunks.jsonl"))
        for elapsed in (1.0, 30.0):
            if row["seconds"] <= elapsed:
                continue
            prior = predictions[row["job_id"], elapsed]
            if row["task"] in prior["training_tasks"]:
                raise ValueError("Reader prior includes the evaluation task")
            at = case["timing"]["started_at"] + round(elapsed * 1e9)
            end = max(
                (c["byte_offset"] + c["byte_length"] for c in chunks if c["chunk_time"] <= at),
                default=0,
            )
            with (root / case["timing"]["output_ref"]).open("rb") as raw:
                start = max(0, end - 4096)
                raw.seek(start)
                text = raw.read(end - start).decode("utf-8", errors="replace")
            prompts.append(
                {
                    "task": row["task"],
                    "job_id": row["job_id"],
                    "elapsed": elapsed,
                    "command": case["command"],
                    "observed_prefix_bytes": end,
                    "log_excerpt": text,
                    "history_prior": {k: prior[k] for k in ("p10", "p50", "p90")},
                }
            )
    if len(prompts) < 3:
        raise ValueError("Insufficient eligible focused-tool prefixes for resource qualification")
    out = Output(output, kind="experiment", identity=data["identity"], command=command)
    out.write("prompts.json", prompts)
    out.write(
        "protocol.json",
        {
            "schema_version": 1,
            "kind": "experiment",
            "identity": data["identity"],
            "protocol_id": "local-log-reader-resource-v1",
            "status": "preregistered-unexecuted",
            "model_path": str(model),
            "assets": assets,
            "device": 3,
            "gpu_peak_limit_bytes": 20 * 1024**3,
            "host_rss_limit_bytes": 32 * 1024**3,
            "torch_threads": 4,
            "warmup_requests": 2,
            "repetitions": 3,
            "max_input_tokens": 1536,
            "max_new_tokens": 96,
            "latency_p95_limit_seconds": 0.5,
            "format_valid_rate_required": 1.0,
            "backend": "transformers eager SDPA; BF16; isolated GPU 3",
            "backend_source_hashes": {
                "log_reader.py": sha256_file(Path(__file__)),
                "qualify_log_reader.py": sha256_file(
                    Path(__file__).resolve().parents[1] / "scripts/qualify_log_reader.py"
                ),
            },
            "latency_rationale": "Inline arrival forecasts must fit the existing 0.5-second planning horizon",
            "resource_probe_selection": "first three task/job/elapsed-sorted eligible focused-tool prefixes",
            "quality_selection": "all focused jobs at elapsed 1/30 seconds while still observed running; at most 24",
            "quality_execution": "only after format, memory and latency qualification; otherwise stop with a nondeployment result",
            "prompt_sha256": sha256_file(out.path / "prompts.json"),
            "sources": {
                str(corpus / "manifest.json"): sha256_file(corpus / "manifest.json"),
                str(evaluation / "predictions.json"): sha256_file(evaluation / "predictions.json"),
            },
            "scope": "exploratory collector-prefix replay; no live receipt or GPU solver interference measurement",
            "sampling": {"do_sample": False, "seed": 42},
        },
    )
    out.finish(evidence="local-log-reader-resource-preregistration", reader_executed=False)
    return out.path
