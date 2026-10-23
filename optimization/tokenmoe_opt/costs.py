"""Measured rank-local copy profiles; short-prompt timing remains diagnostic."""

import json
from pathlib import Path

import numpy as np
from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.ids import content_id
from tokenmoe_collect.jsonl import read_jsonl

from .artifacts import Output, cache_execution, manifest, read_manifest
from .metrics import token_metrics
from .validation import validate_cache_events


def fit(native_root, cache_root, inputs_path, output, *, command):
    native_root, cache_root = Path(native_root), Path(cache_root)
    inputs = read_manifest(inputs_path, "input")
    native = json.loads((native_root / "result.json").read_text())
    cached = json.loads((cache_root / "result.json").read_text())
    if not native.get("source_stable") or not cached.get("source_stable"):
        raise ValueError("Copy profiles require source-stable live executions")
    if [t["tokens"] for t in native["trials"]] != [t["tokens"] for t in cached["trials"]]:
        raise ValueError("Cost fitting requires matching baseline/cached greedy outputs")
    for key in (
        "model",
        "tensor_parallel_size",
        "dtype",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_memory_bytes",
        "enforce_eager",
        "async_scheduling",
        "enable_prefix_caching",
        "kernel_config",
    ):
        if native["config"][key] != cached["config"][key]:
            raise ValueError(f"Cost fitting configurations differ: {key}")
    if native["config"].get("generation_config", "auto") != cached["config"].get(
        "generation_config", "auto"
    ):
        raise ValueError("Cost fitting configurations differ: generation_config")
    model_path = Path(cached["config"]["model"])
    metadata = model_path / ".cache/huggingface/download/config.json.metadata"
    config_path, tokenizer_path = model_path / "config.json", model_path / "tokenizer.json"
    revision = (
        metadata.read_text().splitlines()[0]
        if metadata.exists()
        else content_id(json.loads(config_path.read_text()))
    )
    if (
        revision != inputs["identity"]["weight_revision"]
        or sha256_file(tokenizer_path) != inputs["identity"]["tokenizer_hash"]
    ):
        raise ValueError("Measured checkpoint/tokenizer differs from predictor input identity")
    out = Output(output, kind="cost-profile", identity=inputs["identity"], command=command)
    sources = {
        str(path.resolve()): sha256_file(path)
        for path in (
            inputs_path,
            native_root / "result.json",
            cache_root / "result.json",
            config_path,
            tokenizer_path,
            *([metadata] if metadata.exists() else []),
        )
    }
    ranks, durations, critical = {}, [], []
    try:
        for path in sorted((cache_root / "observations").glob("*/cache-rank-*.jsonl")):
            sources[str(path)] = sha256_file(path)
            starts, evictions, sizes, times, waits = {}, {}, set(), [], []

            def events():
                for event in read_jsonl(path):
                    kind, layer = event["kind"], event.get("layer")
                    if kind == "eviction":
                        evictions[(layer, event["slot"], event["generation"] + 1)] = event[
                            "source_at"
                        ]
                    elif kind == "load_start":
                        key = (layer, event["transfer_id"])
                        starts[key] = {
                            **event,
                            "admitted_at": evictions.pop(
                                (layer, event["slot"], event["generation"]), None
                            ),
                        }
                    elif kind == "load_complete":
                        start = starts.pop((layer, event["transfer_id"]))
                        if start["trigger"] == "demand":
                            sizes.add(start["bytes"])
                            times.append(event["device_duration_ns"] / 1e9)
                            if start["admitted_at"] is not None:
                                waits.append((event["source_at"] - start["admitted_at"]) / 1e9)
                    yield event

            status_path = path.with_suffix(".status.json")
            sources[str(status_path)] = sha256_file(status_path)
            status = json.loads(status_path.read_text())
            validation = validate_cache_events(
                events(), status=status, expected_layers=inputs["identity"]["layers"]
            )
            if (
                not validation["exact_replay_available"]
                or len(sizes) != 1
                or not times
                or not waits
            ):
                raise ValueError(
                    "Copy fitting requires complete detailed equal-size demand transfers"
                )
            size = sizes.pop()
            rank = int(path.stem.rsplit("-", 1)[1])
            ranks[str(rank)] = {
                "expert_bytes": size,
                "samples": len(times),
                "device_copy_p50_seconds": float(np.median(times)),
                "device_copy_p90_seconds": float(np.percentile(times, 90)),
                "recovery_p50_seconds": float(np.median(waits)),
                "bandwidth_bytes_second": size / float(np.median(times)),
            }
            durations.extend(times)
            critical.extend(waits)
        requests = []
        for path in (cache_root / "observations").glob("*/token-readiness.jsonl"):
            sources[str(path)] = sha256_file(path)
            requests.extend(token_metrics(read_jsonl(path)))
        ttfts = [r["engine_ttft_seconds"] for r in requests if r["engine_ttft_seconds"] is not None]
        tpots = [
            r["endpoint_average_tpot_seconds"]
            for r in requests
            if r["endpoint_average_tpot_seconds"] is not None
        ]
        if not ranks or not ttfts or not tpots:
            raise ValueError("Measured copy and phase observation points are required")
        controller = {
            "version": "copy-recovery-v1",
            "measured": True,
            "cache_execution": cache_execution(cached["config"]["expert_cache_config"]),
            "model_hash": cached["final_memory"][0]["model_hash"],
            "device_uuids": [d["uuid"] for d in cached["resources"]["metadata"]["devices"]],
            "copy_seconds": float(np.percentile(durations, 90)),
            "demand_wait_seconds": float(np.percentile(critical, 90)),
            "shared_bandwidth_bytes_second": min(
                r["bandwidth_bytes_second"] for r in ranks.values()
            ),
            "prefill_seconds": float(np.median(ttfts)),
            "decode_token_seconds": float(np.median(tpots)),
            "hide_probability": 0,
        }
        first_scheduled, computed, native_requests = {}, {}, []
        for path in sorted((native_root / "observations").glob("*/token-readiness.jsonl")):
            sources[str(path)] = sha256_file(path)
            status_path = path.with_suffix(".status.json")
            status = json.loads(status_path.read_text())
            sources[str(status_path)] = sha256_file(status_path)
            if not status.get("flush_complete") or status.get("dropped") or status.get("errors"):
                raise ValueError("Native model service requires loss-free complete export")
            sequence = [0]

            def native_events():
                for event in read_jsonl(path):
                    if event["sequence"] != sequence[0] + 1:
                        raise ValueError("Native scheduling observations contain an event gap")
                    sequence[0] = event["sequence"]
                    if event["kind"] == "schedule":
                        for rid, count in event["scheduled"].items():
                            if count:
                                first_scheduled.setdefault(rid, event["source_at"])
                                computed[rid] = computed.get(rid, 0) + count
                    yield event

            native_requests.extend(token_metrics(native_events()))
            if sequence[0] != status["last_sequence"]:
                raise ValueError("Native scheduling observations have an incomplete export tail")
        native_requests.sort(key=lambda r: r.get("received_at", 0))
        warmup = native.get("measurement", {}).get("warmup_requests", 0)
        measured_count = sum(len(t["tokens"]) for t in native["trials"])
        service = []
        for row in native_requests[warmup : warmup + measured_count]:
            rid = row["request_id"]
            if not row["complete_token_coverage"] or rid not in first_scheduled:
                raise ValueError(
                    "Native model service requires complete tokens and actual scheduling"
                )
            service.append(
                {
                    "request_id": rid,
                    "computed_tokens": computed[rid],
                    "model_service_seconds": (row["last_token_at"] - first_scheduled[rid]) / 1e9,
                    "input_tokens": row["num_prompt_tokens"],
                    "output_tokens": row["expected_tokens"],
                }
            )
        if len(service) != measured_count:
            raise ValueError("Native measured model-service requests are incomplete")
        out.write(
            "cost-profile.json",
            manifest(
                "cost-profile",
                inputs["identity"],
                ranks=ranks,
                controller=controller,
                native_service={
                    "layers": inputs["identity"]["layers"],
                    "requests": service,
                    "observation_point": "first scheduled host observation to last token readiness",
                    "scope": "measured fixed-input requests only; no pure GPU or layer-time claim",
                },
                sources=sources,
                copy={
                    "bytes_per_expert": next(iter(ranks.values()))["expert_bytes"],
                    "bandwidth_bytes_second": controller["shared_bandwidth_bytes_second"],
                    "setup_seconds": 0,
                },
                timing_status="diagnostic: short-prompt service times; held-out live calibration outstanding",
                replay_setup_assumption="zero device-copy setup only; controller uses measured full recovery cost",
                shared_link_assumption="conservative single link at minimum measured rank bandwidth",
            ),
        )
        out.finish(evidence="measured-copy-diagnostic", timing_simulation_qualified=False)
        return out.path
    except Exception as exc:
        out.fail(exc)
        raise
