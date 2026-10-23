"""Held-out fixed-input N1 calibration; cache counts and service errors stay separate."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.jsonl import read_jsonl

from .artifacts import Output, read_manifest
from .metrics import token_metrics
from .replay import CopyCost, profile_costs, replay
from .timing import simulate
from .validation import validate_cache_events


def measure_run(run, cost_path, protocol_path):
    run = Path(run).resolve()
    cost = read_manifest(cost_path, "cost-profile")
    protocol = json.loads((Path(protocol_path) / "protocol.json").read_text())
    if str(run) != protocol["held_out_run"] or sha256_file(cost_path) != protocol["cost_sha256"]:
        raise ValueError("Calibration run/cost differs from preregistration")
    if str(run / "result.json") in cost["sources"]:
        raise ValueError("Calibration execution was included in cost fitting")
    core = Path(__file__).parent
    if any(sha256_file(core / name) != digest for name, digest in protocol["core_hashes"].items()):
        raise ValueError("Calibration model changed after preregistration; freeze a separate run")
    record = json.loads((run / "result.json").read_text())
    if not record.get("source_stable") or any(len(t["tokens"]) != 1 for t in record["trials"]):
        raise ValueError("Calibration requires source-stable N1 fixed-input executions")
    if record["final_memory"][0]["model_hash"] != cost["controller"]["model_hash"]:
        raise ValueError("Calibration model/configuration differs from measured costs")
    readiness = next((run / "observations").glob("*/token-readiness.jsonl"))
    status = json.loads(readiness.with_suffix(".status.json").read_text())
    if not status["flush_complete"] or status["dropped"] or status["errors"]:
        raise ValueError("Calibration token export is incomplete")
    first_scheduled = {}

    def token_events():
        for event in read_jsonl(readiness):
            if event["kind"] == "schedule":
                for rid, count in event["scheduled"].items():
                    if count:
                        first_scheduled.setdefault(rid, event["source_at"])
            yield event

    requests = sorted(token_metrics(token_events()), key=lambda r: r["received_at"])
    start = record["measurement"]["warmup_requests"]
    measured = requests[start : start + len(record["trials"])]
    if start != protocol["warmup_requests"] or len(measured) != protocol["measured_requests"]:
        raise ValueError("Calibration repetition/warmup boundaries differ")
    if not all(r["complete_token_coverage"] for r in requests):
        raise ValueError("Calibration contains incomplete token observations")
    cutoff = measured[0]["received_at"]
    if any(r["finished_at"] >= cutoff for r in requests[:start]):
        raise ValueError("Warmup is not closed before the measured window")
    infos, validations = {}, []
    capacity = protocol["capacity"]
    for path in sorted((run / "observations").glob("*/cache-rank-*.jsonl")):
        rank = int(path.stem.rsplit("-", 1)[1])
        groups, step_requests, token_counts = [], defaultdict(set), defaultdict(int)
        counts, loaded, current = defaultdict(Counter), set(), []

        def cache_events():
            for event in read_jsonl(path):
                kind, step, layer = event["kind"], event.get("step"), event.get("layer")
                if kind == "initial_snapshot":
                    if len(event["slots"]) != capacity or any(
                        s["state"] != "FREE" for s in event["slots"]
                    ):
                        raise ValueError("Calibration requires complete known-cold initialization")
                elif kind == "feedback":
                    step_requests[step].add(event["request_id"])
                    token_counts[step, layer] += event["end"] - event["start"]
                elif kind == "demand":
                    key = step, layer, event["source_at"]
                    if current and current[0] != key:
                        raise ValueError("Unclosed authoritative microbatch demand")
                    if not current:
                        current.extend([key, []])
                    current[1].append(event["expert"])
                elif kind == "demand_ready":
                    key = step, layer, event["demand_at"]
                    if not current or current[0] != key:
                        raise ValueError("Demand recovery lacks its authoritative expert set")
                    groups.append(
                        {
                            "step": step,
                            "layer": layer,
                            "experts": list(current[1]),
                            "time_ns": event["demand_at"],
                        }
                    )
                    current.clear()
                    counts[step]["microbatches"] += 1
                if kind == "load_start":
                    if event["trigger"] != "demand":
                        raise ValueError("This calibration scope excludes speculative transfers")
                    key = layer, event["expert"]
                    counts[step]["bytes"] += event["bytes"]
                    counts[step]["reloads"] += key in loaded
                    loaded.add(key)
                if kind in {"hit", "miss", "eviction"}:
                    counts[step][
                        {
                            "hit": "unique_demand_hits",
                            "miss": "unique_demand_misses",
                            "eviction": "evictions",
                        }[kind]
                    ] += 1
                yield event

        cache_status = json.loads(path.with_suffix(".status.json").read_text())
        checked = validate_cache_events(
            cache_events(),
            status=cache_status,
            expected_layers=cost["identity"]["layers"],
            checkpoint_at=cutoff,
        )
        if not checked["exact_replay_available"] or current:
            raise ValueError("Calibration requires complete validated cache lifecycles")
        if any(step is not None and len(ids) != 1 for step, ids in step_requests.items()):
            raise ValueError("This timing calibration supports N1 only")
        infos[rank] = {
            "groups": groups,
            "steps": step_requests,
            "tokens": token_counts,
            "counts": counts,
            "checkpoint": checked.pop("checkpoint"),
        }
        validations.append(checked)
    if len(infos) != len(cost["ranks"]):
        raise ValueError("Missing calibration rank")
    canonical = infos[min(infos)]

    def geometry(info):
        return [(g["step"], g["layer"], g["experts"]) for g in info["groups"]]

    if any(geometry(info) != geometry(canonical) for info in infos.values()):
        raise ValueError("TP authoritative demand order differs")
    measured_ids = {r["request_id"] for r in measured}
    steps = {step for step, ids in canonical["steps"].items() if next(iter(ids)) in measured_ids}
    snapshot = {"slots": [], "previously_loaded": []}
    for info in infos.values():
        for slot in info["checkpoint"]["slots"]:
            if slot["state"] == "LOADING":
                raise ValueError("N1 warmup has an unresolved copy at its closed boundary")
            if slot["state"] == "IN_USE":
                slot = {**slot, "state": "READY", "pins": 0}
            snapshot["slots"].append(slot)
        snapshot["previously_loaded"].extend(info["checkpoint"]["previously_loaded"])
    selected = [g for g in canonical["groups"] if g["step"] in steps]
    frozen = [{"kind": "snapshot", "time_seconds": 0, **snapshot}] + [
        {
            "layer": g["layer"],
            "experts": g["experts"],
            "partition_source_capacity": capacity,
            "time_seconds": (g["time_ns"] - cutoff) / 1e9,
        }
        for g in selected
    ]
    actual = Counter()
    for rank, info in infos.items():
        for step in steps:
            actual.update(
                {
                    k: v
                    for k, v in info["counts"][step].items()
                    if k != "microbatches" or rank == min(infos)
                }
            )
    predicted = replay(
        frozen, capacity=capacity, cost=CopyCost(**cost["copy"]), rank_costs=profile_costs(cost)
    )
    nodes = []
    for i, row in enumerate(measured):
        layers, counted = [], set()
        for group in selected:
            if row["request_id"] not in canonical["steps"][group["step"]]:
                continue
            key = group["step"], group["layer"]
            layers.append(
                {
                    "layer": group["layer"],
                    "experts": group["experts"],
                    "partition_source_capacity": capacity,
                    "token_count_source": "validated_compact_feedback",
                    "observed_token_count": canonical["tokens"][key] if key not in counted else 0,
                }
            )
            counted.add(key)
        nodes.append(
            {
                "id": f"request-{i}",
                "kind": "model",
                "layers": layers,
                "input_tokens": row["num_prompt_tokens"],
                "output_tokens": row["expected_tokens"],
                "service_profile_index": i,
                "dependencies": [f"request-{i - 1}"] if i else [],
            }
        )
    timing = simulate(
        nodes,
        capacity=capacity,
        costs=profile_costs(cost),
        native_service=cost["native_service"],
        initial_snapshot=snapshot,
    )
    errors = []
    for i, row in enumerate(measured):
        actual_seconds = (row["last_token_at"] - first_scheduled[row["request_id"]]) / 1e9
        node = timing["nodes"][f"request-{i}"]
        seconds = node["finished"] - node["started"]
        errors.append(
            {
                "request": row["request_id"],
                "actual_service_seconds": actual_seconds,
                "predicted_service_seconds": seconds,
                "relative_error": abs(seconds - actual_seconds) / actual_seconds,
            }
        )
    counts = {
        key: {
            "actual": actual[key],
            "predicted": predicted[key],
            "absolute_error": abs(actual[key] - predicted[key]),
        }
        for key in (
            "bytes",
            "unique_demand_hits",
            "unique_demand_misses",
            "evictions",
            "reloads",
            "microbatches",
        )
    }
    median = float(np.median([r["relative_error"] for r in errors]))
    p95 = float(np.percentile([r["relative_error"] for r in errors], 95))
    qualified = (
        all(v["absolute_error"] == 0 for v in counts.values())
        and median <= protocol["model_service_median_relative_error_limit"]
        and p95 <= protocol["model_service_p95_relative_error_limit"]
    )
    validation = {
        "cache": validations,
        "scope": protocol["scope"],
        "cache_count_errors": counts,
        "service_errors": errors,
        "median_service_relative_error": median,
        "p95_service_relative_error": p95,
        "qualified_fixed_input_service": qualified,
        "snapshot_note": "N1 warmup completion implies prior kernel reads are complete; derived READY state retains observed generations and replacement history",
    }
    return {
        "validation.json": validation,
        "graph.json": {"nodes": nodes, "initial_snapshot": snapshot},
        "frozen-replay.json": predicted,
        "timing.json": timing,
    }


def validate_run(run, cost_path, protocol_path, output, *, command):
    cost = read_manifest(cost_path, "cost-profile")
    out = Output(output, kind="result", identity=cost["identity"], command=command)
    try:
        result = measure_run(run, cost_path, protocol_path)
        for name, value in result.items():
            out.write(name, value)
        out.finish(
            evidence="held-out-fixed-input-calibration",
            qualified_fixed_input_service=result["validation.json"][
                "qualified_fixed_input_service"
            ],
            agent_workflow_timing_qualified=False,
        )
    except Exception as exc:
        out.fail(exc)
        raise
    return out.path
