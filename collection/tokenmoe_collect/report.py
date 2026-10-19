"""Runtime summaries and versioned labels derived from finalized real traces."""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .assets import sha256_file
from .estimate import measured
from .jsonl import read_jsonl, write_json


def rows(root, name):
    path = Path(root) / "runtime" / f"{name}.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def parquet_rows(path):
    if Path(path).exists():
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
            yield from batch.to_pylist()


def distribution(values):
    values = [float(x) for x in values if x is not None]
    return {
        "count": len(values),
        "p50": float(np.percentile(values, 50)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "reason": None if values else "No observed values with complete endpoints",
    }


def interval(row, start, end):
    a, b = row.get(start), row.get(end)
    return (b - a) / 1e9 if a is not None and b is not None and b >= a else None


def tool_category(command):
    if re.search(r"\b(pytest|unittest|tox|nosetests|test)\b", command):
        return "test"
    if re.search(r"\b(grep|rg|find|ls|head|tail|cat|sed)\b", command):
        return "inspect"
    if re.search(r"\b(patch|apply_patch|tee)\b|>\s*\S+|\b(write_text|write_bytes)\b", command):
        return "edit"
    if re.search(r"\b(pip|conda|apt|make|build)\b", command):
        return "build_or_install"
    return "unknown"


def classify_units(root, *, min_location_rate=0.99):
    """Separate actual task outcomes from collection integrity, including startup faults."""
    root = Path(root)
    applications = {x["application_run_id"]: x for x in rows(root, "application_runs")}
    requests = defaultdict(list)
    for row in rows(root, "llm_requests"):
        requests[row["session_id"]].append(row)
    validation_path = root / "validation.json"
    validation = (
        json.loads(validation_path.read_text()) if validation_path.exists() else {"passed": False}
    )
    alignment_path = root / "alignment.json"
    alignment = json.loads(alignment_path.read_text()) if alignment_path.exists() else {}
    rate = alignment.get("rendered_location_rate", alignment.get("location_rate"))
    request_sessions = {
        r["llm_request_id"]: r["session_id"] for group in requests.values() for r in group
    }
    tool_sessions = {t["tool_call_id"]: t["session_id"] for t in rows(root, "tool_calls")}
    session_records = rows(root, "sessions")
    known_sessions = {s["session_id"] for s in session_records}
    run_sessions = {s["application_run_id"]: s["session_id"] for s in session_records}
    affected, global_failure = (
        set(),
        not validation.get("passed", False) and not validation.get("rules"),
    )
    for rule in validation.get("rules", {}).values():
        if rule["status"] != "failed":
            continue
        examples = rule.get("examples", [])
        if len(examples) < rule.get("violation_count", 0) or not examples:
            global_failure = True
        for identifier in examples:
            sid = (
                request_sessions.get(identifier)
                or tool_sessions.get(identifier)
                or run_sessions.get(identifier)
            )
            if identifier in known_sessions:
                sid = identifier
            if sid:
                affected.add(sid)
            else:
                global_failure = True
    fields = defaultdict(dict)
    for segment in parquet_rows(root / "runtime/prompt_segments.parquet"):
        sid = request_sessions.get(segment["llm_request_id"])
        if sid and segment.get("message_index") is not None and segment.get("message_field"):
            key = (segment["llm_request_id"], segment["message_index"], segment["message_field"])
            status = (
                "omitted"
                if segment.get("alignment_status") == "template_omitted"
                else "failed"
                if segment.get("alignment_error")
                else "located"
            )
            fields[sid][key] = status
    results = []
    for session in session_records:
        application = applications[session["application_run_id"]]
        reqs = requests[session["session_id"]]
        reasons = []
        if not reqs or not any(
            r.get("routing_ref")
            and (r.get("row_end") or 0) > (r.get("row_start", r.get("num_cached_tokens")) or 0)
            for r in reqs
        ):
            reasons.append("no_real_inference")
        if global_failure or session["session_id"] in affected:
            reasons.append("structural_validation_failed")
        session_fields = Counter(fields[session["session_id"]].values())
        denominator = session_fields["located"] + session_fields["failed"]
        session_rate = session_fields["located"] / denominator if denominator else rate
        if session_rate is None or session_rate < min_location_rate:
            reasons.append("prompt_alignment_failed")
        if any(
            (not r.get("routing_ref") or not (root / r["routing_ref"]).is_file())
            for r in reqs
            if r.get("engine_admission_status") != "rejected_before_engine"
        ):
            reasons.append("routing_missing")
        if session.get("finished_at") is None or session.get("final_status") not in {
            "succeeded",
            "failed",
            "timeout",
            "cancelled",
        }:
            reasons.append("terminal_state_missing")
        results.append(
            {
                "benchmark_item_id": application["benchmark_item_id"],
                "session_id": session["session_id"],
                "application_run_id": session["application_run_id"],
                "task_status": session["final_status"],
                "exit_reason": session["exit_reason"],
                "requests": len(reqs),
                "rendered_location_rate": session_rate,
                "valid": not reasons,
                "invalid_reasons": reasons,
                "artifact": str(root.resolve()),
            }
        )
    return results


def summarize_runtime(root):
    root = Path(root).resolve()
    requests, sessions, tools = (
        rows(root, "llm_requests"),
        rows(root, "sessions"),
        rows(root, "tool_calls"),
    )
    profile = next(read_jsonl(root / "static/model_profiles.jsonl"))
    experiment = next(read_jsonl(root / "static/experiment_configs.jsonl"))
    by_request = {r["llm_request_id"]: r for r in requests}
    chunks = defaultdict(list)
    for row in parquet_rows(root / "runtime/tool_output_chunks.parquet"):
        chunks[row["tool_call_id"]].append(row)
    lookups = {
        phase: np.zeros(
            (len(profile["moe_layer_ids"]), profile["num_routed_experts"]), dtype=np.int64
        )
        for phase in ["new_prefill", "decode"]
    }
    counts = Counter(sessions=len(sessions), requests=len(requests), tools=len(tools))
    for request in requests:
        counts["pre_engine_context_rejections"] += (
            request.get("engine_admission_status") == "rejected_before_engine"
        )
        for key in [
            "num_prompt_tokens",
            "num_output_tokens",
            "num_cached_tokens",
            "num_preemptions",
        ]:
            counts[key] += request.get(key) or 0
        if request.get("routing_ref"):
            with np.load(root / request["routing_ref"], allow_pickle=False) as data:
                positions, experts = data["token_positions"], data["experts"]
                counts["first_computed_tokens"] += len(positions)
                for phase, mask in [
                    ("new_prefill", positions < request["num_prompt_tokens"]),
                    ("decode", positions >= request["num_prompt_tokens"]),
                ]:
                    counts[f"{phase}_tokens"] += int(mask.sum())
                    for layer in range(experts.shape[1]):
                        lookups[phase][layer] += np.bincount(
                            experts[mask, layer].ravel(), minlength=profile["num_routed_experts"]
                        )
    batch_sizes, running, waiting = [], [], []
    for step in parquet_rows(root / "runtime/engine_steps.parquet"):
        counts["engine_steps"] += 1
        batch_sizes.append(len({e["vllm_request_id"] for e in step["entries"]}))
        running.append(step.get("num_running_reqs", step.get("num_running", 0)))
        waiting.append(step.get("num_waiting_reqs", step.get("num_waiting", 0)))
        for entry in step["entries"]:
            if entry["phase"] == "recompute":
                counts["recompute_tokens"] += entry["token_end"] - entry["token_start"]
    payloads = Counter()
    for segment in parquet_rows(root / "runtime/prompt_segments.parquet"):
        if segment["segment_type"] == "tool_output":
            if segment.get("alignment_error"):
                payloads["unaligned_segments"] += 1
            elif segment.get("token_start") is not None:
                req = by_request[segment["llm_request_id"]]
                payloads["located_segments"] += 1
                payloads["overlapping_new_prefill_segments"] += int(
                    segment["token_end"] > req["num_cached_tokens"]
                )
                payloads["cached_segments"] += int(segment["token_end"] <= req["num_cached_tokens"])
    labels, durations, first_output, silence, consume_delay = [], [], [], [], []
    for tool in tools:
        tcid = tool["tool_call_id"]
        command = (root / tool["tool_args_ref"]).read_text()
        category = tool_category(command)
        outcome = (
            "timeout"
            if tool.get("timed_out")
            else "success"
            if tool.get("exit_status") == 0
            else "failure"
            if tool.get("exit_status") is not None
            else "unknown"
        )
        labels.append(
            {
                "tool_call_id": tcid,
                "tool_category": category,
                "outcome_category": outcome,
                "category_available_at": tool.get("issued_at"),
                "outcome_available_at": tool["finished_at"],
                "outcome_is_supervision_at_issue": True,
                "rules_version": "command-outcome-v1",
            }
        )
        durations.append(interval(tool, "started_at", "finished_at"))
        observed = sorted(chunks[tcid], key=lambda r: r["chunk_index"])
        times = [tool["started_at"], *[x["chunk_time"] for x in observed], tool["finished_at"]]
        silence.append(max(b - a for a, b in zip(times, times[1:])) / 1e9)
        first_output.append(
            (observed[0]["chunk_time"] - tool["started_at"]) / 1e9 if observed else None
        )
        counts["chunks"] += len(observed)
        counts["tool_raw_bytes"] += tool["output_bytes_raw"]
        counts["tool_seen_bytes"] += tool["output_bytes_seen"]
        counts["tools_with_consuming_requests"] += bool(tool.get("llm_request_ids_consuming"))
        for rid in tool.get("llm_request_ids_consuming", []):
            consume_delay.append(
                (by_request[rid]["request_created_at"] - tool["finished_at"]) / 1e9
            )
    transitions = []
    previous = {}
    by_tool = {x["tool_call_id"]: x for x in labels}
    for request in sorted(requests, key=lambda r: r["request_created_at"]):
        sid = request["session_id"]
        categories = [
            by_tool[t]["tool_category"] for t in request["prev_tool_call_ids"] if t in by_tool
        ]
        transitions.append(
            {
                "llm_request_id": request["llm_request_id"],
                "first_step": sid not in previous,
                "preceding_tool_categories": categories,
                "tool_switch": sid in previous and bool(categories) and previous[sid] != categories,
                "available_at": request["request_created_at"],
                "rules_version": "command-outcome-v1",
            }
        )
        if categories:
            previous[sid] = categories
        else:
            previous.setdefault(sid, [])
    groups = sorted({x["benchmark_item_id"] for x in rows(root, "application_runs")})
    write_json(root / "derived/labels_v1/tool_labels.json", labels)
    write_json(root / "derived/labels_v1/transitions.json", transitions)
    write_json(
        root / "derived/labels_v1/g1_groups.json",
        {
            "group_by": "benchmark_item_id",
            "tasks": groups,
            "G2": "not_covered",
            "G3": "not_covered",
        },
    )

    def overlap(items, start, end):
        events = sorted(
            (timestamp, delta)
            for row in items
            if row.get(start) is not None and row.get(end) is not None
            for timestamp, delta in [(row[start], 1), (row[end], -1)]
        )
        active = maximum = 0
        for _, delta in events:
            active += delta
            maximum = max(maximum, active)
        return maximum

    hosts = list(parquet_rows(root / "runtime/host_load.parquet"))
    counts["host_samples"] = len(hosts)
    counts["missing_gpu_samples"] = sum(x.get("gpu_util") is None for x in hosts)
    result = {
        "run_dir": str(root),
        "profile": experiment.get("collection_profile"),
        "counts": dict(counts),
        "outcomes": dict(Counter(x["final_status"] for x in sessions)),
        "exit_reasons": dict(Counter(x["exit_reason"] for x in sessions)),
        "concurrency": experiment["concurrency"],
        "engine_config_id": experiment["engine_config_id"],
        "max_session_overlap": overlap(sessions, "started_at", "finished_at"),
        "max_request_overlap": overlap(requests, "request_sent_at", "response_received_at"),
        "batch_request_counts": distribution(batch_sizes),
        "tool_categories": dict(Counter(x["tool_category"] for x in labels)),
        "tool_outcomes": dict(Counter(x["outcome_category"] for x in labels)),
        "first_step_samples": sum(x["first_step"] for x in transitions),
        "tool_switch_samples": sum(x["tool_switch"] for x in transitions),
        "tool_payload_coverage": dict(payloads),
        "expert_lookup_counts": {phase: values.tolist() for phase, values in lookups.items()},
        "layer_ids": profile["moe_layer_ids"],
        "seconds": {
            "tool_duration": distribution(durations),
            "tool_first_output": distribution(first_output),
            "tool_longest_silence": distribution(silence),
            "tool_to_consumer": distribution(consume_delay),
            "queueing": distribution(
                [interval(r, "engine_received_at", "first_scheduled_at") for r in requests]
            ),
            "time_to_first_token": distribution(
                [interval(r, "engine_received_at", "first_token_at") for r in requests]
            ),
            "generation_interval": distribution(
                [interval(r, "first_token_at", "inference_finished_at") for r in requests]
            ),
        },
        "g1_task_groups": groups,
        "scope": "One generalist role and bash; G2/G3 and unseen-tool generalization are not covered. Timing is wall-clock, not per-layer kernel or per-token latency.",
        "alignment": json.loads((root / "alignment.json").read_text()),
        "validation": json.loads((root / "validation.json").read_text()),
        "units": classify_units(root),
        "storage": measured(
            root,
            layers=len(profile["moe_layer_ids"]),
            top_k=profile["top_k"],
            experts=profile["num_routed_experts"],
        ),
    }
    write_json(root / "reports/runtime_summary.json", result)
    text = f"# Local runtime data\n\nRun: {root}\n\nValid units: {sum(x['valid'] for x in result['units'])}/{len(result['units'])}.\n\n"
    text += (
        "```json\n"
        + json.dumps(
            {k: v for k, v in result.items() if k not in {"expert_lookup_counts", "storage"}},
            indent=2,
        )
        + "\n```\n"
    )
    (root / "reports/runtime_summary.md").write_text(text)
    return result


def artifact_checksums(root):
    root = Path(root)
    result, seen, excluded = {}, {}, []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_checksums.json":
            continue
        if path.name in {
            "collection_manifest.json",
            "recovery_events.jsonl",
            "health_checks.jsonl",
            "latest_health_check.json",
            ".collection.lock",
        } or path.name.startswith("supervisor-"):
            excluded.append(str(path.relative_to(root)))
            continue
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity not in seen:
            seen[identity] = sha256_file(path)
        result[str(path.relative_to(root))] = {"bytes": stat.st_size, "sha256": seen[identity]}
    write_json(
        root / "reports/artifact_checksums.json",
        {"algorithm": "sha256", "artifacts": result, "excluded_mutable_paths": excluded},
    )
    return {"files": len(result), "unique_inodes": len(seen)}
