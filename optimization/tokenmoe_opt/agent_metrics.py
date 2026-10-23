"""Actual agent outcomes and engine/client boundaries, independent of full capture."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.jsonl import read_jsonl
from tokenmoe_collect.report import distribution, rows

from .artifacts import Output, digest
from .data import model_identity
from .metrics import cache_metrics, resource_metrics, token_metrics


def seconds(start, end):
    if start is None or end is None:
        return None
    if end < start:
        raise ValueError("Metric endpoints regress in the observation clock")
    return (end - start) / 1e9


def table(root, name, key):
    records = (
        rows(root, name)
        if (root / "runtime").exists()
        else [
            row
            for path in sorted(root.glob(f"raw/harness/*/{name}.jsonl"))
            for row in read_jsonl(path)
        ]
    )
    if len({row[key] for row in records}) != len(records):
        raise ValueError(f"Duplicate {key} in agent metrics")
    return {row[key]: row for row in records}


def task_uncertainty(records, field, *, seed=42, repetitions=1000):
    groups = defaultdict(list)
    for row in records:
        if row.get(field) is not None:
            groups[row["task"]].append(row[field])
    values = np.array([np.mean(group) for group in groups.values()])
    interval = None
    if len(values) >= 2:
        indices = np.random.default_rng(seed).integers(
            0, len(values), size=(repetitions, len(values))
        )
        interval = np.percentile(values[indices].mean(axis=1), [2.5, 97.5]).tolist()
    return {
        "task_macro_mean": float(values.mean()) if len(values) else None,
        "task_ci95": interval,
        "tasks": len(values),
        "bootstrap_unit": "whole task groups; equal weight for each task's mean",
        "reason": "At least two task groups are required for uncertainty"
        if interval is None
        else None,
    }


def measure(directory):
    outer = Path(directory).resolve()
    manifest_path = outer / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text())["status"] == "running":
        raise ValueError("Agent pilot is still running; wait for its terminal record")
    root = outer / "agent" if (outer / "agent").is_dir() else outer
    applications = table(root, "application_runs", "application_run_id")
    sessions = table(root, "sessions", "session_id")
    requests = table(root, "llm_requests", "llm_request_id")
    tools = table(root, "tool_calls", "tool_call_id")
    cell_path, boundary_path = outer / "cell.json", outer / "boundaries.json"
    cell = json.loads(cell_path.read_text()) if cell_path.exists() else {}
    boundary = json.loads(boundary_path.read_text()) if boundary_path.exists() else {}
    warmup = set(boundary.get("warmup_request_ids", []))
    warmup_sessions = set(boundary.get("warmup_session_ids", []))
    if warmup_sessions - sessions.keys():
        raise ValueError("Warmup sessions are absent from actual observations")
    clock_files = [*root.glob("static/engine_meta.json"), *root.glob("raw/harness/*/producer.json")]
    clock_files.extend(outer.glob("observations/*/engine_meta.json"))
    clocks = {json.loads(p.read_text())["clock_domain_id"] for p in clock_files}
    if len(clocks) > 1:
        raise ValueError("Agent and engine observation clocks differ")

    def events(path):
        for event in read_jsonl(path):
            clocks.add(event["clock_domain_id"])
            if len(clocks) > 1:
                raise ValueError("Agent and engine observation clocks differ")
            yield event

    engine = {}
    for path in sorted(outer.glob("observations/*/token-readiness.jsonl")):
        for row in token_metrics(events(path)):
            alias = row.get("llm_request_id")
            if alias:
                if alias in engine:
                    raise ValueError("A client attempt maps to multiple engine requests")
                engine[alias] = row
    if warmup - requests.keys() - engine.keys():
        raise ValueError("Warmup IDs are absent from actual observations")
    selected = {rid: r for rid, r in requests.items() if r["session_id"] not in warmup_sessions}
    if set(selected) & warmup:
        raise ValueError("Mixed warmup/measured agent sessions require separate session boundaries")
    request_rows, outcomes = [], []
    for rid, row in selected.items():
        if row["session_id"] not in sessions:
            raise ValueError("Incomplete agent session has no recorded terminal state")
        session = sessions[row["session_id"]]
        task = applications[session["application_run_id"]]["benchmark_item_id"]
        observed = engine.get(rid, {})
        received = observed.get("received_at", row.get("engine_received_at"))
        first = observed.get("first_token_at", row.get("first_token_at"))
        finished = observed.get("finished_at", row.get("inference_finished_at"))
        tokens = observed.get("expected_tokens", row.get("num_output_tokens"))
        if tokens is None:
            tokens = (row.get("usage") or {}).get("completion_tokens")
        dependencies = [tools[tid] for tid in row["prev_tool_call_ids"]]
        complete_join = all(t.get("finished_at") is not None for t in dependencies)
        joined = (
            max((t["finished_at"] for t in dependencies), default=None) if complete_join else None
        )
        if any(t["session_id"] != row["session_id"] for t in dependencies):
            raise ValueError("A request consumes another session's tool result")
        if joined is not None and joined > row["request_created_at"]:
            raise ValueError("A request consumes an unfinished dependency")
        created, delivered = row["request_created_at"], row.get("response_received_at")
        if boundary.get("measured_started_at", created) > created or (
            delivered is not None and delivered > boundary.get("measured_finished_at", delivered)
        ):
            raise ValueError("A measured request lies outside its frozen boundaries")
        window_tokens = tokens
        if observed:
            window_tokens = sum(
                boundary.get("measured_started_at", t)
                <= t
                <= boundary.get("measured_finished_at", t)
                for t in observed["token_observation_times_ns"]
            )
        request_rows.append(
            {
                "request": rid,
                "session": row["session_id"],
                "task": task,
                "status": row["request_status"],
                "finish_reason": row.get("finish_reason"),
                "truncated": row.get("finish_reason") == "length",
                "observed_output_tokens": tokens,
                "output_tokens_in_measurement": window_tokens,
                "engine_ttft_seconds": seconds(received, first),
                "request_to_complete_response_seconds": seconds(created, delivered),
                "tools_to_engine_first_token_seconds": seconds(joined, first),
                "tools_to_complete_response_seconds": seconds(joined, delivered),
                "tools_to_first_visible_token_seconds": seconds(joined, delivered),
                "tool_join_complete": complete_join,
                "endpoint_average_tpot_seconds": observed.get("endpoint_average_tpot_seconds"),
                "historical_endpoint_tpot_upper_bound_seconds": seconds(first, finished)
                / (tokens - 1)
                if rid not in engine
                and tokens is not None
                and tokens > 1
                and first is not None
                and finished is not None
                else None,
                "observed_itl_seconds": observed.get("observed_itl_seconds"),
            }
        )
    for sid, row in sessions.items():
        if sid in warmup_sessions:
            continue
        application = applications[row["application_run_id"]]
        group = [r for r in request_rows if r["session"] == sid]
        outcomes.append(
            {
                "task": application["benchmark_item_id"],
                "session": sid,
                "status": row["final_status"],
                "exit_reason": row["exit_reason"],
                "submitted": row["final_status"] == "succeeded"
                and row["exit_reason"] == "Submitted",
                "benchmark_score": application.get("benchmark_score"),
                "task_seconds": seconds(row.get("created_at"), row.get("finished_at")),
                "agent_execution_seconds": seconds(row.get("started_at"), row.get("finished_at")),
                "truncated_requests": sum(r["truncated"] for r in group),
            }
        )
    for task in set(cell.get("tasks", [])) - {row["task"] for row in outcomes}:
        outcomes.append(
            {
                "task": task,
                "session": None,
                "status": "not_started",
                "submitted": False,
                "benchmark_score": None,
                "task_seconds": None,
            }
        )
    started = boundary.get("measured_started_at")
    ended = boundary.get("measured_finished_at")
    measured_sessions = [s for sid, s in sessions.items() if sid not in warmup_sessions]
    if not boundary and measured_sessions:
        started = min(s["created_at"] for s in measured_sessions)
        ends = [s["finished_at"] for s in measured_sessions if s.get("finished_at") is not None]
        ended = max(ends) if len(ends) == len(measured_sessions) else None
    duration = seconds(started, ended)
    window = (started, ended) if started is not None and ended is not None else None
    memory = resource_metrics(outer / "resources", window=window)
    if window is not None and memory is not None:
        if memory["metadata"]["clock"]["clock_domain_id"] not in clocks:
            raise ValueError("Resource samples use another observation clock")
    plans = outer / "controller/plans.jsonl"
    controller = [row for row in read_jsonl(plans)] if plans.exists() else []
    if window is not None:
        controller = [r for r in controller if started <= r["created_at"] <= ended]
    statuses = [p.with_suffix(".status.json") for p in outer.glob("observations/*/*.jsonl")]
    statuses.extend(
        p.with_name("control_export_status.json")
        for p in root.glob("raw/harness/*/control_events.jsonl")
    )
    observer_status = {
        str(p.relative_to(outer)): json.loads(p.read_text()) if p.exists() else None
        for p in statuses
    }
    exported = all(
        s
        and s.get("flush_complete")
        and not (s.get("dropped") or s.get("errors") or s.get("export_errors"))
        for s in observer_status.values()
    )
    return {
        "schema_version": 1,
        "evidence": "observed-agent-trial",
        "clock_domain_id": next(iter(clocks), None),
        "cell": cell,
        "warmup_request_ids": sorted(warmup),
        "boundaries": boundary,
        "measurement_seconds": duration,
        "requests": request_rows,
        "outcomes": outcomes,
        "outcome_counts": dict(Counter(r["status"] for r in outcomes)),
        "request_counts": dict(Counter(r["status"] for r in request_rows)),
        "truncated_requests": sum(r["truncated"] for r in request_rows),
        "unknown_output_token_requests": sum(
            r["observed_output_tokens"] is None for r in request_rows
        ),
        "observed_output_tokens_per_second": sum(
            r["output_tokens_in_measurement"] or 0 for r in request_rows
        )
        / duration
        if duration
        else None,
        "submitted_agent_tasks_per_second": sum(r["submitted"] for r in outcomes) / duration
        if duration
        else None,
        "benchmark_solved_tasks_per_second": None,
        "memory": memory,
        "cache": cache_metrics(outer / "observations", window=window),
        "controller_wall_seconds": distribution(
            [r["controller_wall_ns"] / 1e9 for r in controller if "controller_wall_ns" in r]
        ),
        "observer_status": observer_status,
        "side_channel_export_complete": bool(exported) if observer_status else None,
        "metrics": {
            field: {
                "observed": distribution([r[field] for r in request_rows]),
                "task_uncertainty": task_uncertainty(request_rows, field),
            }
            for field in (
                "engine_ttft_seconds",
                "tools_to_complete_response_seconds",
                "endpoint_average_tpot_seconds",
            )
        },
        "task_latency": distribution([r["task_seconds"] for r in outcomes]),
        "task_latency_uncertainty": task_uncertainty(outcomes, "task_seconds"),
        "limitations": [
            "Submitted agent termination is not benchmark correctness; no solved-task throughput claimed",
            "Nonstreaming client exposes the whole response together; network first-byte time is unobserved",
            "Historical inference-finished endpoint is an upper bound on final token readiness",
            "Missing, rejected and truncated requests remain in status and coverage counts",
            "Transfer/cache cost attribution is separate from request metrics",
        ],
    }


def extract(directory, output, *, command):
    path = Path(directory).resolve()
    if Path(output).resolve().is_relative_to(path):
        raise ValueError("Metric output must be outside the immutable source run")
    source = path / "manifest.json"
    identity = json.loads(source.read_text())["identity"] if source.exists() else None
    agent = path / "agent" if (path / "agent").exists() else path
    profiles = agent / "static/model_profiles.jsonl"
    if profiles.exists():
        profiles = list(read_jsonl(profiles))
        templates = list(read_jsonl(agent / "static/role_templates.jsonl"))
        if len(profiles) != 1 or len(templates) != 1:
            raise ValueError("Agent metrics require one exact model/template identity")
        actual = model_identity(profiles[0], templates[0])
        actual["id"] = digest(actual)
        if identity is not None and identity != actual:
            raise ValueError("Actual agent template differs from its experiment identity")
        identity = actual
    if identity is None:
        raise ValueError("Source model/template identity is required")
    out = Output(output, kind="result", identity=identity, command=command)
    try:

        def source_hashes():
            return {
                str(p.relative_to(path)): sha256_file(p)
                for p in path.rglob("*")
                if p.suffix in {".jsonl", ".json"}
            }

        hashes = source_hashes()
        out.write("metrics.json", measure(path))
        if hashes != source_hashes():
            raise ValueError("Agent observations changed during metric extraction")
        out.write("source-hashes.json", hashes)
        out.finish(evidence="observed-agent-trial", controlled_comparison_qualified=False)
    except Exception as exc:
        out.fail(exc)
        raise
    return out.path
