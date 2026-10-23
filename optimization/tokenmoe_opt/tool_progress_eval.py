"""Exploratory held-task scoring of collector-observed prefixes; no live gain claim."""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.jsonl import read_jsonl
from tokenmoe_collect.report import distribution

from .agent_metrics import task_uncertainty
from .artifacts import Output, read_manifest
from .tool_progress import (
    FEATURE_NAMES,
    ProgressPredictor,
    StreamState,
    command_family,
    probability,
)


def freeze(corpus, output, *, command):
    corpus = Path(corpus).resolve()
    source = read_manifest(corpus / "manifest.json", "result")
    if source.get("evidence") != "validated-real-test-build-corpus" or not source.get(
        "coverage_passed"
    ):
        raise ValueError("Progress evaluation requires a validated coverage-passing corpus")
    out = Output(output, kind="experiment", identity=source["identity"], command=command)
    out.write(
        "protocol.json",
        {
            "schema_version": 1,
            "kind": "experiment",
            "identity": source["identity"],
            "protocol_id": "observed-tool-progress-v1",
            "status": "frozen-before-scoring",
            "corpus": str(corpus),
            "input_hashes": {
                name: sha256_file(corpus / name)
                for name in (
                    "manifest.json",
                    "tools.json",
                    "protocol.json",
                    "coverage.json",
                    "validation.json",
                )
            },
            "methods": ["elapsed-history", "explicit-progress", "observed-stream"],
            "feature_version": "collector-prefix-v1",
            "model_code_hashes": {
                name: sha256_file(Path(__file__).parent / name)
                for name in (
                    "tool_progress.py",
                    "tool_progress_eval.py",
                    "arrival.py",
                    "agent_metrics.py",
                )
            },
            "feature_names": list(FEATURE_NAMES),
            "checkpoints_seconds": [
                0.0,
                0.25,
                0.5,
                1.0,
                2.0,
                5.0,
                10.0,
                20.0,
                30.0,
                60.0,
                90.0,
                120.0,
                180.0,
                240.0,
            ],
            "horizons_seconds": [0.5, 1.0, 5.0],
            "neighbors": 8,
            "min_support": 4,
            "runtime_repetitions": 3,
            "bootstrap": {
                "unit": "task; equal job means within task",
                "repetitions": 1000,
                "seed": 42,
            },
            "receipt_rule": "same-process collector observation replay proxy; controller receipt/export delay not measured",
            "target": "natural completion uncertainty; timeout lower bounds are censored, never exact completed durations",
            "training": "frozen offline leave-one-task-out; every job/copy of the held task excluded",
            "design_status": "exploratory; initial corpus examples inspected before freezing generic feature rules; no held-out scoring performed",
            "unsupported": "stream neighbors fall back when censored training prefixes are unavailable or task support is insufficient",
            "action_rule": "probability of completion within five seconds at least 0.8; potential lead only, no prefetch execution",
        },
    )
    out.finish(evidence="stream-progress-preanalysis-plan", predictions_executed=False)
    return out.path


def snapshots(record, checkpoints):
    root = Path(record["case_root"])
    for relative, expected in record["source_hashes"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError("Raw tool observations changed after corpus validation")
    case = json.loads((root / "case.json").read_text())
    chunks = list(read_jsonl(root / case["producer"] / "tool_output_chunks.jsonl"))
    started = case["timing"]["started_at"]
    state, index, result, times = StreamState(), 0, {}, []
    with (root / case["timing"]["output_ref"]).open("rb") as raw:
        for elapsed in checkpoints:
            if elapsed >= record["seconds"]:
                continue
            at = started + round(elapsed * 1e9)
            while index < len(chunks) and chunks[index]["chunk_time"] <= at:
                chunk = chunks[index]
                if raw.tell() != chunk["byte_offset"]:
                    raise ValueError("Prefix reader cannot skip unknown bytes")
                data = raw.read(chunk["byte_length"])
                began = time.perf_counter_ns()
                state.observe(data, chunk["chunk_time"])
                times.append((time.perf_counter_ns() - began) / 1e9)
                index += 1
            result[elapsed] = state.features(decision_at=at, elapsed=elapsed)
    return {**record, "family": command_family(case["command"]), "features": result}, times


def score(prediction, record, elapsed, horizons):
    remaining = record["seconds"] - elapsed
    uncensored = not record["censored"]
    point = prediction["p50"]
    low, high = prediction["p10"], prediction["p90"]
    result = {
        "point_absolute_error_seconds": abs(point - remaining)
        if point is not None and uncensored
        else None,
        "finite_interval": float(low is not None and high is not None) if uncensored else None,
        "covered_80_interval": float(low <= remaining <= high)
        if low is not None and high is not None and uncensored
        else None,
    }
    for horizon in horizons:
        predicted = probability(prediction, horizon)
        observed = (
            None if record["censored"] and remaining <= horizon else float(remaining <= horizon)
        )
        result[f"brier_{horizon:g}s"] = (
            (predicted - observed) ** 2 if predicted is not None and observed is not None else None
        )
    actionable = probability(prediction, 5)
    known = uncensored or remaining > 5
    result["action_precision"] = (
        float(uncensored and remaining <= 5)
        if actionable is not None and actionable >= 0.8 and known
        else None
    )
    result["potential_lead_seconds"] = remaining if result["action_precision"] == 1 else None
    return result


def evaluate(protocol_path, output, *, command):
    protocol = read_manifest(protocol_path, "experiment")
    if protocol.get("protocol_id") != "observed-tool-progress-v1":
        raise ValueError("Unsupported progress evaluation protocol")
    if any(
        sha256_file(Path(__file__).parent / name) != expected
        for name, expected in protocol["model_code_hashes"].items()
    ):
        raise ValueError("Progress model changed after freezing; create a separate evaluation plan")
    corpus = Path(protocol["corpus"])
    for name, expected in protocol["input_hashes"].items():
        if sha256_file(corpus / name) != expected:
            raise ValueError("Progress evaluation inputs changed after freezing")
    records, feature_times = [], []
    for record in json.loads((corpus / "tools.json").read_text()):
        row, times = snapshots(record, protocol["checkpoints_seconds"])
        records.append(row)
        feature_times.extend(times)
    out = Output(output, kind="result", identity=protocol["identity"], command=command)
    decisions, runtime, cells = [], defaultdict(list), defaultdict(list)
    try:
        for task in sorted({r["task"] for r in records}):
            model = ProgressPredictor(
                [r for r in records if r["task"] != task],
                neighbors=protocol["neighbors"],
                min_support=protocol["min_support"],
            )
            for row in (r for r in records if r["task"] == task):
                for elapsed, features in row["features"].items():
                    for method in protocol["methods"]:
                        for _ in range(protocol["runtime_repetitions"]):
                            began = time.perf_counter_ns()
                            predicted = model.predict(
                                method,
                                task=task,
                                family=row["family"],
                                elapsed=elapsed,
                                features=features,
                            )
                            runtime[method].append((time.perf_counter_ns() - began) / 1e9)
                        if task in predicted["training_tasks"]:
                            raise ValueError("Held task leaked into prediction training")
                        values = score(predicted, row, elapsed, protocol["horizons_seconds"])
                        for metric, value in values.items():
                            if value is not None:
                                cells[method, metric, task, row["job_id"]].append(value)
                        decisions.append(
                            {
                                "task": task,
                                "job_id": row["job_id"],
                                "method": method,
                                "elapsed": elapsed,
                                "visible_features": features.tolist(),
                                "feature_names": list(FEATURE_NAMES),
                                "prediction": predicted,
                                "censored": row["censored"],
                                "scores": values,
                            }
                        )
        summary = []
        for method, metric in sorted({(m, metric) for m, metric, _, _ in cells}):
            jobs = [
                {"task": task, "value": float(np.mean(values))}
                for (m, name, task, _), values in cells.items()
                if (m, name) == (method, metric)
            ]
            summary.append(
                {
                    "method": method,
                    "metric": metric,
                    "jobs": len(jobs),
                    **task_uncertainty(
                        jobs,
                        "value",
                        seed=protocol["bootstrap"]["seed"],
                        repetitions=protocol["bootstrap"]["repetitions"],
                    ),
                }
            )
        out.write("protocol.json", protocol)
        out.write("predictions.json", decisions)
        out.write(
            "metrics.json",
            {
                "evidence": "exploratory-task-held-out-stream-progress-replay",
                "summary": summary,
                "decisions": len(decisions),
                "prediction_runtime_seconds": {m: distribution(v) for m, v in runtime.items()},
                "feature_update_runtime_seconds": distribution(feature_times),
                "explicit_interval_scope": "Linear explicit-progress points have no calibrated interval; history fallback is labeled",
                "limits": [
                    "No live consumer receipt or prefetch overlap measured",
                    "Natural timeouts remain censored; known wrapper deadlines are a separate return mechanism",
                    "Exploratory authored workflows; no confirmation or agent solve claim",
                ],
            },
        )
        lines = [
            "# Exploratory held-task stream predictions",
            "",
            "Job means are averaged within each task; confidence intervals bootstrap task groups.",
            "",
            "| Method | Metric | Task macro | Task 95% interval |",
            "|---|---|---:|---|",
        ]
        lines.extend(
            f"| {r['method']} | {r['metric']} | {r['task_macro_mean']} | {r['task_ci95']} |"
            for r in summary
        )
        lines.extend(
            [
                "",
                "Collector-observation replay proxy; transport and live prefetch gains are unmeasured.",
            ]
        )
        (out.path / "report.md").write_text("\n".join(lines) + "\n")
        out.finish(
            evidence="exploratory-task-held-out-stream-progress-replay",
            live_prefetch_gain_measured=False,
        )
    except BaseException as exc:
        out.fail(exc)
        raise
    return out.path
