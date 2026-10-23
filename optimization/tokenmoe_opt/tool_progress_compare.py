"""Paired exploratory statistics from immutable saved forecasts; no refitting."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenmoe_collect.assets import sha256_file

from .agent_metrics import task_uncertainty
from .artifacts import Output, read_manifest


def compare(evaluation, output, *, command):
    evaluation = Path(evaluation).resolve()
    source = read_manifest(evaluation / "manifest.json", "result")
    if (
        source.get("status") != "complete"
        or source.get("evidence") != "exploratory-task-held-out-stream-progress-replay"
    ):
        raise ValueError("Paired analysis requires completed held-task stream forecasts")
    records = json.loads((evaluation / "predictions.json").read_text())
    index = defaultdict(dict)
    for row in records:
        key = row["task"], row["job_id"], row["elapsed"]
        if row["method"] in index[key]:
            raise ValueError("Duplicate method/decision in forecasts")
        index[key][row["method"]] = row
    methods = {"elapsed-history", "explicit-progress", "observed-stream"}
    if any(set(rows) != methods for rows in index.values()):
        raise ValueError("Methods do not cover the same causal decisions")
    for rows in index.values():
        reference = rows["elapsed-history"]
        if any(
            row["visible_features"] != reference["visible_features"]
            or row["censored"] != reference["censored"]
            for row in rows.values()
        ):
            raise ValueError("Methods did not receive the same observed prefix/target")
    cells, support, alerts = (
        defaultdict(list),
        defaultdict(int),
        defaultdict(lambda: defaultdict(int)),
    )
    for (task, job, _), rows in index.items():
        for method, row in rows.items():
            support[method, "all"] += 1
            support[method, "point"] += row["prediction"]["p50"] is not None
            support[method, "interval"] += (
                row["prediction"]["p10"] is not None and row["prediction"]["p90"] is not None
            )
            alerts[method]["correct"] += row["scores"]["action_precision"] == 1
            alerts[method]["false"] += row["scores"]["action_precision"] == 0
        for baseline in ("elapsed-history", "explicit-progress"):
            actual, reference = rows["observed-stream"]["scores"], rows[baseline]["scores"]
            for metric in actual:
                if actual[metric] is not None and reference[metric] is not None:
                    cells[baseline, metric, task, job].append(actual[metric] - reference[metric])
    summary = []
    for baseline, metric in sorted({(b, m) for b, m, _, _ in cells}):
        jobs = [
            {"task": task, "value": float(np.mean(values))}
            for (b, m, task, _), values in cells.items()
            if (b, m) == (baseline, metric)
        ]
        summary.append(
            {
                "baseline": baseline,
                "metric": metric,
                "paired_jobs": len(jobs),
                **task_uncertainty(jobs, "value"),
            }
        )
    out = Output(output, kind="result", identity=source["identity"], command=command)
    out.write(
        "paired.json",
        {
            "evidence": "post-hoc-exploratory-paired-forecast-analysis",
            "difference": "observed-stream minus baseline; same decision and shared nonmissing score",
            "summary": summary,
            "support": [
                {
                    "method": m,
                    "decisions": support[m, "all"],
                    "point_predictions": support[m, "point"],
                    "finite_intervals": support[m, "interval"],
                    "alerts": dict(alerts[m]),
                }
                for m in sorted(methods)
            ],
            "source_sha256": sha256_file(evaluation / "predictions.json"),
            "live_gain_measured": False,
        },
    )
    lines = [
        "# Paired exploratory forecast differences",
        "",
        "Observed-stream minus baseline, using the same decision and jointly available scores.",
        "",
        "| Baseline | Metric | Task macro difference | Task 95% interval |",
        "|---|---|---:|---|",
    ]
    lines.extend(
        f"| {r['baseline']} | {r['metric']} | {r['task_macro_mean']} | {r['task_ci95']} |"
        for r in summary
    )
    (out.path / "report.md").write_text("\n".join(lines) + "\n")
    out.finish(
        evidence="post-hoc-exploratory-paired-forecast-analysis", live_prefetch_gain_measured=False
    )
    return out.path
