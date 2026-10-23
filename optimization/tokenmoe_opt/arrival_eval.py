"""Task-held-out historical duration diagnostics, without receipt-latency claims."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenmoe_collect.jsonl import read_jsonl

from .arrival import Duration, Survival
from .artifacts import Output, input_files, read_manifest
from .features import command_categories


def duration_observations(roots, *, label_version="shell-command-v2"):
    observations = []
    for root in map(Path, roots):
        runs = {
            r["application_run_id"]: r for r in read_jsonl(root / "runtime/application_runs.jsonl")
        }
        sessions = {
            s["session_id"]: runs[s["application_run_id"]]["benchmark_item_id"]
            for s in read_jsonl(root / "runtime/sessions.jsonl")
        }
        for tool in read_jsonl(root / "runtime/tool_calls.jsonl"):
            duration = (tool["finished_at"] - tool["started_at"]) / 1e9
            category = "+".join(
                command_categories((root / tool["tool_args_ref"]).read_text(), label_version)
            )
            observations.append(
                (
                    tool,
                    Duration(
                        sessions[tool["session_id"]], category, duration, bool(tool["timed_out"])
                    ),
                )
            )
    return observations


def evaluate(inputs_path, output, *, command):
    inputs = read_manifest(inputs_path, "input")
    roots = [Path(root) for root in inputs["sources"]]
    if input_files(roots) != inputs["sources"]:
        raise ValueError("Historical input hashes changed")
    if not all(r["validation"]["passed"] for r in inputs["validation"]):
        raise ValueError("Arrival diagnostics require validated inputs")
    out = Output(output, kind="result", identity=inputs["identity"], command=command)
    try:
        observations = duration_observations(roots)
        tasks = sorted({d.task for _, d in observations})
        records, calibration, coverage = [], defaultdict(list), defaultdict(list)
        for task in tasks:
            model = Survival([d for _, d in observations if d.task != task])
            for tool, duration in observations:
                if duration.task != task:
                    continue
                for elapsed in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
                    if duration.seconds <= elapsed:
                        continue
                    prediction = model.predict(duration.category, elapsed=elapsed)
                    lo, hi = prediction["p10"], prediction["p90"]
                    remaining = duration.seconds - elapsed
                    covered = (
                        (lo <= remaining <= hi)
                        if lo is not None and hi is not None and not duration.censored
                        else None
                    )
                    if covered is not None:
                        coverage[task].append(covered)
                    for horizon in (0.5, 1.0, 5.0):
                        predicted = max(
                            (cdf for time, cdf in prediction["cdf"] if time <= horizon), default=0.0
                        )
                        observed = (
                            None
                            if duration.censored and remaining <= horizon
                            else int(remaining <= horizon)
                        )
                        if observed is not None:
                            calibration[horizon].append((predicted, observed))
                    records.append(
                        {
                            "task": task,
                            "tool_id": tool["tool_call_id"],
                            "elapsed": elapsed,
                            "decision_at": tool["started_at"] + int(elapsed * 1e9),
                            "category_observed_at": tool.get("parsed_at") or tool["started_at"],
                            "receipt_provenance": "historical host observation proxy; transport unmeasured",
                            "prediction": prediction,
                            "covered_80_interval": covered,
                            "censored": duration.censored,
                        }
                    )
        out.write("predictions.json", records)
        out.write(
            "metrics.json",
            {
                "schema_version": 1,
                "evidence": "historical-task-held-out-arrival-diagnostic",
                "tools": len(observations),
                "decisions": len(records),
                "finite_uncensored_intervals": sum(map(len, coverage.values())),
                "task_macro_80_interval_coverage": float(
                    np.mean([np.mean(v) for v in coverage.values()])
                )
                if coverage
                else None,
                "calibration": {
                    str(horizon): {
                        "scorable": len(pairs),
                        "mean_predicted": float(np.mean([p for p, _ in pairs])),
                        "mean_observed": float(np.mean([o for _, o in pairs])),
                    }
                    for horizon, pairs in calibration.items()
                },
                "limitations": [
                    "One-role short-tool exploratory data",
                    "Intervals with unknown tails remain null",
                    "Censoring prevents some horizon scores; transport receipt delay is unmeasured",
                    "Repeated elapsed decisions from a tool are correlated; no live arrival benefit measured",
                ],
            },
        )
        out.write("inputs.json", inputs)
        out.finish(evidence="historical-task-held-out-arrival-diagnostic")
        return out.path
    except Exception as exc:
        out.fail(exc)
        raise
