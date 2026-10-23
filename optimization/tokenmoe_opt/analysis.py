"""Grouped lookup coverage, task uncertainty, and explicitly labeled hindsight."""

import json
from collections import defaultdict

import numpy as np

from .artifacts import Output, input_files, manifest
from .data import PHASES, load_runs
from .features import FEATURE_VERSION, LABEL_VERSION
from .predictors import POLICIES, History, Priors, selected
from .splits import grouped_splits, tune_mixture


def subgroups(features):
    result = ["all", features["stage"]]
    if features["no_tool"]:
        result.append("no_tool")
    if features["multiple_tools"]:
        result.append("multiple_tools")
    if features["unknown_category"]:
        result.append("unknown_category")
    if not features["first_request"]:
        result.append("tool_switch" if features["tool_switch"] else "no_tool_switch")
    return result


def uncertainty(hits, totals, *, seed=42, repetitions=1000):
    valid = totals > 0
    values = hits[valid] / totals[valid]
    if not len(values):
        return {
            "task_macro": None,
            "task_ci95": [None, None],
            "tasks": 0,
            "reason": "No observed target lookups",
        }
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return {
        "task_macro": float(values.mean()),
        "task_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "tasks": len(values),
        "reason": None,
    }


def analyze(
    roots,
    output,
    *,
    command,
    budgets=(16, 32, 64, 96, 128),
    label_version=LABEL_VERSION,
    tune=False,
    held_out=None,
):
    out = Output(
        output, kind="result", identity={"pending": "validated input identity"}, command=command
    )
    try:
        roots = [root.resolve() for root in roots]
        hashes = input_files(roots)
        samples, identity, validation = load_runs(roots, label_version=label_version)
        out.record["identity"] = identity
        out.write("inputs.json", manifest("input", identity, sources=hashes, validation=validation))
        split = grouped_splits(samples, identity, held_out=held_out)
        out.write("splits.json", split)
        budgets = tuple(budgets)
        if not budgets or any(not 0 < b <= identity["experts"] for b in budgets):
            raise ValueError("All budgets must be within the model's expert count")
        tasks = sorted({s.task for s in samples})
        task_index = {t: i for i, t in enumerate(tasks)}
        # Each cell retains per-task and per-layer counts, not means of request means.
        cells = defaultdict(
            lambda: np.zeros((2, len(tasks), len(identity["layers"])), dtype=np.int64)
        )
        counts = defaultdict(int)
        parameters = []
        with (out.path / "requests.jsonl").open("x") as requests_file:
            for fold_index, fold in enumerate(split["folds"]):
                test = set(fold["test"])
                train = [s for s in samples if s.task not in test]
                prior = Priors(train)
                alpha, inner_scores = tune_mixture(train, Priors) if tune else (0.5, None)
                params = manifest(
                    "predictor",
                    identity,
                    held_out=sorted(test),
                    feature_version=FEATURE_VERSION,
                    label_version=label_version,
                    history_alpha=alpha,
                    ewma_alpha=0.5,
                    recent_window=32,
                    recent_window_scope="trailing computed positions within the corresponding previous phase",
                    tie_break="stable ascending expert ID; final K entries",
                    cold_start="global history; retain configured prior mixture",
                    protocol="frozen training priors + chronological same-session history",
                    test_session_sharing=False,
                    inner_scores=inner_scores,
                    parameters=prior.artifact(),
                    training_tasks=sorted({sample.task for sample in train}),
                )
                out.write(f"predictors/fold-{fold_index:02d}.json", params)
                parameters.append({k: v for k, v in params.items() if k != "parameters"})
                history = History(alpha=alpha)
                for sample in sorted(
                    (s for s in samples if s.task in test), key=lambda s: s.decision
                ):
                    provenance, metrics = {}, []
                    for policy in POLICIES:
                        distribution, provenance[policy] = history.predict(sample, prior, policy)
                        for budget in budgets:
                            hits = np.take_along_axis(
                                sample.hist, selected(distribution, budget), axis=-1
                            ).sum(axis=-1)
                            totals = sample.hist.sum(axis=-1)
                            for phase_index, phase in enumerate(PHASES):
                                numerator, denominator = hits[phase_index], totals[phase_index]
                                metrics.append(
                                    {
                                        "policy": policy,
                                        "budget": budget,
                                        "phase": phase,
                                        "hits": int(numerator.sum()),
                                        "lookups": int(denominator.sum()),
                                        "coverage": float(numerator.sum() / denominator.sum())
                                        if denominator.sum()
                                        else None,
                                        "reason": None
                                        if denominator.sum()
                                        else "No observed target lookups",
                                    }
                                )
                                for group in subgroups(sample.features):
                                    key = (policy, budget, phase, group)
                                    cell = cells[key]
                                    cell[0, task_index[sample.task]] += numerator
                                    cell[1, task_index[sample.task]] += denominator
                                    counts[key] += bool(denominator.sum())
                    requests_file.write(
                        json.dumps(
                            {
                                "request": sample.request,
                                "session": sample.session,
                                "task": sample.task,
                                "identity": identity["id"],
                                "features": sample.features,
                                "observed_at": sample.observed,
                                "provenance": provenance,
                                "metrics": metrics,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    history.defer(sample)
        summary = []
        for (policy, budget, phase, group), values in sorted(cells.items()):
            hits, totals = values.sum(axis=-1)
            layer_hits, layer_totals = values.sum(axis=1)
            summary.append(
                {
                    "policy": policy,
                    "budget": budget,
                    "phase": phase,
                    "subgroup": group,
                    "lookup_weighted": float(hits.sum() / totals.sum()) if totals.sum() else None,
                    "hits": int(hits.sum()),
                    "lookups": int(totals.sum()),
                    "requests": counts[(policy, budget, phase, group)],
                    **uncertainty(hits, totals),
                    "task_counts": {
                        task: {"hits": int(hits[i]), "lookups": int(totals[i])}
                        for i, task in enumerate(tasks)
                    },
                    "layers": [
                        {
                            "layer": layer,
                            "coverage": float(h / d) if d else None,
                            "hits": int(h),
                            "lookups": int(d),
                        }
                        for layer, h, d in zip(identity["layers"], layer_hits, layer_totals)
                    ],
                }
            )
        out.write(
            "metrics.json",
            {
                "schema_version": 1,
                "evidence": "fixed-set-lookup-analysis",
                "bootstrap": {"unit": "task", "repetitions": 1000, "seed": 42},
                "summary": summary,
            },
        )
        rows = [r for r in summary if r["subgroup"] == "all" and r["budget"] == 64]
        lines = [
            "# Exploratory fixed-set lookup coverage",
            "",
            "Held-out task groups include every execution copy. These tasks informed the design.",
            "No cache events, latency improvement, or speedup are measured.",
            "",
            "| Predictor | Phase | Lookup coverage | Task macro | Task 95% interval |",
            "|---|---|---:|---:|---:|",
        ]

        def percent(value):
            return f"{100 * value:.4f}%" if value is not None else "null"

        for row in rows:
            lo, hi = row["task_ci95"]
            lines.append(
                f"| {row['policy']} | {row['phase']} | {percent(row['lookup_weighted'])} | "
                f"{percent(row['task_macro'])} | {percent(lo)} – {percent(hi)} |"
            )
        lines.extend(
            [
                "",
                "Agent-feature increments are percentage-point differences at budget 64:",
                "",
                "| Subgroup | Phase | Tool minus history+global | Tool minus history+stage |",
                "|---|---|---:|---:|",
            ]
        )
        index = {(r["policy"], r["phase"], r["subgroup"]): r for r in summary if r["budget"] == 64}
        increments = []
        for group in ("all", "subsequent", "tool_switch", "multiple_tools"):
            for phase in PHASES:
                tool = index.get(("previous+tool", phase, group))
                if tool is None:
                    continue
                diffs = {}
                for base in ("previous+global", "previous+stage"):
                    baseline = index[(base, phase, group)]
                    if (
                        tool["lookup_weighted"] is not None
                        and baseline["lookup_weighted"] is not None
                    ):
                        diffs[base] = 100 * (tool["lookup_weighted"] - baseline["lookup_weighted"])
                    else:
                        diffs[base] = None
                increments.append({"phase": phase, "subgroup": group, "difference_pp": diffs})
                lines.append(
                    f"| {group} | {phase} | {diffs['previous+global']} | {diffs['previous+stage']} |"
                )
        out.write("increments.json", increments)
        (out.path / "report.md").write_text("\n".join(lines) + "\n")
        # Rehash after validation and analysis to detect accidental source mutation.
        if input_files(roots) != hashes:
            raise RuntimeError("Source data changed during analysis")
        out.finish(
            evidence="fixed-set-lookup-analysis",
            predictors=parameters,
            milestones={**out.record["milestones"], "analysis": "complete-exploratory"},
        )
        return out.path
    except Exception as exc:
        out.fail(exc)
        raise
