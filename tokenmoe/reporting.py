from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from tokenmoe.metrics import (
    evaluate_metadata_ablations,
    evaluate_segment_predictors,
    layer_histogram,
    locality_summary,
)
from tokenmoe.schema import WorkloadRecord
from tokenmoe.simulators import simulator_summary
from tokenmoe.trace import TraceRecord, validate_current_stage_traces


def _save_role_expert_heatmap(records: list[TraceRecord], path: Path) -> None:
    role_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        for layer in record.layers:
            role_counts[record.metadata.role].update(
                layer_histogram(record, layer.layer_id)
            )
    roles = sorted(role_counts)
    experts = sorted({expert for counts in role_counts.values() for expert in counts})
    data = np.zeros((len(roles), len(experts)), dtype=float)
    for i, role in enumerate(roles):
        total = sum(role_counts[role].values()) or 1
        for j, expert in enumerate(experts):
            data[i, j] = role_counts[role][expert] / total
    fig, ax = plt.subplots(figsize=(max(8, len(experts) * 0.45), max(4, len(roles) * 0.35)))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(roles)), roles)
    ax.set_xticks(range(len(experts)), experts)
    ax.set_xlabel("Expert ID")
    ax.set_title("Role-conditioned expert distribution")
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_topm_plot(summary: dict[str, object], path: Path) -> None:
    rows = summary["top_m_hit_rate"]
    if not rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Top-M results unavailable", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    labels = [f"{row['predictor']}@{row['top_m']}" for row in rows]  # type: ignore[index]
    values = [row["hit_rate"] for row in rows]  # type: ignore[index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(values)), values, color="#2f6f9f")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Expert-label hit rate")
    ax.set_title("RouteSig vs baseline top-M prediction")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _topm_rows_from_prediction(prediction_summary: dict[str, object]) -> list[dict[str, object]]:
    rows = prediction_summary.get("results", [])
    if not isinstance(rows, list):
        return []
    selected = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("predictor") not in {"global_frequency", "routesig"}:
            continue
        selected.append(
            {
                "predictor": row["predictor"],
                "top_m": row["top_m"],
                "hit_rate": row["expert_label_hit_rate"],
                "budget": row["budget"],
            }
        )
    return selected


def _layer_sensitivity_from_prediction(
    prediction_summary: dict[str, object],
    *,
    budget: str = "2x",
) -> dict[int, dict[str, float]]:
    rows = prediction_summary.get("results", [])
    if not isinstance(rows, list):
        return {}
    by_predictor = {
        row.get("predictor"): row
        for row in rows
        if isinstance(row, dict) and row.get("budget") == budget
    }
    global_row = by_predictor.get("global_frequency")
    routesig_row = by_predictor.get("routesig")
    if not isinstance(global_row, dict) or not isinstance(routesig_row, dict):
        return {}
    global_layers = global_row.get("per_layer", {})
    routesig_layers = routesig_row.get("per_layer", {})
    if not isinstance(global_layers, dict) or not isinstance(routesig_layers, dict):
        return {}
    result: dict[int, dict[str, float]] = {}
    for layer_key, global_metrics in global_layers.items():
        routesig_metrics = routesig_layers.get(layer_key)
        if not isinstance(global_metrics, dict) or not isinstance(routesig_metrics, dict):
            continue
        layer_id = int(layer_key)
        global_rate = float(global_metrics.get("expert_label_hit_rate", 0.0))
        routesig_rate = float(routesig_metrics.get("expert_label_hit_rate", 0.0))
        result[layer_id] = {
            "global_hit_rate": global_rate,
            "routesig_hit_rate": routesig_rate,
            "delta": routesig_rate - global_rate,
        }
    return result


def _save_layer_sensitivity(summary: dict[str, object], path: Path) -> None:
    raw = summary["layer_sensitivity"]
    if not raw:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Layer sensitivity unavailable", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return
    layers = sorted(int(layer) for layer in raw)
    global_vals = [raw[layer]["global_hit_rate"] for layer in layers]  # type: ignore[index]
    routesig_vals = [raw[layer]["routesig_hit_rate"] for layer in layers]  # type: ignore[index]
    x = np.arange(len(layers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, global_vals, width, label="Global")
    ax.bar(x + width / 2, routesig_vals, width, label="RouteSig")
    ax.set_xticks(x, layers)
    ax.set_ylim(0, 1)
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("Top-M hit rate")
    ax.set_title("Layer sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_simulator_plot(sim: dict[str, object], path: Path) -> None:
    scheduler = sim["scheduler"]  # type: ignore[index]
    policies = scheduler["policy_results"]  # type: ignore[index]
    labels = list(policies)
    values = [policies[name]["actual_mean_fanout"] for name in labels]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(labels, values, color=["#7d8ca3", "#2f6f9f", "#3b8f5a", "#cc7a29"])
    ax.set_title("Prediction-based scheduler replay")
    ax.set_ylabel("Actual active expert fanout")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_reports(
    *,
    records: list[TraceRecord],
    workloads: Iterable[WorkloadRecord],
    output_dir: str | Path = "analysis",
    report_dir: str | Path = "reports",
    require_current_stage: bool = True,
) -> dict[str, object]:
    started = time.perf_counter()

    def log_stage(stage: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"[analysis] {stage} at {elapsed:.2f}s", flush=True)

    if require_current_stage:
        log_stage("validating traces")
        validate_current_stage_traces(records)
    output = Path(output_dir)
    figures = output / "figures"
    reports = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    log_stage("computing locality summary")
    summary = locality_summary(records)
    log_stage("evaluating segment predictors (primary split 0.7)")
    prediction_summary = evaluate_segment_predictors(
        records,
        require_current_stage=require_current_stage,
    )
    summary["top_m_hit_rate"] = _topm_rows_from_prediction(prediction_summary)
    summary["layer_sensitivity"] = _layer_sensitivity_from_prediction(prediction_summary)
    log_stage("evaluating segment predictors (second split 0.5)")
    prediction_second_split = evaluate_segment_predictors(
        records,
        train_fraction=0.5,
        require_current_stage=require_current_stage,
    )
    log_stage("evaluating metadata ablations (2x budget)")
    ablation_summary = evaluate_metadata_ablations(
        records,
        require_current_stage=require_current_stage,
    )
    log_stage("running scheduler replay")
    sim = simulator_summary(records, workloads)
    metrics_payload = {
        "locality": summary,
        "prediction": prediction_summary,
        "prediction_second_split": prediction_second_split,
        "metadata_ablations": ablation_summary,
        "simulators": sim,
    }
    (output / "locality_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    log_stage("writing figures")
    heatmap = figures / "role_expert_heatmap.png"
    topm = figures / "topm_hit_rate.png"
    layer = figures / "layer_sensitivity.png"
    simulator = figures / "simulator_summary.png"
    _save_role_expert_heatmap(records, heatmap)
    _save_topm_plot(summary, topm)
    _save_layer_sensitivity(summary, layer)
    _save_simulator_plot(sim, simulator)

    log_stage("writing markdown reports")
    topm_rows = summary["top_m_hit_rate"]  # type: ignore[index]
    routesig_rows = [row for row in topm_rows if row["predictor"] == "routesig"]
    global_rows = [
        row for row in topm_rows if row["predictor"] == "global_frequency"
    ]
    routesig_best = (
        max(routesig_rows, key=lambda row: row["hit_rate"])
        if routesig_rows
        else None
    )
    global_best = (
        max(global_rows, key=lambda row: row["hit_rate"])
        if global_rows
        else None
    )
    routesig_best_text = (
        f"**{routesig_best['hit_rate']:.3f}** at top-{routesig_best['top_m']}"
        if routesig_best
        else "**n/a** at top-n/a"
    )
    global_best_text = (
        f"**{global_best['hit_rate']:.3f}** at top-{global_best['top_m']}"
        if global_best
        else "**n/a** at top-n/a"
    )
    scheduler = sim["scheduler"]  # type: ignore[index]
    policy_results = scheduler["policy_results"]  # type: ignore[index]
    prediction_rows = prediction_summary.get("results", []) if isinstance(prediction_summary, dict) else []
    main_prediction = next(
        (
            row
            for row in prediction_rows
            if row["predictor"] == "routesig" and row["budget"] == "2x"
        ),
        None,
    )
    main_global = next(
        (
            row
            for row in prediction_rows
            if row["predictor"] == "global_frequency" and row["budget"] == "2x"
        ),
        None,
    )

    locality_md = f"""# TokenMoE Locality Report

Records: **{len(records)}**. Scope: prompt-only vLLM MoE traces and offline
prediction replay.

![Role expert heatmap](figures/role_expert_heatmap.png)

![Top-M hit rate](figures/topm_hit_rate.png)

![Layer sensitivity](figures/layer_sensitivity.png)

## Metrics

- RouteSig best hit rate: {routesig_best_text}.
- Global-frequency best hit rate: {global_best_text}.
- Cross-agent JSD: **{summary['cross_agent_divergence']['mean_jsd']:.4f}**.
- Mean route entropy: **{summary['route_entropy']['mean_entropy']:.4f}**.

## Simulator Snapshot

![Simulator summary](figures/simulator_summary.png)

- Scheduler replay kind: **{scheduler['replay_kind']}**.
- Scheduler mean active expert fanout: FIFO **{scheduler['baseline_mean_fanout']:.2f}**, RouteSig prediction **{scheduler['tokenmoe_mean_fanout']:.2f}**.
- Low-confidence gated fraction: **{scheduler['low_confidence_gated_fraction']:.3f}**.
- Prefetch and EPLB replay outputs are archived/excluded from this change.
"""
    (output / "locality_report.md").write_text(locality_md, encoding="utf-8")

    final_report = f"""# TokenMoE vLLM Prototype Report

## Executive Summary

This prototype validates a current-stage TokenMoE path: pre-router metadata and
prompt block spans are joined with prompt-only vLLM MoE routed-expert traces,
RouteSig predicts per-segment/per-layer expert working sets, and offline
scheduler replay uses predicted demand before any runtime vLLM scheduler change.

Trace records: **{len(records)}**. Model backend: **{records[0].backend if records else 'n/a'}**.

## Agent-Router Locality

![Role-conditioned expert distribution](../analysis/figures/role_expert_heatmap.png)

The heatmap shows that roles and phases do not activate experts uniformly.
This is the signal RouteSig exploits before request admission.

## Predictor Results

![Top-M predictor comparison](../analysis/figures/topm_hit_rate.png)

Main 2x budget RouteSig expert-label hit rate: **{main_prediction['expert_label_hit_rate'] if main_prediction else 'n/a'}**.
Main 2x budget global-frequency expert-label hit rate: **{main_global['expert_label_hit_rate'] if main_global else 'n/a'}**.
Oracle rows remain separate in `analysis/locality_metrics.json` and are not used
as prediction results.

## Layer Sensitivity

![Layer sensitivity](../analysis/figures/layer_sensitivity.png)

Layer sensitivity identifies where metadata is useful. Runtime integration
should prioritize high-delta MoE layers and leave low-confidence layers on the
baseline path.

## Offline Scheduler Replay

![Simulator summary](../analysis/figures/simulator_summary.png)

- FIFO actual fanout: **{policy_results['fifo']['actual_mean_fanout']:.2f}**.
- Temporal-window actual fanout: **{policy_results['temporal_window_frequency']['actual_mean_fanout']:.2f}**.
- RouteSig prediction actual fanout: **{policy_results['routesig']['actual_mean_fanout']:.2f}**.
- Oracle future-demand actual fanout: **{policy_results['oracle_future_demand']['actual_mean_fanout']:.2f}**.
- Replay kind: **{scheduler['replay_kind']}**; DAG scheduling available: **{scheduler['dag_scheduling_available']}**.

Prefetch replay, online expert prefetch, vLLM scheduler modification, and EPLB
replica placement are excluded from this validation stage and are next-stage
work.

## Correctness Boundary

All prototype optimizations are confidence-gated and replay-only. Real router
outputs remain the source of truth. When metadata is missing, confidence is low,
or a prediction misses, the runtime falls back to normal vLLM routing/dispatch.
"""
    (reports / "tokenmoe_report.md").write_text(final_report, encoding="utf-8")
    log_stage("done")
    return metrics_payload
