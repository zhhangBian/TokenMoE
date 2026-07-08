from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from tokenmoe.metrics import locality_summary
from tokenmoe.schema import WorkloadRecord
from tokenmoe.simulators import simulator_summary
from tokenmoe.trace import TraceRecord


def _save_role_expert_heatmap(records: list[TraceRecord], path: Path) -> None:
    role_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        for layer in record.layers:
            role_counts[record.metadata.role].update(
                int(expert) for expert in layer.selected_array().reshape(-1)
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


def _save_layer_sensitivity(summary: dict[str, object], path: Path) -> None:
    raw = summary["layer_sensitivity"]
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
    prefetch = sim["prefetch"]  # type: ignore[index]
    scheduler = sim["scheduler"]  # type: ignore[index]
    eplb = sim["eplb"]  # type: ignore[index]
    labels = [
        "prefetch hit",
        "fallback hit",
        "fanout ratio",
        "EPLB p95 ratio",
    ]
    values = [
        prefetch["hit_rate"],
        prefetch["fallback_hit_rate"],
        scheduler["tokenmoe_mean_fanout"]
        / max(1e-9, scheduler["baseline_mean_fanout"]),
        eplb["tokenmoe_p95_tail"] / max(1e-9, eplb["moving_average_p95_tail"]),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(labels, values, color=["#2f6f9f", "#7d8ca3", "#cc7a29", "#3b8f5a"])
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Simulator summary")
    ax.set_ylabel("Rate or ratio")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_reports(
    *,
    records: list[TraceRecord],
    workloads: Iterable[WorkloadRecord],
    output_dir: str | Path = "analysis",
    report_dir: str | Path = "reports",
) -> dict[str, object]:
    output = Path(output_dir)
    figures = output / "figures"
    reports = Path(report_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    summary = locality_summary(records)
    sim = simulator_summary(records, workloads)
    metrics_payload = {"locality": summary, "simulators": sim}
    (output / "locality_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    heatmap = figures / "role_expert_heatmap.png"
    topm = figures / "topm_hit_rate.png"
    layer = figures / "layer_sensitivity.png"
    simulator = figures / "simulator_summary.png"
    _save_role_expert_heatmap(records, heatmap)
    _save_topm_plot(summary, topm)
    _save_layer_sensitivity(summary, layer)
    _save_simulator_plot(sim, simulator)

    topm_rows = summary["top_m_hit_rate"]  # type: ignore[index]
    routesig_best = max(
        (row for row in topm_rows if row["predictor"] == "routesig"),
        key=lambda row: row["hit_rate"],
    )
    global_best = max(
        (row for row in topm_rows if row["predictor"] == "global_frequency"),
        key=lambda row: row["hit_rate"],
    )
    prefetch = sim["prefetch"]  # type: ignore[index]
    scheduler = sim["scheduler"]  # type: ignore[index]
    eplb = sim["eplb"]  # type: ignore[index]

    locality_md = f"""# TokenMoE Locality Report

Records: **{len(records)}**.

![Role expert heatmap](figures/role_expert_heatmap.png)

![Top-M hit rate](figures/topm_hit_rate.png)

![Layer sensitivity](figures/layer_sensitivity.png)

## Metrics

- RouteSig best hit rate: **{routesig_best['hit_rate']:.3f}** at top-{routesig_best['top_m']}.
- Global-frequency best hit rate: **{global_best['hit_rate']:.3f}** at top-{global_best['top_m']}.
- Cross-agent JSD: **{summary['cross_agent_divergence']['mean_jsd']:.4f}**.
- Mean route entropy: **{summary['route_entropy']['mean_entropy']:.4f}**.

## Simulator Snapshot

![Simulator summary](figures/simulator_summary.png)

- Prefetch hit rate: **{prefetch['hit_rate']:.3f}**; wasted prefetch rate: **{prefetch['wasted_prefetch_rate']:.3f}**.
- Scheduler mean active expert fanout: baseline **{scheduler['baseline_mean_fanout']:.2f}**, TokenMoE replay **{scheduler['tokenmoe_mean_fanout']:.2f}**.
- EPLB p95 tail proxy: moving average **{eplb['moving_average_p95_tail']:.2f}**, TokenMoE future-demand replay **{eplb['tokenmoe_p95_tail']:.2f}**.
"""
    (output / "locality_report.md").write_text(locality_md, encoding="utf-8")

    final_report = f"""# TokenMoE vLLM Prototype Report

## Executive Summary

This prototype validates a trace-first TokenMoE path: agent metadata is joined
with real MoE router traces, RouteSig predicts per-layer expert working sets,
and prefetch/scheduling/EPLB policies are evaluated in replay before any
correctness-sensitive vLLM runtime changes.

Trace records: **{len(records)}**. Model backend: **{records[0].backend if records else 'n/a'}**.

## Agent-Router Locality

![Role-conditioned expert distribution](../analysis/figures/role_expert_heatmap.png)

The heatmap shows that roles and phases do not activate experts uniformly.
This is the signal RouteSig exploits before request admission.

## Predictor Results

![Top-M predictor comparison](../analysis/figures/topm_hit_rate.png)

RouteSig best hit rate: **{routesig_best['hit_rate']:.3f}**.
Global-frequency best hit rate: **{global_best['hit_rate']:.3f}**.
The delta is the first-order evidence for or against agent-conditioned
expert prediction on this workload/model pair.

## Layer Sensitivity

![Layer sensitivity](../analysis/figures/layer_sensitivity.png)

Layer sensitivity identifies where metadata is useful. Runtime integration
should prioritize high-delta MoE layers and leave low-confidence layers on the
baseline path.

## System Replay

![Simulator summary](../analysis/figures/simulator_summary.png)

- Prefetch hit rate: **{prefetch['hit_rate']:.3f}** with **{prefetch['wasted_prefetch_rate']:.3f}** wasted-prefetch rate.
- Estimated stall reduction proxy: **{prefetch['estimated_stall_reduction_ms']:.2f} ms** over the replay window.
- Expert-overlap scheduler fanout: baseline **{scheduler['baseline_mean_fanout']:.2f}**, TokenMoE **{scheduler['tokenmoe_mean_fanout']:.2f}**.
- EPLB p95 tail proxy: moving-average **{eplb['moving_average_p95_tail']:.2f}**, TokenMoE **{eplb['tokenmoe_p95_tail']:.2f}**.

## Correctness Boundary

All prototype optimizations are confidence-gated and replay-only. Real router
outputs remain the source of truth. When metadata is missing, confidence is low,
or a prediction misses, the runtime falls back to normal vLLM routing/dispatch.
"""
    (reports / "tokenmoe_report.md").write_text(final_report, encoding="utf-8")
    return metrics_payload
