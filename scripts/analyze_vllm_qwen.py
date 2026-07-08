#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze vLLM/Qwen TokenMoE runs.")
    parser.add_argument("--data-dir", default="data/vllm_qwen")
    parser.add_argument("--analysis-dir", default="analysis/vllm_qwen")
    parser.add_argument("--report-dir", default="reports/vllm_qwen")
    return parser.parse_args()


def write_figures(df: pd.DataFrame, analysis_dir: Path) -> dict[str, str]:
    sns.set_theme(style="whitegrid", context="paper")
    fig_dir = analysis_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

    plt.figure(figsize=(7.0, 4.0))
    sns.barplot(
        data=df,
        x="batch_size",
        y="latency_ms",
        hue="mode",
        errorbar=("pi", 95),
    )
    plt.ylabel("Per-request latency (ms)")
    plt.xlabel("vLLM batch size")
    plt.tight_layout()
    path = fig_dir / "latency_by_batch.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["latency_by_batch"] = str(path)

    summary = (
        df.groupby(["mode", "batch_size"])
        .agg(
            requests=("request_id", "count"),
            total_batch_ms=("batch_latency_ms", "sum"),
            output_tokens=("output_tokens", "sum"),
        )
        .reset_index()
    )
    summary["estimated_wall_s"] = (
        summary["total_batch_ms"] / summary["batch_size"] / 1000.0
    )
    summary["throughput_req_s"] = summary["requests"] / summary["estimated_wall_s"]
    summary["throughput_tok_s"] = (
        summary["output_tokens"] / summary["estimated_wall_s"]
    )

    plt.figure(figsize=(7.0, 4.0))
    sns.lineplot(
        data=summary,
        x="batch_size",
        y="throughput_req_s",
        hue="mode",
        marker="o",
    )
    plt.ylabel("Throughput (requests/s)")
    plt.xlabel("vLLM batch size")
    plt.tight_layout()
    path = fig_dir / "throughput_by_batch.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["throughput_by_batch"] = str(path)

    sidecar = df[df["mode"] == "tokenmoe_sidecar"].copy()
    plt.figure(figsize=(7.0, 4.0))
    sns.boxplot(data=sidecar, x="batch_size", y="sidecar_elapsed_us")
    plt.ylabel("Sidecar decision time (us)")
    plt.xlabel("vLLM batch size")
    plt.tight_layout()
    path = fig_dir / "sidecar_overhead.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["sidecar_overhead"] = str(path)

    plt.figure(figsize=(8.0, 4.2))
    role_order = (
        df.groupby("role")["latency_ms"].median().sort_values(ascending=False).index
    )
    sns.boxplot(data=df, x="role", y="latency_ms", hue="mode", order=role_order)
    plt.ylabel("Per-request latency (ms)")
    plt.xlabel("Agent role")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path = fig_dir / "latency_by_role.png"
    plt.savefig(path, dpi=180)
    plt.close()
    figures["latency_by_role"] = str(path)

    return figures


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["mode", "batch_size"])
        .agg(
            requests=("request_id", "count"),
            mean_latency_ms=("latency_ms", "mean"),
            p50_latency_ms=("latency_ms", "median"),
            p95_latency_ms=("latency_ms", lambda s: s.quantile(0.95)),
            mean_sidecar_us=("sidecar_elapsed_us", "mean"),
            output_tokens=("output_tokens", "sum"),
            total_batch_ms=("batch_latency_ms", "sum"),
        )
        .reset_index()
    )
    # batch_latency_ms is stored once per request in the batch. Divide by the
    # batch size to recover total wall time for this fixed-size experiment.
    grouped["estimated_wall_s"] = grouped["total_batch_ms"] / grouped[
        "batch_size"
    ] / 1000.0
    grouped["throughput_req_s"] = grouped["requests"] / (
        grouped["estimated_wall_s"]
    )
    grouped["throughput_output_tok_s"] = grouped["output_tokens"] / (
        grouped["estimated_wall_s"]
    )
    return grouped


def output_stability(data_dir: Path) -> pd.DataFrame:
    generations_path = data_dir / "generations.jsonl"
    if not generations_path.exists() or generations_path.stat().st_size == 0:
        return pd.DataFrame(
            columns=["batch_size", "requests", "exact_output_match_rate"]
        )
    gen = pd.read_json(generations_path, lines=True)
    rows = []
    for batch_size, group in gen.groupby("batch_size"):
        base = group[group["mode"] == "baseline"][
            ["request_id", "output_text"]
        ].rename(columns={"output_text": "baseline_output"})
        sidecar = group[group["mode"] == "tokenmoe_sidecar"][
            ["request_id", "output_text"]
        ].rename(columns={"output_text": "sidecar_output"})
        joined = base.merge(sidecar, on="request_id", how="inner")
        if joined.empty:
            match_rate = 0.0
        else:
            match_rate = (
                joined["baseline_output"] == joined["sidecar_output"]
            ).mean()
        rows.append(
            {
                "batch_size": int(batch_size),
                "requests": int(len(joined)),
                "exact_output_match_rate": float(match_rate),
            }
        )
    return pd.DataFrame(rows).sort_values("batch_size")


def write_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    capability: dict,
    figures: dict[str, str],
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    cap = capability["capability"]
    best_batch = summary.sort_values("throughput_req_s", ascending=False).iloc[0]
    sidecar = df[df["mode"] == "tokenmoe_sidecar"]
    fallback_reasons = (
        sidecar["sidecar_fallback_reason"].replace("", "none").value_counts().to_dict()
    )
    batch_sizes = sorted(int(x) for x in df["batch_size"].unique())
    modes = sorted(str(x) for x in df["mode"].unique())
    unique_requests = int(df["request_id"].nunique())
    exact_match_rate = (
        stability["exact_output_match_rate"].mean() if not stability.empty else 0.0
    )
    lines = [
        "# TokenMoE vLLM/Qwen End-to-End Report",
        "",
        "## Setup",
        "",
        f"- Model: `{capability['model']}`",
        f"- vLLM: `{capability['vllm_version']}` from `{capability['vllm_file']}`",
        f"- CUDA_VISIBLE_DEVICES: `{capability.get('cuda_visible_devices')}`",
        f"- Unique workload requests: {unique_requests}",
        f"- Batch sizes: {batch_sizes}",
        f"- Modes: {modes}",
        f"- Model type: `{cap['model_type']}`",
        f"- Architectures: `{cap['architectures']}`",
        f"- MoE routed expert capture supported: `{cap['supports_routed_expert_capture']}`",
        f"- Fallback reason: `{cap.get('fallback_reason')}`",
        "",
        "## Key Results",
        "",
        (
            f"- Best observed throughput was {best_batch['throughput_req_s']:.2f} "
            f"requests/s at batch size {int(best_batch['batch_size'])} "
            f"in `{best_batch['mode']}` mode."
        ),
        (
            f"- TokenMoE sidecar metadata decisions averaged "
            f"{sidecar['sidecar_elapsed_us'].mean():.2f} us/request "
            f"(p95 {sidecar['sidecar_elapsed_us'].quantile(0.95):.2f} us)."
        ),
        f"- Sidecar fallback reasons: `{fallback_reasons}`.",
        f"- Baseline vs sidecar exact output match averaged {exact_match_rate:.3f} across separately sharded batch-size runs.",
        "",
        "The requested Qwen2.5-7B-Instruct model is a dense Qwen2 CausalLM model, "
        "so the vLLM path was exercised as a real serving run while routed-expert "
        "capture was correctly disabled by capability detection. The sidecar does "
        "not alter prompts, sampling parameters, or model weights; exact-output "
        "matching is reported as a measurement because vLLM batch execution is not "
        "bitwise stable across independently restarted shards.",
        "",
        "## Figures",
        "",
        f"![Latency by batch]({Path(figures['latency_by_batch']).relative_to(report_dir.parent.parent)})",
        "",
        f"![Throughput by batch]({Path(figures['throughput_by_batch']).relative_to(report_dir.parent.parent)})",
        "",
        f"![Sidecar overhead]({Path(figures['sidecar_overhead']).relative_to(report_dir.parent.parent)})",
        "",
        f"![Latency by role]({Path(figures['latency_by_role']).relative_to(report_dir.parent.parent)})",
        "",
        "## Summary Table",
        "",
        summary[
            [
                "mode",
                "batch_size",
                "requests",
                "mean_latency_ms",
                "p95_latency_ms",
                "throughput_req_s",
                "throughput_output_tok_s",
                "mean_sidecar_us",
            ]
        ].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Output Stability",
        "",
        stability.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Data",
        "",
        "- Raw generations: `data/vllm_qwen/generations.jsonl`",
        "- Sidecar decisions: `data/vllm_qwen/sidecar_decisions.jsonl`",
        "- Latency CSV: `data/vllm_qwen/latency.csv`",
        "- Capability JSON: `analysis/vllm_qwen/model_capability.json`",
        "- Summary JSON: `analysis/vllm_qwen/experiment_summary.json`",
    ]
    (report_dir / "e2e_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    analysis_dir = Path(args.analysis_dir)
    report_dir = Path(args.report_dir)
    df = pd.read_csv(data_dir / "latency.csv")
    capability = json.loads((analysis_dir / "model_capability.json").read_text())
    figures = write_figures(df, analysis_dir)
    summary = summarize(df)
    stability = output_stability(data_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(analysis_dir / "latency_summary.csv", index=False)
    (analysis_dir / "latency_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    stability.to_csv(analysis_dir / "output_stability.csv", index=False)
    combined = {
        "model": capability["model"],
        "vllm_version": capability["vllm_version"],
        "vllm_file": capability["vllm_file"],
        "cuda_visible_devices": capability.get("cuda_visible_devices"),
        "capability": capability["capability"],
        "batch_sizes": sorted(int(x) for x in df["batch_size"].unique()),
        "modes": sorted(str(x) for x in df["mode"].unique()),
        "unique_requests": int(df["request_id"].nunique()),
        "rows": int(len(df)),
        "sidecar_fallback_reasons": df[df["mode"] == "tokenmoe_sidecar"][
            "sidecar_fallback_reason"
        ]
        .replace("", "none")
        .value_counts()
        .to_dict(),
        "latency_summary": json.loads(summary.to_json(orient="records")),
        "output_stability": json.loads(stability.to_json(orient="records")),
    }
    (analysis_dir / "experiment_summary.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(df, summary, stability, capability, figures, report_dir)
    print(f"wrote {analysis_dir / 'latency_summary.csv'}")
    print(f"wrote {report_dir / 'e2e_report.md'}")


if __name__ == "__main__":
    main()
