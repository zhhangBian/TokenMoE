"""Leakage-resistant evaluation for expert working-set predictors."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from tokenmoe.prediction import (
    PredictedExpertSet,
    SegmentPredictor,
    build_predictors,
    segment_scores,
    segment_selected,
    top_m_budgets,
)
from tokenmoe.trace import TraceRecord, validate_traces


BOOTSTRAP_RESAMPLES = 1_000


def _group(record: TraceRecord) -> str:
    return record.source_group_id


def split_by_group(
    records: Iterable[TraceRecord], train_fraction: float
) -> tuple[list[TraceRecord], list[TraceRecord], dict[str, object]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    materialized = list(records)
    groups = list(dict.fromkeys(_group(record) for record in materialized))
    if len(groups) < 2:
        raise ValueError("evaluation requires at least two source groups")
    cutoff = min(len(groups) - 1, max(1, int(len(groups) * train_fraction)))
    train_groups = set(groups[:cutoff])
    train = [record for record in materialized if _group(record) in train_groups]
    evaluation = [
        record for record in materialized if _group(record) not in train_groups
    ]
    return (
        train,
        evaluation,
        {
            "train_fraction": train_fraction,
            "train_groups": cutoff,
            "evaluation_groups": len(groups) - cutoff,
            "train_records": len(train),
            "evaluation_records": len(evaluation),
        },
    )


@dataclass(slots=True)
class ScoreAccumulator:
    expert_hits: int = 0
    expert_total: int = 0
    exact_token_hits: int = 0
    token_total: int = 0
    score_covered: float = 0.0
    score_total: float = 0.0
    confidence_total: float = 0.0
    predictions: int = 0
    unaligned_segments: int = 0
    group_stats: dict[str, list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )

    def update(
        self,
        prediction: PredictedExpertSet,
        selected: np.ndarray,
        scores: np.ndarray | None,
        group: str,
    ) -> None:
        predicted = set(prediction.expert_ids)
        flat = selected.reshape(-1)
        hits = int(sum(int(expert_id) in predicted for expert_id in flat))
        self.expert_hits += hits
        self.expert_total += int(flat.size)
        self.token_total += int(selected.shape[0])
        self.exact_token_hits += int(
            sum(set(int(item) for item in row).issubset(predicted) for row in selected)
        )
        self.confidence_total += prediction.confidence
        self.predictions += 1
        self.group_stats[group][0] += hits
        self.group_stats[group][1] += int(flat.size)
        if scores is not None:
            score_array = np.asarray(scores, dtype=np.float64)
            self.score_total += float(score_array.sum())
            self.score_covered += float(
                score_array[
                    np.isin(selected, np.fromiter(predicted, dtype=np.int64))
                ].sum()
            )

    def summary(self) -> dict[str, object]:
        return {
            "expert_label_hit_rate": (
                self.expert_hits / self.expert_total if self.expert_total else 0.0
            ),
            "exact_token_hit_rate": (
                self.exact_token_hits / self.token_total if self.token_total else 0.0
            ),
            "weighted_coverage": (
                self.score_covered / self.score_total if self.score_total else None
            ),
            "mean_confidence": (
                self.confidence_total / self.predictions if self.predictions else 0.0
            ),
            "expert_labels": self.expert_total,
            "tokens": self.token_total,
            "predictions": self.predictions,
            "unaligned_segments": self.unaligned_segments,
        }


def paired_group_bootstrap(
    groups: Iterable[str],
    candidate: dict[str, list[int]],
    baseline: dict[str, list[int]],
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, object]:
    group_list = list(groups)
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    candidate_values = np.asarray(
        [candidate.get(group, [0, 0]) for group in group_list], dtype=np.float64
    )
    baseline_values = np.asarray(
        [baseline.get(group, [0, 0]) for group in group_list], dtype=np.float64
    )

    def rate(values: np.ndarray) -> float:
        total = values[:, 1].sum()
        return float(values[:, 0].sum() / total) if total else 0.0

    result: dict[str, object] = {
        "delta": rate(candidate_values) - rate(baseline_values),
        "groups": len(group_list),
        "resamples": n_resamples,
    }
    if len(group_list) < 2:
        return {**result, "ci_low": None, "ci_high": None, "significant": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(group_list), size=(n_resamples, len(group_list)))
    candidate_sample = candidate_values[indices]
    baseline_sample = baseline_values[indices]
    candidate_rates = candidate_sample[:, :, 0].sum(axis=1) / np.maximum(
        candidate_sample[:, :, 1].sum(axis=1), 1.0
    )
    baseline_rates = baseline_sample[:, :, 0].sum(axis=1) / np.maximum(
        baseline_sample[:, :, 1].sum(axis=1), 1.0
    )
    low, high = np.percentile(candidate_rates - baseline_rates, [2.5, 97.5])
    return {
        **result,
        "ci_low": float(low),
        "ci_high": float(high),
        "significant": bool(low > 0.0 or high < 0.0),
    }


def _score_predictor(
    predictor: SegmentPredictor, records: Iterable[TraceRecord], top_m: int
) -> ScoreAccumulator:
    accumulator = ScoreAccumulator()
    for record in records:
        for segment in record.prompt_segments:
            if segment.token_start is None or segment.token_end is None:
                accumulator.unaligned_segments += 1
                continue
            for layer in record.layers:
                prediction = predictor.predict(record, segment, layer.layer_id, top_m)
                accumulator.update(
                    prediction,
                    segment_selected(record, segment, layer.layer_id),
                    segment_scores(record, segment, layer.layer_id),
                    _group(record),
                )
        predictor.update(record)
    return accumulator


def evaluate_split(
    records: Iterable[TraceRecord],
    *,
    train_fraction: float,
    temporal_window: int = 64,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    materialized = list(records)
    validate_traces(materialized)
    train, evaluation, split = split_by_group(materialized, train_fraction)
    router_top_k = materialized[0].router_top_k
    num_experts = materialized[0].num_experts
    rows: list[dict[str, object]] = []
    for budget, requested_top_m in top_m_budgets(router_top_k).items():
        top_m = min(requested_top_m, num_experts)
        accumulators: dict[str, ScoreAccumulator] = {}
        for predictor in build_predictors(
            train, top_m=top_m, temporal_window=temporal_window
        ):
            accumulator = _score_predictor(predictor, evaluation, top_m)
            accumulators[predictor.name] = accumulator
            rows.append(
                {
                    "predictor": predictor.name,
                    "budget": budget,
                    "top_m": top_m,
                    **accumulator.summary(),
                }
            )
        groups = list(dict.fromkeys(_group(record) for record in evaluation))
        for row in rows:
            if row["budget"] != budget:
                continue
            candidate = accumulators[str(row["predictor"])]
            comparisons = {}
            for baseline_name in ("global_frequency", "temporal_window"):
                if row["predictor"] == baseline_name:
                    continue
                comparisons[baseline_name] = paired_group_bootstrap(
                    groups,
                    candidate.group_stats,
                    accumulators[baseline_name].group_stats,
                    n_resamples=bootstrap_resamples,
                )
            row["delta_vs"] = comparisons
    return {"split": split, "results": rows}


def evaluate_traces(
    records: Iterable[TraceRecord],
    *,
    train_fractions: tuple[float, ...] = (0.7, 0.5),
    temporal_window: int = 64,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    materialized = list(records)
    if not materialized:
        raise ValueError("trace file is empty")
    validate_traces(materialized)
    first = materialized[0]
    return {
        "schema_version": first.schema_version,
        "backend": first.backend,
        "model_id": first.model_id,
        "records": len(materialized),
        "source_groups": len({_group(record) for record in materialized}),
        "prompt_tokens": sum(record.prompt_token_count for record in materialized),
        "moe_layers": len(first.moe_layer_ids),
        "router_top_k": first.router_top_k,
        "num_experts": first.num_experts,
        "router_scores_available": first.has_router_scores,
        "splits": {
            str(fraction): evaluate_split(
                materialized,
                train_fraction=fraction,
                temporal_window=temporal_window,
                bootstrap_resamples=bootstrap_resamples,
            )
            for fraction in train_fractions
        },
    }
