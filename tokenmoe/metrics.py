from __future__ import annotations

import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from tokenmoe.routesig import RouteSigStore, route_entropy_from_probabilities
from tokenmoe.schema import AgentNodeMeta
from tokenmoe.trace import TraceRecord, validate_current_stage_traces
from tokenmoe.prediction import (
    PredictedExpertSet,
    build_predictors,
    record_segments,
    selected_array_for_layer,
    top_m_budgets,
)


def split_records(
    records: list[TraceRecord], train_fraction: float = 0.7
) -> tuple[list[TraceRecord], list[TraceRecord]]:
    if not records:
        return [], []
    split = max(1, min(len(records) - 1, int(len(records) * train_fraction)))
    return records[:split], records[split:]


def layer_histogram(record: TraceRecord, layer_id: int) -> Counter[int]:
    selected = selected_array_for_layer(record, layer_id)
    valid = selected.reshape(-1)
    valid = valid[valid >= 0]
    if valid.size == 0:
        return Counter()
    experts, counts = np.unique(valid.astype(np.int64, copy=False), return_counts=True)
    return Counter({int(expert): int(count) for expert, count in zip(experts, counts)})


def distribution(counter: Counter[int]) -> dict[int, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {expert: count / total for expert, count in counter.items()}


def jensen_shannon_divergence(
    p: dict[int, float], q: dict[int, float], eps: float = 1e-12
) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    p_arr = np.asarray([p.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    q_arr = np.asarray([q.get(k, 0.0) for k in keys], dtype=np.float64) + eps
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    m = 0.5 * (p_arr + q_arr)
    return float(0.5 * np.sum(p_arr * np.log(p_arr / m)) + 0.5 * np.sum(q_arr * np.log(q_arr / m)))


def group_layer_distributions(
    records: Iterable[TraceRecord], group_level: str = "role"
) -> dict[tuple[str, int], dict[int, float]]:
    counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
    for record in records:
        group = record.metadata.group_key(group_level)
        for layer in record.layers:
            counts[(group, layer.layer_id)].update(layer_histogram(record, layer.layer_id))
    return {key: distribution(counter) for key, counter in counts.items()}


def agent_expert_overlap(
    records: Iterable[TraceRecord], group_level: str = "role"
) -> dict[str, float]:
    group_active: dict[tuple[str, int], set[int]] = defaultdict(set)
    for record in records:
        group = record.metadata.group_key(group_level)
        for layer in record.layers:
            group_active[(group, layer.layer_id)].update(layer.active_expert_histogram.keys())
    same_layer_scores = []
    by_layer: dict[int, list[tuple[str, set[int]]]] = defaultdict(list)
    for (group, layer_id), active in group_active.items():
        by_layer[layer_id].append((group, {int(x) for x in active}))
    for items in by_layer.values():
        for i, (_, a) in enumerate(items):
            for _, b in items[i + 1 :]:
                union = a | b
                if union:
                    same_layer_scores.append(len(a & b) / len(union))
    return {
        "mean_cross_group_jaccard": float(np.mean(same_layer_scores))
        if same_layer_scores
        else 0.0,
        "num_pairs": float(len(same_layer_scores)),
    }


def cross_agent_divergence(
    records: Iterable[TraceRecord], group_level: str = "role"
) -> dict[str, float]:
    dists = group_layer_distributions(records, group_level)
    by_layer: dict[int, list[tuple[str, dict[int, float]]]] = defaultdict(list)
    for (group, layer_id), dist in dists.items():
        by_layer[layer_id].append((group, dist))
    layer_values: dict[int, list[float]] = defaultdict(list)
    for layer_id, items in by_layer.items():
        for i, (_, p) in enumerate(items):
            for _, q in items[i + 1 :]:
                layer_values[layer_id].append(jensen_shannon_divergence(p, q))
    per_layer = {
        layer: float(np.mean(values)) if values else 0.0
        for layer, values in layer_values.items()
    }
    all_values = [value for values in layer_values.values() for value in values]
    return {
        "mean_jsd": float(np.mean(all_values)) if all_values else 0.0,
        "per_layer": per_layer,
    }


def route_entropy(records: Iterable[TraceRecord]) -> dict[str, float]:
    entropies = []
    per_layer: dict[int, list[float]] = defaultdict(list)
    for record in records:
        for layer in record.layers:
            hist = layer_histogram(record, layer.layer_id)
            entropy = route_entropy_from_probabilities(distribution(hist).values())
            entropies.append(entropy)
            per_layer[layer.layer_id].append(entropy)
    return {
        "mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "per_layer": {
            layer: float(np.mean(values)) for layer, values in sorted(per_layer.items())
        },
    }


class Predictor(Protocol):
    name: str

    def predict(self, meta: AgentNodeMeta, layer_id: int, top_m: int) -> list[int]:
        ...

    def update(self, record: TraceRecord) -> None:
        ...


class GlobalFrequencyPredictor:
    name = "global_frequency"

    def __init__(self, train_records: Iterable[TraceRecord]) -> None:
        self.counts: dict[int, Counter[int]] = defaultdict(Counter)
        for record in train_records:
            self.update(record)

    def predict(self, meta: AgentNodeMeta, layer_id: int, top_m: int) -> list[int]:
        return [expert for expert, _ in self.counts[layer_id].most_common(top_m)]

    def update(self, record: TraceRecord) -> None:
        for layer in record.layers:
            self.counts[layer.layer_id].update(layer_histogram(record, layer.layer_id))


class LRUPredictor:
    name = "request_lru"

    def __init__(self, train_records: Iterable[TraceRecord], maxlen: int = 128) -> None:
        self.history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.global_predictor = GlobalFrequencyPredictor(train_records)
        for record in train_records:
            self.update(record)

    def predict(self, meta: AgentNodeMeta, layer_id: int, top_m: int) -> list[int]:
        seen = []
        for expert in reversed(self.history[layer_id]):
            if expert not in seen:
                seen.append(expert)
            if len(seen) >= top_m:
                return seen
        for expert in self.global_predictor.predict(meta, layer_id, top_m):
            if expert not in seen:
                seen.append(expert)
            if len(seen) >= top_m:
                break
        return seen

    def update(self, record: TraceRecord) -> None:
        for layer in record.layers:
            for expert in layer_histogram(record, layer.layer_id):
                self.history[layer.layer_id].append(expert)


class SequenceHistoryPredictor:
    name = "sequence_history"

    def __init__(self, train_records: Iterable[TraceRecord]) -> None:
        train_list = list(train_records)
        self.global_predictor = GlobalFrequencyPredictor(train_list)
        self.history: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
        for record in train_list:
            self.update(record)

    def predict(self, meta: AgentNodeMeta, layer_id: int, top_m: int) -> list[int]:
        counter = self.history.get((meta.agent_id, layer_id))
        if not counter:
            counter = self.history.get((meta.role, layer_id))
        result = (
            [expert for expert, _ in counter.most_common(top_m)] if counter else []
        )
        for expert in self.global_predictor.predict(meta, layer_id, top_m):
            if expert not in result:
                result.append(expert)
            if len(result) >= top_m:
                break
        return result

    def update(self, record: TraceRecord) -> None:
        for layer in record.layers:
            hist = layer_histogram(record, layer.layer_id)
            self.history[(record.metadata.agent_id, layer.layer_id)].update(hist)
            self.history[(record.metadata.role, layer.layer_id)].update(hist)


class RouteSigPredictor:
    name = "routesig"

    def __init__(
        self,
        train_records: Iterable[TraceRecord],
        *,
        top_m: int = 4,
        min_samples: int = 12,
    ) -> None:
        self.store = RouteSigStore(top_m=top_m, min_samples=min_samples)
        self.store.update_many(train_records)

    def predict(self, meta: AgentNodeMeta, layer_id: int, top_m: int) -> list[int]:
        return self.store.top_experts(meta, layer_id, top_m=top_m)

    def update(self, record: TraceRecord) -> None:
        self.store.update_trace(record)


@dataclass(frozen=True)
class TopMResult:
    predictor: str
    top_m: int
    hit_rate: float
    exact_token_hit_rate: float
    total_expert_labels: int
    total_tokens: int


def evaluate_top_m(
    predictor: Predictor,
    eval_records: Iterable[TraceRecord],
    *,
    top_m: int = 4,
    online_update: bool = True,
) -> TopMResult:
    expert_hits = 0
    expert_total = 0
    exact_token_hits = 0
    token_total = 0
    for record in eval_records:
        for layer in record.layers:
            predicted = set(predictor.predict(record.metadata, layer.layer_id, top_m))
            selected = selected_array_for_layer(record, layer.layer_id)
            valid = selected[selected >= 0]
            expert_total += int(valid.size)
            token_total += int(selected.shape[0])
            if not predicted or valid.size == 0:
                continue
            predicted_arr = np.fromiter(predicted, dtype=np.int64)
            expert_hits += int(np.isin(valid, predicted_arr).sum())
            valid_mask = selected >= 0
            token_has_label = valid_mask.any(axis=1)
            token_hits = np.isin(selected, predicted_arr) | ~valid_mask
            exact_token_hits += int((token_has_label & token_hits.all(axis=1)).sum())
        if online_update:
            predictor.update(record)
    return TopMResult(
        predictor=predictor.name,
        top_m=top_m,
        hit_rate=expert_hits / expert_total if expert_total else 0.0,
        exact_token_hit_rate=exact_token_hits / token_total if token_total else 0.0,
        total_expert_labels=expert_total,
        total_tokens=token_total,
    )


def evaluate_predictors(
    records: list[TraceRecord],
    *,
    top_m_values: Iterable[int] = (2, 4, 6),
    train_fraction: float = 0.7,
) -> list[TopMResult]:
    train, eval_records = split_records(records, train_fraction=train_fraction)
    results: list[TopMResult] = []
    for top_m in top_m_values:
        predictors: list[Predictor] = [
            GlobalFrequencyPredictor(train),
            LRUPredictor(train),
            SequenceHistoryPredictor(train),
            RouteSigPredictor(train, top_m=top_m, min_samples=max(4, top_m * 2)),
        ]
        for predictor in predictors:
            results.append(evaluate_top_m(predictor, eval_records, top_m=top_m))
    return results


def layer_sensitivity(
    records: list[TraceRecord], *, top_m: int = 4, train_fraction: float = 0.7
) -> dict[int, dict[str, float]]:
    train, eval_records = split_records(records, train_fraction=train_fraction)
    global_pred = GlobalFrequencyPredictor(train)
    routesig_pred = RouteSigPredictor(train, top_m=top_m, min_samples=max(4, top_m * 2))
    stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"global_hits": 0, "routesig_hits": 0, "total": 0}
    )
    for record in eval_records:
        for layer in record.layers:
            global_set = set(global_pred.predict(record.metadata, layer.layer_id, top_m))
            routesig_set = set(
                routesig_pred.predict(record.metadata, layer.layer_id, top_m)
            )
            selected = selected_array_for_layer(record, layer.layer_id)
            valid = selected[selected >= 0]
            stats[layer.layer_id]["total"] += int(valid.size)
            if valid.size == 0:
                continue
            if global_set:
                global_arr = np.fromiter(global_set, dtype=np.int64)
                stats[layer.layer_id]["global_hits"] += int(np.isin(valid, global_arr).sum())
            if routesig_set:
                routesig_arr = np.fromiter(routesig_set, dtype=np.int64)
                stats[layer.layer_id]["routesig_hits"] += int(np.isin(valid, routesig_arr).sum())
        global_pred.update(record)
        routesig_pred.update(record)
    result = {}
    for layer, values in stats.items():
        total = max(1, values["total"])
        global_rate = values["global_hits"] / total
        routesig_rate = values["routesig_hits"] / total
        result[layer] = {
            "global_hit_rate": global_rate,
            "routesig_hit_rate": routesig_rate,
            "delta": routesig_rate - global_rate,
        }
    return result


def locality_summary(records: list[TraceRecord]) -> dict[str, object]:
    return {
        "num_records": len(records),
        "agent_expert_overlap": agent_expert_overlap(records),
        "cross_agent_divergence": cross_agent_divergence(records),
        "route_entropy": route_entropy(records),
        "top_m_hit_rate": [],
        "layer_sensitivity": {},
    }


def normalized_entropy_from_hist(hist: Counter[int]) -> float:
    if not hist:
        return 0.0
    probs = distribution(hist)
    entropy = route_entropy_from_probabilities(probs.values())
    return entropy / math.log(max(2, len(probs)))


def split_records_by_group(
    records: list[TraceRecord], train_fraction: float = 0.7
) -> tuple[list[TraceRecord], list[TraceRecord], dict[str, object]]:
    if not records:
        return [], [], {"available": False, "reason": "empty_records"}
    ordered = sorted(
        records,
        key=lambda record: (
            "" if record.timestamp is None else str(record.timestamp),
            "" if record.source_index is None else str(record.source_index),
            record.request_id,
        ),
    )
    groups: list[str] = []
    group_to_records: dict[str, list[TraceRecord]] = defaultdict(list)
    for record in ordered:
        group = record.source_group_id
        if not group:
            return [], [], {
                "available": False,
                "reason": "missing_source_group_id",
            }
        if group not in group_to_records:
            groups.append(group)
        group_to_records[group].append(record)
    if len(groups) < 2:
        return [], [], {
            "available": False,
            "reason": "not_enough_source_groups",
            "num_groups": len(groups),
        }
    split = max(1, min(len(groups) - 1, int(len(groups) * train_fraction)))
    train_groups = set(groups[:split])
    train = [record for group in groups[:split] for record in group_to_records[group]]
    eval_records = [
        record for group in groups[split:] for record in group_to_records[group]
    ]
    return train, eval_records, {
        "available": True,
        "num_groups": len(groups),
        "train_groups": len(train_groups),
        "eval_groups": len(groups) - len(train_groups),
        "train_records": len(train),
        "eval_records": len(eval_records),
    }


def _segment_length_bucket(segment) -> str:
    count = segment.token_count
    if count < 16:
        return "<16"
    if count < 64:
        return "16-63"
    if count < 256:
        return "64-255"
    return ">=256"


def _score_prediction(
    prediction: PredictedExpertSet,
    record: TraceRecord,
    segment,
) -> dict[str, float | int | str | None]:
    predicted = set(prediction.expert_ids)
    if segment.token_start is None or segment.token_end is None:
        expert_total = 0
        expert_hits = 0
        exact_token_hits = 0
        token_total = 0
    else:
        selected = selected_array_for_layer(record, prediction.layer_id)[
            segment.token_start : segment.token_end
        ]
        valid = selected[selected >= 0]
        expert_total = int(valid.size)
        token_total = int(selected.shape[0])
        if predicted and valid.size:
            predicted_arr = np.fromiter(predicted, dtype=np.int64)
            expert_hits = int(np.isin(valid, predicted_arr).sum())
            valid_mask = selected >= 0
            token_has_label = valid_mask.any(axis=1)
            token_hits = np.isin(selected, predicted_arr) | ~valid_mask
            exact_token_hits = int((token_has_label & token_hits.all(axis=1)).sum())
        else:
            expert_hits = 0
            exact_token_hits = 0
    return {
        "expert_hits": expert_hits,
        "expert_total": expert_total,
        "exact_token_hits": exact_token_hits,
        "token_total": token_total,
        "confidence": prediction.confidence,
        "fallback_key": prediction.fallback_key,
        "block_type": prediction.block_type,
        "layer_id": prediction.layer_id,
        "segment_length_bucket": _segment_length_bucket(segment),
        "claim_scope": record.claim_scope,
    }


def _summarize_score_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    expert_hits = sum(int(row["expert_hits"]) for row in rows)
    expert_total = sum(int(row["expert_total"]) for row in rows)
    exact_hits = sum(int(row["exact_token_hits"]) for row in rows)
    token_total = sum(int(row["token_total"]) for row in rows)
    confidences = [float(row["confidence"]) for row in rows]
    fallback_counts = Counter(str(row["fallback_key"]) for row in rows)
    return {
        "expert_label_hit_rate": expert_hits / expert_total if expert_total else 0.0,
        "exact_token_hit_rate": exact_hits / token_total if token_total else 0.0,
        "weighted_coverage": None,
        "weighted_coverage_status": "unavailable_router_scores_not_captured_by_vllm",
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "confidence_calibration": _confidence_bins(rows),
        "fallback_key_counts": dict(fallback_counts),
        "total_expert_labels": expert_total,
        "total_tokens": token_total,
        "num_predictions": len(rows),
    }


class ScoreAccumulator:
    def __init__(self) -> None:
        self.expert_hits = 0
        self.expert_total = 0
        self.exact_hits = 0
        self.token_total = 0
        self.confidence_sum = 0.0
        self.count = 0
        self.fallback_counts: Counter[str] = Counter()
        self.confidence_bins = [
            {"lo": 0.0, "hi": 0.25, "hits": 0, "total": 0, "count": 0},
            {"lo": 0.25, "hi": 0.5, "hits": 0, "total": 0, "count": 0},
            {"lo": 0.5, "hi": 0.75, "hits": 0, "total": 0, "count": 0},
            {"lo": 0.75, "hi": 1.01, "hits": 0, "total": 0, "count": 0},
        ]

    def update(self, row: dict[str, float | int | str | None]) -> None:
        expert_hits = int(row["expert_hits"])
        expert_total = int(row["expert_total"])
        self.expert_hits += expert_hits
        self.expert_total += expert_total
        self.exact_hits += int(row["exact_token_hits"])
        self.token_total += int(row["token_total"])
        confidence = float(row["confidence"])
        self.confidence_sum += confidence
        self.count += 1
        self.fallback_counts[str(row["fallback_key"])] += 1
        for item in self.confidence_bins:
            if float(item["lo"]) <= confidence < float(item["hi"]):
                item["hits"] = int(item["hits"]) + expert_hits
                item["total"] = int(item["total"]) + expert_total
                item["count"] = int(item["count"]) + 1
                break

    def summary(self) -> dict[str, object]:
        bins = []
        for item in self.confidence_bins:
            count = int(item["count"])
            if count <= 0:
                continue
            expert_total = int(item["total"])
            bins.append(
                {
                    "confidence_min": float(item["lo"]),
                    "confidence_max": min(1.0, float(item["hi"])),
                    "mean_hit_rate": int(item["hits"]) / expert_total
                    if expert_total
                    else 0.0,
                    "count": float(count),
                }
            )
        return {
            "expert_label_hit_rate": self.expert_hits / self.expert_total
            if self.expert_total
            else 0.0,
            "exact_token_hit_rate": self.exact_hits / self.token_total
            if self.token_total
            else 0.0,
            "weighted_coverage": None,
            "weighted_coverage_status": "unavailable_router_scores_not_captured_by_vllm",
            "mean_confidence": self.confidence_sum / self.count if self.count else 0.0,
            "confidence_calibration": bins,
            "fallback_key_counts": dict(self.fallback_counts),
            "total_expert_labels": self.expert_total,
            "total_tokens": self.token_total,
            "num_predictions": self.count,
        }


def _confidence_bins(rows: list[dict[str, object]]) -> list[dict[str, float]]:
    bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    result = []
    for lo, hi in bins:
        selected = [
            row
            for row in rows
            if lo <= float(row["confidence"]) < hi
        ]
        if not selected:
            continue
        expert_hits = sum(int(row["expert_hits"]) for row in selected)
        expert_total = sum(int(row["expert_total"]) for row in selected)
        result.append(
            {
                "confidence_min": lo,
                "confidence_max": min(1.0, hi),
                "mean_hit_rate": expert_hits / expert_total if expert_total else 0.0,
                "count": float(len(selected)),
            }
        )
    return result


def evaluate_segment_predictors(
    records: list[TraceRecord],
    *,
    router_top_k: int | None = None,
    train_fraction: float = 0.7,
    require_current_stage: bool = True,
    temporal_window: int = 64,
) -> dict[str, object]:
    if require_current_stage:
        validate_current_stage_traces(records)
    if not records:
        return {"available": False, "reason": "empty_records"}
    effective_top_k = router_top_k or records[0].router_top_k or records[0].top_k
    train, eval_records, split_info = split_records_by_group(
        records, train_fraction=train_fraction
    )
    if not split_info.get("available"):
        return {"available": False, "split": split_info}
    results: dict[str, object] = {
        "available": True,
        "router_top_k": effective_top_k,
        "top_m_budgets": top_m_budgets(effective_top_k),
        "split": split_info,
        "results": [],
    }
    started = time.perf_counter()
    for budget_name, top_m in top_m_budgets(effective_top_k).items():
        print(
            f"[analysis] segment budget {budget_name} top_m={top_m} start "
            f"at {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        predictors = build_predictors(
            train,
            top_m=top_m,
            temporal_window=temporal_window,
            include_oracle=True,
        )
        for predictor in predictors:
            predictor_started = time.perf_counter()
            print(
                f"[analysis] segment predictor {predictor.name} "
                f"budget={budget_name} start at {predictor_started - started:.2f}s",
                flush=True,
            )
            accumulator = ScoreAccumulator()
            by_block: dict[str, ScoreAccumulator] = defaultdict(ScoreAccumulator)
            by_layer: dict[str, ScoreAccumulator] = defaultdict(ScoreAccumulator)
            by_bucket: dict[str, ScoreAccumulator] = defaultdict(ScoreAccumulator)
            by_scope: dict[str, ScoreAccumulator] = defaultdict(ScoreAccumulator)
            unavailable_spans = 0
            for record in eval_records:
                for segment in record_segments(record):
                    if segment.token_start is None or segment.token_end is None:
                        unavailable_spans += 1
                        continue
                    for layer in record.layers:
                        prediction = predictor.predict(
                            record, segment, layer.layer_id, top_m
                        )
                        row = _score_prediction(prediction, record, segment)
                        accumulator.update(row)
                        by_block[str(row["block_type"])].update(row)
                        by_layer[str(row["layer_id"])].update(row)
                        by_bucket[str(row["segment_length_bucket"])].update(row)
                        by_scope[str(row["claim_scope"])].update(row)
                predictor.update(record)
            summary = accumulator.summary()
            results["results"].append(
                {
                    "predictor": predictor.name,
                    "budget": budget_name,
                    "top_m": top_m,
                    **summary,
                    "per_block_type": {
                        key: value.summary()
                        for key, value in sorted(by_block.items())
                    },
                    "per_layer": {
                        key: value.summary()
                        for key, value in sorted(by_layer.items(), key=lambda item: int(item[0]))
                    },
                    "per_segment_length_bucket": {
                        key: value.summary()
                        for key, value in sorted(by_bucket.items())
                    },
                    "per_claim_scope": {
                        key: value.summary()
                        for key, value in sorted(by_scope.items())
                    },
                    "unavailable_span_count": unavailable_spans,
                }
            )
            print(
                f"[analysis] segment predictor {predictor.name} "
                f"budget={budget_name} done in "
                f"{time.perf_counter() - predictor_started:.2f}s",
                flush=True,
            )
    return results
