from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from tokenmoe.routesig import RouteSigStore, route_entropy_from_probabilities
from tokenmoe.schema import AgentNodeMeta
from tokenmoe.trace import TraceRecord


def split_records(
    records: list[TraceRecord], train_fraction: float = 0.7
) -> tuple[list[TraceRecord], list[TraceRecord]]:
    if not records:
        return [], []
    split = max(1, min(len(records) - 1, int(len(records) * train_fraction)))
    return records[:split], records[split:]


def layer_histogram(record: TraceRecord, layer_id: int) -> Counter[int]:
    selected = record.layer(layer_id).selected_array()
    return Counter(int(item) for item in selected.reshape(-1) if int(item) >= 0)


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
            selected = layer.selected_array()
            for row in selected:
                token_total += 1
                row_set = {int(expert) for expert in row if int(expert) >= 0}
                expert_total += len(row_set)
                expert_hits += len(row_set & predicted)
                if row_set and row_set.issubset(predicted):
                    exact_token_hits += 1
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
            selected = layer.selected_array()
            global_set = set(global_pred.predict(record.metadata, layer.layer_id, top_m))
            routesig_set = set(
                routesig_pred.predict(record.metadata, layer.layer_id, top_m)
            )
            for expert in selected.reshape(-1):
                expert = int(expert)
                if expert < 0:
                    continue
                stats[layer.layer_id]["total"] += 1
                stats[layer.layer_id]["global_hits"] += int(expert in global_set)
                stats[layer.layer_id]["routesig_hits"] += int(expert in routesig_set)
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
    predictor_results = evaluate_predictors(records)
    return {
        "num_records": len(records),
        "agent_expert_overlap": agent_expert_overlap(records),
        "cross_agent_divergence": cross_agent_divergence(records),
        "route_entropy": route_entropy(records),
        "top_m_hit_rate": [result.__dict__ for result in predictor_results],
        "layer_sensitivity": layer_sensitivity(records),
    }


def normalized_entropy_from_hist(hist: Counter[int]) -> float:
    if not hist:
        return 0.0
    probs = distribution(hist)
    entropy = route_entropy_from_probabilities(probs.values())
    return entropy / math.log(max(2, len(probs)))
