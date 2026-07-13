from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol

import numpy as np

from tokenmoe.routesig import RouteSigStore
from tokenmoe.schema import AgentNodeMeta, PromptSegment, default_prompt_segment
from tokenmoe.trace import LayerTrace, TraceRecord


_LAYER_ARRAY_CACHE: dict[int, dict[int, np.ndarray]] = {}
_SEGMENT_LAYER_COUNT_CACHE: dict[tuple[int, str, int, int | None, int | None], Counter[int]] = {}


@dataclass(frozen=True)
class PredictedExpertSet:
    request_id: str
    segment_id: str
    layer_id: int
    expert_ids: list[int]
    expert_weights: list[float]
    confidence: float
    source: str
    fallback_key: str
    scores: dict[str, float] = field(default_factory=dict)
    token_segment: str = "prompt"
    block_type: str = "prompt"
    segment_position: int = 0
    token_start: int | None = None
    token_end: int | None = None
    weight_rule: str = "normalized_counts"
    claim_scope: str | None = None
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if len(self.expert_ids) != len(self.expert_weights):
            raise ValueError("expert_ids and expert_weights must have the same length")
        total = sum(self.expert_weights)
        if self.expert_ids and not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"expert_weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def token_count(self) -> int:
        if self.token_start is None or self.token_end is None:
            return 0
        return max(0, self.token_end - self.token_start)


def top_m_budgets(router_top_k: int) -> dict[str, int]:
    return {
        "1x": int(math.ceil(router_top_k * 1.0)),
        "1.5x": int(math.ceil(router_top_k * 1.5)),
        "2x": int(math.ceil(router_top_k * 2.0)),
        "3x": int(math.ceil(router_top_k * 3.0)),
    }


def _normalize_counts(counter: Counter[int], top_m: int) -> tuple[list[int], list[float]]:
    if not counter or top_m <= 0:
        return [], []
    selected = [expert for expert, _ in counter.most_common(top_m)]
    total = sum(counter[expert] for expert in selected)
    if total <= 0:
        return selected, [1.0 / len(selected) for _ in selected]
    return selected, [counter[expert] / total for expert in selected]


def _uniform(expert_ids: list[int]) -> list[float]:
    if not expert_ids:
        return []
    return [1.0 / len(expert_ids) for _ in expert_ids]


def prediction_from_counter(
    *,
    record: TraceRecord,
    segment: PromptSegment,
    layer_id: int,
    counter: Counter[int],
    top_m: int,
    source: str,
    fallback_key: str,
    confidence: float,
) -> PredictedExpertSet:
    expert_ids, weights = _normalize_counts(counter, top_m)
    return PredictedExpertSet(
        request_id=record.request_id,
        segment_id=segment.segment_id,
        layer_id=layer_id,
        expert_ids=expert_ids,
        expert_weights=weights,
        confidence=float(confidence if expert_ids else 0.0),
        source=source,
        fallback_key=fallback_key,
        block_type=segment.block_type,
        segment_position=segment.segment_position,
        token_start=segment.token_start,
        token_end=segment.token_end,
        weight_rule="normalized_counts",
        claim_scope=record.claim_scope,
        unavailable_reason=None if expert_ids else "no_prediction_history",
    )


def prediction_from_ranked_set(
    *,
    record: TraceRecord,
    segment: PromptSegment,
    layer_id: int,
    expert_ids: list[int],
    source: str,
    fallback_key: str,
    confidence: float,
) -> PredictedExpertSet:
    unique = []
    for expert in expert_ids:
        if expert not in unique:
            unique.append(int(expert))
    return PredictedExpertSet(
        request_id=record.request_id,
        segment_id=segment.segment_id,
        layer_id=layer_id,
        expert_ids=unique,
        expert_weights=_uniform(unique),
        confidence=float(confidence if unique else 0.0),
        source=source,
        fallback_key=fallback_key,
        block_type=segment.block_type,
        segment_position=segment.segment_position,
        token_start=segment.token_start,
        token_end=segment.token_end,
        weight_rule="uniform_ranked_set",
        claim_scope=record.claim_scope,
        unavailable_reason=None if unique else "no_prediction_history",
    )


def record_segments(record: TraceRecord) -> list[PromptSegment]:
    if record.prompt_segments:
        return record.prompt_segments
    return [default_prompt_segment(record.prompt, record.metadata.block_type_key)]


def layer_by_id(record: TraceRecord, layer_id: int) -> LayerTrace:
    return record.layer(layer_id)


def selected_array_for_layer(record: TraceRecord, layer_id: int) -> np.ndarray:
    per_record = _LAYER_ARRAY_CACHE.setdefault(id(record), {})
    if layer_id not in per_record:
        per_record[layer_id] = layer_by_id(record, layer_id).selected_array()
    return per_record[layer_id]


def true_segment_layer_counts(
    record: TraceRecord, segment: PromptSegment, layer_id: int
) -> Counter[int]:
    if segment.token_start is None or segment.token_end is None:
        return Counter()
    cache_key = (
        id(record),
        segment.segment_id,
        int(layer_id),
        segment.token_start,
        segment.token_end,
    )
    cached = _SEGMENT_LAYER_COUNT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    selected = selected_array_for_layer(record, layer_id)[segment.token_start : segment.token_end]
    if selected.size == 0:
        counter: Counter[int] = Counter()
    else:
        valid = selected.reshape(-1)
        valid = valid[valid >= 0]
        if valid.size == 0:
            counter = Counter()
        else:
            experts, counts = np.unique(
                valid.astype(np.int64, copy=False),
                return_counts=True,
            )
            counter = Counter(
                {int(expert): int(count) for expert, count in zip(experts, counts)}
            )
    _SEGMENT_LAYER_COUNT_CACHE[cache_key] = counter
    return counter
    return Counter(int(item) for item in selected.reshape(-1) if int(item) >= 0)


class SegmentPredictor(Protocol):
    name: str

    def predict(
        self,
        record: TraceRecord,
        segment: PromptSegment,
        layer_id: int,
        top_m: int,
    ) -> PredictedExpertSet:
        ...

    def update(self, record: TraceRecord) -> None:
        ...


class GlobalFrequencySegmentPredictor:
    name = "global_frequency"

    def __init__(self, train_records: Iterable[TraceRecord] = ()) -> None:
        self.counts: dict[int, Counter[int]] = defaultdict(Counter)
        for record in train_records:
            self.update(record)

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        return prediction_from_counter(
            record=record,
            segment=segment,
            layer_id=layer_id,
            counter=self.counts[layer_id],
            top_m=top_m,
            source=self.name,
            fallback_key="global",
            confidence=1.0 if self.counts[layer_id] else 0.0,
        )

    def update(self, record: TraceRecord) -> None:
        for segment in record_segments(record):
            if segment.token_start is None or segment.token_end is None:
                continue
            for layer in record.layers:
                self.counts[layer.layer_id].update(
                    true_segment_layer_counts(record, segment, layer.layer_id)
                )


class RequestLRUSegmentPredictor:
    name = "request_lru"

    def __init__(self, train_records: Iterable[TraceRecord] = (), maxlen: int = 512) -> None:
        self.history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=maxlen))
        self.global_predictor = GlobalFrequencySegmentPredictor()
        for record in train_records:
            self.update(record)

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        ranked = []
        for expert in reversed(self.history[layer_id]):
            if expert not in ranked:
                ranked.append(expert)
            if len(ranked) >= top_m:
                break
        if len(ranked) < top_m:
            for expert in self.global_predictor.predict(record, segment, layer_id, top_m).expert_ids:
                if expert not in ranked:
                    ranked.append(expert)
                if len(ranked) >= top_m:
                    break
        return prediction_from_ranked_set(
            record=record,
            segment=segment,
            layer_id=layer_id,
            expert_ids=ranked,
            source=self.name,
            fallback_key="lru_then_global",
            confidence=1.0 if ranked else 0.0,
        )

    def update(self, record: TraceRecord) -> None:
        self.global_predictor.update(record)
        for segment in record_segments(record):
            if segment.token_start is None or segment.token_end is None:
                continue
            for layer in record.layers:
                for expert in true_segment_layer_counts(record, segment, layer.layer_id):
                    self.history[layer.layer_id].append(expert)


class SequenceHistorySegmentPredictor:
    name = "sequence_history"

    def __init__(self, train_records: Iterable[TraceRecord] = ()) -> None:
        self.counts: dict[tuple[str, int], Counter[int]] = defaultdict(Counter)
        self.global_predictor = GlobalFrequencySegmentPredictor()
        for record in train_records:
            self.update(record)

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        keys = [
            (record.metadata.agent_id, layer_id),
            (record.metadata.role, layer_id),
            (segment.block_type, layer_id),
        ]
        for key in keys:
            if self.counts.get(key):
                return prediction_from_counter(
                    record=record,
                    segment=segment,
                    layer_id=layer_id,
                    counter=self.counts[key],
                    top_m=top_m,
                    source=self.name,
                    fallback_key="/".join(str(item) for item in key),
                    confidence=1.0,
                )
        base = self.global_predictor.predict(record, segment, layer_id, top_m)
        return prediction_from_ranked_set(
            record=record,
            segment=segment,
            layer_id=layer_id,
            expert_ids=base.expert_ids,
            source=self.name,
            fallback_key="global",
            confidence=base.confidence,
        )

    def update(self, record: TraceRecord) -> None:
        self.global_predictor.update(record)
        for segment in record_segments(record):
            if segment.token_start is None or segment.token_end is None:
                continue
            for layer in record.layers:
                hist = true_segment_layer_counts(record, segment, layer.layer_id)
                self.counts[(record.metadata.agent_id, layer.layer_id)].update(hist)
                self.counts[(record.metadata.role, layer.layer_id)].update(hist)
                self.counts[(segment.block_type, layer.layer_id)].update(hist)


class TemporalWindowSegmentPredictor:
    name = "temporal_window_frequency"

    def __init__(self, train_records: Iterable[TraceRecord] = (), window_size: int = 64) -> None:
        self.window_size = int(window_size)
        self.windows: dict[int, deque[Counter[int]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        for record in train_records:
            self.update(record)

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        counter: Counter[int] = Counter()
        for hist in self.windows[layer_id]:
            counter.update(hist)
        return prediction_from_counter(
            record=record,
            segment=segment,
            layer_id=layer_id,
            counter=counter,
            top_m=top_m,
            source=self.name,
            fallback_key=f"last_{self.window_size}_requests",
            confidence=1.0 if counter else 0.0,
        )

    def update(self, record: TraceRecord) -> None:
        per_layer: dict[int, Counter[int]] = defaultdict(Counter)
        for segment in record_segments(record):
            if segment.token_start is None or segment.token_end is None:
                continue
            for layer in record.layers:
                per_layer[layer.layer_id].update(
                    true_segment_layer_counts(record, segment, layer.layer_id)
                )
        for layer_id, hist in per_layer.items():
            self.windows[layer_id].append(hist)


class RouteSigSegmentPredictor:
    name = "routesig"

    def __init__(
        self,
        train_records: Iterable[TraceRecord] = (),
        *,
        top_m: int = 4,
        min_samples: int = 12,
    ) -> None:
        self.store = RouteSigStore(top_m=top_m, min_samples=min_samples)
        for record in train_records:
            self.update(record)

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        sig = self.store.lookup_segment(record.metadata, segment, layer_id)
        if sig is None:
            return prediction_from_ranked_set(
                record=record,
                segment=segment,
                layer_id=layer_id,
                expert_ids=[],
                source=self.name,
                fallback_key="unavailable",
                confidence=0.0,
            )
        expert_ids = sig.top_experts[:top_m]
        total = sum(sig.expert_prob.get(expert, 0.0) for expert in expert_ids)
        weights = (
            [sig.expert_prob.get(expert, 0.0) / total for expert in expert_ids]
            if total > 0
            else _uniform(expert_ids)
        )
        return PredictedExpertSet(
            request_id=record.request_id,
            segment_id=segment.segment_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_weights=weights,
            confidence=sig.confidence,
            source=self.name,
            fallback_key="/".join(sig.key),
            scores={str(k): float(v) for k, v in sig.expert_prob.items()},
            block_type=segment.block_type,
            segment_position=segment.segment_position,
            token_start=segment.token_start,
            token_end=segment.token_end,
            weight_rule="routesig_probability",
            claim_scope=record.claim_scope,
        )

    def update(self, record: TraceRecord) -> None:
        self.store.update_segment_trace(record)


class OracleSegmentPredictor:
    name = "oracle_future_demand"

    def __init__(self, train_records: Iterable[TraceRecord] = ()) -> None:
        pass

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        return prediction_from_counter(
            record=record,
            segment=segment,
            layer_id=layer_id,
            counter=true_segment_layer_counts(record, segment, layer_id),
            top_m=top_m,
            source=self.name,
            fallback_key="oracle_current_trace",
            confidence=1.0,
        )

    def update(self, record: TraceRecord) -> None:
        return None


def build_predictors(
    train_records: Iterable[TraceRecord],
    *,
    top_m: int,
    temporal_window: int = 64,
    include_oracle: bool = True,
) -> list[SegmentPredictor]:
    train = list(train_records)
    predictors: list[SegmentPredictor] = [
        GlobalFrequencySegmentPredictor(train),
        RequestLRUSegmentPredictor(train),
        SequenceHistorySegmentPredictor(train),
        TemporalWindowSegmentPredictor(train, window_size=temporal_window),
        RouteSigSegmentPredictor(train, top_m=top_m, min_samples=max(4, top_m * 2)),
    ]
    if include_oracle:
        predictors.append(OracleSegmentPredictor(train))
    return predictors


def aggregate_record_demand(
    predictions: Iterable[PredictedExpertSet],
    *,
    router_top_k: int,
) -> dict[tuple[int, int], float]:
    demand: dict[tuple[int, int], float] = defaultdict(float)
    for prediction in predictions:
        token_count = prediction.token_count
        if token_count <= 0:
            continue
        for expert_id, weight in zip(prediction.expert_ids, prediction.expert_weights):
            demand[(prediction.layer_id, expert_id)] += token_count * router_top_k * weight
    return dict(demand)


def predict_record_demand(
    predictor: SegmentPredictor,
    record: TraceRecord,
    *,
    top_m: int,
    router_top_k: int,
) -> tuple[dict[tuple[int, int], float], list[PredictedExpertSet]]:
    predictions: list[PredictedExpertSet] = []
    for segment in record_segments(record):
        if segment.token_start is None or segment.token_end is None:
            continue
        for layer in record.layers:
            predictions.append(predictor.predict(record, segment, layer.layer_id, top_m))
    return aggregate_record_demand(predictions, router_top_k=router_top_k), predictions


def record_true_demand(record: TraceRecord) -> dict[tuple[int, int], float]:
    demand: dict[tuple[int, int], float] = defaultdict(float)
    for layer in record.layers:
        selected = selected_array_for_layer(record, layer.layer_id).reshape(-1)
        valid = selected[selected >= 0]
        if valid.size == 0:
            continue
        experts, counts = np.unique(
            valid.astype(np.int64, copy=False),
            return_counts=True,
        )
        for expert, count in zip(experts, counts):
            demand[(layer.layer_id, int(expert))] += float(count)
    return dict(demand)
