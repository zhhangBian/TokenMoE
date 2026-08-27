"""Small, explainable expert-set predictors for online evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from tokenmoe.routesig import RouteSigStore
from tokenmoe.schema import PromptSegment
from tokenmoe.trace import TraceRecord


@dataclass(frozen=True, slots=True)
class PredictedExpertSet:
    layer_id: int
    expert_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    confidence: float
    source: str
    fallback_key: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.expert_ids) != len(self.probabilities):
            raise ValueError("expert IDs and probabilities must align")
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("predicted expert IDs must be unique")


class SegmentPredictor(Protocol):
    name: str

    def predict(
        self,
        record: TraceRecord,
        segment: PromptSegment,
        layer_id: int,
        top_m: int,
    ) -> PredictedExpertSet: ...

    def update(self, record: TraceRecord) -> None: ...


def top_m_budgets(router_top_k: int) -> dict[str, int]:
    if router_top_k < 1:
        raise ValueError("router_top_k must be positive")
    return {
        "1x": router_top_k,
        "2x": router_top_k * 2,
        "4x": router_top_k * 4,
    }


def _prediction_from_counts(
    *,
    layer_id: int,
    counts: Counter[int],
    top_m: int,
    source: str,
    confidence: float,
    fallback_key: tuple[str, ...] | None = None,
) -> PredictedExpertSet:
    total = float(sum(counts.values()))
    ranked = sorted(counts, key=lambda expert_id: (-counts[expert_id], expert_id))[
        :top_m
    ]
    probabilities = tuple(counts[item] / total for item in ranked) if total else ()
    return PredictedExpertSet(
        layer_id=layer_id,
        expert_ids=tuple(ranked),
        probabilities=probabilities,
        confidence=confidence if ranked else 0.0,
        source=source,
        fallback_key=fallback_key,
    )


def segment_selected(
    record: TraceRecord, segment: PromptSegment, layer_id: int
) -> np.ndarray:
    if segment.token_start is None or segment.token_end is None:
        return np.empty((0, record.router_top_k), dtype=np.int64)
    return record.layer(layer_id).selected_array()[
        segment.token_start : segment.token_end
    ]


def segment_scores(
    record: TraceRecord, segment: PromptSegment, layer_id: int
) -> np.ndarray | None:
    if segment.token_start is None or segment.token_end is None:
        return None
    scores = record.layer(layer_id).scores_array()
    if scores is None:
        return None
    return scores[segment.token_start : segment.token_end]


def _layer_counts(record: TraceRecord) -> dict[int, Counter[int]]:
    return {
        layer.layer_id: Counter(
            int(item) for item in layer.selected_array().reshape(-1)
        )
        for layer in record.layers
    }


class GlobalFrequencyPredictor:
    name = "global_frequency"

    def __init__(self, train_records: Iterable[TraceRecord]) -> None:
        self._counts: dict[int, Counter[int]] = defaultdict(Counter)
        for record in train_records:
            self.update(record)

    def predict(
        self,
        record: TraceRecord,
        segment: PromptSegment,
        layer_id: int,
        top_m: int,
    ) -> PredictedExpertSet:
        return _prediction_from_counts(
            layer_id=layer_id,
            counts=self._counts[layer_id],
            top_m=top_m,
            source=self.name,
            confidence=1.0,
        )

    def update(self, record: TraceRecord) -> None:
        for layer_id, counts in _layer_counts(record).items():
            self._counts[layer_id].update(counts)


class TemporalWindowPredictor:
    name = "temporal_window"

    def __init__(
        self, train_records: Iterable[TraceRecord], *, window_size: int = 64
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._window: deque[dict[int, Counter[int]]] = deque(maxlen=window_size)
        self._counts: dict[int, Counter[int]] = defaultdict(Counter)
        for record in train_records:
            self.update(record)

    def predict(
        self,
        record: TraceRecord,
        segment: PromptSegment,
        layer_id: int,
        top_m: int,
    ) -> PredictedExpertSet:
        confidence = min(1.0, len(self._window) / max(1, self._window.maxlen or 1))
        return _prediction_from_counts(
            layer_id=layer_id,
            counts=self._counts[layer_id],
            top_m=top_m,
            source=self.name,
            confidence=confidence,
        )

    def update(self, record: TraceRecord) -> None:
        if len(self._window) == self._window.maxlen:
            expired = self._window[0]
            for layer_id, counts in expired.items():
                self._counts[layer_id].subtract(counts)
                self._counts[layer_id] += Counter()
        counts = _layer_counts(record)
        self._window.append(counts)
        for layer_id, values in counts.items():
            self._counts[layer_id].update(values)


class RouteSigPredictor:
    name = "routesig"

    def __init__(
        self,
        train_records: Iterable[TraceRecord],
        *,
        top_m: int,
        min_support: int = 16,
    ) -> None:
        self._store = RouteSigStore(top_m=top_m, min_support=min_support)
        self._store.update_many(train_records)

    def predict(
        self,
        record: TraceRecord,
        segment: PromptSegment,
        layer_id: int,
        top_m: int,
    ) -> PredictedExpertSet:
        signature = self._store.lookup(record.metadata, segment, layer_id)
        if signature is None:
            return PredictedExpertSet(layer_id, (), (), 0.0, self.name)
        experts = signature.top_experts[:top_m]
        return PredictedExpertSet(
            layer_id=layer_id,
            expert_ids=experts,
            probabilities=tuple(signature.probability(item) for item in experts),
            confidence=signature.confidence,
            source=self.name,
            fallback_key=signature.key,
        )

    def update(self, record: TraceRecord) -> None:
        self._store.update_trace(record)


def build_predictors(
    train_records: Iterable[TraceRecord], *, top_m: int, temporal_window: int = 64
) -> tuple[SegmentPredictor, ...]:
    records = tuple(train_records)
    return (
        GlobalFrequencyPredictor(records),
        TemporalWindowPredictor(records, window_size=temporal_window),
        RouteSigPredictor(records, top_m=top_m, min_support=max(4, top_m * 2)),
    )
