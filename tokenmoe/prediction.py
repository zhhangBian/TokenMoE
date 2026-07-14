from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Protocol

import numpy as np

from tokenmoe.routesig import RouteSigStore
from tokenmoe.schema import AgentNodeMeta, PromptSegment, default_prompt_segment
from tokenmoe.trace import LayerTrace, TraceRecord


_LAYER_ARRAY_CACHE: dict[int, dict[int, np.ndarray]] = {}
_LAYER_SCORE_CACHE: dict[int, dict[int, np.ndarray | None]] = {}
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


def scores_array_for_layer(record: TraceRecord, layer_id: int) -> np.ndarray | None:
    per_record = _LAYER_SCORE_CACHE.setdefault(id(record), {})
    if layer_id not in per_record:
        per_record[layer_id] = layer_by_id(record, layer_id).scores_array()
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
        statistics_mode: str = "count",
    ) -> None:
        self.store = RouteSigStore(
            top_m=top_m, min_samples=min_samples, statistics_mode=statistics_mode
        )
        if statistics_mode != "count":
            self.name = f"routesig_{statistics_mode}"
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


def _coverage(expert_ids: Iterable[int], true_counts: Counter[int]) -> float:
    total = sum(true_counts.values())
    if total <= 0:
        return 0.0
    predicted = set(int(e) for e in expert_ids)
    covered = sum(count for expert, count in true_counts.items() if expert in predicted)
    return covered / total


_CONFIDENCE_BIN_EDGES = (0.0, 0.25, 0.5, 0.75, 1.01)


class ConfidenceCalibrator:
    """Train-only per-(block_type, layer) confidence calibration.

    Maps a raw RouteSig confidence to the empirical coverage observed for that
    confidence bin on the train split. Falls back to a layer-agnostic bin, then
    the raw confidence when no data exists.
    """

    def __init__(self, *, min_bin_count: int = 5) -> None:
        self.min_bin_count = int(min_bin_count)
        # (block_type, layer_id, bin_idx) -> [coverage_sum, count]
        self._bins: dict[tuple[str, int, int], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        self._global_bins: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        self.frozen = False

    @staticmethod
    def _bin_index(confidence: float) -> int:
        for idx in range(len(_CONFIDENCE_BIN_EDGES) - 1):
            if _CONFIDENCE_BIN_EDGES[idx] <= confidence < _CONFIDENCE_BIN_EDGES[idx + 1]:
                return idx
        return len(_CONFIDENCE_BIN_EDGES) - 2

    def observe(
        self, block_type: str, layer_id: int, confidence: float, coverage: float
    ) -> None:
        if self.frozen:
            raise RuntimeError("ConfidenceCalibrator is frozen (train-only)")
        bin_idx = self._bin_index(float(confidence))
        stats = self._bins[(str(block_type), int(layer_id), bin_idx)]
        stats[0] += float(coverage)
        stats[1] += 1.0
        global_stats = self._global_bins[bin_idx]
        global_stats[0] += float(coverage)
        global_stats[1] += 1.0

    def freeze(self) -> None:
        self.frozen = True

    def calibrate(self, block_type: str, layer_id: int, confidence: float) -> float:
        bin_idx = self._bin_index(float(confidence))
        stats = self._bins.get((str(block_type), int(layer_id), bin_idx))
        if stats is not None and stats[1] >= self.min_bin_count:
            return max(0.0, min(1.0, stats[0] / stats[1]))
        global_stats = self._global_bins.get(bin_idx)
        if global_stats is not None and global_stats[1] >= self.min_bin_count:
            return max(0.0, min(1.0, global_stats[0] / global_stats[1]))
        return max(0.0, min(1.0, float(confidence)))


class HybridRouteSigTemporalPredictor:
    """Per-layer confidence-weighted fusion of RouteSig and temporal-window.

    Layers whose RouteSig train-split delta_vs_global is non-positive are gated
    off (temporal-only). Gate decisions and calibration are fit on the train
    split only (predict-before-update) and frozen afterwards.
    """

    name = "hybrid_routesig_temporal"

    def __init__(
        self,
        train_records: Iterable[TraceRecord] = (),
        *,
        top_m: int = 4,
        min_samples: int = 12,
        window_size: int = 64,
        statistics_mode: str = "count",
    ) -> None:
        self.routesig = RouteSigSegmentPredictor(
            top_m=top_m, min_samples=min_samples, statistics_mode=statistics_mode
        )
        self.temporal = TemporalWindowSegmentPredictor(window_size=window_size)
        self.calibrator = ConfidenceCalibrator()
        self._gate_top_m = int(top_m)
        # layer_id -> [routesig_cov_sum, global_cov_sum, pair_count]
        self._train_stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        self.gate_decisions: dict[int, bool] = {}
        self._fit(list(train_records))

    def _fit(self, train: list[TraceRecord]) -> None:
        global_predictor = GlobalFrequencySegmentPredictor()
        for record in train:
            for segment in record_segments(record):
                if segment.token_start is None or segment.token_end is None:
                    continue
                for layer in record.layers:
                    layer_id = layer.layer_id
                    true_counts = true_segment_layer_counts(record, segment, layer_id)
                    if not true_counts:
                        continue
                    rs = self.routesig.predict(record, segment, layer_id, self._gate_top_m)
                    gl = global_predictor.predict(record, segment, layer_id, self._gate_top_m)
                    rs_cov = _coverage(rs.expert_ids, true_counts)
                    gl_cov = _coverage(gl.expert_ids, true_counts)
                    stats = self._train_stats[layer_id]
                    stats[0] += rs_cov
                    stats[1] += gl_cov
                    stats[2] += 1.0
                    if rs.expert_ids:
                        self.calibrator.observe(
                            segment.block_type, layer_id, rs.confidence, rs_cov
                        )
            self.routesig.update(record)
            self.temporal.update(record)
            global_predictor.update(record)
        for layer_id, (rs_sum, gl_sum, count) in self._train_stats.items():
            delta = (rs_sum - gl_sum) / count if count else 0.0
            self.gate_decisions[int(layer_id)] = delta > 0.0
        self.calibrator.freeze()

    def gate_report(self) -> dict[str, object]:
        deltas = {
            str(layer_id): ((s[0] - s[1]) / s[2] if s[2] else 0.0)
            for layer_id, s in sorted(self._train_stats.items())
        }
        return {
            "predictor": self.name,
            "gate_metric": "train_split_coverage_delta_vs_global",
            "layer_gates": {
                str(layer_id): bool(passed)
                for layer_id, passed in sorted(self.gate_decisions.items())
            },
            "train_delta_vs_global": deltas,
            "train_pair_count": {
                str(layer_id): int(s[2])
                for layer_id, s in sorted(self._train_stats.items())
            },
        }

    def _from_temporal(
        self,
        record: TraceRecord,
        segment: PromptSegment,
        layer_id: int,
        top_m: int,
        reason: str,
    ) -> PredictedExpertSet:
        tw = self.temporal.predict(record, segment, layer_id, top_m)
        return prediction_from_counter(
            record=record,
            segment=segment,
            layer_id=layer_id,
            counter=Counter(
                {
                    expert: weight
                    for expert, weight in zip(tw.expert_ids, tw.expert_weights)
                }
            ),
            top_m=top_m,
            source=self.name,
            fallback_key=f"{reason}/{tw.fallback_key}",
            confidence=tw.confidence,
        )

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        if not self.gate_decisions.get(int(layer_id), False):
            return self._from_temporal(
                record, segment, layer_id, top_m, "gated_temporal_only"
            )
        rs = self.routesig.predict(record, segment, layer_id, top_m)
        if not rs.expert_ids:
            return self._from_temporal(
                record, segment, layer_id, top_m, "routesig_unavailable"
            )
        tw = self.temporal.predict(record, segment, layer_id, top_m)
        fusion_weight = self.calibrator.calibrate(
            segment.block_type, layer_id, rs.confidence
        )
        combined: dict[int, float] = defaultdict(float)
        for expert, weight in zip(rs.expert_ids, rs.expert_weights):
            combined[int(expert)] += fusion_weight * weight
        for expert, weight in zip(tw.expert_ids, tw.expert_weights):
            combined[int(expert)] += (1.0 - fusion_weight) * weight
        ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:top_m]
        expert_ids = [expert for expert, _ in ranked]
        total = sum(weight for _, weight in ranked)
        weights = (
            [weight / total for _, weight in ranked]
            if total > 0
            else _uniform(expert_ids)
        )
        return PredictedExpertSet(
            request_id=record.request_id,
            segment_id=segment.segment_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_weights=weights,
            confidence=float(fusion_weight),
            source=self.name,
            fallback_key=f"fused/{rs.fallback_key}",
            scores={"fusion_weight": float(fusion_weight)},
            block_type=segment.block_type,
            segment_position=segment.segment_position,
            token_start=segment.token_start,
            token_end=segment.token_end,
            weight_rule="hybrid_convex_fusion",
            claim_scope=record.claim_scope,
        )

    def update(self, record: TraceRecord) -> None:
        self.routesig.update(record)
        self.temporal.update(record)


_RANKER_FEATURE_GROUPS = (
    "global",
    "temporal",
    "role",
    "phase",
    "tool_type",
    "trajectory_phase",
    "block_type",
    "position",
    "length_bucket",
)


def _ranker_length_bucket(segment: PromptSegment) -> str:
    tokens = segment.token_count
    if tokens < 8:
        return "lt8"
    if tokens < 32:
        return "lt32"
    if tokens < 128:
        return "lt128"
    return "ge128"


class LearnedRankerSegmentPredictor:
    """Logistic-regression ranker over pre-router features only.

    Features are conditional expert-frequency probabilities keyed by metadata
    visible before routing (role/phase/tool_type/trajectory_phase/block_type/
    position/length bucket) plus global and temporal history. Weights are fit
    on the train split with predict-before-update; history counters keep
    updating online during evaluation.
    """

    name = "learned_ranker"

    def __init__(
        self,
        train_records: Iterable[TraceRecord] = (),
        *,
        window_size: int = 64,
        learning_rate: float = 0.5,
        candidate_pool: int = 8,
        max_train_updates: int = 200_000,
    ) -> None:
        self.window_size = int(window_size)
        self.learning_rate = float(learning_rate)
        self.candidate_pool = int(candidate_pool)
        self.max_train_updates = int(max_train_updates)
        self._global: dict[int, Counter[int]] = defaultdict(Counter)
        self._windows: dict[int, deque[Counter[int]]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self._conditional: dict[tuple[str, str, int], Counter[int]] = defaultdict(
            Counter
        )
        self.weights = np.zeros(len(_RANKER_FEATURE_GROUPS) + 1, dtype=np.float64)
        self._fit(list(train_records))

    @staticmethod
    def _feature_values(
        record: TraceRecord, segment: PromptSegment
    ) -> dict[str, str]:
        meta = record.metadata
        return {
            "role": meta.role,
            "phase": meta.phase,
            "tool_type": meta.tool_type or "none",
            "trajectory_phase": meta.trajectory_phase or "none",
            "block_type": segment.block_type,
            "position": str(min(int(segment.segment_position), 8)),
            "length_bucket": _ranker_length_bucket(segment),
        }

    def _group_probs(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int
    ) -> dict[str, dict[int, float]]:
        probs: dict[str, dict[int, float]] = {}
        counters: dict[str, Counter[int]] = {"global": self._global[layer_id]}
        temporal: Counter[int] = Counter()
        for hist in self._windows[layer_id]:
            temporal.update(hist)
        counters["temporal"] = temporal
        for group, value in self._feature_values(record, segment).items():
            counters[group] = self._conditional[(group, value, layer_id)]
        for group, counter in counters.items():
            total = sum(counter.values())
            probs[group] = (
                {expert: count / total for expert, count in counter.items()}
                if total > 0
                else {}
            )
        return probs

    def _candidates(
        self, probs: dict[str, dict[int, float]], pool: int
    ) -> list[int]:
        candidates: set[int] = set()
        for group_probs in probs.values():
            top = sorted(group_probs.items(), key=lambda item: (-item[1], item[0]))
            candidates.update(expert for expert, _ in top[:pool])
        return sorted(candidates)

    def _feature_vector(
        self, probs: dict[str, dict[int, float]], expert: int
    ) -> np.ndarray:
        vec = np.empty(len(_RANKER_FEATURE_GROUPS) + 1, dtype=np.float64)
        for idx, group in enumerate(_RANKER_FEATURE_GROUPS):
            vec[idx] = probs.get(group, {}).get(expert, 0.0)
        vec[-1] = 1.0
        return vec

    def _fit(self, train: list[TraceRecord]) -> None:
        updates = 0
        for record in train:
            for segment in record_segments(record):
                if segment.token_start is None or segment.token_end is None:
                    continue
                for layer in record.layers:
                    if updates >= self.max_train_updates:
                        break
                    layer_id = layer.layer_id
                    true_counts = true_segment_layer_counts(record, segment, layer_id)
                    if not true_counts:
                        continue
                    probs = self._group_probs(record, segment, layer_id)
                    candidates = set(self._candidates(probs, self.candidate_pool))
                    candidates.update(true_counts)
                    true_set = set(true_counts)
                    for expert in sorted(candidates):
                        vec = self._feature_vector(probs, expert)
                        label = 1.0 if expert in true_set else 0.0
                        logit = float(self.weights @ vec)
                        pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
                        self.weights += self.learning_rate * (label - pred) * vec
                        updates += 1
            self.update(record)

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        probs = self._group_probs(record, segment, layer_id)
        candidates = self._candidates(probs, max(self.candidate_pool, top_m * 2))
        if not candidates:
            return prediction_from_ranked_set(
                record=record,
                segment=segment,
                layer_id=layer_id,
                expert_ids=[],
                source=self.name,
                fallback_key="unavailable",
                confidence=0.0,
            )
        scored: list[tuple[int, float]] = []
        for expert in candidates:
            logit = float(self.weights @ self._feature_vector(probs, expert))
            sigmoid = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
            scored.append((expert, sigmoid))
        scored.sort(key=lambda item: (-item[1], item[0]))
        top = scored[:top_m]
        expert_ids = [expert for expert, _ in top]
        total = sum(score for _, score in top)
        weights = (
            [score / total for _, score in top] if total > 0 else _uniform(expert_ids)
        )
        confidence = float(np.mean([score for _, score in top])) if top else 0.0
        return PredictedExpertSet(
            request_id=record.request_id,
            segment_id=segment.segment_id,
            layer_id=layer_id,
            expert_ids=expert_ids,
            expert_weights=weights,
            confidence=max(0.0, min(1.0, confidence)),
            source=self.name,
            fallback_key="learned_ranker",
            block_type=segment.block_type,
            segment_position=segment.segment_position,
            token_start=segment.token_start,
            token_end=segment.token_end,
            weight_rule="learned_ranker_sigmoid",
            claim_scope=record.claim_scope,
        )

    def update(self, record: TraceRecord) -> None:
        per_layer: dict[int, Counter[int]] = {}
        for segment in record_segments(record):
            if segment.token_start is None or segment.token_end is None:
                continue
            features = self._feature_values(record, segment)
            for layer in record.layers:
                layer_id = layer.layer_id
                hist = true_segment_layer_counts(record, segment, layer_id)
                if not hist:
                    continue
                self._global[layer_id].update(hist)
                per_layer.setdefault(layer_id, Counter()).update(hist)
                for group, value in features.items():
                    self._conditional[(group, value, layer_id)].update(hist)
        for layer_id, hist in per_layer.items():
            self._windows[layer_id].append(hist)


class MetadataMask:
    """Masks pre-router metadata fields on records/segments for ablations.

    Masked fields are replaced by constants so predictors cannot condition on
    them. `length_only` collapses all metadata to the segment length bucket.
    """

    def __init__(self, mask_fields: Iterable[str] = (), *, length_only: bool = False) -> None:
        valid = {"role", "phase", "tool_type", "block_type", "position"}
        self.mask_fields = frozenset(str(item) for item in mask_fields)
        unknown = self.mask_fields - valid
        if unknown:
            raise ValueError(f"unknown mask fields: {sorted(unknown)}")
        self.length_only = bool(length_only)
        if self.length_only:
            self.mask_fields = frozenset(valid)
        self._record_cache: dict[
            int, tuple[TraceRecord, dict[str, PromptSegment]]
        ] = {}

    def mask_segment(self, segment: PromptSegment) -> PromptSegment:
        block_type = segment.block_type
        position = segment.segment_position
        if self.length_only:
            block_type = _ranker_length_bucket(segment)
            position = 0
        else:
            if "block_type" in self.mask_fields:
                block_type = "masked"
            if "position" in self.mask_fields:
                position = 0
        if block_type == segment.block_type and position == segment.segment_position:
            return segment
        return replace(segment, block_type=block_type, segment_position=position)

    def mask_record(
        self, record: TraceRecord
    ) -> tuple[TraceRecord, dict[str, PromptSegment]]:
        cached = self._record_cache.get(id(record))
        if cached is not None:
            return cached
        meta = record.metadata
        meta_kwargs: dict[str, object] = {}
        if "role" in self.mask_fields:
            # agent_id subsumes role in the most specific fallback keys.
            meta_kwargs["role"] = "masked"
            meta_kwargs["agent_id"] = "masked"
        if "phase" in self.mask_fields:
            meta_kwargs["phase"] = "masked"
            meta_kwargs["trajectory_phase"] = None
        if "tool_type" in self.mask_fields:
            meta_kwargs["tool_type"] = None
        masked_meta = replace(meta, **meta_kwargs) if meta_kwargs else meta
        segment_map: dict[str, PromptSegment] = {}
        masked_segments: list[PromptSegment] = []
        for segment in record.prompt_segments:
            masked = self.mask_segment(segment)
            segment_map[segment.segment_id] = masked
            masked_segments.append(masked)
        masked_record = replace(
            record, metadata=masked_meta, prompt_segments=masked_segments
        )
        result = (masked_record, segment_map)
        self._record_cache[id(record)] = result
        return result

    def mask_records(self, records: Iterable[TraceRecord]) -> list[TraceRecord]:
        return [self.mask_record(record)[0] for record in records]


class MetadataMaskingPredictor:
    """Delegates to an inner predictor after masking metadata (ablations)."""

    def __init__(self, inner: SegmentPredictor, mask: MetadataMask, *, name: str) -> None:
        self.inner = inner
        self.mask = mask
        self.name = name

    def predict(
        self, record: TraceRecord, segment: PromptSegment, layer_id: int, top_m: int
    ) -> PredictedExpertSet:
        masked_record, segment_map = self.mask.mask_record(record)
        masked_segment = segment_map.get(segment.segment_id, segment)
        return self.inner.predict(masked_record, masked_segment, layer_id, top_m)

    def update(self, record: TraceRecord) -> None:
        masked_record, _ = self.mask.mask_record(record)
        self.inner.update(masked_record)


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
    min_samples = max(4, top_m * 2)
    predictors: list[SegmentPredictor] = [
        GlobalFrequencySegmentPredictor(train),
        RequestLRUSegmentPredictor(train),
        SequenceHistorySegmentPredictor(train),
        TemporalWindowSegmentPredictor(train, window_size=temporal_window),
        RouteSigSegmentPredictor(train, top_m=top_m, min_samples=min_samples),
        HybridRouteSigTemporalPredictor(
            train,
            top_m=top_m,
            min_samples=min_samples,
            window_size=temporal_window,
        ),
        LearnedRankerSegmentPredictor(train, window_size=temporal_window),
    ]
    if train and all(record.has_router_scores for record in train):
        predictors.append(
            RouteSigSegmentPredictor(
                train,
                top_m=top_m,
                min_samples=min_samples,
                statistics_mode="score_weighted",
            )
        )
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
