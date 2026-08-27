"""Online route signatures keyed only by pre-router request metadata."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from tokenmoe.schema import AgentNodeMeta, PromptSegment
from tokenmoe.trace import TraceRecord


StatisticsMode = Literal["count", "score_weighted"]
RouteKey = tuple[str, ...]


def fallback_keys(meta: AgentNodeMeta, segment: PromptSegment) -> tuple[RouteKey, ...]:
    """Return stable keys from most specific to global."""

    block = segment.block_type
    position = str(segment.position)
    keys: list[RouteKey] = []
    if meta.trajectory_phase and meta.tool_type:
        keys.extend(
            [
                (
                    meta.agent_id,
                    meta.role,
                    meta.trajectory_phase,
                    meta.tool_type,
                    block,
                    position,
                ),
                (
                    meta.role,
                    meta.trajectory_phase,
                    meta.tool_type,
                    block,
                    position,
                ),
            ]
        )
    if meta.trajectory_phase:
        keys.append((meta.role, meta.trajectory_phase, block, position))
    keys.extend(
        [
            (meta.role, meta.phase, block, position),
            (meta.role, meta.phase, block),
            (meta.role, block),
            (block,),
            (meta.role, meta.phase),
            (meta.role,),
            ("global",),
        ]
    )
    return tuple(dict.fromkeys(keys))


@dataclass(frozen=True, slots=True)
class RouteSignature:
    key: RouteKey
    layer_id: int
    top_experts: tuple[int, ...]
    probabilities: dict[int, float]
    unseen_probability: float
    entropy: float
    confidence: float
    support: int
    smoothed: bool
    statistics_mode: StatisticsMode

    def probability(self, expert_id: int) -> float:
        return self.probabilities.get(expert_id, self.unseen_probability)


class RouteSigStore:
    """Bounded online sufficient statistics for metadata-conditioned routing."""

    def __init__(
        self,
        *,
        top_m: int,
        min_support: int = 16,
        smoothing: float = 0.5,
        stability_window: int = 8,
        statistics_mode: StatisticsMode = "count",
    ) -> None:
        if top_m < 1 or min_support < 1 or smoothing < 0:
            raise ValueError("invalid RouteSig configuration")
        if statistics_mode not in ("count", "score_weighted"):
            raise ValueError(f"unsupported statistics mode {statistics_mode!r}")
        self.top_m = top_m
        self.min_support = min_support
        self.smoothing = smoothing
        self.statistics_mode = statistics_mode
        self._counts: dict[tuple[RouteKey, int], Counter[int]] = defaultdict(Counter)
        self._support: Counter[tuple[RouteKey, int]] = Counter()
        self._recent: dict[tuple[RouteKey, int], deque[frozenset[int]]] = defaultdict(
            lambda: deque(maxlen=stability_window)
        )
        self._num_experts: dict[int, int] = {}
        self._cache: dict[tuple[RouteKey, int], RouteSignature] = {}

    def update_segment(
        self,
        meta: AgentNodeMeta,
        segment: PromptSegment,
        layer_id: int,
        selected_experts: np.ndarray,
        router_scores: np.ndarray | None = None,
        *,
        num_experts: int,
    ) -> None:
        selected = np.asarray(selected_experts, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            return
        if np.any((selected < 0) | (selected >= num_experts)):
            raise ValueError("selected expert is outside the model expert range")
        if self.statistics_mode == "score_weighted":
            if router_scores is None:
                raise ValueError("score_weighted RouteSig requires router scores")
            scores = np.asarray(router_scores, dtype=np.float64).reshape(-1)
            if scores.shape != selected.shape or not np.isfinite(scores).all():
                raise ValueError("router scores are invalid")
            histogram: Counter[int] = Counter()
            for expert_id, score in zip(selected, scores):
                histogram[int(expert_id)] += float(score)
        else:
            ids, counts = np.unique(selected, return_counts=True)
            histogram = Counter(
                {int(expert_id): int(count) for expert_id, count in zip(ids, counts)}
            )
        self._num_experts[layer_id] = num_experts
        active = frozenset(histogram)
        for key in fallback_keys(meta, segment):
            index = (key, layer_id)
            self._counts[index].update(histogram)
            self._support[index] += 1
            self._recent[index].append(active)
            self._cache.pop(index, None)

    def update_trace(self, record: TraceRecord) -> None:
        for segment in record.prompt_segments:
            if segment.token_start is None or segment.token_end is None:
                continue
            token_slice = slice(segment.token_start, segment.token_end)
            for layer in record.layers:
                scores = layer.scores_array()
                self.update_segment(
                    record.metadata,
                    segment,
                    layer.layer_id,
                    layer.selected_array()[token_slice],
                    None if scores is None else scores[token_slice],
                    num_experts=record.num_experts,
                )

    def update_many(self, records: Iterable[TraceRecord]) -> None:
        for record in records:
            self.update_trace(record)

    def lookup(
        self,
        meta: AgentNodeMeta,
        segment: PromptSegment,
        layer_id: int,
        *,
        min_confidence: float = 0.0,
    ) -> RouteSignature | None:
        for key in fallback_keys(meta, segment):
            signature = self.signature(key, layer_id)
            if signature is not None and signature.confidence >= min_confidence:
                return signature
        return None

    def signature(self, key: RouteKey, layer_id: int) -> RouteSignature | None:
        index = (key, layer_id)
        if index in self._cache:
            return self._cache[index]
        counts = self._counts.get(index)
        num_experts = self._num_experts.get(layer_id)
        if not counts or num_experts is None:
            return None
        support = self._support[index]
        alpha = self.smoothing if support < self.min_support else 0.0
        total = float(sum(counts.values()))
        denominator = total + alpha * num_experts
        probabilities = {
            expert_id: (float(value) + alpha) / denominator
            for expert_id, value in counts.items()
        }
        unseen_probability = alpha / denominator if alpha else 0.0
        probability_vector = np.full(num_experts, unseen_probability, dtype=np.float64)
        for expert_id, probability in probabilities.items():
            probability_vector[expert_id] = probability
        positive = probability_vector[probability_vector > 0]
        entropy = float(-(positive * np.log(positive)).sum())
        ranked = sorted(
            range(num_experts),
            key=lambda expert_id: (-probability_vector[expert_id], expert_id),
        )[: self.top_m]
        confidence = self._confidence(index, support, entropy, num_experts)
        signature = RouteSignature(
            key=key,
            layer_id=layer_id,
            top_experts=tuple(ranked),
            probabilities=probabilities,
            unseen_probability=unseen_probability,
            entropy=entropy,
            confidence=confidence,
            support=support,
            smoothed=bool(alpha),
            statistics_mode=self.statistics_mode,
        )
        self._cache[index] = signature
        return signature

    def _confidence(
        self,
        index: tuple[RouteKey, int],
        support: int,
        entropy: float,
        num_experts: int,
    ) -> float:
        support_score = min(1.0, support / self.min_support)
        concentration = 1.0 - min(1.0, entropy / math.log(max(2, num_experts)))
        recent = self._recent[index]
        if len(recent) < 2:
            stability = 0.5
        else:
            overlaps = []
            for left, right in zip(recent, list(recent)[1:]):
                union = left | right
                overlaps.append(len(left & right) / len(union) if union else 1.0)
            stability = float(np.mean(overlaps))
        return float(0.5 * support_score + 0.3 * concentration + 0.2 * stability)
