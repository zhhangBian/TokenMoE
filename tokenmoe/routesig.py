from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from tokenmoe.schema import AgentNodeMeta
from tokenmoe.schema import PromptSegment
from tokenmoe.trace import TraceRecord


_LAYER_ARRAY_CACHE: dict[int, dict[int, np.ndarray]] = {}


@dataclass
class RouteSignature:
    key: tuple[str, ...]
    layer_id: int
    expert_prob: dict[int, float]
    top_experts: list[int]
    entropy: float
    confidence: float
    miss_cost: dict[int, float]
    sample_count: int
    updated_at: float
    smoothed: bool = False
    statistics_mode: str = "count"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key"] = list(self.key)
        data["expert_prob"] = {str(k): float(v) for k, v in self.expert_prob.items()}
        data["miss_cost"] = {str(k): float(v) for k, v in self.miss_cost.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteSignature":
        return cls(
            key=tuple(str(item) for item in data["key"]),
            layer_id=int(data["layer_id"]),
            expert_prob={int(k): float(v) for k, v in data["expert_prob"].items()},
            top_experts=[int(x) for x in data["top_experts"]],
            entropy=float(data["entropy"]),
            confidence=float(data["confidence"]),
            miss_cost={int(k): float(v) for k, v in data["miss_cost"].items()},
            sample_count=int(data["sample_count"]),
            updated_at=float(data["updated_at"]),
            smoothed=bool(data.get("smoothed", False)),
            statistics_mode=str(data.get("statistics_mode", "count")),
        )


class RouteSigStore:
    """Online agent-conditioned expert distribution store."""

    def __init__(
        self,
        *,
        top_m: int = 4,
        min_samples: int = 32,
        stability_window: int = 12,
        default_miss_cost: float = 1.0,
        statistics_mode: str = "count",
        smoothing_alpha: float = 0.5,
    ) -> None:
        if statistics_mode not in ("count", "score_weighted"):
            raise ValueError(f"unknown statistics_mode {statistics_mode!r}")
        self.top_m = int(top_m)
        self.min_samples = int(min_samples)
        self.stability_window = int(stability_window)
        self.default_miss_cost = float(default_miss_cost)
        self.statistics_mode = statistics_mode
        self.smoothing_alpha = float(smoothing_alpha)
        self._counts: dict[tuple[tuple[str, ...], int], Counter[int]] = defaultdict(
            Counter
        )
        self._sample_counts: dict[tuple[tuple[str, ...], int], int] = defaultdict(int)
        self._recent: dict[tuple[tuple[str, ...], int], deque[set[int]]] = defaultdict(
            lambda: deque(maxlen=self.stability_window)
        )
        self._num_experts_by_layer: dict[int, int] = {}
        self._signature_cache: dict[tuple[tuple[str, ...], int], RouteSignature | None] = {}

    def update(self, meta: AgentNodeMeta, layer_id: int, selected: np.ndarray) -> None:
        self._update_keys(meta.fallback_keys(), layer_id, selected)

    def update_segment(
        self,
        meta: AgentNodeMeta,
        segment: PromptSegment,
        layer_id: int,
        selected: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> None:
        self._update_keys(
            meta.segment_fallback_keys(segment.block_type, segment.segment_position),
            layer_id,
            selected,
            scores=scores,
        )

    def _update_keys(
        self,
        keys: Iterable[tuple[str, ...]],
        layer_id: int,
        selected: np.ndarray,
        scores: np.ndarray | None = None,
    ) -> None:
        selected_arr = np.asarray(selected)
        if selected_arr.size == 0:
            return
        flat = selected_arr.reshape(-1)
        valid_mask = flat >= 0
        valid = flat[valid_mask]
        if valid.size == 0:
            return
        valid = valid.astype(np.int64, copy=False)
        if self.statistics_mode == "score_weighted" and scores is not None:
            score_flat = np.asarray(scores, dtype=np.float64).reshape(-1)
            if score_flat.shape != flat.shape:
                raise ValueError(
                    "scores must have the same shape as selected experts"
                )
            valid_scores = score_flat[valid_mask]
            hist: Counter[int] = Counter()
            for expert in np.unique(valid):
                hist[int(expert)] = float(valid_scores[valid == expert].sum())
        else:
            experts, counts = np.unique(valid, return_counts=True)
            hist = Counter(
                {int(expert): int(count) for expert, count in zip(experts, counts)}
            )
        max_expert = max(hist)
        self._num_experts_by_layer[layer_id] = max(
            self._num_experts_by_layer.get(layer_id, 0), max_expert + 1
        )
        active = set(hist)
        for key in keys:
            idx = (key, int(layer_id))
            self._counts[idx].update(hist)
            self._sample_counts[idx] += int(valid.size)
            self._recent[idx].append(active)
        self._signature_cache.clear()

    def update_trace(self, record: TraceRecord) -> None:
        for layer in record.layers:
            selected = _selected_array_for_layer(record, layer.layer_id)
            self.update(record.metadata, layer.layer_id, selected)

    def update_segment_trace(self, record: TraceRecord) -> None:
        layer_arrays = {
            layer.layer_id: _selected_array_for_layer(record, layer.layer_id)
            for layer in record.layers
        }
        layer_scores: dict[int, np.ndarray | None] = {}
        if self.statistics_mode == "score_weighted":
            layer_scores = {
                layer.layer_id: layer.scores_array() for layer in record.layers
            }
        for segment in record.prompt_segments:
            if segment.token_start is None or segment.token_end is None:
                continue
            for layer in record.layers:
                selected = layer_arrays[layer.layer_id][
                    segment.token_start : segment.token_end
                ]
                scores = layer_scores.get(layer.layer_id)
                if scores is not None:
                    scores = scores[segment.token_start : segment.token_end]
                self.update_segment(
                    record.metadata, segment, layer.layer_id, selected, scores=scores
                )

    def update_many(self, records: Iterable[TraceRecord]) -> None:
        for record in records:
            self.update_trace(record)

    def lookup(
        self,
        meta: AgentNodeMeta,
        layer_id: int,
        *,
        min_confidence: float = 0.0,
    ) -> RouteSignature | None:
        for key in meta.fallback_keys():
            sig = self.signature(key, layer_id)
            if sig is not None and sig.confidence >= min_confidence:
                return sig
        return None

    def lookup_segment(
        self,
        meta: AgentNodeMeta,
        segment: PromptSegment,
        layer_id: int,
        *,
        min_confidence: float = 0.0,
    ) -> RouteSignature | None:
        for key in meta.segment_fallback_keys(segment.block_type, segment.segment_position):
            sig = self.signature(key, layer_id)
            if sig is not None and sig.confidence >= min_confidence:
                return sig
        return None

    def signature(self, key: tuple[str, ...], layer_id: int) -> RouteSignature | None:
        cache_key = (key, int(layer_id))
        if cache_key in self._signature_cache:
            return self._signature_cache[cache_key]
        counts = self._counts.get((key, int(layer_id)))
        if not counts:
            self._signature_cache[cache_key] = None
            return None
        total = float(sum(counts.values()))
        sample_count = self._sample_counts.get((key, int(layer_id)), 0)
        if sample_count <= 0:
            sample_count = int(round(total)) if total > 0 else len(counts)
        num_experts = max(self._num_experts_by_layer.get(layer_id, 0), len(counts), 1)
        smoothed = sample_count < self.min_samples and self.smoothing_alpha > 0.0
        if smoothed:
            denom = total + self.smoothing_alpha * num_experts
            probs = {
                expert: (count + self.smoothing_alpha) / denom
                for expert, count in counts.items()
            }
        else:
            probs = {expert: count / total for expert, count in counts.items()}
        top = [
            expert
            for expert, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[: self.top_m]
        ]
        entropy = route_entropy_from_probabilities(probs.values())
        confidence = self._confidence(key, layer_id, sample_count, entropy, num_experts)
        miss_cost = {
            expert: self.default_miss_cost * (1.0 + (1.0 - prob))
            for expert, prob in probs.items()
        }
        signature = RouteSignature(
            key=key,
            layer_id=int(layer_id),
            expert_prob=probs,
            top_experts=top,
            entropy=float(entropy),
            confidence=float(confidence),
            miss_cost=miss_cost,
            sample_count=int(sample_count),
            updated_at=time.time(),
            smoothed=smoothed,
            statistics_mode=self.statistics_mode,
        )
        self._signature_cache[cache_key] = signature
        return signature

    def _confidence(
        self,
        key: tuple[str, ...],
        layer_id: int,
        sample_count: int,
        entropy: float,
        num_experts: int,
    ) -> float:
        sample_term = min(1.0, sample_count / max(1, self.min_samples))
        max_entropy = math.log(max(2, num_experts))
        concentration = 1.0 - min(1.0, entropy / max_entropy)
        recent = list(self._recent.get((key, layer_id), []))
        if len(recent) < 2:
            stability = 0.5
        else:
            overlaps = []
            for a, b in zip(recent, recent[1:]):
                union = a | b
                overlaps.append(1.0 if not union else len(a & b) / len(union))
            stability = float(np.mean(overlaps)) if overlaps else 0.5
        return max(0.0, min(1.0, 0.45 * sample_term + 0.35 * concentration + 0.20 * stability))

    def top_experts(
        self,
        meta: AgentNodeMeta,
        layer_id: int,
        *,
        top_m: int | None = None,
        min_confidence: float = 0.0,
    ) -> list[int]:
        sig = self.lookup(meta, layer_id, min_confidence=min_confidence)
        if sig is None:
            return []
        return sig.top_experts[: (top_m or self.top_m)]

    def top_experts_for_segment(
        self,
        meta: AgentNodeMeta,
        segment: PromptSegment,
        layer_id: int,
        *,
        top_m: int | None = None,
        min_confidence: float = 0.0,
    ) -> list[int]:
        sig = self.lookup_segment(meta, segment, layer_id, min_confidence=min_confidence)
        if sig is None:
            return []
        return sig.top_experts[: (top_m or self.top_m)]

    def signatures(self) -> list[RouteSignature]:
        result: list[RouteSignature] = []
        for key, layer_id in sorted(self._counts):
            sig = self.signature(key, layer_id)
            if sig is not None:
                result.append(sig)
        return result

    def to_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "top_m": self.top_m,
            "min_samples": self.min_samples,
            "stability_window": self.stability_window,
            "default_miss_cost": self.default_miss_cost,
            "signatures": [sig.to_dict() for sig in self.signatures()],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _selected_array_for_layer(record: TraceRecord, layer_id: int) -> np.ndarray:
    per_record = _LAYER_ARRAY_CACHE.setdefault(id(record), {})
    if layer_id not in per_record:
        per_record[layer_id] = record.layer(layer_id).selected_array()
    return per_record[layer_id]


def route_entropy_from_probabilities(probabilities: Iterable[float]) -> float:
    probs = np.asarray([p for p in probabilities if p > 0], dtype=np.float64)
    if probs.size == 0:
        return 0.0
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs)).sum())


def build_routesig_store(
    records: Iterable[TraceRecord],
    *,
    top_m: int = 4,
    min_samples: int = 32,
    stability_window: int = 12,
) -> RouteSigStore:
    store = RouteSigStore(
        top_m=top_m, min_samples=min_samples, stability_window=stability_window
    )
    store.update_many(records)
    return store
