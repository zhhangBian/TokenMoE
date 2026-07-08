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
from tokenmoe.trace import TraceRecord


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
    ) -> None:
        self.top_m = int(top_m)
        self.min_samples = int(min_samples)
        self.stability_window = int(stability_window)
        self.default_miss_cost = float(default_miss_cost)
        self._counts: dict[tuple[tuple[str, ...], int], Counter[int]] = defaultdict(
            Counter
        )
        self._recent: dict[tuple[tuple[str, ...], int], deque[set[int]]] = defaultdict(
            lambda: deque(maxlen=self.stability_window)
        )
        self._num_experts_by_layer: dict[int, int] = {}

    def update(self, meta: AgentNodeMeta, layer_id: int, selected: np.ndarray) -> None:
        selected_arr = np.asarray(selected)
        if selected_arr.size == 0:
            return
        flat = [int(item) for item in selected_arr.reshape(-1) if int(item) >= 0]
        if not flat:
            return
        max_expert = max(flat)
        self._num_experts_by_layer[layer_id] = max(
            self._num_experts_by_layer.get(layer_id, 0), max_expert + 1
        )
        active = set(flat)
        for key in meta.fallback_keys():
            idx = (key, int(layer_id))
            self._counts[idx].update(flat)
            self._recent[idx].append(active)

    def update_trace(self, record: TraceRecord) -> None:
        for layer in record.layers:
            self.update(record.metadata, layer.layer_id, layer.selected_array())

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

    def signature(self, key: tuple[str, ...], layer_id: int) -> RouteSignature | None:
        counts = self._counts.get((key, int(layer_id)))
        if not counts:
            return None
        total = sum(counts.values())
        probs = {expert: count / total for expert, count in counts.items()}
        top = [
            expert
            for expert, _ in sorted(
                counts.items(), key=lambda item: (-item[1], item[0])
            )[: self.top_m]
        ]
        entropy = route_entropy_from_probabilities(probs.values())
        num_experts = max(self._num_experts_by_layer.get(layer_id, 0), len(probs), 1)
        confidence = self._confidence(key, layer_id, total, entropy, num_experts)
        miss_cost = {
            expert: self.default_miss_cost * (1.0 + (1.0 - prob))
            for expert, prob in probs.items()
        }
        return RouteSignature(
            key=key,
            layer_id=int(layer_id),
            expert_prob=probs,
            top_experts=top,
            entropy=float(entropy),
            confidence=float(confidence),
            miss_cost=miss_cost,
            sample_count=int(total),
            updated_at=time.time(),
        )

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
