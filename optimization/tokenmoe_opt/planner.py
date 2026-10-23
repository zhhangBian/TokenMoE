"""Model-specific expected demand and conservative shared transfer reservations."""

import heapq
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from tokenmoe_collect.clock import clock_metadata


@dataclass(frozen=True)
class Demand:
    session: str
    identity: str
    distribution: np.ndarray
    expected_tokens: float
    arrival_probability: float
    service_probability: float
    scheduler_state: str
    terminal: bool = False


def aggregate(demands, *, identity, shape, top_k=8):
    if top_k < 1 or top_k > shape[-1]:
        raise ValueError("Top-k must fit the model's expert count")
    expected = np.zeros(shape, dtype=float)
    sources = []
    for demand in demands:
        if demand.identity != identity:
            raise ValueError("Demand model/template identity differs from planner")
        if demand.terminal:
            continue
        if demand.scheduler_state not in {"running", "queued", "waiting_tools"}:
            raise ValueError("Actual scheduler/session state must be explicit")
        probabilities = (demand.arrival_probability, demand.service_probability)
        if any(not 0 <= p <= 1 for p in probabilities) or demand.expected_tokens < 0:
            raise ValueError("Invalid probability or token volume")
        dist = demand.distribution
        if dist.shape != shape or not np.isfinite(dist).all() or (dist < 0).any():
            raise ValueError("Invalid phase demand distribution")
        if not np.allclose(dist.sum(axis=-1), 1):
            raise ValueError("Phase demand must be normalized per layer")
        weight = demand.expected_tokens * demand.arrival_probability * demand.service_probability
        expected += top_k * weight * dist
        sources.append(
            {"session": demand.session, "weight": weight, "scheduler_state": demand.scheduler_state}
        )
    return expected, sources


@dataclass
class Ledger:
    rank_free_bytes: dict[int, int]
    shared_link_bytes: int
    speculative_byte_limit: int
    speculative_inflight_limit: int
    expert_bytes: dict[int, int]
    kv_reserved_bytes: int = 0
    inflight: set[tuple[int, int, int]] = field(default_factory=set)
    reserved_bytes: int = 0
    kv_arbitration: str = "unavailable"

    def reserve(self, layer, expert):
        required = {
            (rank, layer, expert): size
            for rank, size in self.expert_bytes.items()
            if (rank, layer, expert) not in self.inflight
        }
        total = sum(required.values())
        if any(size > self.rank_free_bytes[rank] for (rank, _, _), size in required.items()):
            return False, "per_rank_memory"
        if self.reserved_bytes + total > self.speculative_byte_limit:
            return False, "speculative_bytes"
        if len(self.inflight) + len(required) > self.speculative_inflight_limit:
            return False, "speculative_inflight"
        if total > self.shared_link_bytes - self.kv_reserved_bytes:
            return False, "shared_link"
        for key, size in required.items():
            self.rank_free_bytes[key[0]] -= size
            self.inflight.add(key)
        self.shared_link_bytes -= total
        self.reserved_bytes += total
        return True, "reserved" if total else "joined_existing_transfer"


def plan(
    expected,
    *,
    identity,
    epoch,
    now_ns,
    expiry_ns,
    capacity,
    resident,
    pinned,
    demand,
    residency_epoch,
    ledger,
    cost_profile,
    sources=(),
    min_residence_epochs=2,
    max_churn=8,
    confidence=1.0,
):
    if expiry_ns <= now_ns or not 0 <= confidence <= 1:
        raise ValueError("Invalid plan expiry or prediction confidence")
    if not cost_profile.get("measured") or cost_profile.get("identity") != identity:
        raise ValueError("Executable plans require matching measured cost profiles")
    costs = [cost_profile["copy_seconds"], cost_profile["demand_wait_seconds"]]
    if (
        not all(np.isfinite(cost) and cost >= 0 for cost in costs)
        or not 0 <= cost_profile.get("hide_probability", 0) <= 1
        or capacity < 1
        or max_churn < 0
        or min_residence_epochs < 0
    ):
        raise ValueError("Invalid measured costs or planning limits")
    if (expected < 0).any() or not np.isfinite(expected).all():
        raise ValueError("Expected demand must be finite and nonnegative")
    desired, decisions, transfers = {}, [], []
    protected, stable, current, candidates = {}, {}, {}, []
    quanta = np.maximum(1, expected.sum(axis=-1, keepdims=True) / capacity)
    benefits = (
        quanta * (1 - np.exp(-expected / quanta)) * cost_profile["demand_wait_seconds"] * confidence
    )
    exposed_copy = cost_profile["copy_seconds"] * (1 - cost_profile.get("hide_probability", 0))
    for layer, values in enumerate(expected):
        current[layer] = set(resident.get(layer, ()))
        protected[layer] = set(pinned.get(layer, ())) | set(demand.get(layer, ()))
        if len(protected[layer]) > capacity:
            raise ValueError("Scheduled demand needs capacity-fitting microbatch execution")
        desired[layer] = current[layer] | protected[layer]
        stable[layer] = {
            e
            for e in current[layer]
            if epoch - residency_epoch.get((layer, e), epoch) < min_residence_epochs
        }
        while len(desired[layer]) > capacity:
            evictable = desired[layer] - protected[layer]
            if not evictable:
                raise ValueError("Pinned experts leave no feasible demand allocation")
            desired[layer].remove(min(evictable, key=lambda e: (benefits[layer, e], -e)))
        for expert, volume in enumerate(values):
            if volume > 0 and expert not in desired[layer] and expert not in current[layer]:
                heapq.heappush(
                    candidates, (-(benefits[layer, expert] - exposed_copy), layer, expert)
                )
    rejected = Counter()
    while candidates and len(transfers) < max_churn:
        upper, layer, expert = heapq.heappop(candidates)
        victim = None
        if len(desired[layer]) >= capacity:
            evictable = desired[layer] - protected[layer] - stable[layer]
            if not evictable:
                rejected["protected_slots"] += 1
                continue
            victim = min(evictable, key=lambda e: (benefits[layer, e], -e))
        displaced = benefits[layer, victim] if victim is not None else 0
        gain = benefits[layer, expert] - displaced - exposed_copy
        if gain <= 0:
            rejected["nonpositive_gain"] += 1
            continue
        # A prior replacement can raise displaced reuse; re-rank globally.
        if -gain > upper + 1e-12:
            heapq.heappush(candidates, (-gain, layer, expert))
            continue
        accepted, reason = ledger.reserve(layer, expert)
        if accepted:
            if victim is not None:
                desired[layer].remove(victim)
            desired[layer].add(expert)
            stable[layer].add(expert)
            transfers.append({"layer": layer, "expert": expert})
            decisions.append(
                {
                    "layer": layer,
                    "expert": expert,
                    "reason": reason,
                    "gain": float(gain),
                    "displaced": victim,
                }
            )
        else:
            rejected[reason] += 1
    if candidates:
        rejected["churn_limit"] += len(candidates)
    return {
        "schema_version": 1,
        "clock_domain_id": clock_metadata()["clock_domain_id"],
        "epoch": epoch,
        "identity": identity,
        "created_at": now_ns,
        "expires_at": expiry_ns,
        "sources": list(sources),
        "policy_version": "expected-stall-v2",
        "cost_profile_version": cost_profile["version"],
        "confidence": confidence,
        "occurrence_assumption": "independent capacity-scale microbatches; unique loads per quantum",
        "desired": {str(layer): sorted(experts) for layer, experts in desired.items()},
        "transfers": transfers,
        "decisions": decisions,
        "deferred_counts": dict(rejected),
        "limits": {"max_churn": max_churn, "min_residence_epochs": min_residence_epochs},
        "reserved_rank_bytes": ledger.reserved_bytes,
        "kv_arbitration": ledger.kv_arbitration,
    }
