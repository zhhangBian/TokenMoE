"""Finite-capacity frozen-timeline diagnostics; no inferred layer timestamps."""

from dataclasses import dataclass
from math import isfinite


def token_microbatches(routes, capacity):
    """Return deterministic contiguous token partitions with fitting expert unions."""
    result, start, union = [], 0, set()
    for index, token in enumerate(routes):
        demand = set(map(int, token))
        if not demand or len(demand) > capacity:
            raise ValueError("A token's expert demand exceeds available slots")
        if len(union | demand) > capacity:
            result.append((start, index, tuple(sorted(union))))
            start, union = index, set()
        union.update(demand)
    if union:
        result.append((start, len(routes), tuple(sorted(union))))
    return result


@dataclass(frozen=True)
class CopyCost:
    bytes_per_expert: int
    bandwidth_bytes_second: float
    setup_seconds: float = 0

    def __post_init__(self):
        if (
            not isinstance(self.bytes_per_expert, int)
            or self.bytes_per_expert <= 0
            or not isfinite(self.bandwidth_bytes_second)
            or self.bandwidth_bytes_second <= 0
            or not isfinite(self.setup_seconds)
            or self.setup_seconds < 0
        ):
            raise ValueError("Copy costs must have positive bytes/bandwidth and nonnegative setup")

    @property
    def seconds(self):
        return self.setup_seconds + self.bytes_per_expert / self.bandwidth_bytes_second


def replay(records, *, capacity, cost, policy="lru", recompute_policy="reject"):
    if capacity < 1 or policy not in {"lru", "lfu"}:
        raise ValueError("Invalid replay capacity or replacement policy")
    if recompute_policy != "reject":
        raise ValueError("Only explicit rejection of unrecorded recompute routes is supported")
    resident, inflight, use, frequency = {}, {}, {}, {}
    link_ready, serial, transferred, misses, hits, reloads, wait = 0.0, 0, 0, 0, 0, 0, 0.0
    loaded = set()
    last_time = -1.0
    for record in records:
        if record.get("phase") == "recompute" and not record.get("routes_authoritative"):
            raise ValueError("Operational replay requires separately recorded recompute routes")
        t = float(record["time_seconds"])
        if not isfinite(t) or t < last_time:
            raise ValueError("Replay records must be chronological")
        last_time = t
        layer = record["layer"]
        resident.setdefault(layer, set())
        # Timestamp is an external frozen demand observation, never a miss-delayed future lead.
        for _, _, experts in token_microbatches(record["tokens"], capacity):
            needed = set(experts)
            for expert in experts:
                key = (layer, expert)
                serial += 1
                if expert in resident[layer]:
                    hits += 1
                else:
                    misses += 1
                    while len(resident[layer]) >= capacity:
                        victims = [e for e in resident[layer] if e not in needed]
                        if not victims:
                            raise ValueError("Pinned demand leaves no evictable slot")
                        victim = min(
                            victims,
                            key=lambda e: (
                                frequency.get((layer, e), 0) if policy == "lfu" else 0,
                                use.get((layer, e), 0),
                                e,
                            ),
                        )
                        resident[layer].remove(victim)
                    if key not in inflight:
                        frequency[key] = 0
                        link_ready = max(link_ready, t) + cost.seconds
                        inflight[key] = link_ready
                        transferred += cost.bytes_per_expert
                        reloads += key in loaded
                        loaded.add(key)
                    ready = inflight.pop(key)
                    wait += max(0.0, ready - t)
                    t = max(t, ready)
                    resident[layer].add(expert)
                use[key], frequency[key] = serial, frequency.get(key, 0) + 1
    return {
        "schema_version": 1,
        "evidence": "frozen-timeline-cache-replay-diagnostic",
        "calibrated_timing": False,
        "bytes": transferred,
        "unique_demand_hits": hits,
        "unique_demand_misses": misses,
        "reloads": reloads,
        "serialized_readiness_wait_seconds": wait,
        "limitation": "No compute/tool-dependency timing propagation or live calibration",
    }
