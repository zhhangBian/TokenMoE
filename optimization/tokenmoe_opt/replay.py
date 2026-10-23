"""Finite-capacity frozen-timeline diagnostics; no inferred layer timestamps."""

from dataclasses import dataclass
from math import isfinite
from operator import index as integer


def token_microbatches(routes, capacity, max_tokens=64):
    """Return deterministic contiguous token partitions with fitting expert unions."""
    result, start, union = [], 0, set()
    for index, token in enumerate(routes):
        try:
            demand = {integer(expert) for expert in token if type(expert) is not bool}
        except TypeError as exc:
            raise ValueError("Routes require integer expert IDs") from exc
        if not demand or len(demand) != len(token) or min(demand) < 0 or len(demand) > capacity:
            raise ValueError("A token's expert demand exceeds available slots")
        if len(union | demand) > capacity or index - start >= max_tokens:
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


def profile_costs(profile):
    copy = profile["copy"]
    return (
        {
            int(rank): CopyCost(**{**copy, "bytes_per_expert": row["expert_bytes"]})
            for rank, row in profile["ranks"].items()
        }
        if profile.get("ranks")
        else {0: CopyCost(**copy)}
    )


def replay(
    records,
    *,
    capacity,
    cost,
    policy="lru",
    recompute_policy="reject",
    rank_costs=None,
    timeline_kind="contains-demand-stalls",
):
    from .replay_cache import CacheModel

    if recompute_policy != "reject":
        raise ValueError("Only explicit rejection of unrecorded recompute routes is supported")
    cache = CacheModel(capacity, rank_costs or {0: cost}, policy=policy)
    last_time = -1.0
    for record in records:
        if record.get("phase") == "recompute" and not record.get("routes_authoritative"):
            raise ValueError("Operational replay requires separately recorded recompute routes")
        t = float(record["time_seconds"])
        if not isfinite(t) or t < last_time:
            raise ValueError("Replay records must be chronological")
        last_time = t
        if record.get("kind") == "snapshot":
            if t != 0:
                raise ValueError("Initial residency snapshots must precede the replay at time zero")
            cache.initialize(record["slots"], record.get("previously_loaded", ()))
        elif record.get("kind", "demand") == "prefetch":
            if timeline_kind != "neutral":
                raise ValueError("Prefetch cannot reuse timestamps containing demand stalls")
            cache.prefetch(record, t)
        elif record.get("kind", "demand") == "demand":
            cache.demand(record, t)
        else:
            raise ValueError("Unsupported frozen replay record kind")
    result = cache.result(max(0, last_time))
    return {
        **{
            key: cache.counts[key]
            for key in (
                "bytes",
                "unique_demand_hits",
                "unique_demand_misses",
                "reloads",
                "serialized_readiness_wait_seconds",
            )
        },
        **result,
        "schema_version": 1,
        "evidence": "frozen-timeline-cache-replay-diagnostic",
        "calibrated_timing": False,
        "timeline_kind": timeline_kind,
        "limitation": "No compute/tool-dependency timing propagation or live calibration",
    }
