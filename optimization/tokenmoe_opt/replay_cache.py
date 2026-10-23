"""Finite shared-link cache state, with conservative busy-time prefetch admission."""

import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite

from .replay import token_microbatches


@dataclass
class Slot:
    index: int
    expert: int | None = None
    generation: int = 0
    state: str = "FREE"
    pins: int = 0
    last_use: int = 0
    frequency: int = 0
    prefetched: bool = False


class CacheModel:
    def __init__(
        self,
        capacity,
        costs,
        *,
        policy="lru",
        experts=128,
        speculative_bytes=128 * 1024**2,
        inflight_limit=2,
        byte_rate=128 * 1024**2,
    ):
        if capacity < 1 or capacity > experts or policy not in {"lru", "lfu"} or not costs:
            raise ValueError("Invalid cache capacity, expert count, policy or rank costs")
        if min(speculative_bytes, byte_rate) < 0 or inflight_limit < 1:
            raise ValueError("Invalid speculative resource limits")
        self.capacity, self.costs, self.policy, self.experts = capacity, costs, policy, experts
        self.speculative_bytes, self.inflight_limit, self.byte_rate = (
            speculative_bytes,
            inflight_limit,
            byte_rate,
        )
        self.slots = defaultdict(lambda: [Slot(i) for i in range(capacity)])
        self.mapping, self.loaded = {}, set()
        self.copies, self.pending, self.desired = [], [], {}
        self.counts = Counter()
        self.time = self.model_clock = self.link_free = 0.0
        self.serial, self.epoch, self.expiry = 0, 0, 0.0
        self.plan_bytes = defaultdict(int)
        self.rate = {rank: [0.0, 0] for rank in costs}
        self.wake_at = None
        self.initialized_layers = None

    def _key(self, rank, layer, expert):
        if (
            type(layer) is not int
            or layer < 0
            or type(expert) is not int
            or not 0 <= expert < self.experts
        ):
            raise ValueError("Invalid logical layer/expert identity")
        if self.initialized_layers is not None and (rank, layer) not in self.initialized_layers:
            raise ValueError("Missing initial residency for a required rank/layer")
        return rank, layer, expert

    def initialize(self, records, loaded_history=()):
        if self.initialized_layers is not None or self.time or self.mapping or self.counts["bytes"]:
            raise ValueError("Initial residency must precede cache activity")
        seen = set()
        for item in records:
            rank, layer, index = item["rank"], item["layer"], item["slot"]
            if (
                rank not in self.costs
                or type(index) is not int
                or not 0 <= index < self.capacity
                or type(layer) is not int
                or layer < 0
                or (rank, layer, index) in seen
            ):
                raise ValueError("Invalid or duplicate initial slot identity")
            seen.add((rank, layer, index))
            slot = self.slots[rank, layer][index]
            slot.generation = item["generation"]
            if type(slot.generation) is not int or slot.generation < 0:
                raise ValueError("Invalid initial slot generation")
            if item["state"] == "FREE":
                if item["expert"] is not None:
                    raise ValueError("A free slot cannot own an expert")
                continue
            key = self._key(rank, layer, item["expert"])
            if (
                item["state"] not in {"READY", "LOADING"}
                or item.get("pins", 0)
                or key in self.mapping
            ):
                raise ValueError("Initial state requires unpinned, unique ready/loading experts")
            slot.expert, slot.state = item["expert"], item["state"]
            slot.last_use, slot.frequency = item.get("last_use", 0), item.get("frequency", 0)
            if any(
                type(value) is not int or value < 0 for value in (slot.last_use, slot.frequency)
            ):
                raise ValueError("Invalid initial replacement metadata")
            slot.prefetched = item.get("prefetched", False)
            self.serial = max(self.serial, slot.last_use)
            self.mapping[key] = slot
            self.loaded.add(key)
            if slot.state == "LOADING":
                ready = item.get("ready_at_seconds")
                if ready is None or not isfinite(ready) or ready < 0:
                    raise ValueError(
                        "Initial in-flight copies require authoritative readiness times"
                    )
                self.serial += 1
                heapq.heappush(
                    self.copies, (ready, self.serial, rank, layer, slot, slot.generation)
                )
                self.link_free = max(self.link_free, ready)
                self.counts["initial_inflight_bytes"] += self.costs[rank].bytes_per_expert
        layers = {(rank, layer) for rank, layer, _ in seen}
        if any(
            {i for rank, layer, i in seen if (rank, layer) == key} != set(range(self.capacity))
            for key in layers
        ):
            raise ValueError("Initial residency requires complete slot snapshots")
        self.initialized_layers = layers
        for row in loaded_history:
            self.loaded.add(self._key(row["rank"], row["layer"], row["expert"]))

    def _expire(self, time):
        if self.expiry and time >= self.expiry:
            self.pending, self.desired = [], {}
            self.expiry = 0.0
            self.counts["expired_plans"] += 1

    def _advance(self, target, *, busy):
        if target < self.time:
            raise ValueError("Cache observation clock regressed")
        while True:
            ready = self.copies[0][0] if self.copies else float("inf")
            wake = self.wake_at if self.wake_at is not None else float("inf")
            moment = min(ready, wake)
            if moment > target:
                break
            self.time = moment
            if wake == moment:
                self.wake_at = None
            while self.copies and self.copies[0][0] <= moment:
                _, _, rank, layer, slot, generation = heapq.heappop(self.copies)
                if slot.generation != generation or slot.state != "LOADING":
                    raise ValueError("Stale simulated copy completion")
                slot.state = "READY"
                self.counts["completed_copy_bytes"] += self.costs[rank].bytes_per_expert
            self._expire(moment)
            if not busy and moment < target:
                self._admit(moment)
        self.time = target
        self._expire(target)

    def _reserve(self, rank, layer, expert, protected, *, prefetch):
        key = self._key(rank, layer, expert)
        if key in self.mapping:
            return self.mapping[key]
        candidates = [
            s
            for s in self.slots[rank, layer]
            if s.state == "FREE"
            or (s.state == "READY" and not s.pins and s.expert not in protected)
        ]
        if not candidates:
            return None
        preferred = self.desired.get(layer, set())
        slot = min(
            candidates,
            key=lambda s: (
                s.state != "FREE",
                s.expert in preferred,
                s.frequency if self.policy == "lfu" else 0,
                s.last_use,
                s.index,
            ),
        )
        cost = self.costs[rank]
        if slot.expert is not None:
            self.mapping.pop((rank, layer, slot.expert))
            self.counts["evictions"] += 1
            if slot.prefetched:
                self.counts["evicted_unused_prefetch_bytes"] += cost.bytes_per_expert
        slot.expert, slot.generation, slot.state = expert, slot.generation + 1, "LOADING"
        slot.frequency, slot.prefetched = 0, prefetch
        self.mapping[key] = slot
        self.counts["reloads"] += key in self.loaded
        self.loaded.add(key)
        self.counts["bytes"] += cost.bytes_per_expert
        self.counts["prefetch_bytes" if prefetch else "demand_bytes"] += cost.bytes_per_expert
        self.link_free = max(self.link_free, self.time) + cost.seconds
        self.serial += 1
        heapq.heappush(
            self.copies, (self.link_free, self.serial, rank, layer, slot, slot.generation)
        )
        return slot

    def _admit(self, time):
        self._expire(time)
        remaining = []
        for rank, layer, expert in self.pending:
            key = rank, layer, expert
            if key in self.mapping:
                continue
            cost = self.costs[rank]
            start, used = self.rate[rank]
            if time >= start + 1:
                start, used = time, 0
                self.rate[rank] = [start, used]
            if self.plan_bytes[rank] + cost.bytes_per_expert > self.speculative_bytes:
                remaining.append(key)
                continue
            if used + cost.bytes_per_expert > self.byte_rate:
                remaining.append(key)
                if self.byte_rate >= cost.bytes_per_expert:
                    self.wake_at = min(self.wake_at or float("inf"), start + 1)
                continue
            if (
                sum(s.state == "LOADING" for (r, _, _), s in self.mapping.items() if r == rank)
                >= self.inflight_limit
            ):
                remaining.append(key)
                continue
            slot = self._reserve(rank, layer, expert, self.desired.get(layer, set()), prefetch=True)
            if slot is None:
                remaining.append(key)
            else:
                self.plan_bytes[rank] += cost.bytes_per_expert
                self.rate[rank][1] += cost.bytes_per_expert
        self.pending = remaining

    def prefetch(self, record, time):
        if not isfinite(time) or time < 0:
            raise ValueError("Invalid plan observation time")
        if time < self.model_clock:
            self.counts["busy_plan_suppression"] += 1
            return
        self._advance(time, busy=False)
        epoch, expiry = record["epoch"], record["expires_at_seconds"]
        if epoch <= self.epoch or not isfinite(expiry) or expiry <= time:
            self.counts["stale_or_expired_plans"] += 1
            return
        desired = {int(layer): set(experts) for layer, experts in record["desired"].items()}
        for layer, experts in desired.items():
            if len(experts) > self.capacity:
                raise ValueError("Desired residency exceeds capacity")
            for expert in experts:
                self._key(next(iter(self.costs)), layer, expert)
        transfers = sorted({(t["layer"], t["expert"]) for t in record["transfers"]})
        if any(expert not in desired.get(layer, ()) for layer, expert in transfers):
            raise ValueError("Prefetch transfers must belong to desired residency")
        self.epoch, self.expiry, self.desired = epoch, expiry, desired
        self.plan_bytes = defaultdict(int)
        self.pending = [(rank, layer, expert) for layer, expert in transfers for rank in self.costs]
        self._admit(time)

    def demand(self, record, time):
        if not isfinite(time) or time < 0:
            raise ValueError("Invalid demand observation time")
        start = max(time, self.model_clock)
        self._advance(start, busy=time <= self.model_clock)
        self.counts["queue_seconds"] += start - time
        layer = record["layer"]
        if record.get("phase") == "recompute" and not record.get("routes_authoritative"):
            raise ValueError("Operational replay requires separately recorded recompute routes")
        if "tokens" in record:
            chunks = token_microbatches(record["tokens"], self.capacity)
        else:
            if record.get("partition_source_capacity") != self.capacity:
                raise ValueError(
                    "Prepartitioned demands cannot change capacity without token routes"
                )
            experts = tuple(sorted(set(record["experts"])))
            chunks = [(0, 1, experts)]
        compute = record.get("compute_seconds", 0.0)
        if not isfinite(compute) or compute < 0:
            raise ValueError("Invalid declared model compute duration")
        for begin, end, experts in chunks:
            if not experts or len(experts) > self.capacity:
                raise ValueError("Demand does not fit one microbatch")
            requested, indices = self.time, []
            needed = set(experts)
            for expert in experts:
                for rank in self.costs:
                    key = self._key(rank, layer, expert)
                    slot = self.mapping.get(key)
                    ready = slot is not None and slot.state == "READY"
                    self.counts["unique_demand_hits" if ready else "unique_demand_misses"] += 1
                    self.counts["joined_inflight_demands"] += (
                        slot is not None and slot.state == "LOADING"
                    )
                    while slot is None:
                        slot = self._reserve(rank, layer, expert, needed, prefetch=False)
                        if slot is None:
                            if not self.copies:
                                raise ValueError(
                                    "No evictable slot or completion can satisfy demand"
                                )
                            self._advance(self.copies[0][0], busy=True)
                    slot.pins += 1
                    while slot.state == "LOADING":
                        self._advance(self.copies[0][0], busy=True)
                    self.serial += 1
                    slot.last_use, slot.frequency = self.serial, slot.frequency + 1
                    slot.prefetched, slot.state = False, "IN_USE"
                    indices.append(slot)
            self.counts["serialized_readiness_wait_seconds"] += self.time - requested
            fraction = (end - begin) / len(record["tokens"]) if "tokens" in record else 1
            self._advance(self.time + compute * fraction, busy=True)
            for slot in indices:
                slot.pins -= 1
                slot.state = "READY"
            self.counts["microbatches"] += 1
        self.model_clock = self.time
        self.wake_at = self.time if self.pending else self.wake_at
        return self.time

    def result(self, boundary=None):
        if boundary is not None:
            self._advance(max(boundary, self.time), busy=False)
        resident_unused = inflight = 0
        for (rank, _, _), slot in self.mapping.items():
            size = self.costs[rank].bytes_per_expert
            inflight += size if slot.state == "LOADING" else 0
            resident_unused += size if slot.state == "READY" and slot.prefetched else 0
        return {
            **self.counts,
            "finish_seconds": self.time,
            "resident_unused_prefetch_bytes_at_end": resident_unused,
            "inflight_bytes_at_end": inflight,
            "pending_speculative_shards_at_end": len(self.pending),
            "initial_state": "cold-empty"
            if self.initialized_layers is None
            else "explicit snapshots",
            "slots": [
                {
                    "rank": r,
                    "layer": layer,
                    "slot": s.index,
                    "expert": s.expert,
                    "generation": s.generation,
                    "state": s.state,
                }
                for (r, layer), slots in self.slots.items()
                for s in slots
            ],
            "assumptions": [
                "serialized conservative shared copy link",
                "one model execution lane",
                "busy-time plans suppressed; speculation never acquires miss-created extra lead",
            ],
        }
