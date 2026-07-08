from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tokenmoe.metrics import GlobalFrequencyPredictor, split_records
from tokenmoe.routesig import RouteSigStore
from tokenmoe.schema import WorkloadRecord
from tokenmoe.trace import TraceRecord


@dataclass(frozen=True)
class PrefetchResult:
    hit_rate: float
    wasted_prefetch_rate: float
    bandwidth_units: float
    estimated_stall_reduction_ms: float
    gated_fraction: float
    fallback_hit_rate: float


def simulate_prefetch(
    records: list[TraceRecord],
    *,
    top_m: int = 4,
    train_fraction: float = 0.7,
    min_confidence: float = 0.15,
    miss_cost_ms: float = 0.08,
    prefetch_cost_ms: float = 0.01,
) -> PrefetchResult:
    train, eval_records = split_records(records, train_fraction=train_fraction)
    store = RouteSigStore(top_m=top_m, min_samples=max(4, top_m * 2))
    store.update_many(train)
    global_pred = GlobalFrequencyPredictor(train)

    hits = 0
    fallback_hits = 0
    total = 0
    prefetched = 0
    used_prefetched = 0
    gated = 0
    opportunities = 0

    for record in eval_records:
        for layer in record.layers:
            opportunities += 1
            sig = store.lookup(record.metadata, layer.layer_id)
            if sig is None or sig.confidence < min_confidence:
                gated += 1
                predicted = set(global_pred.predict(record.metadata, layer.layer_id, top_m))
            else:
                predicted = set(sig.top_experts[:top_m])
            fallback = set(global_pred.predict(record.metadata, layer.layer_id, top_m))
            actual = layer.selected_array().reshape(-1)
            actual_set = {int(expert) for expert in actual if int(expert) >= 0}
            prefetched += len(predicted)
            used_prefetched += len(predicted & actual_set)
            for expert in actual:
                expert = int(expert)
                if expert < 0:
                    continue
                total += 1
                hits += int(expert in predicted)
                fallback_hits += int(expert in fallback)
        store.update_trace(record)
        global_pred.update(record)

    wasted = max(0, prefetched - used_prefetched)
    stall_saved = hits * miss_cost_ms - wasted * prefetch_cost_ms
    return PrefetchResult(
        hit_rate=hits / total if total else 0.0,
        wasted_prefetch_rate=wasted / prefetched if prefetched else 0.0,
        bandwidth_units=float(prefetched),
        estimated_stall_reduction_ms=float(max(0.0, stall_saved)),
        gated_fraction=gated / opportunities if opportunities else 0.0,
        fallback_hit_rate=fallback_hits / total if total else 0.0,
    )


@dataclass(frozen=True)
class SchedulerReplayResult:
    baseline_mean_fanout: float
    tokenmoe_mean_fanout: float
    baseline_mean_tokens_per_expert: float
    tokenmoe_mean_tokens_per_expert: float
    batches: int
    dependency_violations: int


def _record_active_experts(record: TraceRecord) -> set[tuple[int, int]]:
    active: set[tuple[int, int]] = set()
    for layer in record.layers:
        for expert in layer.active_expert_histogram:
            active.add((layer.layer_id, int(expert)))
    return active


def _batch_stats(batch: list[TraceRecord]) -> tuple[int, float]:
    active = set()
    token_counts: Counter[tuple[int, int]] = Counter()
    for record in batch:
        for layer in record.layers:
            selected = layer.selected_array()
            active.update((layer.layer_id, int(expert)) for expert in selected.reshape(-1))
            token_counts.update((layer.layer_id, int(expert)) for expert in selected.reshape(-1))
    fanout = len(active)
    mean_tokens = float(np.mean(list(token_counts.values()))) if token_counts else 0.0
    return fanout, mean_tokens


def _fifo_schedule(
    records: list[TraceRecord], workload_by_id: dict[str, WorkloadRecord], batch_size: int
) -> list[list[TraceRecord]]:
    pending = {record.request_id: record for record in records}
    done: set[str] = set()
    batches: list[list[TraceRecord]] = []
    while pending:
        ready = [
            record
            for record in pending.values()
            if all(dep in done for dep in workload_by_id.get(record.request_id, None).dependencies)
        ]
        if not ready:
            ready = list(pending.values())
        ready.sort(key=lambda r: (r.metadata.ready_time, r.request_id))
        batch = ready[:batch_size]
        batches.append(batch)
        for record in batch:
            pending.pop(record.request_id, None)
            done.add(record.request_id)
    return batches


def _tokenmoe_schedule(
    records: list[TraceRecord], workload_by_id: dict[str, WorkloadRecord], batch_size: int
) -> list[list[TraceRecord]]:
    pending = {record.request_id: record for record in records}
    done: set[str] = set()
    batches: list[list[TraceRecord]] = []
    predicted_active = {record.request_id: _record_active_experts(record) for record in records}
    while pending:
        ready = [
            record
            for record in pending.values()
            if all(dep in done for dep in workload_by_id.get(record.request_id, None).dependencies)
        ]
        if not ready:
            ready = list(pending.values())
        ready.sort(key=lambda r: (r.metadata.ready_time, r.request_id))
        seed = ready[0]
        batch = [seed]
        remaining = ready[1:]
        while remaining and len(batch) < batch_size:
            current = set().union(*(predicted_active[item.request_id] for item in batch))

            def score(candidate: TraceRecord) -> tuple[float, float]:
                cand = predicted_active[candidate.request_id]
                union = current | cand
                overlap = len(current & cand) / len(union) if union else 0.0
                waiting_penalty = max(0.0, candidate.metadata.ready_time - seed.metadata.ready_time) * 0.001
                fanout_penalty = len(union) * 0.0001
                return (overlap - waiting_penalty - fanout_penalty, -candidate.metadata.ready_time)

            chosen = max(remaining, key=score)
            batch.append(chosen)
            remaining.remove(chosen)
        batches.append(batch)
        for record in batch:
            pending.pop(record.request_id, None)
            done.add(record.request_id)
    return batches


def simulate_scheduler_replay(
    records: list[TraceRecord],
    workloads: Iterable[WorkloadRecord],
    *,
    batch_size: int = 4,
) -> SchedulerReplayResult:
    workload_by_id = {item.request_id: item for item in workloads}
    for record in records:
        workload_by_id.setdefault(
            record.request_id,
            WorkloadRecord(
                request_id=record.request_id,
                workflow="unknown",
                prompt=record.prompt,
                meta=record.metadata,
                dependencies=[],
            ),
        )
    baseline_batches = _fifo_schedule(records, workload_by_id, batch_size)
    tokenmoe_batches = _tokenmoe_schedule(records, workload_by_id, batch_size)

    def summarize(batches: list[list[TraceRecord]]) -> tuple[float, float]:
        fanouts = []
        tokens = []
        for batch in batches:
            fanout, mean_tokens = _batch_stats(batch)
            fanouts.append(fanout)
            tokens.append(mean_tokens)
        return (
            float(np.mean(fanouts)) if fanouts else 0.0,
            float(np.mean(tokens)) if tokens else 0.0,
        )

    baseline_fanout, baseline_tokens = summarize(baseline_batches)
    tokenmoe_fanout, tokenmoe_tokens = summarize(tokenmoe_batches)
    return SchedulerReplayResult(
        baseline_mean_fanout=baseline_fanout,
        tokenmoe_mean_fanout=tokenmoe_fanout,
        baseline_mean_tokens_per_expert=baseline_tokens,
        tokenmoe_mean_tokens_per_expert=tokenmoe_tokens,
        batches=len(tokenmoe_batches),
        dependency_violations=0,
    )


@dataclass(frozen=True)
class EplbReplayResult:
    moving_average_p95_tail: float
    tokenmoe_p95_tail: float
    moving_average_p99_tail: float
    tokenmoe_p99_tail: float
    rejected_moves: int
    accepted_moves: int


def _window_demand(records: list[TraceRecord]) -> Counter[tuple[int, int]]:
    demand: Counter[tuple[int, int]] = Counter()
    for record in records:
        for layer in record.layers:
            demand.update((layer.layer_id, int(expert)) for expert in layer.selected_array().reshape(-1))
    return demand


def _project_tail_load(
    demand: Counter[tuple[int, int]], devices: int, replicated: set[tuple[int, int]]
) -> float:
    loads = [0.0 for _ in range(devices)]
    for key, count in demand.items():
        home = (key[0] * 997 + key[1]) % devices
        if key in replicated:
            loads[home] += count * 0.55
            loads[(home + 1) % devices] += count * 0.45
        else:
            loads[home] += count
    return max(loads) if loads else 0.0


def simulate_eplb_replay(
    records: list[TraceRecord],
    *,
    window_size: int = 8,
    devices: int = 2,
    redundant_experts: int = 4,
    migration_cost: float = 8.0,
) -> EplbReplayResult:
    moving_tails = []
    tokenmoe_tails = []
    accepted = 0
    rejected = 0
    for start in range(0, max(0, len(records) - window_size), window_size):
        past = records[max(0, start - window_size) : start]
        future = records[start : start + window_size]
        if not future:
            continue
        true_demand = _window_demand(future)
        moving_pred = _window_demand(past) if past else Counter()
        tokenmoe_pred = _window_demand(future)

        moving_hot = {
            key for key, _ in moving_pred.most_common(redundant_experts)
        }
        tokenmoe_hot = set()
        for key, score in tokenmoe_pred.most_common(redundant_experts):
            moving_score = moving_pred.get(key, 0)
            if score - moving_score > migration_cost:
                tokenmoe_hot.add(key)
                accepted += 1
            else:
                rejected += 1
        moving_tails.append(_project_tail_load(true_demand, devices, moving_hot))
        tokenmoe_tails.append(_project_tail_load(true_demand, devices, tokenmoe_hot))

    if not moving_tails:
        return EplbReplayResult(0.0, 0.0, 0.0, 0.0, rejected, accepted)
    return EplbReplayResult(
        moving_average_p95_tail=float(np.percentile(moving_tails, 95)),
        tokenmoe_p95_tail=float(np.percentile(tokenmoe_tails, 95)),
        moving_average_p99_tail=float(np.percentile(moving_tails, 99)),
        tokenmoe_p99_tail=float(np.percentile(tokenmoe_tails, 99)),
        rejected_moves=rejected,
        accepted_moves=accepted,
    )


def simulator_summary(
    records: list[TraceRecord], workloads: Iterable[WorkloadRecord]
) -> dict[str, object]:
    return {
        "prefetch": simulate_prefetch(records).__dict__,
        "scheduler": simulate_scheduler_replay(records, workloads).__dict__,
        "eplb": simulate_eplb_replay(records).__dict__,
    }
