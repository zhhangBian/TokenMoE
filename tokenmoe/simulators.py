from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from tokenmoe.metrics import GlobalFrequencyPredictor, split_records
from tokenmoe.metrics import split_records_by_group
from tokenmoe.prediction import (
    OracleSegmentPredictor,
    RouteSigSegmentPredictor,
    TemporalWindowSegmentPredictor,
    predict_record_demand,
    record_true_demand,
    selected_array_for_layer,
)
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
    replay_kind: str = "unknown"
    dag_scheduling_available: bool = False
    policy_results: dict[str, dict[str, float | int | str]] = None  # type: ignore[assignment]
    low_confidence_gated_fraction: float = 0.0
    missing_span_fraction: float = 0.0
    unavailable_layer_fraction: float = 0.0
    unavailable_metric_fraction: float = 0.0


def _batch_stats(batch: list[TraceRecord]) -> tuple[int, float]:
    active = set()
    token_counts: Counter[tuple[int, int]] = Counter()
    for record in batch:
        for layer in record.layers:
            selected = selected_array_for_layer(record, layer.layer_id)
            valid = selected.reshape(-1)
            valid = valid[valid >= 0]
            if valid.size == 0:
                continue
            experts, counts = np.unique(
                valid.astype(np.int64, copy=False),
                return_counts=True,
            )
            active.update((layer.layer_id, int(expert)) for expert in experts)
            token_counts.update(
                {
                    (layer.layer_id, int(expert)): int(count)
                    for expert, count in zip(experts, counts)
                }
            )
    fanout = len(active)
    mean_tokens = float(np.mean(list(token_counts.values()))) if token_counts else 0.0
    return fanout, mean_tokens


def _dependencies_satisfied(
    record: TraceRecord,
    workload_by_id: dict[str, WorkloadRecord],
    pending: dict[str, TraceRecord],
    done: set[str],
) -> bool:
    workload = workload_by_id.get(record.request_id)
    dependencies = workload.dependencies if workload is not None else []
    return all(dep in done or dep not in pending for dep in dependencies)


def _ready_records(
    pending: dict[str, TraceRecord],
    workload_by_id: dict[str, WorkloadRecord],
    done: set[str],
    step: int,
) -> list[TraceRecord]:
    ready = [
        record
        for record in pending.values()
        if _dependencies_satisfied(record, workload_by_id, pending, done)
        and record.metadata.ready_time <= step
    ]
    if ready:
        return sorted(ready, key=lambda r: (r.metadata.ready_time, r.request_id))
    future = [
        record
        for record in pending.values()
        if _dependencies_satisfied(record, workload_by_id, pending, done)
    ]
    return sorted(future, key=lambda r: (r.metadata.ready_time, r.request_id))


def _dependency_violations(
    batches: list[list[TraceRecord]], workload_by_id: dict[str, WorkloadRecord]
) -> int:
    done: set[str] = set()
    scheduled_ids = {record.request_id for batch in batches for record in batch}
    violations = 0
    for batch in batches:
        for record in batch:
            workload = workload_by_id.get(record.request_id)
            for dep in workload.dependencies if workload is not None else []:
                if dep in scheduled_ids and dep not in done:
                    violations += 1
        for record in batch:
            done.add(record.request_id)
    return violations


def _candidate_cost(
    demand: dict[tuple[int, int], float],
    *,
    waiting_steps: float,
    mean_confidence: float,
    confidence_threshold: float,
) -> tuple[float, float, float, float]:
    active = [value for value in demand.values() if value > 0]
    fanout_cost = float(len(active))
    low_density_cost = float(sum(1.0 / max(value, 1.0) for value in active))
    waiting_cost = max(0.0, waiting_steps) * 0.05
    confidence_penalty = 2.0 if mean_confidence < confidence_threshold else 0.0
    return fanout_cost, low_density_cost, waiting_cost, confidence_penalty


def _run_policy(
    *,
    policy_name: str,
    records: list[TraceRecord],
    workload_by_id: dict[str, WorkloadRecord],
    batch_size: int,
    predictor,
    top_m: int,
    router_top_k: int,
    confidence_threshold: float,
    max_delay_steps: int,
) -> tuple[list[list[TraceRecord]], dict[str, float | int | str]]:
    pending = {record.request_id: record for record in records}
    done = {record.request_id for record in records if record.request_id not in pending}
    batches: list[list[TraceRecord]] = []
    low_confidence = 0
    prediction_ops = 0
    step = 0
    while pending:
        ready = _ready_records(pending, workload_by_id, done, step)
        if not ready:
            break
        batch: list[TraceRecord] = []
        if policy_name == "fifo":
            batch = ready[:batch_size]
        else:
            candidates = list(ready)
            demand_cache: dict[
                str, tuple[Counter[tuple[int, int]], list[float]]
            ] = {}
            batch_demand: Counter[tuple[int, int]] = Counter()
            batch_confidences: list[float] = []

            def cached_demand(record: TraceRecord) -> tuple[Counter[tuple[int, int]], list[float]]:
                cached = demand_cache.get(record.request_id)
                if cached is not None:
                    return cached
                demand, predictions = predict_record_demand(
                    predictor,
                    record,
                    top_m=top_m,
                    router_top_k=router_top_k,
                )
                payload = (
                    Counter(demand),
                    [pred.confidence for pred in predictions],
                )
                demand_cache[record.request_id] = payload
                return payload

            while candidates and len(batch) < batch_size:
                starved = [
                    record
                    for record in candidates
                    if step - record.metadata.ready_time >= max_delay_steps
                ]
                if starved:
                    chosen = sorted(starved, key=lambda r: (r.metadata.ready_time, r.request_id))[0]
                    batch.append(chosen)
                    candidates.remove(chosen)
                    continue
                best_record = None
                best_score = None
                for candidate in candidates:
                    candidate_demand, candidate_confidences = cached_demand(candidate)
                    combined_demand = Counter(batch_demand)
                    combined_demand.update(candidate_demand)
                    confidences = batch_confidences + candidate_confidences
                    mean_conf = float(np.mean(confidences)) if confidences else 0.0
                    prediction_ops += 1
                    low_confidence += int(mean_conf < confidence_threshold)
                    score = _candidate_cost(
                        dict(combined_demand),
                        waiting_steps=max(0.0, step - candidate.metadata.ready_time),
                        mean_confidence=mean_conf,
                        confidence_threshold=confidence_threshold,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_record = candidate
                if best_record is None:
                    break
                batch.append(best_record)
                chosen_demand, chosen_confidences = cached_demand(best_record)
                batch_demand.update(chosen_demand)
                batch_confidences.extend(chosen_confidences)
                candidates.remove(best_record)
        batches.append(batch)
        for record in batch:
            pending.pop(record.request_id, None)
            done.add(record.request_id)
            if predictor is not None and policy_name != "oracle":
                predictor.update(record)
        step += 1
    fanouts = []
    mean_tokens = []
    overlaps = []
    for batch in batches:
        fanout, tokens = _batch_stats(batch)
        fanouts.append(fanout)
        mean_tokens.append(tokens)
        demands = [set(record_true_demand(record)) for record in batch]
        if len(demands) > 1:
            union = set().union(*demands)
            intersection = set.intersection(*demands) if demands else set()
            overlaps.append(len(intersection) / len(union) if union else 0.0)
    return batches, {
        "policy": policy_name,
        "batches": len(batches),
        "actual_mean_fanout": float(np.mean(fanouts)) if fanouts else 0.0,
        "actual_mean_tokens_per_expert": float(np.mean(mean_tokens)) if mean_tokens else 0.0,
        "batch_expert_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "dependency_violations": _dependency_violations(batches, workload_by_id),
        "low_confidence_gated_fraction": low_confidence / prediction_ops if prediction_ops else 0.0,
        "mean_added_waiting_steps": 0.0,
        "p95_added_waiting_steps": 0.0,
        "max_delay": float(max_delay_steps),
    }


def simulate_scheduler_replay(
    records: list[TraceRecord],
    workloads: Iterable[WorkloadRecord],
    *,
    batch_size: int = 4,
    router_top_k: int | None = None,
    top_m: int | None = None,
    train_fraction: float = 0.7,
    confidence_threshold: float = 0.1,
    max_delay_steps: int = 8,
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
    effective_top_k = router_top_k or (records[0].router_top_k if records else None) or (records[0].top_k if records else 1)
    effective_top_m = top_m or (2 * effective_top_k)
    train, eval_records, split_info = split_records_by_group(records, train_fraction)
    if not split_info.get("available"):
        train, eval_records = split_records(records, train_fraction)
    dag_available = any(item.dag_available for item in workload_by_id.values())
    claim_scopes = {record.claim_scope for record in records if record.claim_scope}
    replay_kind = (
        "agent_dag"
        if dag_available and claim_scopes <= {"real_agent_metadata"}
        else "prompt_or_domain_locality"
    )
    temporal = TemporalWindowSegmentPredictor(train, window_size=64)
    routesig = RouteSigSegmentPredictor(
        train, top_m=effective_top_m, min_samples=max(4, effective_top_m * 2)
    )
    oracle = OracleSegmentPredictor(train)
    _, fifo_summary = _run_policy(
        policy_name="fifo",
        records=eval_records,
        workload_by_id=workload_by_id,
        batch_size=batch_size,
        predictor=None,
        top_m=effective_top_m,
        router_top_k=effective_top_k,
        confidence_threshold=confidence_threshold,
        max_delay_steps=max_delay_steps,
    )
    _, temporal_summary = _run_policy(
        policy_name="temporal_window_frequency",
        records=eval_records,
        workload_by_id=workload_by_id,
        batch_size=batch_size,
        predictor=temporal,
        top_m=effective_top_m,
        router_top_k=effective_top_k,
        confidence_threshold=confidence_threshold,
        max_delay_steps=max_delay_steps,
    )
    _, routesig_summary = _run_policy(
        policy_name="routesig",
        records=eval_records,
        workload_by_id=workload_by_id,
        batch_size=batch_size,
        predictor=routesig,
        top_m=effective_top_m,
        router_top_k=effective_top_k,
        confidence_threshold=confidence_threshold,
        max_delay_steps=max_delay_steps,
    )
    _, oracle_summary = _run_policy(
        policy_name="oracle",
        records=eval_records,
        workload_by_id=workload_by_id,
        batch_size=batch_size,
        predictor=oracle,
        top_m=effective_top_m,
        router_top_k=effective_top_k,
        confidence_threshold=confidence_threshold,
        max_delay_steps=max_delay_steps,
    )
    policy_results = {
        "fifo": fifo_summary,
        "temporal_window_frequency": temporal_summary,
        "routesig": routesig_summary,
        "oracle_future_demand": oracle_summary,
    }
    missing_segments = sum(
        1
        for record in eval_records
        for segment in record.prompt_segments
        if segment.token_start is None or segment.token_end is None
    )
    total_segments = sum(len(record.prompt_segments) for record in eval_records)
    unavailable_layers = sum(len(record.unavailable_layer_ids) for record in eval_records)
    total_layers = sum(len(record.moe_layer_ids or []) for record in eval_records)
    return SchedulerReplayResult(
        baseline_mean_fanout=float(fifo_summary["actual_mean_fanout"]),
        tokenmoe_mean_fanout=float(routesig_summary["actual_mean_fanout"]),
        baseline_mean_tokens_per_expert=float(fifo_summary["actual_mean_tokens_per_expert"]),
        tokenmoe_mean_tokens_per_expert=float(routesig_summary["actual_mean_tokens_per_expert"]),
        batches=int(routesig_summary["batches"]),
        dependency_violations=int(routesig_summary["dependency_violations"]),
        replay_kind=replay_kind,
        dag_scheduling_available=dag_available,
        policy_results=policy_results,
        low_confidence_gated_fraction=float(routesig_summary["low_confidence_gated_fraction"]),
        missing_span_fraction=missing_segments / total_segments if total_segments else 0.0,
        unavailable_layer_fraction=unavailable_layers / total_layers if total_layers else 0.0,
        unavailable_metric_fraction=0.0 if eval_records else 1.0,
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
            selected = selected_array_for_layer(record, layer.layer_id)
            valid = selected.reshape(-1)
            valid = valid[valid >= 0]
            if valid.size == 0:
                continue
            experts, counts = np.unique(
                valid.astype(np.int64, copy=False),
                return_counts=True,
            )
            demand.update(
                {
                    (layer.layer_id, int(expert)): int(count)
                    for expert, count in zip(experts, counts)
                }
            )
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
        "scheduler": simulate_scheduler_replay(records, workloads).__dict__,
        "archived_excluded": {
            "prefetch": "excluded_current_stage_next_stage_runtime_integration",
            "eplb": "excluded_current_stage_next_stage_replica_placement",
        },
    }
