from __future__ import annotations

import numpy as np

from tokenmoe.metrics import evaluate_predictors, locality_summary
from tokenmoe.routesig import RouteSigStore
from tokenmoe.simulators import simulate_eplb_replay, simulate_prefetch, simulate_scheduler_replay
from tokenmoe.trace import read_trace_jsonl, trace_from_selected_experts, write_trace_jsonl
from tokenmoe.workloads import generate_workloads


def _records(n: int = 16):
    workloads = generate_workloads(per_workflow=4, workflows=["planner-coder-tester"])[:n]
    records = []
    role_base = {"planner": 0, "coder": 2, "tester": 4, "critic": 6}
    for item in workloads:
        selected = np.zeros((6, 2, 2), dtype=np.int64)
        base = role_base.get(item.meta.role, 0)
        for token in range(selected.shape[0]):
            for layer in range(selected.shape[1]):
                selected[token, layer] = [(base + layer) % 8, (base + layer + 1) % 8]
        records.append(
            trace_from_selected_experts(
                request_id=item.request_id,
                metadata=item.meta,
                model_id="unit-test",
                backend="synthetic",
                prompt=item.prompt,
                selected_experts=selected,
            )
        )
    return workloads, records


def test_trace_jsonl_roundtrip(tmp_path):
    _, records = _records(4)
    path = tmp_path / "traces.jsonl"
    write_trace_jsonl(records, path)
    loaded = read_trace_jsonl(path)
    assert len(loaded) == 4
    assert loaded[0].metadata.role == records[0].metadata.role
    assert loaded[0].layers[0].active_expert_histogram


def test_routesig_fallback_and_confidence():
    _, records = _records(12)
    store = RouteSigStore(top_m=3, min_samples=4)
    for record in records[:8]:
        store.update_trace(record)
    sig = store.lookup(records[-1].metadata, layer_id=0)
    assert sig is not None
    assert sig.key in records[-1].metadata.fallback_keys()
    assert 0 <= sig.confidence <= 1
    assert sig.top_experts


def test_predictors_and_locality_summary():
    _, records = _records(16)
    results = evaluate_predictors(records, top_m_values=(2, 4))
    assert {r.predictor for r in results} >= {
        "global_frequency",
        "request_lru",
        "sequence_history",
        "routesig",
    }
    summary = locality_summary(records)
    assert summary["num_records"] == len(records)
    assert summary["route_entropy"]["mean_entropy"] >= 0


def test_simulators_return_metrics():
    workloads, records = _records(16)
    prefetch = simulate_prefetch(records, top_m=3)
    scheduler = simulate_scheduler_replay(records, workloads, batch_size=3)
    eplb = simulate_eplb_replay(records, window_size=4)
    assert 0 <= prefetch.hit_rate <= 1
    assert scheduler.batches > 0
    assert scheduler.dependency_violations == 0
    assert eplb.moving_average_p95_tail >= 0
