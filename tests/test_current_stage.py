from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from dataset_adapters import common
from dataset_adapters.sharegpt import convert as convert_sharegpt
from tokenmoe.metrics import evaluate_segment_predictors
from tokenmoe.prediction import (
    RouteSigSegmentPredictor,
    aggregate_record_demand,
    prediction_from_ranked_set,
    top_m_budgets,
)
from tokenmoe.schema import (
    CLAIM_SCOPE_REAL_AGENT,
    WORKLOAD_SCHEMA_V2,
    AgentNodeMeta,
    PromptSegment,
    WorkloadRecord,
    read_workload_jsonl,
    write_workload_jsonl,
)
from tokenmoe.simulators import simulate_scheduler_replay, simulator_summary
from tokenmoe.trace import (
    CURRENT_TRACE_BACKEND,
    TRACE_SCHEMA_V2,
    TraceValidationError,
    trace_from_selected_experts,
    validate_current_stage_traces,
    write_trace_jsonl,
    read_trace_jsonl,
)


def _records(n: int = 8):
    workloads = []
    traces = []
    for idx in range(n):
        prompt = "[system]\nagent\n[instruction]\nsolve task\n"
        segments = [
            PromptSegment(
                "system-000",
                "system",
                0,
                0,
                15,
                token_start=0,
                token_end=2,
                alignment_status="aligned",
            ),
            PromptSegment(
                "instruction-001",
                "instruction",
                1,
                15,
                len(prompt),
                token_start=2,
                token_end=6,
                alignment_status="aligned",
            ),
        ]
        dep = [f"req-{idx - 1:03d}"] if idx % 2 == 1 else []
        meta = AgentNodeMeta(
            request_id=f"req-{idx:03d}",
            agent_id=f"agent:{idx // 2}",
            role="coder" if idx % 2 else "planner",
            phase="act" if idx % 2 else "plan",
            tool_type="python" if idx % 2 else None,
            graph_node_type="worker" if idx % 2 else "root",
            prompt_block_types=["system", "instruction"],
            ready_time=float(idx),
        )
        workloads.append(
            WorkloadRecord(
                request_id=meta.request_id,
                workflow="unit-agent",
                prompt=prompt,
                meta=meta,
                dependencies=dep,
                source="unit",
                expected_output_tokens=1,
                schema_version=WORKLOAD_SCHEMA_V2,
                prompt_segments=segments,
                source_dataset="unit",
                source_index=idx,
                source_group_id=f"group-{idx // 2}",
                claim_scope=CLAIM_SCOPE_REAL_AGENT,
                dag_available=True,
                dependency_edges=[(dep[0], meta.request_id)] if dep else [],
            )
        )
        selected = np.zeros((6, 2, 2), dtype=np.int64)
        base = 2 if idx % 2 else 0
        selected[:, 0, :] = [base, base + 1]
        selected[:, 1, :] = [base + 2, base + 3]
        traces.append(
            trace_from_selected_experts(
                request_id=meta.request_id,
                metadata=meta,
                model_id="unit-moe",
                backend=CURRENT_TRACE_BACKEND,
                prompt=prompt,
                selected_experts=selected,
                token_ids=list(range(6)),
                output_token_count=1,
                generated_token_ids=[99],
                prompt_segments=segments,
                moe_layer_ids=[0, 1],
                router_top_k=2,
                num_experts=8,
                source_dataset="unit",
                source_group_id=f"group-{idx // 2}",
                source_index=idx,
                claim_scope=CLAIM_SCOPE_REAL_AGENT,
            )
        )
    return workloads, traces


def test_workload_v2_roundtrip_and_manifest(tmp_path, monkeypatch):
    workloads, _ = _records(2)
    path = tmp_path / "workloads.jsonl"
    write_workload_jsonl(workloads, path)
    loaded = read_workload_jsonl(path)
    assert loaded[0].schema_version == WORKLOAD_SCHEMA_V2
    assert loaded[0].prompt_segments[0].char_start == 0
    assert loaded[1].dependency_edges == [(workloads[0].request_id, workloads[1].request_id)]

    monkeypatch.setattr(common, "MANIFEST_DIR", tmp_path / "manifests")
    manifest = common.write_manifest(
        adapter_name="unit",
        source_dataset="unit/dataset",
        source_path=tmp_path,
        output_path=path,
        sample_count=2,
        conversion_command=["unit"],
        unavailable_fields=["timestamp"],
        claim_scope=CLAIM_SCOPE_REAL_AGENT,
        dag_available=True,
    )
    payload = json.loads(manifest.read_text())
    assert payload["sample_count"] == 2
    assert payload["field_mapping_version"] == common.FIELD_MAPPING_VERSION


def test_sharegpt_adapter_emits_claim_scope_and_spans(tmp_path):
    raw = [
        {
            "id": "conv-1",
            "conversations": [
                {"from": "human", "value": "hello"},
                {"from": "gpt", "value": "hi"},
            ],
        }
    ]
    (tmp_path / "data.json").write_text(json.dumps(raw), encoding="utf-8")
    records, unavailable, dag_available = convert_sharegpt(tmp_path, 1, "sharegpt/unit")
    assert len(records) == 1
    assert records[0].claim_scope == "chat_prompt_only"
    assert records[0].prompt_segments
    assert dag_available is False
    assert "ready_times" in unavailable


def test_trace_v2_validation_and_rejection(tmp_path):
    _, traces = _records(3)
    validate_current_stage_traces(traces)
    path = tmp_path / "traces.jsonl"
    write_trace_jsonl(traces, path)
    loaded = read_trace_jsonl(path)
    assert loaded[0].schema_version == TRACE_SCHEMA_V2
    bad_backend = [replace(loaded[0], backend="transformers-router-logits")]
    try:
        validate_current_stage_traces(bad_backend)
    except TraceValidationError:
        pass
    else:
        raise AssertionError("non-vLLM backend should be rejected")


def test_prediction_shape_budgets_and_demand():
    _, traces = _records(8)
    predictor = RouteSigSegmentPredictor(traces[:6], top_m=4, min_samples=2)
    segment = traces[6].prompt_segments[0]
    prediction = predictor.predict(traces[6], segment, layer_id=0, top_m=4)
    assert prediction.expert_ids
    assert abs(sum(prediction.expert_weights) - 1.0) < 1e-9
    assert top_m_budgets(8) == {"1x": 8, "1.5x": 12, "2x": 16, "3x": 24}

    ranked = prediction_from_ranked_set(
        record=traces[0],
        segment=traces[0].prompt_segments[0],
        layer_id=0,
        expert_ids=[1, 2],
        source="unit",
        fallback_key="unit",
        confidence=1.0,
    )
    demand = aggregate_record_demand([ranked], router_top_k=2)
    assert demand[(0, 1)] == 2.0
    assert demand[(0, 2)] == 2.0


def test_online_eval_and_prediction_scheduler_replay():
    workloads, traces = _records(10)
    report = evaluate_segment_predictors(traces, require_current_stage=True)
    assert report["available"] is True
    assert {row["predictor"] for row in report["results"]} >= {
        "global_frequency",
        "temporal_window_frequency",
        "routesig",
        "oracle_future_demand",
    }
    replay = simulate_scheduler_replay(traces, workloads, batch_size=2)
    assert replay.dependency_violations == 0
    assert replay.policy_results is not None
    assert set(replay.policy_results) == {
        "fifo",
        "temporal_window_frequency",
        "routesig",
        "oracle_future_demand",
    }
    summary = simulator_summary(traces, workloads)
    assert "prefetch" not in summary
    assert summary["archived_excluded"]["eplb"].startswith("excluded")
