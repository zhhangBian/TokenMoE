from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from dataset_adapters import common
from dataset_adapters.sharegpt import convert as convert_sharegpt
from dataset_adapters.swe_agent import HEURISTIC_VERSION, convert as convert_swe_agent
from tokenmoe.metrics import (
    bootstrap_group_delta,
    evaluate_metadata_ablations,
    evaluate_segment_predictors,
    evaluate_segment_predictors_with_splits,
)
from tokenmoe.prediction import (
    ConfidenceCalibrator,
    HybridRouteSigTemporalPredictor,
    LearnedRankerSegmentPredictor,
    RouteSigSegmentPredictor,
    aggregate_record_demand,
    prediction_from_ranked_set,
    top_m_budgets,
)
from tokenmoe.routesig import RouteSigStore
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
    TRACE_SCHEMA_V3,
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


def _swe_trajectory_raw() -> dict:
    return {
        "instance_id": "acme__proj-1",
        "trajectory": [
            {"role": "system", "text": "You are an autonomous programmer."},
            {"role": "user", "text": "ISSUE: TypeError in foo() when bar is None"},
            {"role": "ai", "text": "Let's look at the file.\n```\nopen foo.py 10\n```"},
            {"role": "user", "text": "(Open file: foo.py)\n10: def foo(bar):"},
            {"role": "ai", "text": "Fix the bug.\n```\nedit 10:12\nif bar is None:\nend_of_edit\n```"},
            {
                "role": "user",
                "text": "Your proposed edit has introduced new syntax error(s).\nERRORS:\n- E999 SyntaxError",
            },
            {"role": "ai", "text": "Run the tests.\n```\npytest tests/test_foo.py\n```"},
            {"role": "user", "text": "==== 2 failed, 3 passed ===="},
            {"role": "ai", "text": "Submitting.\n```\nsubmit\n```"},
        ],
    }


def test_swe_agent_adapter_enriched_metadata(tmp_path, monkeypatch):
    (tmp_path / "data.json").write_text(
        json.dumps([_swe_trajectory_raw()]), encoding="utf-8"
    )
    records, unavailable, dag_available = convert_swe_agent(
        tmp_path, 64, "swe-agent/unit"
    )
    assert dag_available is True
    # system + leading issue events are lifted into context, 7 steps remain.
    assert len(records) == 7
    assert all(r.claim_scope == CLAIM_SCOPE_REAL_AGENT for r in records)
    assert len({r.source_group_id for r in records}) == 1

    metas = [r.meta for r in records]
    # Refined segment roles only.
    for record in records:
        assert {seg.block_type for seg in record.prompt_segments} <= {
            "system",
            "code_context",
            "trajectory_event",
            "tool_result",
        }
        # Issue statement lifted into context blocks of every record.
        assert any(
            seg.block_type == "code_context" for seg in record.prompt_segments
        )

    # tool_type heuristics.
    assert metas[0].tool_type == "read_file"
    assert metas[2].tool_type == "edit"
    assert metas[3].tool_type == "inspect_error"
    assert metas[4].tool_type == "run_test"

    # event_outcome heuristics (observations only).
    assert metas[0].event_outcome is None
    assert metas[1].event_outcome == "none"
    assert metas[3].event_outcome == "error_observed"
    assert metas[5].event_outcome == "test_failed"

    # trajectory_phase heuristics.
    assert metas[0].trajectory_phase == "issue_understanding"
    assert metas[2].trajectory_phase == "edit"
    assert metas[4].trajectory_phase == "debug"
    assert metas[6].trajectory_phase == "finalize"

    # DAG fields: linear chain.
    for idx, (record, meta) in enumerate(zip(records, metas)):
        assert meta.dag_depth == idx
        assert meta.group_local_step_index == idx
        assert meta.on_critical_path is True
        assert meta.ready_time == float(idx)
        if idx:
            assert record.dependencies == [records[idx - 1].request_id]
            assert record.dependency_edges == [
                (records[idx - 1].request_id, record.request_id)
            ]

    # Round-trip preserves enriched fields.
    path = tmp_path / "swe.jsonl"
    write_workload_jsonl(records, path)
    loaded = read_workload_jsonl(path)
    assert loaded[3].meta.event_outcome == "error_observed"
    assert loaded[6].meta.trajectory_phase == "finalize"
    assert loaded[5].meta.dag_depth == 5

    # Manifest records the heuristic version.
    monkeypatch.setattr(common, "MANIFEST_DIR", tmp_path / "manifests")
    manifest = common.write_manifest(
        adapter_name="swe_agent",
        source_dataset="swe-agent/unit",
        source_path=tmp_path,
        output_path=path,
        sample_count=len(records),
        conversion_command=["unit"],
        unavailable_fields=unavailable,
        claim_scope=CLAIM_SCOPE_REAL_AGENT,
        dag_available=dag_available,
        manifest_extra={"metadata_heuristic_version": HEURISTIC_VERSION},
    )
    payload = json.loads(manifest.read_text())
    assert payload["metadata_heuristic_version"] == HEURISTIC_VERSION


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


def _v3_trace(meta, prompt, segments, *, with_scores=True, reason=None, idx=0):
    selected = np.zeros((6, 2, 2), dtype=np.int64)
    base = 2 if idx % 2 else 0
    selected[:, 0, :] = [base, base + 1]
    selected[:, 1, :] = [base + 2, base + 3]
    scores = None
    if with_scores:
        scores = np.full(selected.shape, 0.5, dtype=np.float32)
        scores[:, :, 0] = 0.75
        scores[:, :, 1] = 0.25
    return trace_from_selected_experts(
        request_id=meta.request_id,
        metadata=meta,
        model_id="unit-moe",
        backend=CURRENT_TRACE_BACKEND,
        prompt=prompt,
        selected_experts=selected,
        router_scores=scores,
        schema_version=TRACE_SCHEMA_V3,
        router_score_semantics="unit_moe:softmax_topk_renormalized"
        if with_scores
        else None,
        router_scores_unavailable_reason=reason,
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


def test_trace_v3_roundtrip_scores_and_semantics(tmp_path):
    workloads, v2_traces = _records(2)
    v3_traces = [
        _v3_trace(
            t.metadata, t.prompt, t.prompt_segments, with_scores=True, idx=i
        )
        for i, t in enumerate(v2_traces)
    ]
    validate_current_stage_traces(v3_traces)
    path = tmp_path / "traces_v3.jsonl"
    write_trace_jsonl(v3_traces, path)
    loaded = read_trace_jsonl(path)
    assert loaded[0].schema_version == TRACE_SCHEMA_V3
    assert loaded[0].has_router_scores
    assert loaded[0].router_score_semantics == "unit_moe:softmax_topk_renormalized"
    scores = loaded[0].layers[0].scores_array()
    ids = loaded[0].layers[0].selected_array()
    assert scores is not None and scores.shape == ids.shape
    validate_current_stage_traces(loaded)


def test_trace_v3_fail_closed_and_mixed_rejection():
    _, v2_traces = _records(2)
    template = v2_traces[0]

    # v3 without scores must carry an explicit unavailable reason.
    no_reason = _v3_trace(
        template.metadata,
        template.prompt,
        template.prompt_segments,
        with_scores=False,
    )
    try:
        validate_current_stage_traces([no_reason])
    except TraceValidationError:
        pass
    else:
        raise AssertionError("v3 without scores and without reason must be rejected")

    with_reason = _v3_trace(
        template.metadata,
        template.prompt,
        template.prompt_segments,
        with_scores=False,
        reason="vllm_output_missing_routed_expert_scores",
    )
    validate_current_stage_traces([with_reason])

    # v3 with scores but no semantics must be rejected.
    ok = _v3_trace(template.metadata, template.prompt, template.prompt_segments)
    no_semantics = replace(ok, router_score_semantics=None)
    try:
        validate_current_stage_traces([no_semantics])
    except TraceValidationError:
        pass
    else:
        raise AssertionError("v3 scores without semantics must be rejected")

    # Mixed v2 + v3 must be rejected.
    try:
        validate_current_stage_traces([v2_traces[0], ok])
    except TraceValidationError:
        pass
    else:
        raise AssertionError("mixed v2/v3 traces must be rejected")


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


def test_weighted_coverage_from_v3_scores_and_v2_unavailable():
    _, v2_traces = _records(10)
    v3_traces = [
        _v3_trace(t.metadata, t.prompt, t.prompt_segments, with_scores=True, idx=i)
        for i, t in enumerate(v2_traces)
    ]
    report = evaluate_segment_predictors(v3_traces, require_current_stage=True)
    assert report["available"] is True
    oracle_rows = [
        row for row in report["results"] if row["predictor"] == "oracle_future_demand"
    ]
    assert oracle_rows
    for row in oracle_rows:
        assert row["weighted_coverage_status"] == "computed_from_v3_router_scores"
        assert row["weighted_coverage"] is not None
        assert 0.0 < row["weighted_coverage"] <= 1.0 + 1e-6
    # Oracle at the largest budget covers all used experts -> full score mass.
    top_row = max(oracle_rows, key=lambda row: row["top_m"])
    assert abs(top_row["weighted_coverage"] - 1.0) < 1e-3

    v2_report = evaluate_segment_predictors(v2_traces, require_current_stage=True)
    for row in v2_report["results"]:
        assert row["weighted_coverage"] is None
        assert (
            row["weighted_coverage_status"]
            == "unavailable_router_scores_not_captured_by_vllm"
        )


def test_routesig_store_smoothing_and_score_weighted_mode():
    _, traces = _records(4)
    sparse = RouteSigStore(top_m=4, min_samples=1000, smoothing_alpha=0.5)
    # Opposite-parity traces make experts 2/3 known to the layer while the
    # planner key only ever observes experts 0/1.
    sparse.update_segment_trace(traces[0])
    sparse.update_segment_trace(traces[1])
    meta = traces[0].metadata
    segment = traces[0].prompt_segments[0]
    sig = sparse.lookup_segment(meta, segment, 0)
    assert sig is not None
    assert sig.smoothed is True
    assert sig.statistics_mode == "count"
    # Smoothing reserves probability mass for unseen experts.
    assert sum(sig.expert_prob.values()) < 1.0 - 1e-9
    for prob in sig.expert_prob.values():
        assert 0.0 < prob < 1.0

    dense = RouteSigStore(top_m=4, min_samples=2, smoothing_alpha=0.5)
    for trace in traces:
        dense.update_segment_trace(trace)
    dense_sig = dense.lookup_segment(meta, segment, 0)
    assert dense_sig is not None
    assert dense_sig.smoothed is False
    assert abs(sum(dense_sig.expert_prob.values()) - 1.0) < 1e-9

    v3 = [
        _v3_trace(t.metadata, t.prompt, t.prompt_segments, with_scores=True, idx=i)
        for i, t in enumerate(traces)
    ]
    weighted = RouteSigStore(top_m=4, min_samples=2, statistics_mode="score_weighted")
    for trace in v3:
        weighted.update_segment_trace(trace)
    weighted_sig = weighted.lookup_segment(v3[0].metadata, segment, 0)
    assert weighted_sig is not None
    assert weighted_sig.statistics_mode == "score_weighted"
    # Slot 0 carries score 0.75, slot 1 carries 0.25 -> unequal expert mass.
    probs = sorted(weighted_sig.expert_prob.values(), reverse=True)
    assert probs[0] > probs[1]
    try:
        RouteSigStore(statistics_mode="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid statistics_mode must be rejected")


def test_confidence_calibrator_boundaries():
    calibrator = ConfidenceCalibrator(min_bin_count=3)
    # Not enough observations -> raw confidence (clamped).
    assert calibrator.calibrate("system", 0, 0.9) == 0.9
    assert calibrator.calibrate("system", 0, 1.7) == 1.0
    assert calibrator.calibrate("system", 0, -0.2) == 0.0
    for coverage in (0.2, 0.4, 0.6):
        calibrator.observe("system", 0, 0.9, coverage)
    # Bin mean replaces the raw high confidence.
    assert abs(calibrator.calibrate("system", 0, 0.95) - 0.4) < 1e-9
    # Other block types fall back to the layer-agnostic global bin.
    assert abs(calibrator.calibrate("instruction", 5, 0.8) - 0.4) < 1e-9
    # Different bin still falls back to raw confidence.
    assert calibrator.calibrate("system", 0, 0.1) == 0.1
    calibrator.freeze()
    try:
        calibrator.observe("system", 0, 0.9, 1.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("frozen calibrator must reject observe")


def test_hybrid_fusion_math_and_gating():
    workloads, traces = _records(8)
    # top_m=2 < experts-per-layer(4): global cannot cover both parities, so
    # role-keyed RouteSig has a positive train delta and gates pass.
    hybrid = HybridRouteSigTemporalPredictor(traces[:6], top_m=2, min_samples=2)
    report = hybrid.gate_report()
    assert report["layer_gates"] == {"0": True, "1": True}
    assert all(float(v) > 0 for v in report["train_delta_vs_global"].values())

    record = traces[6]
    segment = record.prompt_segments[0]
    prediction = hybrid.predict(record, segment, 0, top_m=4)
    assert prediction.source == "hybrid_routesig_temporal"
    assert prediction.weight_rule == "hybrid_convex_fusion"
    assert prediction.fallback_key.startswith("fused/")
    assert abs(sum(prediction.expert_weights) - 1.0) < 1e-9
    fusion_weight = prediction.scores["fusion_weight"]
    assert 0.0 <= fusion_weight <= 1.0

    # Fusion math: combined = w * routesig + (1 - w) * temporal on the union.
    rs = hybrid.routesig.predict(record, segment, 0, 4)
    tw = hybrid.temporal.predict(record, segment, 0, 4)
    combined: dict[int, float] = {}
    for expert, weight in zip(rs.expert_ids, rs.expert_weights):
        combined[expert] = combined.get(expert, 0.0) + fusion_weight * weight
    for expert, weight in zip(tw.expert_ids, tw.expert_weights):
        combined[expert] = combined.get(expert, 0.0) + (1.0 - fusion_weight) * weight
    ranked = sorted(combined.items(), key=lambda item: (-item[1], item[0]))[:4]
    expected_ids = [expert for expert, _ in ranked]
    total = sum(weight for _, weight in ranked)
    assert prediction.expert_ids == expected_ids
    assert np.allclose(
        prediction.expert_weights, [weight / total for _, weight in ranked]
    )

    # Gated-off layer falls back to temporal-only.
    hybrid.gate_decisions[0] = False
    gated = hybrid.predict(record, segment, 0, top_m=4)
    assert gated.source == "hybrid_routesig_temporal"
    assert gated.fallback_key.startswith("gated_temporal_only/")
    assert gated.expert_ids == tw.expert_ids

    # Unknown layer (never trained) defaults to gated-off.
    unknown = hybrid.predict(record, segment, 99, top_m=4)
    assert unknown.fallback_key.startswith("gated_temporal_only/")


def test_learned_ranker_uses_no_post_router_inputs():
    _, traces = _records(8)
    ranker = LearnedRankerSegmentPredictor(traces[:6])
    record = traces[6]
    segment = record.prompt_segments[0]
    prediction = ranker.predict(record, segment, 0, top_m=4)
    assert prediction.expert_ids
    assert abs(sum(prediction.expert_weights) - 1.0) < 1e-9

    # Same metadata/prompt but completely different routing must not change
    # the prediction: the ranker sees only pre-router inputs.
    altered_selected = np.full((6, 2, 2), 7, dtype=np.int64)
    altered = trace_from_selected_experts(
        request_id=record.request_id,
        metadata=record.metadata,
        model_id="unit-moe",
        backend=CURRENT_TRACE_BACKEND,
        prompt=record.prompt,
        selected_experts=altered_selected,
        token_ids=list(range(6)),
        output_token_count=1,
        generated_token_ids=[99],
        prompt_segments=record.prompt_segments,
        moe_layer_ids=[0, 1],
        router_top_k=2,
        num_experts=8,
        source_dataset="unit",
        source_group_id=record.source_group_id,
        source_index=record.source_index,
        claim_scope=CLAIM_SCOPE_REAL_AGENT,
    )
    altered_prediction = ranker.predict(altered, segment, 0, top_m=4)
    assert altered_prediction.expert_ids == prediction.expert_ids
    assert altered_prediction.expert_weights == prediction.expert_weights


def test_bootstrap_group_delta_cis():
    groups = [f"g{i}" for i in range(20)]
    better = {g: [90.0, 100.0] for g in groups}
    worse = {g: [50.0, 100.0] for g in groups}
    result = bootstrap_group_delta(groups, better, worse, n_resamples=500, seed=7)
    assert abs(float(result["delta"]) - 0.4) < 1e-9
    assert result["significant"] is True
    assert float(result["ci_low"]) <= float(result["delta"]) <= float(result["ci_high"])
    assert result["method"] == "paired_group_bootstrap_percentile_95"

    tied = bootstrap_group_delta(groups, better, dict(better), n_resamples=500)
    assert tied["delta"] == 0.0
    assert tied["significant"] is False

    tiny = bootstrap_group_delta(["only"], better, worse)
    assert tiny["ci_low"] is None and tiny["significant"] is None


def test_online_eval_reports_deltas_splits_and_signal_map():
    _, traces = _records(20)
    report = evaluate_segment_predictors(traces, n_bootstrap_resamples=200)
    assert report["available"] is True
    assert "per_layer_signal_map" in report
    assert set(report["per_layer_signal_map"]) == set(report["top_m_budgets"])
    for row in report["results"]:
        if row["predictor"] in ("temporal_window_frequency",):
            assert "global_frequency" in row.get("delta_vs", {})
            continue
        delta_vs = row.get("delta_vs", {})
        assert "temporal_window_frequency" in delta_vs
        entry = delta_vs["temporal_window_frequency"]
        assert entry["n_resamples"] == 200
        if entry["ci_low"] is not None:
            assert entry["ci_low"] <= entry["delta"] <= entry["ci_high"]
    # Oracle strictly beats temporal on this synthetic parity data.
    oracle_1x = next(
        row
        for row in report["results"]
        if row["predictor"] == "oracle_future_demand" and row["budget"] == "1x"
    )
    assert oracle_1x["delta_vs"]["temporal_window_frequency"]["delta"] > 0

    multi = evaluate_segment_predictors_with_splits(
        traces, train_fractions=(0.7, 0.5), n_bootstrap_resamples=100
    )
    assert set(multi["splits"]) == {"group_time_0.7", "group_time_0.5"}
    for split_report in multi["splits"].values():
        assert split_report["available"] is True
        assert split_report["results"]


def test_metadata_ablation_matrix():
    _, traces = _records(20)
    report = evaluate_metadata_ablations(traces, n_bootstrap_resamples=200)
    assert report["available"] is True
    assert report["budget"] == "2x"
    names = [row["ablation"] for row in report["results"]]
    assert names == [
        "full",
        "minus_role",
        "minus_phase",
        "minus_tool_type",
        "minus_block_type",
        "minus_positions",
        "length_only",
        "temporal_only",
        "routesig_only",
    ]
    for row in report["results"]:
        assert 0.0 <= row["expert_label_hit_rate"] <= 1.0
        if row["ablation"] == "full":
            assert "delta_vs_full" not in row
        else:
            entry = row["delta_vs_full"]
            assert entry["n_resamples"] == 200
            if entry["ci_low"] is not None:
                assert entry["ci_low"] <= entry["delta"] <= entry["ci_high"]


def test_online_eval_and_prediction_scheduler_replay():
    workloads, traces = _records(10)
    report = evaluate_segment_predictors(traces, require_current_stage=True)
    assert report["available"] is True
    assert {row["predictor"] for row in report["results"]} >= {
        "global_frequency",
        "temporal_window_frequency",
        "routesig",
        "hybrid_routesig_temporal",
        "learned_ranker",
        "oracle_future_demand",
    }
    hybrid_rows = [
        row
        for row in report["results"]
        if row["predictor"] == "hybrid_routesig_temporal"
    ]
    assert hybrid_rows
    for row in hybrid_rows:
        assert "layer_gates" in row["gate_report"]
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
