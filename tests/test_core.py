from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from dataset_adapters.sharegpt import convert as convert_sharegpt
from dataset_adapters.swe_agent import convert as convert_swe
from tokenmoe.evaluation import evaluate_traces, paired_group_bootstrap
from tokenmoe.prediction import build_predictors
from tokenmoe.routesig import RouteSigStore
from tokenmoe.schema import (
    CLAIM_SCOPE_REAL_AGENT,
    WORKLOAD_SCHEMA,
    AgentNodeMeta,
    PromptSegment,
    WorkloadRecord,
    read_workload_jsonl,
    write_workload_jsonl,
)
from tokenmoe.trace import (
    TRACE_BACKEND,
    TRACE_SCHEMA,
    TraceValidationError,
    read_trace_jsonl,
    trace_from_vllm,
    validate_trace,
    validate_traces,
    write_trace_jsonl,
)
from tokenmoe.vllm import model_capability, validate_capture_profile


def _metadata(index: int) -> AgentNodeMeta:
    return AgentNodeMeta(
        request_id=f"request-{index}",
        agent_id=f"agent-{index % 3}",
        role="solver" if index % 2 else "planner",
        phase="act",
        graph_node_type="agent_llm_request",
        prompt_block_types=("user_message",),
        trajectory_phase="edit" if index % 2 else "issue_understanding",
        dag_depth=index,
        group_local_step_index=index,
    )


def _workload(index: int) -> WorkloadRecord:
    prompt = f"User:\nrequest {index}"
    meta = _metadata(index)
    return WorkloadRecord(
        request_id=meta.request_id,
        workflow="test-agent",
        prompt=prompt,
        meta=meta,
        prompt_segments=(PromptSegment("user-000", "user_message", 0, 0, len(prompt)),),
        source_dataset="test/source",
        source_index=index,
        source_group_id=f"group-{index // 2}",
        claim_scope=CLAIM_SCOPE_REAL_AGENT,
    )


def _trace(index: int, *, scores: bool = True):
    workload = _workload(index)
    segment = replace(workload.prompt_segments[0], token_start=0, token_end=2)
    base = index % 4
    selected = np.asarray(
        [
            [[base, (base + 1) % 8], [(base + 2) % 8, (base + 3) % 8]],
            [[base, (base + 2) % 8], [(base + 1) % 8, (base + 3) % 8]],
        ],
        dtype=np.int64,
    )
    router_scores = np.full(selected.shape, 0.5, dtype=np.float32) if scores else None
    return trace_from_vllm(
        request_id=workload.request_id,
        metadata=workload.meta,
        model_id="test-moe",
        prompt=workload.prompt,
        prompt_token_ids=(10, 11),
        generated_token_ids=(12,),
        prompt_segments=(segment,),
        selected_experts=selected,
        router_scores=router_scores,
        moe_layer_ids=(1, 3),
        router_top_k=2,
        num_experts=8,
        router_score_semantics="test:softmax_topk_renormalized" if scores else None,
        router_scores_unavailable_reason=None if scores else "capture_disabled",
        source_dataset=workload.source_dataset,
        source_group_id=workload.source_group_id,
        source_index=workload.source_index,
        claim_scope=workload.claim_scope,
    )


def test_workload_roundtrip_and_old_schema_rejected(tmp_path) -> None:
    path = tmp_path / "workloads.jsonl"
    write_workload_jsonl([_workload(0)], path)
    loaded = read_workload_jsonl(path)
    assert loaded[0] == _workload(0)
    assert loaded[0].schema_version == WORKLOAD_SCHEMA

    payload = loaded[0].to_dict()
    payload["schema_version"] = "tokenmoe.workload.v1"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported workload schema"):
        read_workload_jsonl(path)


def test_metadata_boolean_is_strict() -> None:
    payload = _metadata(0).to_dict()
    payload["on_critical_path"] = "false"
    with pytest.raises(TypeError, match="on_critical_path"):
        AgentNodeMeta.from_mapping(payload)


def test_sharegpt_omits_target_answer(tmp_path) -> None:
    source = tmp_path / "sharegpt.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "conversation-1",
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "target answer"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records, _ = convert_sharegpt(source, 1, "sharegpt/test")
    assert "question" in records[0].prompt
    assert "target answer" not in records[0].prompt


def test_swe_adapter_reconstructs_pre_action_prompts(tmp_path) -> None:
    source = tmp_path / "swe.jsonl"
    source.write_text(
        json.dumps(
            {
                "instance_id": "task-1",
                "problem_statement": "fix the bug",
                "trajectory": [
                    {"role": "assistant", "content": "```\ncat app.py\n```"},
                    {"role": "tool", "content": "file contents"},
                    {"role": "assistant", "content": "```\npytest\n```"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records, unavailable = convert_swe(source, 10, "swe/test")
    assert len(records) == 2
    assert "cat app.py" not in records[0].prompt
    assert "cat app.py" in records[1].prompt
    assert "file contents" in records[1].prompt
    assert "pytest" not in records[1].prompt
    assert records[1].dependencies == (records[0].request_id,)
    assert records[0].meta.tool_type is None
    assert "next_action_tool_type" in unavailable


def test_trace_roundtrip_and_strict_boundaries(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    traces = [_trace(0), _trace(1)]
    write_trace_jsonl(traces, path)
    loaded = read_trace_jsonl(path)
    assert loaded == traces
    assert loaded[0].schema_version == TRACE_SCHEMA
    assert loaded[0].backend == TRACE_BACKEND
    assert loaded[0].has_router_scores

    with pytest.raises(TraceValidationError, match="unsupported trace schema"):
        validate_trace(replace(loaded[0], schema_version="tokenmoe.trace.v2"))
    with pytest.raises(TraceValidationError, match="not captured by vLLM"):
        validate_trace(replace(loaded[0], backend="transformers"))


def test_trace_requires_score_reason_and_alignment() -> None:
    id_only = _trace(0, scores=False)
    validate_trace(id_only)
    with pytest.raises(TraceValidationError, match="explicit reason"):
        validate_trace(replace(id_only, router_scores_unavailable_reason=None))
    bad_layer = replace(
        id_only.layers[0],
        selected_experts=((0, 1),),
    )
    with pytest.raises(TraceValidationError, match="shape mismatch"):
        validate_trace(replace(id_only, layers=(bad_layer, id_only.layers[1])))
    with pytest.raises(TraceValidationError, match="homogeneous Router capture"):
        validate_traces([_trace(1), id_only])


def test_routesig_and_predictors_use_aligned_segments() -> None:
    traces = [_trace(index) for index in range(8)]
    store = RouteSigStore(top_m=4, min_support=4)
    store.update_many(traces)
    signature = store.lookup(
        traces[0].metadata, traces[0].prompt_segments[0], layer_id=1
    )
    assert signature is not None
    assert len(signature.top_experts) == 4
    assert 0.0 <= signature.confidence <= 1.0

    predictors = build_predictors(traces[:6], top_m=4, temporal_window=4)
    predictions = [
        predictor.predict(traces[6], traces[6].prompt_segments[0], 1, 4)
        for predictor in predictors
    ]
    assert {prediction.source for prediction in predictions} == {
        "global_frequency",
        "temporal_window",
        "routesig",
    }


def test_evaluation_has_group_splits_and_bootstrap() -> None:
    traces = [_trace(index) for index in range(16)]
    result = evaluate_traces(traces, bootstrap_resamples=50)
    assert result["records"] == 16
    assert result["router_scores_available"] is True
    assert set(result["splits"]) == {"0.7", "0.5"}
    for split in result["splits"].values():
        assert len(split["results"]) == 9
        assert split["split"]["train_groups"] >= 1

    bootstrap = paired_group_bootstrap(
        ["a", "b"],
        {"a": [8, 10], "b": [7, 10]},
        {"a": [4, 10], "b": [3, 10]},
        n_resamples=50,
    )
    assert bootstrap["delta"] > 0
    assert bootstrap["ci_low"] is not None


def test_vllm_model_profile_is_explicit() -> None:
    capability = model_capability(
        {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "num_hidden_layers": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 1,
        }
    )
    assert capability.moe_layer_ids == (1, 2, 3)
    validate_capture_profile(
        capability=capability,
        tensor_parallel_size=1,
        expert_parallel=True,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        kv_transfer_enabled=False,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.9,
    )
    with pytest.raises(ValueError, match="not a supported routed MoE"):
        model_capability({"model_type": "qwen2", "num_hidden_layers": 4})
