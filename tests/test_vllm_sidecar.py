from tokenmoe.schema import AgentNodeMeta
from tokenmoe.vllm_sidecar import infer_model_capability, metadata_sidecar_decision


def test_dense_qwen_capability_disables_routed_capture() -> None:
    capability = infer_model_capability(
        {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "num_hidden_layers": 28,
        }
    )

    assert capability.is_moe is False
    assert capability.supports_routed_expert_capture is False
    assert capability.fallback_reason == "dense_model_no_moe_router"


def test_moe_config_enables_routed_capture() -> None:
    capability = infer_model_capability(
        {
            "model_type": "mixtral",
            "architectures": ["MixtralForCausalLM"],
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
        }
    )

    assert capability.is_moe is True
    assert capability.supports_routed_expert_capture is True
    assert capability.moe_fields["num_local_experts"] == 8


def test_sidecar_dense_fallback_records_metadata_keys() -> None:
    meta = AgentNodeMeta(
        request_id="req-1",
        agent_id="planner:0",
        role="planner",
        phase="plan",
        tool_type=None,
        graph_node_type="root",
        prompt_block_types=["system", "instruction"],
    )
    capability = infer_model_capability({"model_type": "qwen2"})

    decision = metadata_sidecar_decision(meta, capability)

    assert decision.enabled is False
    assert decision.route_capture_requested is False
    assert decision.fallback_reason == "dense_model_no_moe_router"
    assert decision.fallback_keys[-1] == "global"
    assert decision.role_phase == "planner/plan"
