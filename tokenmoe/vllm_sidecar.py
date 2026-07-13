from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tokenmoe.schema import AgentNodeMeta


MOE_CONFIG_FIELDS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_routed_experts",
    "num_experts_per_tok",
    "num_selected_experts",
    "moe_intermediate_size",
)


@dataclass(frozen=True)
class ModelCapability:
    model_type: str
    architectures: list[str]
    is_moe: bool
    moe_fields: dict[str, Any]
    supports_routed_expert_capture: bool
    fallback_reason: str | None
    num_hidden_layers: int | None = None
    moe_layer_ids: list[int] | None = None
    router_top_k: int | None = None
    num_experts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SidecarDecision:
    enabled: bool
    route_capture_requested: bool
    fallback_reason: str | None
    metadata_hash: str
    fallback_keys: list[str]
    role_phase: str
    prompt_block_key: str
    elapsed_us: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_model_capability(config: Mapping[str, Any]) -> ModelCapability:
    """Infer whether a Hugging Face config exposes routed MoE structure."""

    architectures = [str(item) for item in config.get("architectures", []) or []]
    model_type = str(config.get("model_type", "unknown"))
    moe_fields = {
        field: config.get(field)
        for field in MOE_CONFIG_FIELDS
        if config.get(field) not in (None, 0, [], {})
    }
    architecture_says_moe = any("moe" in arch.lower() for arch in architectures)
    model_type_says_moe = "moe" in model_type.lower()
    is_moe = bool(moe_fields or architecture_says_moe or model_type_says_moe)
    fallback_reason = None if is_moe else "dense_model_no_moe_router"
    moe_layer_ids = derive_moe_layer_ids(config) if is_moe else []
    router_top_k = config.get("num_experts_per_tok", config.get("num_selected_experts"))
    num_experts = (
        config.get("num_experts")
        or config.get("num_local_experts")
        or config.get("n_routed_experts")
        or config.get("num_routed_experts")
    )
    return ModelCapability(
        model_type=model_type,
        architectures=architectures,
        is_moe=is_moe,
        moe_fields=moe_fields,
        supports_routed_expert_capture=is_moe,
        fallback_reason=fallback_reason,
        num_hidden_layers=None
        if config.get("num_hidden_layers") in (None, "", "null")
        else int(config["num_hidden_layers"]),
        moe_layer_ids=moe_layer_ids,
        router_top_k=None if router_top_k in (None, "", "null") else int(router_top_k),
        num_experts=None if num_experts in (None, "", "null") else int(num_experts),
    )


def load_hf_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def derive_moe_layer_ids(config: Mapping[str, Any]) -> list[int]:
    """Infer MoE layer IDs from common HF MoE config fields."""

    num_layers = config.get("num_hidden_layers")
    if num_layers in (None, "", "null"):
        return []
    total_layers = int(num_layers)
    mlp_only_layers = {int(item) for item in config.get("mlp_only_layers", []) or []}
    first_dense = int(config.get("first_k_dense_replace", 0) or 0)
    moe_freq = int(config.get("moe_layer_freq", config.get("decoder_sparse_step", 1)) or 1)
    layer_ids = []
    for layer_id in range(total_layers):
        if layer_id in mlp_only_layers:
            continue
        if layer_id < first_dense:
            continue
        if moe_freq > 1 and (layer_id - first_dense) % moe_freq != 0:
            continue
        layer_ids.append(layer_id)
    return layer_ids


def validate_vllm_profile(
    *,
    capability: ModelCapability,
    enable_return_routed_experts: bool,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    kv_transfer_enabled: bool,
    tensor_parallel_size: int | None,
    enable_expert_parallel: bool | None,
    dtype: str | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
) -> None:
    if not capability.is_moe:
        raise ValueError(capability.fallback_reason or "selected model is not MoE")
    if not enable_return_routed_experts:
        raise ValueError("enable_return_routed_experts must be true")
    if pipeline_parallel_size != 1:
        raise ValueError("pipeline parallelism must be disabled for routed expert capture")
    if context_parallel_size != 1:
        raise ValueError("context parallelism must be disabled for routed expert capture")
    if kv_transfer_enabled:
        raise ValueError("KV transfer/connectors must be disabled")
    if tensor_parallel_size is None:
        raise ValueError("tensor_parallel_size must be explicit")
    if enable_expert_parallel is None:
        raise ValueError("enable_expert_parallel must be explicit")
    if not dtype:
        raise ValueError("dtype must be explicit")
    if max_model_len is None:
        raise ValueError("max_model_len must be explicit")
    if gpu_memory_utilization is None:
        raise ValueError("gpu_memory_utilization must be explicit")
    if not capability.moe_layer_ids:
        raise ValueError("could not derive moe_layer_ids from model config")
    if capability.router_top_k is None:
        raise ValueError("could not derive router top-k from model config")


def metadata_sidecar_decision(
    meta: AgentNodeMeta,
    capability: ModelCapability,
) -> SidecarDecision:
    """Make the pre-decode TokenMoE sidecar decision for one request.

    The dense-model path intentionally performs no router intervention. It still
    records the same metadata key material used by the MoE predictor so the
    fallback contract is measurable in a real vLLM run.
    """

    start = time.perf_counter()
    key_material = "|".join(
        [
            meta.agent_id,
            meta.role,
            meta.phase,
            meta.tool_type or "none",
            meta.graph_node_type,
            meta.block_type_key,
        ]
    )
    metadata_hash = hashlib.sha1(key_material.encode("utf-8")).hexdigest()[:16]
    fallback_keys = ["/".join(key) for key in meta.fallback_keys()]
    enabled = capability.supports_routed_expert_capture
    elapsed_us = (time.perf_counter() - start) * 1_000_000.0
    return SidecarDecision(
        enabled=enabled,
        route_capture_requested=enabled,
        fallback_reason=capability.fallback_reason,
        metadata_hash=metadata_hash,
        fallback_keys=fallback_keys,
        role_phase=f"{meta.role}/{meta.phase}",
        prompt_block_key=meta.block_type_key,
        elapsed_us=elapsed_us,
    )
