from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
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
    return ModelCapability(
        model_type=model_type,
        architectures=architectures,
        is_moe=is_moe,
        moe_fields=moe_fields,
        supports_routed_expert_capture=is_moe,
        fallback_reason=fallback_reason,
    )


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
