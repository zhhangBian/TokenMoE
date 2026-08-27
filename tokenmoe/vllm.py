"""Validation helpers for the TokenMoE vLLM capture fork."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelCapability:
    model_type: str
    architectures: tuple[str, ...]
    num_hidden_layers: int
    moe_layer_ids: tuple[int, ...]
    router_top_k: int
    num_experts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "num_hidden_layers": self.num_hidden_layers,
            "moe_layer_ids": list(self.moe_layer_ids),
            "router_top_k": self.router_top_k,
            "num_experts": self.num_experts,
        }


def load_model_config(model_path: str | Path) -> dict[str, Any]:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"model config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def derive_moe_layer_ids(config: Mapping[str, Any]) -> tuple[int, ...]:
    try:
        num_layers = int(config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("model config has no valid num_hidden_layers") from exc
    dense_layers = {int(item) for item in config.get("mlp_only_layers", []) or []}
    first_moe_layer = int(config.get("first_k_dense_replace", 0) or 0)
    frequency = int(
        config.get("moe_layer_freq", config.get("decoder_sparse_step", 1)) or 1
    )
    if frequency < 1:
        raise ValueError("MoE layer frequency must be positive")
    return tuple(
        layer_id
        for layer_id in range(num_layers)
        if layer_id not in dense_layers
        and layer_id >= first_moe_layer
        and (layer_id - first_moe_layer) % frequency == 0
    )


def model_capability(config: Mapping[str, Any]) -> ModelCapability:
    model_type = str(config.get("model_type", ""))
    architectures = tuple(str(item) for item in config.get("architectures", []) or [])
    num_experts_raw = next(
        (
            config.get(name)
            for name in (
                "num_experts",
                "num_local_experts",
                "n_routed_experts",
                "num_routed_experts",
            )
            if config.get(name) is not None
        ),
        None,
    )
    top_k_raw = next(
        (
            config.get(name)
            for name in ("num_experts_per_tok", "num_selected_experts")
            if config.get(name) is not None
        ),
        None,
    )
    if num_experts_raw is None or top_k_raw is None:
        raise ValueError("selected model is not a supported routed MoE model")
    capability = ModelCapability(
        model_type=model_type,
        architectures=architectures,
        num_hidden_layers=int(config["num_hidden_layers"]),
        moe_layer_ids=derive_moe_layer_ids(config),
        router_top_k=int(top_k_raw),
        num_experts=int(num_experts_raw),
    )
    if not capability.moe_layer_ids:
        raise ValueError("model config produced no MoE layers")
    if not 0 < capability.router_top_k < capability.num_experts:
        raise ValueError("model router top-k/expert count is invalid")
    return capability


def validate_capture_profile(
    *,
    capability: ModelCapability,
    tensor_parallel_size: int,
    expert_parallel: bool,
    pipeline_parallel_size: int,
    context_parallel_size: int,
    kv_transfer_enabled: bool,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> None:
    del capability, expert_parallel
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if pipeline_parallel_size != 1 or context_parallel_size != 1:
        raise ValueError("router capture currently requires PP=1 and CP=1")
    if kv_transfer_enabled:
        raise ValueError("router capture currently requires KV transfer disabled")
    if not dtype:
        raise ValueError("dtype must be explicit")
    if max_model_len < 1:
        raise ValueError("max_model_len must be positive")
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
