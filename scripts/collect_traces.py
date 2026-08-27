#!/usr/bin/env python3
"""Collect prompt-only routed-expert traces from the TokenMoE vLLM fork."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tokenmoe.schema import WorkloadRecord, iter_workload_jsonl
from tokenmoe.trace import (
    align_prompt_segments,
    decode_vllm_routed_experts_b64,
    iter_trace_jsonl,
    trace_from_vllm,
)
from tokenmoe.vllm import (
    ModelCapability,
    load_model_config,
    model_capability,
    validate_capture_profile,
)


DEFAULT_ARTIFACT_ROOT = Path(
    os.environ.get(
        "TOKENMOE_ARTIFACT_ROOT", "/home/youwei/bzh/dataset/tokenmoe_artifacts"
    )
)
SCORE_SEMANTICS = {
    "qwen3_moe": "softmax_topk_renormalized",
    "qwen2_moe": "softmax_topk_renormalized",
    "deepseek_v2": "softmax_topk_scaled_unnormalized",
    "deepseek_v3": "sigmoid_topk_scaled",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_ARTIFACT_ROOT / "traces" / "tokenmoe_traces.jsonl"),
    )
    parser.add_argument(
        "--env-report",
        default=str(DEFAULT_ARTIFACT_ROOT / "logs" / "trace_collection_env.json"),
    )
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--expert-parallel", choices=("on", "off"), required=True)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--kv-transfer-enabled", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--router-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture aligned router scores (default: enabled).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing request IDs to an existing valid trace file.",
    )
    return parser.parse_args()


def _supports_parameter(callable_object: Any, name: str) -> bool:
    signature = inspect.signature(callable_object)
    return name in signature.parameters or any(
        value.kind == inspect.Parameter.VAR_KEYWORD
        for value in signature.parameters.values()
    )


def _llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    required = ("enable_return_routed_experts",)
    if any(not _supports_parameter(LLM, name) for name in required):
        raise RuntimeError("the imported vLLM is not the TokenMoE capture fork")
    kwargs: dict[str, Any] = {
        "model": args.model,
        "enforce_eager": True,
        "enable_return_routed_experts": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "hf_overrides": {"sliding_window": None},
    }
    if args.router_scores:
        if not _supports_parameter(LLM, "enable_return_routed_expert_scores"):
            raise RuntimeError("the imported vLLM fork has no router-score capture")
        kwargs["enable_return_routed_expert_scores"] = True
    if _supports_parameter(LLM, "enable_expert_parallel"):
        kwargs["enable_expert_parallel"] = args.expert_parallel == "on"
    elif args.expert_parallel == "on":
        raise RuntimeError("the imported vLLM has no expert-parallel option")
    return LLM(**kwargs)


def _sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    if not _supports_parameter(SamplingParams, "routed_experts_prompt_start"):
        raise RuntimeError("the imported vLLM has no routed-expert output support")
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        routed_experts_prompt_start=0,
    )


def _tokenizer(llm: Any) -> Any:
    if hasattr(llm, "get_tokenizer"):
        return llm.get_tokenizer()
    raise RuntimeError("vLLM did not expose its tokenizer")


def _array(payload: Any) -> np.ndarray:
    return (
        decode_vllm_routed_experts_b64(payload)
        if isinstance(payload, str)
        else np.asarray(payload)
    )


def _prompt_array(payload: Any, prompt_tokens: int, name: str) -> np.ndarray:
    array = _array(payload)
    if array.ndim != 3 or array.shape[0] < prompt_tokens:
        raise ValueError(
            f"{name} must have shape [tokens >= prompt_tokens, layers, top_k]"
        )
    return array[:prompt_tokens]


def _semantics(capability: ModelCapability) -> str:
    try:
        return f"{capability.model_type}:{SCORE_SEMANTICS[capability.model_type]}"
    except KeyError as exc:
        raise ValueError(
            f"router score semantics are unknown for {capability.model_type!r}"
        ) from exc


def _existing_request_ids(path: Path, resume: bool) -> set[str]:
    if not resume or not path.exists():
        return set()
    request_ids: set[str] = set()
    for record in iter_trace_jsonl(path):
        if record.request_id in request_ids:
            raise ValueError(
                f"duplicate request ID in existing trace: {record.request_id}"
            )
        request_ids.add(record.request_id)
    return request_ids


def collect(args: argparse.Namespace, capability: ModelCapability) -> int:
    if args.limit < 1 or args.batch_size < 1:
        raise ValueError("limit and batch-size must be positive")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _existing_request_ids(output, args.resume)
    records = [
        record
        for record in itertools.islice(iter_workload_jsonl(args.workload), args.limit)
        if record.request_id not in completed
    ]
    if not records:
        print("all requested workload records are already present")
        return 0

    llm = _llm(args)
    tokenizer = _tokenizer(llm)
    params = _sampling_params(args)
    mode = "a" if args.resume and output.exists() else "w"
    written = 0
    with output.open(mode, encoding="utf-8") as stream:
        for batch_start in range(0, len(records), args.batch_size):
            batch = records[batch_start : batch_start + args.batch_size]
            outputs = llm.generate(
                [record.prompt for record in batch], params, use_tqdm=False
            )
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of requests")
            for workload, request_output in zip(batch, outputs):
                if not request_output.outputs:
                    raise RuntimeError(f"{workload.request_id}: empty vLLM output")
                completion = request_output.outputs[0]
                routed = getattr(completion, "routed_experts", None)
                if routed is None:
                    raise RuntimeError(f"{workload.request_id}: routed experts missing")
                if not request_output.prompt_token_ids:
                    raise RuntimeError(
                        f"{workload.request_id}: vLLM prompt token IDs missing"
                    )
                prompt_ids = tuple(
                    int(item) for item in request_output.prompt_token_ids
                )
                selected = _prompt_array(routed, len(prompt_ids), "routed_experts")
                scores = None
                score_reason = "capture_disabled"
                semantics = None
                if args.router_scores:
                    payload = getattr(completion, "routed_expert_scores", None)
                    if payload is None:
                        score_reason = "vllm_output_missing_routed_expert_scores"
                    else:
                        scores = _prompt_array(
                            payload, len(prompt_ids), "routed_expert_scores"
                        ).astype(np.float32)
                        if scores.shape != selected.shape:
                            raise RuntimeError(
                                "router scores and expert IDs are misaligned"
                            )
                        semantics = _semantics(capability)
                        score_reason = None
                aligned_segments = align_prompt_segments(
                    prompt=workload.prompt,
                    segments=workload.prompt_segments,
                    tokenizer=tokenizer,
                    prompt_token_ids=prompt_ids,
                )
                trace = trace_from_vllm(
                    request_id=workload.request_id,
                    metadata=workload.meta,
                    model_id=args.model,
                    prompt=workload.prompt,
                    prompt_token_ids=prompt_ids,
                    generated_token_ids=completion.token_ids or (),
                    prompt_segments=aligned_segments,
                    selected_experts=selected,
                    router_scores=scores,
                    moe_layer_ids=capability.moe_layer_ids,
                    router_top_k=capability.router_top_k,
                    num_experts=capability.num_experts,
                    router_score_semantics=semantics,
                    router_scores_unavailable_reason=score_reason,
                    source_dataset=workload.source_dataset,
                    source_group_id=workload.source_group_id,
                    source_index=workload.source_index,
                    claim_scope=workload.claim_scope,
                    timestamp=workload.timestamp,
                )
                stream.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
                written += 1
            stream.flush()
            print(f"collected {written}/{len(records)} new traces", flush=True)
    return written


def main() -> None:
    args = parse_args()
    capability = model_capability(load_model_config(args.model))
    validate_capture_profile(
        capability=capability,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel=args.expert_parallel == "on",
        pipeline_parallel_size=args.pipeline_parallel_size,
        context_parallel_size=args.context_parallel_size,
        kv_transfer_enabled=args.kv_transfer_enabled,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    written = collect(args, capability)
    report = {
        "model": args.model,
        "capability": capability.to_dict(),
        "workload": args.workload,
        "output": args.output,
        "new_trace_records": written,
        "router_scores_requested": args.router_scores,
        "profile": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel": args.expert_parallel,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "context_parallel_size": args.context_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "batch_size": args.batch_size,
        },
    }
    path = Path(args.env_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote collection report to {path}")


if __name__ == "__main__":
    main()
