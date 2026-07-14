#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np

from tokenmoe.schema import WorkloadRecord, read_workload_jsonl
from tokenmoe.trace import (
    CURRENT_TRACE_BACKEND,
    TRACE_SCHEMA_V2,
    TRACE_SCHEMA_V3,
    align_prompt_segments_with_tokenizer,
    decode_vllm_routed_experts_b64,
    trace_from_selected_experts,
    validate_current_stage_traces,
    write_trace_jsonl,
    write_trace_parquet,
)
from tokenmoe.vllm_sidecar import (
    infer_model_capability,
    load_hf_config,
    validate_vllm_profile,
)


DEFAULT_ARTIFACT_ROOT = "/home/youwei/bzh/dataset/tokenmoe_artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect TokenMoE prompt-only routed expert traces with vLLM."
    )
    parser.add_argument(
        "--workload",
        required=True,
        help="Normalized v2 workload JSONL produced by dataset_adapters.",
    )
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_ARTIFACT_ROOT}/traces/tokenmoe_traces.jsonl",
    )
    parser.add_argument("--parquet-output", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument(
        "--expert-parallel",
        choices=["on", "off"],
        required=True,
        help="Explicit expert parallel setting for the vLLM profile.",
    )
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--kv-transfer-enabled", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--router-scores",
        choices=["on", "off"],
        default="off",
        help=(
            "Capture aligned router top-k scores via the modified vLLM "
            "fork; 'on' writes tokenmoe.trace.v3 records, 'off' keeps "
            "ID-only tokenmoe.trace.v2 records."
        ),
    )
    parser.add_argument("--env-report", default=f"{DEFAULT_ARTIFACT_ROOT}/logs/trace_collection_env.json")
    return parser.parse_args()


def _llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    from vllm import LLM

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
    signature = inspect.signature(LLM)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs and "enable_return_routed_experts" not in signature.parameters:
        raise RuntimeError("installed vLLM does not expose enable_return_routed_experts")
    if args.router_scores == "on":
        if (
            not accepts_kwargs
            and "enable_return_routed_expert_scores" not in signature.parameters
        ):
            raise RuntimeError(
                "installed vLLM does not expose enable_return_routed_expert_scores; "
                "rerun with --router-scores off for an ID-only v2 collection"
            )
        kwargs["enable_return_routed_expert_scores"] = True
    if accepts_kwargs or "enable_expert_parallel" in signature.parameters:
        kwargs["enable_expert_parallel"] = args.expert_parallel == "on"
    elif args.expert_parallel == "on":
        raise RuntimeError("installed vLLM does not expose enable_expert_parallel")
    if accepts_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams

    signature = inspect.signature(SamplingParams)
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_kwargs and "routed_experts_prompt_start" not in signature.parameters:
        raise RuntimeError(
            "installed vLLM SamplingParams does not expose routed_experts_prompt_start"
        )
    return SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        routed_experts_prompt_start=0,
    )


def _tokenizer_from_llm(llm: Any) -> Any:
    if hasattr(llm, "get_tokenizer"):
        return llm.get_tokenizer()
    engine = getattr(llm, "llm_engine", None)
    tokenizer = getattr(engine, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    raise RuntimeError("could not access vLLM tokenizer for prompt span alignment")


def _routed_experts_array(payload: Any) -> np.ndarray:
    if isinstance(payload, str):
        return decode_vllm_routed_experts_b64(payload)
    return np.asarray(payload)


def _prompt_only_routing(
    routed_experts: Any,
    *,
    prompt_token_count: int,
) -> tuple[np.ndarray, bool]:
    routed = _routed_experts_array(routed_experts).astype(np.int64)
    if routed.ndim != 3:
        raise ValueError(
            f"routed_experts must have shape [tokens, layers, top_k], got {routed.shape}"
        )
    if routed.shape[0] < prompt_token_count:
        raise ValueError(
            f"routed_experts token dimension {routed.shape[0]} < prompt_token_count {prompt_token_count}"
        )
    decode_excluded = routed.shape[0] > prompt_token_count
    return routed[:prompt_token_count, :, :], decode_excluded


# Per-model-family interpretation of the captured router top-k weights.
# The capture point is BaseRouter.select_experts after _compute_routing,
# i.e. the final combine weights before EPLB mapping.
ROUTER_SCORE_SEMANTICS_BY_MODEL_TYPE = {
    # softmax over all experts -> top-k -> renormalize (norm_topk_prob=True)
    "qwen3_moe": "softmax_topk_renormalized",
    "qwen2_moe": "softmax_topk_renormalized",
    # softmax -> (group-limited) top-k; V2-Lite has norm_topk_prob=False and
    # routed_scaling_factor=1.0, so weights are raw softmax probabilities of
    # the selected experts (do not necessarily sum to 1 per token).
    "deepseek_v2": "softmax_topk_scaled_unnormalized",
    "deepseek_v3": "sigmoid_topk_scaled",
}


def _router_score_semantics(capability: Any) -> str:
    semantics = ROUTER_SCORE_SEMANTICS_BY_MODEL_TYPE.get(capability.model_type)
    if semantics is None:
        raise RuntimeError(
            "router score semantics are not documented for model_type "
            f"{capability.model_type!r}; add it to "
            "ROUTER_SCORE_SEMANTICS_BY_MODEL_TYPE or rerun with "
            "--router-scores off"
        )
    return f"{capability.model_type}:{semantics}"


def _collect_vllm(
    records: list[WorkloadRecord],
    *,
    args: argparse.Namespace,
    capability: Any,
) -> list[Any]:
    from vllm import LLM

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    llm = LLM(**_llm_kwargs(args))
    tokenizer = _tokenizer_from_llm(llm)
    params = _sampling_params(args)
    capture_scores = args.router_scores == "on"
    score_semantics = _router_score_semantics(capability) if capture_scores else None
    traces = []
    for batch_start in range(0, len(records), args.batch_size):
        batch = records[batch_start : batch_start + args.batch_size]
        outputs = llm.generate([record.prompt for record in batch], params, use_tqdm=False)
        if len(outputs) != len(batch):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts"
            )
        for offset, (record, request_output) in enumerate(zip(batch, outputs), start=1):
            idx = batch_start + offset
            if not request_output.outputs:
                raise RuntimeError(f"vLLM returned no completion for {record.request_id}")
            completion = request_output.outputs[0]
            routed_experts = getattr(completion, "routed_experts", None)
            if routed_experts is None:
                raise RuntimeError("vLLM output did not include routed_experts")
            prompt_token_ids = list(request_output.prompt_token_ids or [])
            if not prompt_token_ids:
                raise RuntimeError("vLLM RequestOutput did not include prompt_token_ids")
            aligned_segments, segment_reason = align_prompt_segments_with_tokenizer(
                prompt=record.prompt,
                prompt_segments=record.prompt_segments,
                tokenizer=tokenizer,
                prompt_token_ids=[int(item) for item in prompt_token_ids],
            )
            selected, decode_excluded = _prompt_only_routing(
                routed_experts,
                prompt_token_count=len(prompt_token_ids),
            )
            router_scores = None
            scores_unavailable_reason = None
            if capture_scores:
                raw_scores = getattr(completion, "routed_expert_scores", None)
                if raw_scores is None:
                    # Fail-closed: keep IDs, mark scores unavailable.
                    scores_unavailable_reason = (
                        "vllm_output_missing_routed_expert_scores"
                    )
                else:
                    scores = np.asarray(raw_scores, dtype=np.float32)
                    full_shape = _routed_experts_array(routed_experts).shape
                    if scores.shape != full_shape:
                        raise RuntimeError(
                            f"{record.request_id}: routed_expert_scores shape "
                            f"{scores.shape} misaligned with routed_experts "
                            f"{full_shape}"
                        )
                    router_scores = scores[: len(prompt_token_ids), :, :]
            generated = [int(item) for item in (completion.token_ids or [])]
            trace = trace_from_selected_experts(
                request_id=record.request_id,
                metadata=record.meta,
                model_id=args.model,
                backend=CURRENT_TRACE_BACKEND,
                prompt=record.prompt,
                selected_experts=selected,
                router_scores=router_scores,
                schema_version=TRACE_SCHEMA_V3 if capture_scores else TRACE_SCHEMA_V2,
                router_score_semantics=score_semantics
                if router_scores is not None
                else None,
                router_scores_unavailable_reason=scores_unavailable_reason,
                token_ids=[int(item) for item in prompt_token_ids],
                generated_token_ids=generated,
                output_token_count=len(generated),
                prompt_segments=aligned_segments,
                moe_layer_ids=capability.moe_layer_ids or [],
                routing_scope="prompt_only",
                prompt_routing_start=0,
                decode_routing_excluded=decode_excluded,
                backend_fallback_used=False,
                segment_unavailable_reason=segment_reason,
                router_top_k=capability.router_top_k,
                num_experts=capability.num_experts,
                source_dataset=record.source_dataset,
                source_group_id=record.source_group_id,
                source_index=record.source_index,
                timestamp=record.timestamp,
                claim_scope=record.claim_scope,
            )
            traces.append(trace)
            if idx % 10 == 0 or idx == len(records):
                print(f"collected {idx}/{len(records)} vLLM prompt traces")
    validate_current_stage_traces(traces)
    return traces


def main() -> None:
    args = parse_args()
    workload_records = read_workload_jsonl(args.workload)[: args.limit]
    config = load_hf_config(args.model)
    capability = infer_model_capability(config)
    validate_vllm_profile(
        capability=capability,
        enable_return_routed_experts=True,
        pipeline_parallel_size=args.pipeline_parallel_size,
        context_parallel_size=args.context_parallel_size,
        kv_transfer_enabled=args.kv_transfer_enabled,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_expert_parallel=args.expert_parallel == "on",
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    env_report: dict[str, Any] = {
        "backend": CURRENT_TRACE_BACKEND,
        "model": args.model,
        "limit": len(workload_records),
        "capability": capability.to_dict(),
        "profile": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "expert_parallel": args.expert_parallel,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "context_parallel_size": args.context_parallel_size,
            "dtype": args.dtype,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "kv_transfer_enabled": args.kv_transfer_enabled,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
            "router_scores": args.router_scores,
        },
    }
    if args.router_scores == "on":
        env_report["router_score_semantics"] = _router_score_semantics(capability)
    traces = _collect_vllm(workload_records, args=args, capability=capability)
    write_trace_jsonl(traces, args.output)
    if args.parquet_output:
        write_trace_parquet(traces, args.parquet_output)
    env_report["trace_records"] = len(traces)
    env_report["schema_version"] = traces[0].schema_version if traces else None
    Path(args.env_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.env_report).write_text(json.dumps(env_report, indent=2), encoding="utf-8")
    print(f"wrote {len(traces)} prompt trace records to {args.output}")
    print(f"wrote environment report to {args.env_report}")


if __name__ == "__main__":
    main()
