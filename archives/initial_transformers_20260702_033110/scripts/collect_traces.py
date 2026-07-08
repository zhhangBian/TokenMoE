#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from tokenmoe.schema import WorkloadRecord, read_workload_jsonl
from tokenmoe.trace import (
    TraceRecord,
    trace_from_selected_experts,
    write_trace_jsonl,
    write_trace_parquet,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TokenMoE routed expert traces.")
    parser.add_argument("--workload", default="data/workloads/agent_workloads.jsonl")
    parser.add_argument("--output", default="data/traces/tokenmoe_traces.jsonl")
    parser.add_argument("--parquet-output", default="data/traces/tokenmoe_traces.parquet")
    parser.add_argument(
        "--backend",
        choices=["vllm", "transformers", "deterministic"],
        default="transformers",
    )
    parser.add_argument("--model", default="TitanML/tiny-mixtral")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--env-report", default="analysis/trace_collection_env.json")
    return parser.parse_args()


def _collect_transformers(
    records: list[WorkloadRecord],
    *,
    model_id: str,
    device: str,
    max_length: int,
) -> list[TraceRecord]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    model.eval().to(device)

    traces: list[TraceRecord] = []
    with torch.inference_mode():
        for idx, record in enumerate(records, start=1):
            encoded = tokenizer(
                record.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)
            output = model(**encoded, output_router_logits=True, return_dict=True)
            router_logits = getattr(output, "router_logits", None)
            if not router_logits:
                raise RuntimeError(f"model {model_id} did not return router_logits")
            selected_by_layer = []
            scores_by_layer = []
            for logits in router_logits:
                probs = torch.softmax(logits.float(), dim=-1)
                top_k = getattr(model.config, "num_experts_per_tok", 2)
                scores, expert_ids = torch.topk(probs, k=top_k, dim=-1)
                selected_by_layer.append(expert_ids.detach().cpu().numpy())
                scores_by_layer.append(scores.detach().cpu().numpy())
            # Transformers router_logits are [tokens, experts] per MoE layer.
            selected = np.stack(selected_by_layer, axis=1)
            scores = np.stack(scores_by_layer, axis=1)
            token_ids = encoded["input_ids"][0].detach().cpu().tolist()
            traces.append(
                trace_from_selected_experts(
                    request_id=record.request_id,
                    metadata=record.meta,
                    model_id=model_id,
                    backend="transformers-router-logits",
                    prompt=record.prompt,
                    selected_experts=selected,
                    router_scores=scores,
                    token_ids=token_ids,
                    output_token_count=0,
                )
            )
            if idx % 10 == 0:
                print(f"collected {idx}/{len(records)} traces")
    return traces


def _collect_vllm(
    records: list[WorkloadRecord],
    *,
    model_id: str,
    max_tokens: int,
) -> list[TraceRecord]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_id,
        enforce_eager=True,
        enable_return_routed_experts=True,
        hf_overrides={"sliding_window": None},
    )
    traces: list[TraceRecord] = []
    for idx, record in enumerate(records, start=1):
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            routed_experts_prompt_start=0,
        )
        outputs = llm.generate([record.prompt], params, use_tqdm=False)
        if not outputs:
            raise RuntimeError(f"vLLM returned no output for {record.request_id}")
        completion = outputs[0].outputs[0]
        routed_experts = getattr(completion, "routed_experts", None)
        if routed_experts is None:
            raise RuntimeError(
                "vLLM output did not include routed_experts; verify "
                "enable_return_routed_experts and MoE model support"
            )
        token_ids = list(outputs[0].prompt_token_ids or [])
        generated = list(completion.token_ids or [])
        traces.append(
            trace_from_selected_experts(
                request_id=record.request_id,
                metadata=record.meta,
                model_id=model_id,
                backend="vllm-routed-experts",
                prompt=record.prompt,
                selected_experts=np.asarray(routed_experts),
                router_scores=None,
                token_ids=token_ids,
                output_token_count=len(generated),
            )
        )
        if idx % 10 == 0:
            print(f"collected {idx}/{len(records)} vLLM traces")
    return traces


def _stable_hash(text: str) -> int:
    value = 2166136261
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _collect_deterministic(
    records: list[WorkloadRecord],
    *,
    model_id: str,
    max_length: int,
) -> list[TraceRecord]:
    traces: list[TraceRecord] = []
    role_bias = {
        "planner": 0,
        "coder": 2,
        "tester": 4,
        "critic": 6,
        "searcher": 1,
        "summarizer": 3,
        "tool_caller": 5,
        "assistant": 7,
        "moderator": 0,
        "analyst": 2,
        "merger": 4,
        "debugger": 6,
    }
    num_layers = 2
    num_experts = 8
    top_k = 2
    for record in records:
        tokens = record.prompt.split()[:max(4, min(max_length, 48))]
        selected = np.zeros((len(tokens), num_layers, top_k), dtype=np.int64)
        scores = np.zeros((len(tokens), num_layers, top_k), dtype=np.float32)
        base = role_bias.get(record.meta.role, 0)
        phase = _stable_hash(record.meta.phase) % num_experts
        block = _stable_hash(record.meta.block_type_key) % num_experts
        for t, token in enumerate(tokens):
            token_hash = _stable_hash(token)
            for layer in range(num_layers):
                first = (base + layer + (token_hash % 3)) % num_experts
                second = (phase + block + layer + (token_hash % 5)) % num_experts
                if second == first:
                    second = (second + 1) % num_experts
                selected[t, layer, :] = [first, second]
                scores[t, layer, :] = [0.62, 0.38]
        traces.append(
            trace_from_selected_experts(
                request_id=record.request_id,
                metadata=record.meta,
                model_id=model_id,
                backend="deterministic-trace-only",
                prompt=record.prompt,
                selected_experts=selected,
                router_scores=scores,
                token_ids=list(range(len(tokens))),
            )
        )
    return traces


def main() -> None:
    args = parse_args()
    workload_records = read_workload_jsonl(args.workload)[: args.limit]
    Path(args.env_report).parent.mkdir(parents=True, exist_ok=True)
    env_report: dict[str, object] = {
        "requested_backend": args.backend,
        "model": args.model,
        "limit": len(workload_records),
        "fallback_used": False,
        "error": None,
    }
    try:
        if args.backend == "vllm":
            traces = _collect_vllm(
                workload_records,
                model_id=args.model,
                max_tokens=8,
            )
        elif args.backend == "transformers":
            traces = _collect_transformers(
                workload_records,
                model_id=args.model,
                device=args.device,
                max_length=args.max_length,
            )
        else:
            traces = _collect_deterministic(
                workload_records,
                model_id=args.model,
                max_length=args.max_length,
            )
    except Exception as exc:
        if not args.allow_fallback:
            raise
        env_report["fallback_used"] = True
        env_report["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        print(f"primary trace backend failed, using deterministic fallback: {env_report['error']}")
        traces = _collect_deterministic(
            workload_records,
            model_id=args.model,
            max_length=args.max_length,
        )

    write_trace_jsonl(traces, args.output)
    write_trace_parquet(traces, args.parquet_output)
    env_report["trace_records"] = len(traces)
    env_report["actual_backend"] = traces[0].backend if traces else "none"
    Path(args.env_report).write_text(json.dumps(env_report, indent=2), encoding="utf-8")
    print(f"wrote {len(traces)} trace records to {args.output}")
    print(f"wrote parquet trace rows to {args.parquet_output}")
    print(f"wrote environment report to {args.env_report}")


if __name__ == "__main__":
    main()
