"""Frozen local protocol; an experiment plan is never execution evidence."""

from pathlib import Path

from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.jsonl import read_jsonl

from .artifacts import cache_execution, manifest


def training_sampling(inputs):
    """Recover sampling from the hashed template records used to fit priors."""
    selected = []
    for root, hashes in inputs["sources"].items():
        relative = "static/role_templates.jsonl"
        path = Path(root) / relative
        if sha256_file(path) != hashes[relative]:
            raise ValueError("Training template records changed after analysis")
        for role in read_jsonl(path):
            if (
                role["agent_template_id"] == inputs["identity"]["template_id"]
                and role["version_hash"] == inputs["identity"]["template_version"]
            ):
                selected.append(role["sampling_params"])
    if not selected or any(params != selected[0] for params in selected):
        raise ValueError("Training templates have missing or inconsistent sampling")
    if selected[0].get("max_tokens") != 4096:
        raise ValueError("Training template differs from the selected 4096-token budget")
    return selected[0]


def local_protocol(identity, tasks, confirmation_tasks, *, sampling, prefill_execution="tokens"):
    if not tasks or not confirmation_tasks:
        raise ValueError("Exploration and untouched confirmation task groups are required")
    if set(tasks) & set(confirmation_tasks):
        raise ValueError("Confirmation tasks overlap exploration groups")
    execution = cache_execution({"prefill_execution": prefill_execution})
    result = manifest(
        "experiment",
        identity,
        protocol_id="qwen-residency-local-v4",
        supersedes=[
            "qwen-residency-local-v1 (unexecuted 64-token diagnostic budget)",
            "qwen-residency-local-v2 (unexecuted; confirmation groups not selected)",
            "qwen-residency-local-v3 (unexecuted; greedy sampling mismatched fitted template)",
        ],
        status="preregistered-unexecuted",
        devices=[2, 3],
        tensor_parallel=[1, 2],
        primary_tensor_parallel=2,
        concurrency=[1, 4],
        expert_slots=[16, 32, 64, 96, 128],
        exploration_tasks=sorted(set(tasks)),
        untouched_confirmation_tasks=sorted(set(confirmation_tasks)),
        confirmation_status="frozen-unexecuted; excluded from fitting and policy selection",
        warmup={"requests": 2, "excluded": True},
        measured_repetitions=3,
        policy_order="rotate left by repetition; paired task and sampling seed",
        policies=[
            "full",
            "uva",
            "layer-prefetch",
            "lru",
            "lfu",
            "global",
            "stage",
            "previous",
            "ewma",
            "previous+global",
            "previous+stage",
            "previous+tool",
        ],
        comparison_arms={"layer-ahead": "unqualified", "semantic": "unqualified"},
        execution={
            "dtype": "bfloat16",
            "graph": "eager",
            "native_graph_reference": "required",
            "scheduler": "synchronous-fcfs",
            "ep": 1,
            "dp": 1,
            "speculation": False,
        },
        mechanism={
            "kv_cache_memory_bytes_per_rank": 8 * 1024**3,
            "max_model_len": 32768,
            "max_num_batched_tokens": 4096,
            "prefix_caching": True,
            "max_num_seqs": 4,
        },
        service_capacity={
            "total_gpu_limit_bytes_per_rank": 64 * 1024**3,
            "kv": "reclaimed-expert-memory; separate study",
        },
        latency_targets={
            "engine_ttft_p95_seconds": 1.0,
            "tools_to_complete_response_p95_seconds": 30.0,
            "tools_boundary": "latest required tool completion before request creation",
            "response_boundary": "batched client response completely received",
            "task_p95_seconds": 900.0,
        },
        correctness={
            "forced_tensor_atol": 0.02,
            "forced_tensor_rtol": 0.02,
            "router_mismatches": 0,
            "greedy_token_mismatches": 0,
            "repeat_baseline": "required",
            "change_thresholds_after_run": False,
        },
        outcomes="retain all failures and truncations; successful throughput needs task success evidence",
        sampling={**sampling, "seed": 42},
        sampling_source="hashed training role template; per-attempt seeds derive from base seed 42",
        diagnostic_sampling={
            "max_tokens": [8, 64],
            "eligibility": "backend checks only; excluded from agent service comparisons",
        },
        kv_transfer_arbitration="unavailable; separate contention experiment required",
        transfer_contention={
            "status": "preregistered-unexecuted; connector compatibility must pass first",
            "arbitration": "unavailable; independent shared PCIe transfers",
            "arms": ["native", "expert-only", "kv-only", "expert+kv"],
            "kv_offloading_size_gib": 4,
            "kv_offloading_backend": "native",
            "kv_gpu_bytes_per_rank": 8 * 1024**3,
            "expert_slots": 32,
            "concurrency": [1, 4],
            "repetitions": 3,
            "workload": "paired agent tasks with reused prefixes; retain actual KV hits/preemptions",
            "comparison": "interaction in bytes, exposed wait, TTFT and task latency",
            "kv_eviction_policy": "existing connector default; unchanged",
        },
        evidence_levels=["lookup", "frozen-replay", "calibrated-simulation", "live"],
        milestones={
            key: "outstanding"
            for key in ("backend", "local_evaluation", "long_tools", "multi_role", "migration")
        },
    )
    if execution["prefill_execution"] == "expert-major":
        result.update(
            protocol_id="qwen-residency-local-v5-expert-major-mechanism",
            parent_protocol="qwen-residency-local-v4; token-executor agent comparisons remain separate",
            policies=["full", "uva", "layer-prefetch", "lru", "lfu"],
            predictive_policy_status="unqualified: the token-microbatch avoided-stall model cannot be reused for expert-major transfers",
        )
        result["execution"].update(execution)
        result["mechanism"]["cache_scratch_bytes_per_rank"] = 192 * 1024**2
    return result
