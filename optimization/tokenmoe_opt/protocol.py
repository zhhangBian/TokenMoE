"""Frozen local protocol; an experiment plan is never execution evidence."""

from .artifacts import manifest


def local_protocol(identity, tasks):
    if not tasks:
        raise ValueError("Task groups are required before freezing the protocol")
    return manifest(
        "experiment",
        identity,
        protocol_id="qwen-residency-local-v2",
        supersedes="qwen-residency-local-v1 (unexecuted 64-token diagnostic budget)",
        status="preregistered-unexecuted",
        devices=[2, 3],
        tensor_parallel=[1, 2],
        primary_tensor_parallel=2,
        concurrency=[1, 4],
        expert_slots=[16, 32, 64, 96, 128],
        exploration_tasks=sorted(set(tasks)),
        untouched_confirmation_tasks=[],
        confirmation_status="outstanding: select untouched task groups before confirmation",
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
        sampling={"temperature": 0, "seed": 42, "max_tokens": 4096},
        diagnostic_sampling={
            "max_tokens": [8, 64],
            "eligibility": "backend checks only; excluded from agent service comparisons",
        },
        kv_transfer_arbitration="unavailable; separate contention experiment required",
        evidence_levels=["lookup", "frozen-replay", "calibrated-simulation", "live"],
        milestones={
            key: "outstanding"
            for key in ("backend", "local_evaluation", "long_tools", "multi_role", "migration")
        },
    )
