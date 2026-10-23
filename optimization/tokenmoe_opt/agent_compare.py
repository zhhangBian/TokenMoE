"""Descriptive pilot tables with explicit missing comparison prerequisites."""

import json
from collections import defaultdict
from pathlib import Path

from tokenmoe_collect.assets import sha256_file

from .artifacts import Output, cache_execution, digest, read_manifest


def preconditions(rows, repetitions):
    groups = defaultdict(list)
    for row in rows:
        groups[row["policy"], row["slots"], row["concurrency"], tuple(row["tasks"])].append(
            row["repetition"]
        )
    return {
        "same_model_template": len({r["identity"] for r in rows}) == 1,
        "same_sampling_kv_scheduler_graph": len({r["configuration"] for r in rows}) == 1,
        "same_runtime_sources": len({r["runtime_sources"] for r in rows}) == 1,
        "same_cache_execution": len(
            {r.get("cache_execution") for r in rows if r.get("cache_execution") is not None}
        )
        <= 1,
        "fork_files_hashed": all(r["fork_files_hashed"] for r in rows),
        "same_devices": len({tuple(r["devices"]) for r in rows}) == 1,
        "same_task_and_concurrency_groups": len(
            {(tuple(r["tasks"]), r["concurrency"]) for r in rows}
        )
        == 1,
        "same_protocol": len({r["protocol"] for r in rows}) == 1,
        "complete_repetitions": all(
            sorted(values) == list(range(repetitions)) for values in groups.values()
        ),
        "complete_side_channels": all(r["side_channel_complete"] for r in rows),
        "stable_sources": all(r["source_stable"] for r in rows),
        "native_reference_present": any(r["policy"] == "full" for r in rows),
        "rotated_order_matrix_verified": False,
    }


def summarize(trials, output, *, command):
    rows, sources = [], {}
    for run, metrics in trials:
        run, metrics = Path(run).resolve(), Path(metrics).resolve()
        if any(Path(output).resolve().is_relative_to(path) for path in (run, metrics)):
            raise ValueError("Comparison output must be outside immutable trial and metric inputs")
        experiment = read_manifest(run / "manifest.json", "experiment")
        result = read_manifest(metrics / "manifest.json", "result")
        if (
            result.get("status") != "complete"
            or result.get("evidence") != "observed-agent-trial"
            or experiment["identity"] != result["identity"]
        ):
            raise ValueError("Pilot summary requires completed metrics with matching identity")
        hashes = json.loads((metrics / "source-hashes.json").read_text())
        values = {}
        for name in ("manifest.json", "model.json", "protocol.json", "cell.json"):
            path = run / name
            if hashes.get(name) != sha256_file(path):
                raise ValueError(f"Pilot metadata changed after metric extraction: {path}")
            values[name] = json.loads(path.read_text())
            sources[str(path)] = hashes[name]
        for name in ("manifest.json", "metrics.json", "source-hashes.json"):
            sources[str(metrics / name)] = sha256_file(metrics / name)
        measured = json.loads((metrics / "metrics.json").read_text())
        cell, model, protocol = (values[n] for n in ("cell.json", "model.json", "protocol.json"))
        serve = model["serve"]
        configuration = {
            key: serve.get(key)
            for key in (
                "tensor_parallel_size",
                "dtype",
                "max_model_len",
                "max_num_seqs",
                "max_num_batched_tokens",
                "enforce_eager",
                "async_scheduling",
                "kv_cache_memory_bytes",
                "kernel_config",
                "seed",
                "enable_prefix_caching",
            )
        }
        configuration["sampling"] = protocol["sampling"]
        configuration["warmup"] = protocol["warmup"]
        files = experiment["source"]["files"]
        cache = serve.get("expert_cache_config", {})
        rows.append(
            {
                **cell,
                "tasks": sorted(cell["tasks"]),
                "run": str(run),
                "metrics": str(metrics),
                "identity": digest(experiment["identity"]),
                "configuration": digest(configuration),
                "protocol": digest(protocol),
                "runtime_sources": digest(
                    {"files": files, "fork": experiment["source"]["fork_commit"]}
                ),
                "fork_files_hashed": any(p.startswith("vllm/vllm/") for p in files),
                "cache_execution": digest(cache_execution(cache)) if cache.get("enabled") else None,
                "devices": sorted(d["uuid"] for d in measured["memory"]["metadata"]["devices"]),
                "side_channel_complete": measured["side_channel_export_complete"] is True,
                "source_stable": experiment.get("source_stable", False),
                "metrics_observed": measured["metrics"],
                "memory": measured["memory"],
                "outcomes": measured["outcomes"],
                "benchmark_solved_tasks_per_second": measured["benchmark_solved_tasks_per_second"],
            }
        )
    if not rows or len({row["run"] for row in rows}) != len(rows):
        raise ValueError("Pilot summary requires unique executions")
    checks = preconditions(rows, protocol["measured_repetitions"])
    out = Output(output, kind="result", identity=result["identity"], command=command)
    out.write(
        "comparison.json",
        {
            "evidence": "descriptive-agent-pilot-summary",
            "controlled_comparison_qualified": False,
            "preconditions": checks,
            "unmet_preconditions": [key for key, passed in checks.items() if not passed],
            "rows": rows,
            "sources": sources,
            "limitations": [
                "Raw cell observations only; no paired policy speed ratio or agent-feature attribution",
                "Rotation and full matrix coverage require a separate verified matrix manifest",
                "Metric source hashes describe the extraction boundary; raw streams are not re-extracted",
                "Submitted termination is not a benchmark solve",
            ],
        },
    )
    out.finish(evidence="descriptive-agent-pilot-summary", controlled_comparison_qualified=False)
    return out.path
