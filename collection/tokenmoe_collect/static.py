"""Immutable static records, assembled from producer metadata and local inputs."""

import hashlib
import json
import os
import platform
import re
import struct
from pathlib import Path

import psutil
import yaml

from .clock import clock_metadata
from .ids import content_id
from .jsonl import JsonlWriter, write_json


def load_yaml(path):
    path = Path(path).resolve()
    data = yaml.safe_load(os.path.expandvars(path.read_text()))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return data


def load_benchmark(path, instances):
    path = Path(path)
    if path.suffix == ".jsonl":
        from .jsonl import read_jsonl

        items = list(read_jsonl(path))
    elif path.suffix == ".json":
        items = json.loads(path.read_text())
    else:
        import pyarrow.parquet as pq

        files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No benchmark parquet found under {path}")
        items = [row for file in files for row in pq.read_table(file).to_pylist()]
    by_id = {row["instance_id"]: row for row in items}
    missing = sorted(set(instances) - by_id.keys())
    if missing:
        raise ValueError(f"Unknown benchmark instance IDs: {missing}")
    if len(set(instances)) != len(instances):
        raise ValueError("Duplicate benchmark instance IDs in run config")
    return [by_id[i] for i in instances]


def minisweagent_config():
    import minisweagent

    path = Path(minisweagent.__file__).parent / "config/benchmarks/swebench.yaml"
    config = yaml.safe_load(path.read_text())
    return config


def stored_expert_bytes(model_path, layers, experts):
    """Read safetensors headers only; include packed weights, scales and biases."""
    sizes = {layer: 0 for layer in layers}
    for path in sorted(Path(model_path).glob("*.safetensors")):
        with path.open("rb") as source:
            length = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(length))
        for name, entry in header.items():
            match = re.search(r"(?:^|\.)layers\.(\d+)\..*\.experts\.", name)
            if match and int(match.group(1)) in sizes:
                a, b = entry["data_offsets"]
                sizes[int(match.group(1))] += b - a
    return (
        {str(layer): total / experts for layer, total in sizes.items()}
        if all(sizes.values())
        else None
    )


def write_static(run_dir, run_config, model_config, engine_dir, items, mini_config):
    from .minisweagent_adapter import BASH_TOOL

    root = Path(run_dir)
    static = root / "static"
    static.mkdir(parents=True)
    engine_dir = Path(engine_dir)
    meta = json.loads((engine_dir / "engine_meta.json").read_text())
    layer_map = json.loads((engine_dir / "layer_map.json").read_text())
    model_path = Path(model_config["model_path"])
    hf = (
        json.loads((model_path / "config.json").read_text())
        if (model_path / "config.json").exists()
        else model_config.get("facts", {})
    )
    facts = hf | model_config.get("facts", {})
    tokenizer_path = Path(model_config.get("tokenizer_path", model_path / "tokenizer.json"))
    tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    layers = [entry["layer_id"] for entry in layer_map]
    experts = facts.get(
        "num_routed_experts", facts.get("num_experts", facts.get("n_routed_experts"))
    )
    top_k = facts.get("top_k", facts.get("num_experts_per_tok"))
    if not layers or experts is None or top_k is None:
        raise ValueError("Missing captured layers or model expert facts")
    expert_bytes = stored_expert_bytes(model_path, layers, experts)
    if expert_bytes is None:
        expert_bytes = facts.get("expert_bytes_per_layer")
    revision_file = model_path / ".cache/huggingface/download/config.json.metadata"
    model_revision = model_config.get("revision")
    if model_revision is None and revision_file.exists():
        model_revision = revision_file.read_text().splitlines()[0]
    profile = {
        "model_id": model_config.get("model_id", model_config["name"]),
        "revision_hash": model_revision or content_id(hf),
        "architecture": facts.get("architectures", [facts.get("architecture")])[0],
        "num_layers": facts.get("num_hidden_layers", facts.get("num_layers")),
        "moe_layer_ids": layers,
        "num_routed_experts": experts,
        "top_k": top_k,
        "num_shared_experts": facts.get("n_shared_experts", facts.get("num_shared_experts", 0))
        or 0,
        "expert_bytes_per_layer": expert_bytes,
        "tokenizer_hash": tokenizer_hash,
        "tokenizer_path": str(tokenizer_path.resolve()),
        "mtp_disabled": True,
        "multimodal_disabled": True,
        "layer_latency_profile_ref": None,
    }
    profile["model_profile_id"] = content_id(profile)
    template_identity = {"harness": "mini-swe-agent", "role_type": "generalist"}
    template_id = content_id(template_identity)
    template_content = {
        "agent": mini_config["agent"],
        "model": mini_config["model"],
        "tools": [BASH_TOOL],
        "sampling": model_config["sampling"],
    }
    version = content_id(template_content)
    template_dir = static / "role_templates" / template_id / version
    template_dir.mkdir(parents=True)
    (template_dir / "system_prompt.txt").write_text(
        mini_config["agent"]["system_template"], encoding="utf-8"
    )
    (template_dir / "instance_template.txt").write_text(
        mini_config["agent"]["instance_template"], encoding="utf-8"
    )
    write_json(template_dir / "tools.json", [BASH_TOOL])
    write_json(template_dir / "config.json", template_content)
    role = {
        "agent_template_id": template_id,
        "version_hash": version,
        **template_identity,
        "harness_version": "2.4.6",
        "system_prompt_ref": str((template_dir / "system_prompt.txt").relative_to(root)),
        "tool_schema_ref": str((template_dir / "tools.json").relative_to(root)),
        "sampling_params": model_config["sampling"],
        "description_ref": str((template_dir / "system_prompt.txt").relative_to(root)),
    }
    clock = clock_metadata()
    host = {
        "host_id": clock["host_id"],
        "cpu_model": platform.processor(),
        "dram_gb": psutil.virtual_memory().total / 1e9,
        "gpu_model": None,
        "gpu_count": None,
        "link_type": None,
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            host["gpu_count"] = pynvml.nvmlDeviceGetCount()
            host["gpu_model"] = [
                pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
                for i in range(host["gpu_count"])
            ]
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    experiment = {
        "hosts": [host],
        "engine_config_id": meta["engine_config_id"],
        "model_profile_id": profile["model_profile_id"],
        "engine": meta["engine_config"],
        "harness": {
            "name": "mini-swe-agent",
            "version": "2.4.6",
            "sandbox_type": run_config["runtime"],
            "cpu_quota": run_config.get("cpu_quota"),
            "memory_limit": run_config.get("memory_limit"),
        },
        "concurrency": {
            "num_concurrent_sessions": run_config["num_concurrent_sessions"],
            "arrival_pattern": "closed_loop",
        },
        "seed": run_config["base_seed"],
        "instances": run_config["instances"],
        "limits": {
            "step_limit": run_config["step_limit"],
            "time_limit_seconds": run_config["time_limit_seconds"],
        },
        "agent_template_id": template_id,
        "version_hash": version,
    }
    experiment["experiment_config_id"] = content_id(experiment)
    dataset_path = Path(run_config["dataset_path"])
    dataset_files = (
        [dataset_path] if dataset_path.is_file() else sorted(dataset_path.rglob("*.parquet"))
    )
    dataset_hash = hashlib.sha256()
    for file in dataset_files:
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                dataset_hash.update(chunk)
    benchmark_version = run_config.get("dataset_revision") or dataset_hash.hexdigest()[:16]
    benchmarks = [
        {
            "benchmark_id": run_config.get("benchmark_id", "princeton-nlp/SWE-bench_Verified"),
            "benchmark_version": benchmark_version,
            "benchmark_item_id": item["instance_id"],
            "task_type": item["repo"],
            "difficulty": item.get("difficulty"),
        }
        for item in items
    ]
    for name, records in [
        ("experiment_configs", [experiment]),
        ("model_profiles", [profile]),
        ("role_templates", [role]),
        ("benchmark_items", benchmarks),
        ("application_definitions", []),
    ]:
        with JsonlWriter(static / f"{name}.jsonl") as writer:
            for record in records:
                writer.write(record)
    write_json(static / "engine_meta.json", meta)
    write_json(static / "layer_map.json", layer_map)
    return {"experiment": experiment, "profile": profile, "role": role, "engine": meta}
