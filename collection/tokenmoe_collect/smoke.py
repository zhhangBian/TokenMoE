"""One real request to verify the bound layers, routing and engine step records."""

import json
import time
from pathlib import Path

import httpx
import numpy as np

from .ids import new_id
from .jsonl import read_jsonl, write_json
from .static import load_yaml


def smoke(model_config, trace_dir, *, url="http://127.0.0.1:8000", timeout=1200):
    config = load_yaml(model_config)
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=120) as client:
        while time.monotonic() < deadline:
            try:
                if client.get(url + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise TimeoutError("Server did not become ready")
        engines = sorted(
            Path(trace_dir).glob("*/engine_meta.json"), key=lambda p: p.stat().st_mtime_ns
        )
        if not engines:
            raise FileNotFoundError("No engine metadata under trace directory")
        engine = engines[-1].parent
        layers = json.loads((engine / "layer_map.json").read_text())
        expected = config.get("facts", {}).get("moe_layers")
        if expected is None:
            expected = json.loads((Path(config["model_path"]) / "config.json").read_text())[
                "num_hidden_layers"
            ]
        if len(layers) != expected:
            raise ValueError(f"Expected {expected} bound layers, got {len(layers)}")
        rid = new_id("req")
        response = client.post(
            url + "/v1/chat/completions",
            json={
                "model": config["name"],
                "messages": [
                    {"role": "user", "content": "Reply with the number 42 only. /no_think"}
                ],
                "max_tokens": 32,
                "temperature": 0,
                "vllm_xargs": {"tokenmoe_llm_request_id": rid},
            },
        )
        response.raise_for_status()
        body = response.json()
        if body["choices"][0].get("routed_experts") is not None:
            raise ValueError("Routing unexpectedly returned in API payload")
    path = engine / f"routing/{rid}.npz"
    for _ in range(100):
        if path.exists():
            break
        time.sleep(0.1)
    with np.load(path, allow_pickle=False) as data:
        experts = data["experts"]
        facts = config["facts"]
        valid = experts.shape[1:] == (expected, facts["top_k"])
        valid &= bool(np.all(experts < facts["num_routed_experts"]))
        valid &= bool(np.all(np.diff(np.sort(experts, axis=-1), axis=-1) > 0))
        if not valid:
            raise ValueError("Invalid captured expert rows")
        rows = len(experts)
    steps = [
        step
        for step in read_jsonl(engine / "steps.jsonl")
        if any(entry[0] == rid for entry in step["entries"])
    ]
    if not steps:
        raise ValueError("No attributed engine steps")
    result = {
        "passed": True,
        "llm_request_id": rid,
        "engine_dir": str(engine),
        "bound_layers": len(layers),
        "capture_paths": sorted({layer["capture_path"] for layer in layers}),
        "routing_rows": rows,
        "engine_steps": len(steps),
        "response": body["choices"][0]["message"],
    }
    write_json(Path(trace_dir) / "smoke.json", result)
    return result
