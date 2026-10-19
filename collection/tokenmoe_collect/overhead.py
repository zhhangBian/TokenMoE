"""Bounded capture overhead comparisons, separate from benchmark outcomes."""

import os
import subprocess
import time
from pathlib import Path

import httpx

from .equivalence import fixed_prompts, stop_server, wait_ready
from .ids import new_id
from .jsonl import write_json
from .serving import command
from .static import load_yaml


def measure_overhead(model_config, dataset_path, output_dir, *, port=18241, startup_timeout=1200):
    from .minisweagent_adapter import BASH_TOOL

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = load_yaml(model_config)
    prompts = fixed_prompts(dataset_path)[:12]
    write_json(root / "prompts.json", prompts)
    rounds = []
    for number, capture in enumerate([False, True, False, True]):
        stage = root / f"round-{number}"
        stage.mkdir()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "vllm")
        env.pop("TOKENMOE_TRACE_DIR", None)
        env.pop("TOKENMOE_TRACE_ENGINE_DIR", None)
        if capture:
            env["TOKENMOE_TRACE_DIR"] = str(stage / "engine")
        records = []
        with (stage / "server.log").open("x") as log:
            process = subprocess.Popen(
                command(model_config, port=port, capture=capture),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                wait_ready(f"http://127.0.0.1:{port}", process, timeout=startup_timeout)
                with httpx.Client(timeout=600) as client:
                    for index, prompt in enumerate(prompts):
                        body = {
                            "model": config["name"],
                            "messages": prompt["messages"],
                            "tools": [BASH_TOOL],
                            "temperature": 0,
                            "max_tokens": 64,
                            "seed": index,
                            "return_token_ids": True,
                            "vllm_xargs": {"tokenmoe_llm_request_id": new_id("req")},
                        }
                        started = time.perf_counter()
                        response = client.post(
                            f"http://127.0.0.1:{port}/v1/chat/completions", json=body
                        )
                        elapsed = time.perf_counter() - started
                        response.raise_for_status()
                        result = response.json()
                        records.append(
                            {
                                "benchmark_item_id": prompt["benchmark_item_id"],
                                "elapsed_seconds": elapsed,
                                "warmup": index < 2,
                                "token_ids": result["choices"][0]["token_ids"],
                                "usage": result.get("usage"),
                            }
                        )
            finally:
                stop_server(process)
        round_result = {
            "capture": capture,
            "records": records,
            "measured_seconds": sum(r["elapsed_seconds"] for r in records if not r["warmup"]),
        }
        rounds.append(round_result)
        write_json(stage / "measurements.json", round_result)
    off = sum(r["measured_seconds"] for r in rounds if not r["capture"]) / 2
    on = sum(r["measured_seconds"] for r in rounds if r["capture"]) / 2
    mismatch = sum(
        a["token_ids"] != b["token_ids"] for a, b in zip(rounds[0]["records"], rounds[1]["records"])
    )
    result = {
        "scope": "Sequential HTTP request wall time; excludes startup and two warmup requests per round. Shared-host load can affect results.",
        "hardware": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], text=True
        ),
        "selected_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "model_config": config,
        "repetitions_per_mode": 2,
        "measured_requests_per_round": 10,
        "max_tokens": 64,
        "capture_off_seconds": off,
        "capture_on_seconds": on,
        "overhead_fraction": on / off - 1 if off else None,
        "token_sequence_disagreements_first_pair": mismatch,
        "formal_target_met": on < off * 1.05,
        "local_data_validity_gate": False,
        "rounds": rounds,
    }
    write_json(root / "overhead.json", result)
    return {"passed": True, **result}
