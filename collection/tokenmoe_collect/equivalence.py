"""Fixed-prompt off/off/on/on capture equivalence protocol."""

import json
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np
from jinja2 import StrictUndefined, Template

from .ids import content_id, new_id
from .jsonl import write_json
from .serving import command
from .static import load_yaml, minisweagent_config


def compare(rounds, engine_config_id, num_experts, top_k):
    if len(rounds) != 4 or any(len(r) != 50 for r in rounds):
        raise ValueError("Equivalence requires four rounds of exactly 50 prompts")
    floor = sum(a["token_ids"] != b["token_ids"] for a, b in zip(rounds[0], rounds[1]))
    disagreements = {
        f"off{o + 1}_on{n - 1}": [
            i
            for i, (a, b) in enumerate(zip(rounds[o], rounds[n]))
            if a["token_ids"] != b["token_ids"]
        ]
        for o in [0, 1]
        for n in [2, 3]
    }
    invalid_rows, routing_mismatch = [], []
    for round_index in [2, 3]:
        for i, request in enumerate(rounds[round_index]):
            with np.load(request["routing_file"], allow_pickle=False) as routing:
                experts = routing["experts"]
                valid = (
                    experts.ndim == 3
                    and experts.shape[0] > 0
                    and experts.shape[1] > 0
                    and experts.shape[0] == len(routing["token_positions"])
                    and experts.shape[-1] == top_k
                    and len(routing["layer_ids"]) == experts.shape[1]
                )
                valid &= bool(np.all((experts >= 0) & (experts < num_experts)))
                valid &= bool(np.all(np.diff(np.sort(experts, axis=-1), axis=-1) > 0))
                if not valid:
                    invalid_rows.append([round_index, i])
    for i, (a, b) in enumerate(zip(rounds[2], rounds[3])):
        with (
            np.load(a["routing_file"], allow_pickle=False) as left,
            np.load(b["routing_file"], allow_pickle=False) as right,
        ):
            if not np.array_equal(left["layer_ids"], right["layer_ids"]):
                routing_mismatch.append(i)
                continue
            positions, li, ri = np.intersect1d(
                left["token_positions"], right["token_positions"], return_indices=True
            )
            agree = left["token_ids"][positions] == right["token_ids"][positions]
            if not np.array_equal(left["experts"][li[agree]], right["experts"][ri[agree]]):
                routing_mismatch.append(i)
    return {
        "engine_config_id": engine_config_id,
        "num_prompts": 50,
        "protocol": ["off", "off", "on", "on"],
        "off_off_disagreements": floor,
        "on_off_disagreements": disagreements,
        "invalid_routing": invalid_rows,
        "routing_disagreements": routing_mismatch,
        "passed": max(map(len, disagreements.values())) <= floor + 1
        and not invalid_rows
        and not routing_mismatch,
    }


def fixed_prompts(dataset_path):
    import pyarrow.parquet as pq

    files = sorted(Path(dataset_path).rglob("*.parquet"))
    rows = [row for path in files for row in pq.read_table(path).to_pylist()]
    selected = sorted(rows, key=lambda row: row["instance_id"])[:50]
    if len(selected) != 50:
        raise ValueError("Need at least 50 benchmark instances")
    config = minisweagent_config()["agent"]
    return [
        {
            "benchmark_item_id": item["instance_id"],
            "messages": [
                {
                    "role": "system",
                    "content": Template(
                        config["system_template"], undefined=StrictUndefined
                    ).render(task=item["problem_statement"]),
                },
                {
                    "role": "user",
                    "content": Template(
                        config["instance_template"], undefined=StrictUndefined
                    ).render(task=item["problem_statement"]),
                },
            ],
        }
        for item in selected
    ]


def wait_ready(url, process, timeout=1200):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Model server exited with code {process.returncode}")
            try:
                if client.get(url + "/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
    raise TimeoutError("Model server startup timed out")


def stop_server(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def equivalence(model_config, dataset_path, output_dir, *, port=8001, startup_timeout=1200):
    from .minisweagent_adapter import BASH_TOOL

    config = load_yaml(model_config)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    prompts = fixed_prompts(dataset_path)
    write_json(root / "prompts.json", prompts)
    rounds, config_ids, commands = [], [], []
    repo = Path(__file__).resolve().parents[2]
    with httpx.Client(timeout=600) as client:
        for number, capture in enumerate([False, False, True, True]):
            stage = root / f"round-{number}"
            stage.mkdir()
            args = command(model_config, port=port, capture=capture)
            commands.append(args)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo / "vllm")
            env.pop("TOKENMOE_TRACE_DIR", None)
            env.pop("TOKENMOE_TRACE_ENGINE_DIR", None)
            if capture:
                env["TOKENMOE_TRACE_DIR"] = str(stage / "engine")
            print(shlex.join(args), flush=True)
            with (stage / "server.log").open("w") as log:
                process = subprocess.Popen(
                    args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
                try:
                    wait_ready(f"http://127.0.0.1:{port}", process, timeout=startup_timeout)
                    records = []
                    engine = next((stage / "engine").iterdir()) if capture else None
                    if engine:
                        metadata = json.loads((engine / "engine_meta.json").read_text())
                        config_ids.append(metadata["engine_config_id"])
                    for index, prompt in enumerate(prompts):
                        rid = new_id("req")
                        body = {
                            "model": config["name"],
                            "messages": prompt["messages"],
                            "tools": [BASH_TOOL],
                            "temperature": 0,
                            "max_tokens": 256,
                            "seed": index,
                            "return_token_ids": True,
                            "vllm_xargs": {"tokenmoe_llm_request_id": rid},
                        }
                        response = client.post(
                            f"http://127.0.0.1:{port}/v1/chat/completions", json=body
                        )
                        response.raise_for_status()
                        tokens = response.json()["choices"][0]["token_ids"]
                        if tokens is None:
                            raise ValueError("Server did not return token IDs for equivalence")
                        records.append(
                            {
                                "llm_request_id": rid,
                                "token_ids": tokens,
                                "routing_file": str(engine / f"routing/{rid}.npz")
                                if engine
                                else None,
                            }
                        )
                    rounds.append(records)
                    write_json(stage / "responses.json", records)
                finally:
                    stop_server(process)
    if len(set(config_ids)) != 1:
        raise ValueError("Capture-on rounds resolved to different engine configurations")
    facts = config["facts"]
    result = compare(rounds, config_ids[0], facts["num_routed_experts"], facts["top_k"])
    result.update(prompt_set_hash=content_id(prompts), commands=commands)
    write_json(root / "static/equivalence" / f"{config_ids[0]}.json", result)
    return result
