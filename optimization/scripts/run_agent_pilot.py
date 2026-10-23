"""One preregistered exploratory agent cell; raw failures remain in its output."""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from tokenmoe_collect.clock import now
from tokenmoe_collect.equivalence import stop_server, wait_ready
from tokenmoe_collect.ids import new_id
from tokenmoe_collect.jsonl import write_json
from tokenmoe_collect.launcher import run
from tokenmoe_collect.minisweagent_adapter import BASH_TOOL
from tokenmoe_collect.runtime import check_runtime
from tokenmoe_collect.serving import command as server_command
from tokenmoe_collect.static import load_yaml

from tokenmoe_opt.artifacts import Output, cache_execution, read_manifest, source_version
from tokenmoe_opt.controller import load_predictor


def main():
    parser = argparse.ArgumentParser()
    for name in ("protocol", "run-config", "output", "fork-venv"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--concurrency", type=int, choices=[1, 4], default=1)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--cost-profile", type=Path)
    args = parser.parse_args()
    protocol = read_manifest(args.protocol, "experiment")
    if args.policy not in protocol["policies"] or args.slots not in protocol["expert_slots"]:
        parser.error("Policy and slot capacity must belong to the frozen protocol")
    if not set(args.task) <= set(protocol["exploration_tasks"]):
        parser.error("The bounded pilot accepts exploration tasks only")
    predictive = args.policy not in {"full", "uva", "layer-prefetch", "lru", "lfu"}
    cached = args.policy not in {"full", "uva", "layer-prefetch"}
    if cached and args.cost_profile is None:
        parser.error("Cached cells require a qualified TP cost profile for byte accounting")
    if predictive:
        if args.predictor is None:
            parser.error("Predictive cells require a frozen predictor")
        predictor = load_predictor(args.predictor)
        if predictor.identity != protocol["identity"]:
            parser.error("Predictor and protocol identities differ")
        if set(args.task) & set(predictor.artifact["training_tasks"]):
            parser.error("Pilot tasks overlap predictor training groups")
    costs = None
    execution = cache_execution(protocol["execution"])
    if predictive and execution["prefill_execution"] != "tokens":
        parser.error("Expert-major predictive transfer benefits are not qualified")
    if cached:
        costs = read_manifest(args.cost_profile, "cost-profile")
        if costs["identity"] != protocol["identity"] or len(costs["ranks"]) != 2:
            parser.error("Pilot requires matching measured TP2 costs")
        if cache_execution(costs["controller"].get("cache_execution", {})) != execution:
            parser.error("Pilot requires matching measured cache execution costs")
    config_path = args.run_config.resolve()
    config = load_yaml(config_path)
    model = load_yaml((config_path.parent / config["model_config"]).resolve())
    sampling = {k: v for k, v in protocol["sampling"].items() if k != "seed"}
    if model["sampling"] != sampling:
        parser.error("Model sampling differs from the fitted template and frozen protocol")
    config.update(
        collection_profile="optimization-agent-v1",
        instances=args.task,
        num_concurrent_sessions=args.concurrency,
        dataset_path=str((config_path.parent / config["dataset_path"]).resolve()),
        control_events=True,
    )
    out = Output(
        args.output,
        kind="experiment",
        identity=protocol["identity"],
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, protocol["devices"])),
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "TOKENMOE_FORK_VENV": str(args.fork_venv.resolve()),
        "PYTHONPATH": os.pathsep.join(
            str(root / p) for p in ("vllm", "collection", "optimization")
        ),
    }
    env.pop("TOKENMOE_TRACE_DIR", None)
    env.pop("TOKENMOE_TRACE_ENGINE_DIR", None)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    cache = {"token_readiness_directory": str(out.path / "observations")}
    if cached:
        size = max(r["expert_bytes"] for r in costs["ranks"].values())
        scratch = 64 * 1024**2
        if execution["prefill_execution"] == "expert-major":
            scratch += protocol["mechanism"]["max_num_batched_tokens"] * 8 * 2048 * 2
        cache.update(
            **execution,
            enabled=True,
            slots_per_layer=args.slots,
            policy=args.policy if args.policy in {"lru", "lfu"} else "lru",
            scratch_bytes=scratch,
            budget_bytes_per_rank=48 * args.slots * size + scratch,
            telemetry_level="detailed",
            observation_directory=str(out.path / "observations"),
            control_directory=str(out.path / "workers"),
        )
    mechanism = protocol["mechanism"]
    model["serve"].update(
        tensor_parallel_size=2,
        enforce_eager=True,
        async_scheduling=False,
        max_model_len=mechanism["max_model_len"],
        max_num_batched_tokens=mechanism["max_num_batched_tokens"],
        max_num_seqs=mechanism["max_num_seqs"],
        kv_cache_memory_bytes=mechanism["kv_cache_memory_bytes_per_rank"],
        kernel_config={"moe_backend": "triton"},
        expert_cache_config=cache,
        seed=42,
    )
    if args.policy == "uva":
        model["serve"].update(cpu_offload_gb=24, cpu_offload_params=["experts"])
    elif args.policy == "layer-prefetch":
        model["serve"].update(
            offload_group_size=8,
            offload_num_in_group=7,
            offload_prefetch_step=1,
            offload_params=["experts"],
        )
    model_path = out.path / "model.json"
    write_json(model_path, model)
    config.update(
        model_config=str(model_path),
        output_dir=str(out.path / "agent"),
        server_url=f"http://127.0.0.1:{port}/v1",
    )
    run_path = out.path / "run.json"
    write_json(run_path, config)
    out.write("protocol.json", protocol)
    out.write(
        "cell.json",
        {
            "policy": args.policy,
            "slots": args.slots if cached else None,
            "tasks": args.task,
            "concurrency": args.concurrency,
            "repetition": args.repetition,
            "scope": "bounded exploratory pilot",
        },
    )
    server = resources = controller = None
    try:
        out.write("runtime_preflight.json", check_runtime(run_path))
        with (out.path / "resources.log").open("w") as log:
            resources = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tokenmoe_opt",
                    "resources",
                    "--pid",
                    str(os.getpid()),
                    "--devices",
                    *map(str, protocol["devices"]),
                    "--output",
                    str(out.path / "resources"),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        deadline = time.monotonic() + 10
        while not (out.path / "resources/ready.json").exists():
            if resources.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Resource monitor did not start")
            time.sleep(0.05)
        command = server_command(model_path, port=port, capture=False)
        command[0] = str(args.fork_venv.resolve() / "bin/vllm")
        out.write("server-command.json", command)
        with (out.path / "server.log").open("w") as log:
            server = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=root / "vllm",
                start_new_session=True,
            )
        wait_ready(f"http://127.0.0.1:{port}", server, timeout=1200)
        engine = next((out.path / "observations").glob("*/engine_meta.json")).parent
        config["engine_dir"] = str(engine)
        write_json(run_path, config)
        warmup = []
        write_json(out.path / "resource_phase.json", {"phase": "warmup"})
        with httpx.Client(timeout=config.get("request_timeout", 600)) as client:
            for index in range(protocol["warmup"]["requests"]):
                rid, started = new_id("req"), now()
                response = client.post(
                    config["server_url"] + "/chat/completions",
                    json={
                        **sampling,
                        "seed": 42 + index,
                        "model": model["name"],
                        "messages": [{"role": "user", "content": "Reply with one word: ready."}],
                        "tools": [BASH_TOOL],
                        "stream": False,
                        "vllm_xargs": {"tokenmoe_llm_request_id": rid},
                    },
                )
                response.raise_for_status()
                warmup.append(
                    {
                        "request_id": rid,
                        "started_at": started,
                        "finished_at": now(),
                        "response": response.json(),
                    }
                )
                out.write("warmup.json", warmup)
        if predictive:
            with (out.path / "controller.log").open("w") as log:
                controller = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tokenmoe_opt",
                        "control",
                        "--predictor",
                        str(args.predictor),
                        "--inputs",
                        str(args.predictor.parent.parent / "inputs.json"),
                        "--cost-profile",
                        str(args.cost_profile),
                        "--agent-run",
                        str(out.path / "agent"),
                        "--observations",
                        str(out.path / "observations"),
                        "--workers",
                        str(out.path / "workers"),
                        "--output",
                        str(out.path / "controller"),
                        "--policy",
                        args.policy,
                        "--duration",
                        "1800",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            deadline = time.monotonic() + 60
            while not (out.path / "controller/ready.json").exists():
                if controller.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Controller did not initialize; inspect controller.log")
                time.sleep(0.05)
        write_json(out.path / "resource_phase.json", {"phase": "measured"})
        out.write(
            "boundaries.json",
            {"measured_started_at": now(), "warmup_request_ids": [w["request_id"] for w in warmup]},
        )
        run(run_path)
        boundary = json.loads((out.path / "boundaries.json").read_text())
        out.write("boundaries.json", {**boundary, "measured_finished_at": now()})
        if controller is not None and controller.poll() is not None:
            raise RuntimeError("Controller exited during agent execution")
        stable = source_version()["files"] == out.record["source"]["files"]
        if not stable:
            raise RuntimeError("Runtime source changed during pilot execution")
        out.finish(
            evidence="live-bounded-agent-pilot",
            source_stable=stable,
            controlled_matrix_qualified=False,
        )
    except BaseException as exc:
        out.fail(exc)
        raise
    finally:
        forced = []

        def close_child(process, name):
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                    forced.append(name)

        write_json(out.path / "resource_phase.json", {"phase": "shutdown"})
        try:
            close_child(controller, "controller")
        finally:
            try:
                if server is not None and stop_server(server):
                    forced.append("server")
            finally:
                close_child(resources, "resources")
                peaks = out.path / "resources/peaks.json"
                out.record["resources"] = json.loads(peaks.read_text()) if peaks.exists() else None
                out.record.update(shutdown="complete", forced_cleanup=forced)
                out.write("manifest.json", out.record)


if __name__ == "__main__":
    main()
