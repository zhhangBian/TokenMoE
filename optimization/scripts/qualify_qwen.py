"""Bounded local GPU smoke; this script is not the full qualification gate."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

os.environ.setdefault("OMP_NUM_THREADS", "1")

from vllm import LLM, SamplingParams


def record_sources(directory):
    root = Path(__file__).resolve().parents[2]
    result = {}
    for name, repo in (("main", root), ("fork", root / "vllm")):
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        patch = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--binary"])
        untracked = subprocess.check_output(
            ["git", "-C", str(repo), "ls-files", "--others", "--exclude-standard"], text=True
        ).splitlines()
        for path in untracked:
            diff = subprocess.run(
                ["git", "-C", str(repo), "diff", "--no-index", "--binary", "--", "/dev/null", path],
                capture_output=True,
            )
            if diff.returncode not in (0, 1):
                raise RuntimeError(f"Cannot snapshot source {repo / path}")
            patch += diff.stdout
        (directory / f"{name}.patch").write_bytes(patch)
        result[name] = {"commit": commit, "patch_sha256": hashlib.sha256(patch).hexdigest()}
        directories = (
            [repo / "vllm"]
            if name == "fork"
            else [repo / "optimization", repo / "collection/tokenmoe_collect"]
        )
        hashes = {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for source in directories
            for path in sorted(source.rglob("*.py"))
        }
        result[name]["runtime_files"] = hashes
    return result


def measured_sources(name, files):
    # Live qualification imports the complete controller/predictor dependency tree.
    # Freeze all recorded Python sources instead of maintaining an import allowlist.
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, choices=[1, 2], required=True)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-directory", type=Path)
    parser.add_argument("--observation", choices=["off", "compact", "detailed"], default="off")
    parser.add_argument("--token-readiness", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--native-graph", action="store_true")
    parser.add_argument("--offload", choices=["none", "uva", "prefetch"], default="none")
    parser.add_argument("--live-predictor", type=Path)
    parser.add_argument("--live-inputs", type=Path)
    parser.add_argument("--live-costs", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repetitions < 1:
        parser.error("Warmup must be nonnegative and measured repetitions positive")
    if args.cache and (args.native_graph or args.offload != "none"):
        parser.error("Native graph/offloader references require the cache disabled")
    if (args.control_directory or args.observation != "off") and not args.cache:
        parser.error("Expert observation and control mailboxes require --cache")
    if args.stress and (not args.cache or args.tp != 2):
        parser.error("--stress requires --cache and --tp 2")
    if args.live_predictor and (
        not args.control_directory
        or args.observation == "off"
        or not args.live_inputs
        or not args.live_costs
    ):
        parser.error("Live controller requires control directory, observations, inputs, and costs")
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(
        model="/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B",
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        enforce_eager=not args.native_graph,
        async_scheduling=False,
        kernel_config={"moe_backend": "triton"},
        max_model_len=2048,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        kv_cache_memory_bytes=1024**3,
        enable_prefix_caching=True,
        seed=42,
    )
    if args.offload == "uva":
        config.update(cpu_offload_gb=24, cpu_offload_params=["experts"])
    elif args.offload == "prefetch":
        config.update(
            offload_group_size=8,
            offload_num_in_group=7,
            offload_prefetch_step=1,
            offload_params=["experts"],
        )
    if args.cache:
        config["expert_cache_config"] = dict(
            enabled=True, slots_per_layer=16, budget_bytes_per_rank=8 * 1024**3 // args.tp
        )
        if args.control_directory:
            config["expert_cache_config"]["control_directory"] = str(args.control_directory)
        if args.observation != "off":
            config["expert_cache_config"].update(
                telemetry_level=args.observation,
                observation_directory=str(args.output / "observations"),
                token_readiness_directory=str(args.output / "observations"),
            )
    if args.token_readiness:
        config.setdefault("expert_cache_config", {})["token_readiness_directory"] = str(
            args.output / "observations"
        )
    if args.probe or args.stress:
        scripts = str(Path(__file__).resolve().parent)
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [scripts, str(Path(scripts).parents[0] / "../vllm"), os.environ.get("PYTHONPATH", "")]
        )
        config["worker_extension_cls"] = "qwen_qualification_worker.QwenQualificationWorker"
    record = {
        "schema_version": 1,
        "status": "running",
        "kind": "qwen-bounded-smoke",
        "config": config,
        "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source": record_sources(args.output),
        "host_load": os.getloadavg(),
        "python_executable": sys.executable,
        "environment": subprocess.check_output(
            ["uv", "pip", "freeze", "--python", sys.executable], text=True
        ),
        "gpu_resources": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        ),
        "criteria": {"greedy_token_mismatches": 0, "tensor_atol": 0.02, "tensor_rtol": 0.02},
        "measurement": {
            "warmup_batches": args.warmup,
            "warmup_requests": args.warmup * args.concurrency,
            "warmup_excluded": True,
            "measured_repetitions": args.repetitions,
            "generation_budget": 8,
            "scope": "short backend cost diagnostic; no agent service evidence",
        },
        "limits": "Not full qualification: forced-token/router/cancellation/TP failure checks outstanding",
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")

    save()
    engine = None
    resources = None
    try:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        resources = subprocess.Popen(
            [
                "/home/youwei/bzh/venvs/tokenmoe-harness/bin/python",
                "-m",
                "tokenmoe_opt",
                "resources",
                "--pid",
                str(os.getpid()),
                "--devices",
                *devices,
                "--output",
                str(args.output / "resources"),
            ]
        )
        deadline = time.monotonic() + 10
        while not (args.output / "resources/ready.json").exists():
            if resources.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Resource monitor failed to start")
            time.sleep(0.05)
        began = time.monotonic()
        engine = LLM(**config)
        phase_file = args.output / "resource_phase.tmp"
        phase_file.write_text(json.dumps({"phase": "steady"}))
        phase_file.replace(args.output / "resource_phase.json")
        record["startup_seconds"] = time.monotonic() - began
        record["startup_memory"] = engine.collective_rpc("get_expert_cache_stats", timeout=60)
        if args.probe:
            record["probes"] = engine.collective_rpc(
                "enable_expert_probes", timeout=60, args=(str(args.output / "probes"),)
            )
        if args.probe or args.stress:
            record["worker_environment"] = engine.collective_rpc("qualification_state", timeout=60)
        prompts = [
            "The capital of France is",
            "In Python, a list is",
            "Two plus two equals",
            "A binary tree consists of",
        ][: args.concurrency]
        for _ in range(args.warmup):
            engine.generate(prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False)
        record["trials"] = []
        for repetition in range(args.repetitions):
            began = time.monotonic()
            outputs = engine.generate(
                prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
            )
            record["trials"].append(
                {
                    "repetition": repetition,
                    "seconds": time.monotonic() - began,
                    "tokens": [out.outputs[0].token_ids for out in outputs],
                }
            )
        record["final_memory"] = engine.collective_rpc("get_expert_cache_stats", timeout=60)
        if args.live_predictor:
            controller_dir = args.output / "controller"
            agent_dir = args.output / "controller-fixture"
            process = subprocess.Popen(
                [
                    "/home/youwei/bzh/venvs/tokenmoe-harness/bin/python",
                    "-m",
                    "tokenmoe_opt",
                    "control",
                    "--predictor",
                    str(args.live_predictor),
                    "--inputs",
                    str(args.live_inputs),
                    "--cost-profile",
                    str(args.live_costs),
                    "--agent-run",
                    str(agent_dir),
                    "--observations",
                    str(args.output / "observations"),
                    "--workers",
                    str(args.control_directory),
                    "--duration",
                    "30",
                    "--output",
                    str(controller_dir),
                ]
            )
            try:
                deadline = time.monotonic() + 20
                while not (controller_dir / "manifest.json").exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError("Live controller did not start")
                    time.sleep(0.05)
                subprocess.run(
                    [
                        "/home/youwei/bzh/venvs/tokenmoe-harness/bin/python",
                        str(Path(__file__).with_name("control_fixture.py")),
                        "--predictor",
                        str(args.live_predictor),
                        "--output",
                        str(agent_dir),
                    ],
                    check=True,
                    timeout=30,
                )
                after = engine.collective_rpc("get_expert_cache_stats", timeout=60)
                fixture = json.loads((agent_dir / "fixture.json").read_text())
                starts = [
                    min(
                        (
                            c["counters"]["first_prefetch_started_at"]
                            for c in rank["caches"]
                            if c["counters"].get("first_prefetch_started_at")
                        ),
                        default=0,
                    )
                    for rank in after
                ]
                record["live_controller"] = {
                    "synthetic_session": True,
                    "real_tool": "pytest",
                    "tool_interval": fixture["timing"],
                    "prefetch_started_at": starts,
                    "passed": all(
                        fixture["timing"]["started_at"] <= t <= fixture["timing"]["finished_at"]
                        for t in starts
                    ),
                }
                if not record["live_controller"]["passed"]:
                    raise AssertionError(
                        "Live predictor failed to prefetch during the observed real tool"
                    )
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
        elif args.control_directory:
            before = record["final_memory"]
            tool_started = time.monotonic_ns()
            with (args.output / "idle-tool.log").open("w") as tool_log:
                tool = subprocess.Popen(
                    [
                        "/home/youwei/bzh/venvs/tokenmoe-harness/bin/python",
                        "-m",
                        "pytest",
                        "/home/youwei/bzh/project/TokenMoE/optimization/tests/test_control.py",
                        "-q",
                    ],
                    stdout=tool_log,
                    stderr=subprocess.STDOUT,
                )
                for rank in before:
                    resident = {s["expert"] for s in rank["caches"][0]["snapshot"]["slots"]}
                    expert = next(e for e in range(128) if e not in resident)
                    plan = {
                        "schema_version": 1,
                        "epoch": 1,
                        "backend_hash": rank["backend_hash"],
                        "model_hash": rank["model_hash"],
                        "engine_id": rank["engine_id"],
                        "clock_domain_id": Path("/proc/sys/kernel/random/boot_id")
                        .read_text()
                        .strip(),
                        "expires_at": time.monotonic_ns() + 2_000_000_000,
                        "transfers": [{"layer": 0, "expert": expert}],
                        "desired": {"0": [expert]},
                    }
                    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                        sender.sendto(
                            json.dumps(plan).encode(),
                            str(args.control_directory / f"rank-{rank['rank']}.sock"),
                        )
                tool.wait(timeout=30)
                tool_finished = time.monotonic_ns()
            after = engine.collective_rpc("get_expert_cache_stats", timeout=60)
            starts = [
                rank["caches"][0]["counters"].get("last_prefetch_started_at") for rank in after
            ]
            record["idle_prefetch"] = {
                "tool_started_at": tool_started,
                "tool_finished_at": tool_finished,
                "tool_returncode": tool.returncode,
                "prefetch_started_at": starts,
                "before_next_request": time.monotonic_ns(),
                "passed": all(t is not None and tool_started <= t <= tool_finished for t in starts),
                "scope": "local mechanism using a real pytest tool; not an agent policy comparison",
            }
            if not record["idle_prefetch"]["passed"]:
                raise AssertionError("Prefetch did not start during idle tool execution")
            followup = engine.generate(
                prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
            )
            record["idle_prefetch"]["followup_tokens"] = [
                out.outputs[0].token_ids for out in followup
            ]
            if record["idle_prefetch"]["followup_tokens"] != record["trials"][-1]["tokens"]:
                raise AssertionError("Generation diverged after idle prefetch")
            record["post_control_memory"] = engine.collective_rpc(
                "get_expert_cache_stats", timeout=60
            )
        if args.stress:
            engine.collective_rpc("delay_qualification_shard", timeout=60)
            delayed = engine.generate(
                prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
            )
            record["delayed_shard"] = {
                "workers": engine.collective_rpc("qualification_state", timeout=60),
                "tokens": [out.outputs[0].token_ids for out in delayed],
            }
            if (
                not any(w["delayed_loads"] for w in record["delayed_shard"]["workers"])
                or record["delayed_shard"]["tokens"] != record["trials"][-1]["tokens"]
            ):
                raise AssertionError("Delayed-shard qualification failed")
            core = engine.llm_engine
            cancel_id = "qualification-cancel"
            core.add_request(
                cancel_id, "Count natural numbers:", SamplingParams(temperature=0, max_tokens=64)
            )
            core.step()
            core.abort_request([cancel_id])
            following = engine.generate(
                prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
            )
            record["cancellation"] = {
                "followup_tokens": [out.outputs[0].token_ids for out in following]
            }
            if record["cancellation"]["followup_tokens"] != record["trials"][-1]["tokens"]:
                raise AssertionError("Generation diverged after cancellation")
            workers = engine.collective_rpc("qualification_state", timeout=60)
            engine.collective_rpc("arm_qualification_peer_failure", timeout=60)
            done, timed_out = threading.Event(), threading.Event()

            def watchdog():
                if not done.wait(15):
                    timed_out.set()
                    for process in psutil.Process().children(recursive=True):
                        if process.pid != resources.pid:
                            try:
                                process.kill()
                            except psutil.NoSuchProcess:
                                pass

            guard = threading.Thread(target=watchdog, daemon=True)
            guard.start()
            failed_at, error = time.monotonic(), None
            try:
                engine.generate(
                    prompts, SamplingParams(temperature=0, max_tokens=8), use_tqdm=False
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            finally:

                def worker_alive(worker):
                    try:
                        process = psutil.Process(worker["pid"])
                        return (
                            process.create_time() == worker["create_time"]
                            and process.status() != psutil.STATUS_ZOMBIE
                        )
                    except psutil.NoSuchProcess:
                        return False

                while (
                    any(worker_alive(worker) for worker in workers)
                    and time.monotonic() - failed_at < 15
                ):
                    time.sleep(0.05)
                done.set()
                guard.join(1)
            record["peer_failure"] = {
                "error": error,
                "seconds": time.monotonic() - failed_at,
                "watchdog_required": timed_out.is_set(),
                "worker_pids": workers,
                "workers_released": not any(worker_alive(worker) for worker in workers),
            }
            if (
                error is None
                or timed_out.is_set()
                or not record["peer_failure"]["workers_released"]
            ):
                raise AssertionError("Peer failure did not abort execution within 15 seconds")
        record["status"] = "smoke-complete-unqualified"
        final_source = args.output / "final-source"
        final_source.mkdir()
        record["final_source"] = record_sources(final_source)
        record["source_stable_scope"] = (
            "all fork runtime, qualification scripts, resource monitor and idle-tool test dependencies"
        )
        record["source_stable"] = all(
            measured_sources(name, record["source"][name]["runtime_files"])
            == measured_sources(name, record["final_source"][name]["runtime_files"])
            for name in ("main", "fork")
        )
        if not record["source_stable"]:
            raise AssertionError("Source files changed during GPU measurement")
        save()
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    finally:
        if resources is not None and resources.poll() is None:
            resources.terminate()
            try:
                resources.wait(timeout=5)
            except subprocess.TimeoutExpired:
                resources.kill()
                resources.wait(timeout=5)
                record["resource_monitor_error"] = "Monitor did not stop within five seconds"
        peak_path = args.output / "resources/peaks.json"
        record["resources"] = json.loads(peak_path.read_text()) if peak_path.exists() else None
        save()
        if engine is not None:
            engine.llm_engine.engine_core.shutdown()
            record["shutdown"] = "complete"
            save()


if __name__ == "__main__":
    main()
