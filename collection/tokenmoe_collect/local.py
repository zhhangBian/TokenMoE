"""Persistent local collection plans, recovery provenance, and health snapshots."""

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .assets import inspect_dataset, inspect_model, sha256_file
from .ids import content_id
from .jsonl import JsonlWriter, write_json
from .report import artifact_checksums, summarize_runtime
from .static import load_yaml

PROFILE = "local-qwen3-runtime-v1"
REPO = Path(__file__).resolve().parents[2]


def git_source(directory):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(directory), *args], stderr=subprocess.DEVNULL
        )

    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "patch_sha256": hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest(),
    }


def source_manifest(model_config, dataset_path, instances, devices, fork_venv, output):
    model = load_yaml(model_config)
    assets = inspect_model(model["model_path"])
    dataset = inspect_dataset(dataset_path, instances)
    if not assets["valid"] or not dataset["valid"]:
        raise ValueError(f"Local assets require repair: {assets}; {dataset}")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    available = []
    for line in gpu.splitlines():
        index, uuid, name, total, free = [x.strip() for x in line.split(",")]
        available.append(
            {
                "index": index,
                "uuid": uuid,
                "model": name,
                "total_mib": int(total),
                "free_mib": int(free),
            }
        )
    selected = [
        next((x for x in available if device in {x["index"], x["uuid"]}), None)
        for device in devices
    ]
    if any(x is None for x in selected) or len(selected) != model["serve"]["tensor_parallel_size"]:
        raise ValueError("Selected devices do not match the visible GPUs and TP configuration")
    # A100 BF16 TP=2 requires approximately 30 GiB of weights per rank, plus KV and workspace.
    if any(x["free_mib"] < 40000 for x in selected):
        raise ValueError(
            "Selected GPUs have insufficient free memory; do not terminate other tasks"
        )
    binaries = {
        "harness_python": sys.executable,
        "fork_python": str(Path(fork_venv) / "bin/python"),
    }
    fork_packages = subprocess.check_output(
        [
            binaries["fork_python"],
            "-c",
            "import importlib.metadata as m,json,sys;print(json.dumps({'python':sys.executable,'vllm':m.version('vllm'),'torch':m.version('torch')}))",
        ],
        text=True,
    )
    source = {
        "profile": PROFILE,
        "recorded_at_unix": time.time(),
        "host": platform.node(),
        "clock_domain_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "main": git_source(REPO),
        "fork": git_source(REPO / "vllm"),
        "source_files": {
            str(p.relative_to(REPO)): sha256_file(p)
            for p in sorted((REPO / "collection").rglob("*.py"))
            if "__pycache__" not in p.parts
        },
        "model": assets,
        "dataset": dataset,
        "instances": instances,
        "devices": selected,
        "visible_devices": available,
        "binaries": binaries,
        "fork_packages": json.loads(fork_packages),
        "harness_packages": {
            name: importlib.metadata.version(name)
            for name in ["mini-swe-agent", "httpx", "numpy", "pyarrow", "tokenizers"]
        },
        "disk_free_bytes": shutil.disk_usage(output).free,
    }
    if source["disk_free_bytes"] < 10 * 1024**3:
        raise ValueError("Less than 10 GiB is available for collection artifacts")
    write_json(Path(output) / "source_manifest.json", source)
    return source


def make_plan(config, pilot_config, model_config):
    model, primary, pilot = load_yaml(model_config), load_yaml(config), load_yaml(pilot_config)
    protocol = content_id(
        {
            "profile": PROFILE,
            "model": model,
            "limits": {
                k: primary[k]
                for k in [
                    "step_limit",
                    "time_limit_seconds",
                    "tool_timeout",
                    "request_timeout",
                    "request_retries",
                ]
            },
        }
    )
    units = [
        {
            "benchmark_item_id": item,
            "concurrency": n,
            "base_seed": primary["base_seed"],
            "protocol_id": protocol,
            "status": "pending",
            "attempts": [],
        }
        for n in primary["concurrency_matrix"]
        for item in primary["instances"]
    ]
    tool_smoke = load_yaml(REPO / "collection/configs/runs/smoke-local.yaml")
    for key in ["engine_dir", "server_url", "output_dir"]:
        tool_smoke.pop(key, None)
    tool_smoke.update(
        model_config=str(Path(model_config).resolve()),
        dataset_path=str(REPO / "collection/configs/smoke-items.json"),
        mini_config=str(REPO / "collection/configs/runs/smoke-agent.yaml"),
        collection_profile=PROFILE,
        concurrency_matrix=[1],
        memory_limit="4g",
    )
    return {
        "profile": PROFILE,
        "protocol_id": protocol,
        "model_config": str(Path(model_config).resolve()),
        "primary_config": str(Path(config).resolve()),
        "pilot_config": str(Path(pilot_config).resolve()),
        "model": model,
        "primary": primary,
        "pilot": pilot,
        "tool_smoke": tool_smoke,
        "units": units,
        "review_interval_seconds": 1800,
    }


def health_snapshot(root):
    root = Path(root)
    state_path = root / "collection_manifest.json"
    manifest = json.loads(state_path.read_text()) if state_path.exists() else {}
    active = manifest.get("active_process")
    alive = False
    if active:
        try:
            import psutil

            process = psutil.Process(active["pid"])
            alive = (
                abs(process.create_time() - active["create_time"]) < 0.01 and process.is_running()
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    evidence = []
    for path in sorted(root.rglob("requests.jsonl")):
        stat = path.stat()
        evidence.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": stat.st_size,
                "modified_at_unix": stat.st_mtime,
            }
        )
    return {
        "checked_at_unix": time.time(),
        "status": manifest.get("status", "preparing"),
        "active_process": active,
        "active_process_alive": alive,
        "valid_units": sum(u["status"] == "valid" for u in manifest.get("units", [])),
        "planned_units": len(manifest.get("units", [])),
        "request_logs": evidence,
    }


class LocalCollection:
    def __init__(self, root, plan, *, fork_venv, devices, port=18241, review_interval=1800):
        self.root, self.plan = Path(root).resolve(), plan
        self.fork_venv, self.devices, self.port = str(Path(fork_venv).resolve()), devices, port
        self.review_interval = review_interval
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "collection_manifest.json"
        if (self.root / "collection_plan.json").exists():
            saved = json.loads((self.root / "collection_plan.json").read_text())
            if saved["protocol_id"] != plan["protocol_id"] or saved["primary"] != plan["primary"]:
                raise ValueError("Protocol/configuration changed; use a new experiment directory")
        else:
            write_json(self.root / "collection_plan.json", plan)
        self.state = (
            json.loads(manifest.read_text())
            if manifest.exists()
            else {
                "profile": PROFILE,
                "protocol_id": plan["protocol_id"],
                "status": "preparing",
                "units": plan["units"],
                "stages": {},
            }
        )
        self.stop_monitor = threading.Event()
        self.process = None

    def save(self):
        self.state["updated_at_unix"] = time.time()
        write_json(self.root / "collection_manifest.json", self.state)

    def monitor(self):
        while not self.stop_monitor.is_set():
            try:
                snapshot = health_snapshot(self.root)
                with JsonlWriter(self.root / "health_checks.jsonl") as writer:
                    writer.write(snapshot)
                write_json(self.root / "latest_health_check.json", snapshot)
                print(json.dumps({"health_check": snapshot}), flush=True)
            except (OSError, ValueError) as exc:
                print(f"Health check failed: {exc}", flush=True)
            self.stop_monitor.wait(self.review_interval)

    def execute(self, label, arguments):
        import psutil

        log = self.root / "logs" / f"{label}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        if log.exists():
            raise FileExistsError(f"Preserve existing stage log: {log}")
        source_path = self.root / "source_snapshots" / f"{label}.json"
        snapshot = {
            "recorded_at_unix": time.time(),
            "main": git_source(REPO),
            "fork": git_source(REPO / "vllm"),
            "source_files": {
                str(p.relative_to(REPO)): sha256_file(p)
                for p in sorted((REPO / "collection").rglob("*.py"))
                if "__pycache__" not in p.parts
            },
            "initial_source_manifest_ref": str(self.root / "source_manifest.json"),
        }
        write_json(source_path, snapshot)
        command = [sys.executable, str(REPO / "collection/scripts/run_collection.py"), *arguments]
        started = time.time()
        print(json.dumps({"stage": label, "command": command, "log": str(log)}), flush=True)
        with log.open("x") as stream:
            self.process = subprocess.Popen(
                command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            self.state["active_process"] = {
                "pid": self.process.pid,
                "create_time": psutil.Process(self.process.pid).create_time(),
                "command": command,
                "log": str(log),
                "started_at_unix": started,
                "source_snapshot_ref": str(source_path),
            }
            self.state["status"] = "running"
            self.save()
            try:
                code = self.process.wait()
            finally:
                if self.process.poll() is None:
                    descendants = psutil.Process(self.process.pid).children(recursive=True)
                    self.process.terminate()
                    for child in descendants:
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    _, alive = psutil.wait_procs(descendants, timeout=45)
                    for child in alive:
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                self.state["active_process"] = None
                self.save()
                self.process = None
        self.state["stages"][label] = {
            "returncode": code,
            "started_at_unix": started,
            "finished_at_unix": time.time(),
            "log": str(log),
            "source_snapshot_ref": str(source_path),
        }
        self.save()
        return code

    def gpu_arguments(self):
        return [
            "--fork-venv",
            self.fork_venv,
            "--gpus",
            ",".join(self.devices),
            "--port",
            str(self.port),
        ]

    def next_attempt(self, parent, prefix):
        numbers = []
        for path in [
            *parent.glob("attempt-*"),
            *parent.glob("run-attempt-*.json"),
            *(self.root / "logs").glob(f"{prefix}-*.log"),
        ]:
            suffix = path.stem.rsplit("-", 1)[-1]
            if suffix.isdigit():
                numbers.append(int(suffix))
        return max(numbers, default=0) + 1

    def gate(self):
        for path in sorted((self.root / "equiv").glob("*/static/equivalence/*.json"), reverse=True):
            gate = json.loads(path.read_text())
            if gate.get("passed") and gate.get("num_prompts") == 50:
                return path
        raise ValueError("A real passing equivalence record is required before collection")

    def run_stage(self, name):
        if name == "preflight":
            source_manifest(
                self.plan["model_config"],
                self.plan["primary"]["dataset_path"],
                self.plan["primary"]["instances"],
                self.devices,
                self.fork_venv,
                self.root,
            )
            self.state["stages"]["preflight"] = {"passed": True}
            self.save()
            return
        if name in {"smoke", "equiv", "overhead"}:
            parent = self.root / name
            if any(
                s.get("returncode") == 0
                for label, s in self.state["stages"].items()
                if label.startswith(name + "-")
            ):
                return
            attempt = self.next_attempt(parent, name)
            destination = parent / f"attempt-{attempt:02d}"
            destination.parent.mkdir(exist_ok=True)
            arguments = [
                "--stage",
                name,
                "--model-config",
                self.plan["model_config"],
                "--output-dir",
                str(destination),
                *self.gpu_arguments(),
            ]
            if name in {"equiv", "overhead"}:
                arguments += ["--dataset-dir", self.plan["primary"]["dataset_path"]]
            if self.execute(f"{name}-{attempt:02d}", arguments):
                raise RuntimeError(f"{name} failed; preserve evidence and repair before resuming")
            return
        if name in {"pilot", "tool-smoke"}:
            if any(
                stage.get("returncode") == 0
                for label, stage in self.state["stages"].items()
                if label.startswith(name + "-")
            ):
                return
            parent = self.root / name
            attempt = self.next_attempt(parent, name)
            parent.mkdir(exist_ok=True)
            output = parent / f"attempt-{attempt:02d}"
            config_path = self.plan["pilot_config"]
            if name == "tool-smoke":
                config_path = str(parent / f"run-attempt-{attempt:02d}.json")
                write_json(config_path, {**self.plan["tool_smoke"], "output_dir": str(output)})
            code = self.execute(
                f"{name}-{attempt:02d}",
                [
                    "--stage",
                    "pilot",
                    "--config",
                    config_path,
                    "--output-dir",
                    str(output),
                    "--equivalence-record",
                    str(self.gate()),
                    *self.gpu_arguments(),
                ],
            )
            runs = sorted(output.glob("N*/runtime"))
            valid = bool(runs)
            for runtime in runs:
                report = summarize_runtime(runtime.parent)
                valid &= all(x["valid"] for x in report["units"])
            if code or not valid:
                raise RuntimeError(
                    f"{name} collection integrity failed; diagnose and resume with a new attempt"
                )
            return
        if name == "collect":
            for concurrency in self.plan["primary"]["concurrency_matrix"]:
                pending = [
                    u
                    for u in self.state["units"]
                    if u["concurrency"] == concurrency and u["status"] != "valid"
                ]
                if not pending:
                    continue
                parent = self.root / "collection" / f"N{concurrency}"
                parent.mkdir(parents=True, exist_ok=True)
                attempt = self.next_attempt(parent, f"N{concurrency}")
                destination = parent / f"attempt-{attempt:02d}"
                config = {
                    **self.plan["primary"],
                    "instances": [u["benchmark_item_id"] for u in pending],
                    "concurrency_matrix": [concurrency],
                    "num_concurrent_sessions": concurrency,
                    "model_config": self.plan["model_config"],
                    "output_dir": str(destination),
                    "protocol_id": self.plan["protocol_id"],
                    "source_manifest_ref": str(self.root / "source_manifest.json"),
                }
                config_path = parent / f"run-attempt-{attempt:02d}.json"
                write_json(config_path, config)
                for unit in pending:
                    unit["status"] = "running"
                self.save()
                code = self.execute(
                    f"N{concurrency}-{attempt:02d}",
                    [
                        "--stage",
                        "full",
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(destination),
                        "--equivalence-record",
                        str(self.gate()),
                        *self.gpu_arguments(),
                    ],
                )
                run_root = destination / f"N{concurrency}"
                if (run_root / "runtime").is_dir():
                    report = summarize_runtime(run_root)
                    results = {x["benchmark_item_id"]: x for x in report["units"]}
                else:
                    results = {}
                for unit in pending:
                    old = list(unit["attempts"])
                    result = results.get(
                        unit["benchmark_item_id"],
                        {
                            "valid": False,
                            "invalid_reasons": ["stage_failed_before_finalization"],
                            "artifact": str(run_root),
                        },
                    )
                    unit["attempts"].append(result)
                    unit["status"] = "valid" if result["valid"] else "needs_recollection"
                    if result["valid"]:
                        unit["artifact"] = result["artifact"]
                        unit["session_id"] = result["session_id"]
                    with JsonlWriter(self.root / "recovery_events.jsonl") as writer:
                        writer.write(
                            {
                                "recorded_at_unix": time.time(),
                                "unit": {
                                    k: unit[k]
                                    for k in [
                                        "benchmark_item_id",
                                        "concurrency",
                                        "base_seed",
                                        "protocol_id",
                                    ]
                                },
                                "prior_attempts": old,
                                "new_attempt": result,
                                "returncode": code,
                                "repair_source": git_source(REPO),
                            }
                        )
                self.save()
                if any(u["status"] != "valid" for u in pending):
                    raise RuntimeError(
                        f"N{concurrency} contains invalid units; repair the recorded cause before recollection"
                    )
            return
        if name == "report":
            artifacts = sorted({u["artifact"] for u in self.state["units"] if u.get("artifact")})
            reports = [summarize_runtime(path) for path in artifacts]
            complete = all(u["status"] == "valid" for u in self.state["units"])
            write_json(
                self.root / "reports/runtime_summary.json",
                {
                    "profile": PROFILE,
                    "planned_units": len(self.state["units"]),
                    "valid_units": sum(u["status"] == "valid" for u in self.state["units"]),
                    "complete": complete,
                    "runs": reports,
                },
            )
            (self.root / "reports/runtime_summary.md").write_text(
                f"# Local Qwen runtime collection\n\nValid planned units: {sum(u['status'] == 'valid' for u in self.state['units'])}/{len(self.state['units'])}.\n\nSee runtime_summary.json and per-run reports for timings, demand counts, coverage and failures.\n"
            )
            artifact_checksums(self.root)
            self.state["status"] = "complete" if complete else "incomplete"
            self.save()
            return
        raise ValueError(f"Unknown local stage: {name}")

    def run(self, stages):
        import fcntl

        with (self.root / ".collection.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.save()
            monitor = threading.Thread(target=self.monitor, daemon=True)
            monitor.start()
            try:
                for name in stages:
                    self.run_stage(name)
                if self.state["status"] == "running":
                    self.state["status"] = "ready_for_next_stage"
                    self.save()
            except BaseException as exc:
                self.state["status"] = "needs_repair"
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                self.save()
                raise
            finally:
                self.stop_monitor.set()
                monitor.join(timeout=2)
                write_json(self.root / "latest_health_check.json", health_snapshot(self.root))
