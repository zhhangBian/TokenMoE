"""Bounded concurrent mini-swe-agent sessions with explicit container ownership."""

import concurrent.futures
import copy
import json
import threading
from pathlib import Path

from .client import ChatClient
from .clock import now
from .hostload import HostSampler
from .ids import new_id, sampling_seed
from .jsonl import write_json
from .records import RecordStore
from .static import load_benchmark, load_yaml, minisweagent_config, write_static


def image_name(item):
    return (
        item.get("image_name")
        or item.get("docker_image")
        or (
            "docker.io/swebench/sweb.eval.x86_64."
            + item["instance_id"].replace("__", "_1776_")
            + ":latest"
        ).lower()
    )


def run(config_path, *, sampler_source=None):
    from minisweagent.agents.default import DefaultAgent

    from .minisweagent_adapter import BASH_TOOL, TracedEnvironment, TracedModel

    path = Path(config_path).resolve()
    config = load_yaml(path)
    model_path = (path.parent / config["model_config"]).resolve()
    model = load_yaml(model_path)
    root = Path(config["output_dir"]).expanduser().resolve()
    if root.exists():
        raise FileExistsError(f"Output directory must be new: {root}")
    for key in ["num_concurrent_sessions", "step_limit", "time_limit_seconds"]:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_absolute():
        dataset_path = path.parent / dataset_path
    config["dataset_path"] = str(dataset_path.resolve())
    items = load_benchmark(dataset_path, config["instances"])
    engine_dir = Path(config["engine_dir"]).resolve()
    meta = json.loads((engine_dir / "engine_meta.json").read_text())
    mini = (
        load_yaml(path.parent / config["mini_config"])
        if config.get("mini_config")
        else minisweagent_config()
    )
    mini["agent"].update(
        step_limit=config["step_limit"],
        wall_time_limit_seconds=config["time_limit_seconds"],
        cost_limit=0,
    )
    mini["model"].update(model_name=model["name"], model_kwargs={})
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "run_config.json", config)
    write_json(root / "model_config.json", model)
    write_json(
        root / "engine_source.json",
        {"path": str(engine_dir), "engine_instance_id": meta["engine_instance_id"]},
    )
    static = write_static(root, config, model, engine_dir, items, mini)
    group = new_id("grp")
    live = {}
    lock = threading.Lock()

    with RecordStore(root) as store:
        if store.clock["clock_domain_id"] != meta["clock_domain_id"]:
            raise ValueError("Harness and engine must use the same host boot clock")

        def active():
            with lock:
                return sum(env.executor.active for env in live.values())

        sampler = HostSampler(store, active, source=sampler_source).start()

        def session(item):
            run_id, session_id = new_id("run"), new_id("ses")
            created = now()
            seed = sampling_seed(config["base_seed"], item["instance_id"], 0, 0)
            status, reason, error = "failed", "SetupError", None
            env = client = None
            started = created
            try:
                environment = copy.deepcopy(mini["environment"])
                environment.pop("environment_class", None)
                environment.update(
                    image=image_name(item),
                    pull_timeout=config.get("container_start_timeout", 300),
                    timeout=config.get("tool_timeout", environment["timeout"]),
                )
                if config["runtime"] == "local":
                    environment.update(cwd=config.get("local_cwd", str(root)), env={}, run_args=[])
                else:
                    args = ["--rm", "--pull=never"]
                    if config.get("container_network"):
                        args += ["--network", config["container_network"]]
                    if config.get("cpu_quota"):
                        args += ["--cpus", str(config["cpu_quota"])]
                    if config.get("memory_limit"):
                        args += ["--memory", str(config["memory_limit"])]
                    environment["run_args"] = args
                env = TracedEnvironment(
                    store=store,
                    session_id=session_id,
                    runtime=config["runtime"],
                    create_workdir=config.get("create_workdir", False),
                    **environment,
                )
                with lock:
                    live[session_id] = env
                client = ChatClient(
                    base_url=config["server_url"],
                    model=model["name"],
                    store=store,
                    session_id=session_id,
                    application_run_id=run_id,
                    benchmark_item_id=item["instance_id"],
                    base_seed=config["base_seed"],
                    tools=[BASH_TOOL],
                    sampling=model["sampling"],
                    timeout=config.get("request_timeout", 600),
                    retries=config.get("request_retries", 2),
                )
                traced_model = TracedModel(client, env, mini["model"])
                agent = DefaultAgent(traced_model, env, **mini["agent"])
                started = now()
                result = agent.run(item["problem_statement"])
                reason = result.get("exit_status", "Unknown")
                status = (
                    "succeeded"
                    if reason == "Submitted"
                    else "timeout"
                    if reason in {"LimitsExceeded", "TimeExceeded"}
                    else "failed"
                )
            except KeyboardInterrupt:
                status, reason = "cancelled", "KeyboardInterrupt"
                raise
            except Exception as exc:
                reason, error = type(exc).__name__, str(exc)
            finally:
                try:
                    if env is not None:
                        env.cleanup()
                except Exception as exc:
                    status, reason, error = "failed", "CleanupError", str(exc)
                finally:
                    with lock:
                        live.pop(session_id, None)
                    if client is not None:
                        client.close()
                finished = now()
                store.write(
                    "sessions",
                    {
                        "session_id": session_id,
                        "application_run_id": run_id,
                        "agent_template_id": static["role"]["agent_template_id"],
                        "version_hash": static["role"]["version_hash"],
                        "role_type": "generalist",
                        "parent_session_id": None,
                        "spawned_by_tool_call_id": None,
                        "predecessor_session_ids": [],
                        "created_at": created,
                        "started_at": started,
                        "finished_at": finished,
                        "final_status": status,
                        "exit_reason": reason,
                        "error": error,
                    },
                )
                store.write(
                    "application_runs",
                    {
                        "application_run_id": run_id,
                        "experiment_config_id": static["experiment"]["experiment_config_id"],
                        "application_definition_id": None,
                        "benchmark_id": config.get(
                            "benchmark_id", "princeton-nlp/SWE-bench_Verified"
                        ),
                        "benchmark_item_id": item["instance_id"],
                        "task_type": item["repo"],
                        "seed": seed,
                        "retry_of_application_run_id": None,
                        "concurrency_group_id": group,
                        "run_started_at": created,
                        "run_finished_at": finished,
                        "run_status": status,
                        "exit_reason": reason,
                        "benchmark_score": None,
                        "error": error,
                    },
                )
            return {
                "instance_id": item["instance_id"],
                "status": status,
                "exit_reason": reason,
                "error": error,
            }

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config["num_concurrent_sessions"]
            ) as pool:
                results = list(pool.map(session, items))
        finally:
            sampler.close()
    write_json(root / "run_summary.json", {"sessions": results})
    return root
