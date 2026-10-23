"""Preregister and capture real test/build streams, independent of model inference."""

import concurrent.futures
import copy
import json
import os
import re
import shlex
import signal
import subprocess
import threading
from pathlib import Path

from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.events import emit
from tokenmoe_collect.ids import content_id, new_id
from tokenmoe_collect.jsonl import read_jsonl, write_json
from tokenmoe_collect.runtime import container_run_args, runtime_command, validate_limits
from tokenmoe_collect.static import load_benchmark, minisweagent_config

from .artifacts import Output, read_manifest, source_version

SUITES = (
    ("astropy__astropy-12907", "astropy", ("astropy/coordinates/tests",)),
    ("django__django-10097", "django", ("tests/expressions",)),
    ("matplotlib__matplotlib-13989", "lib/matplotlib", ("lib/matplotlib/tests/test_axes.py",)),
    (
        "mwaskom__seaborn-3069",
        "seaborn",
        ("tests/test_axisgrid.py", "seaborn/tests/test_axisgrid.py"),
    ),
    ("pallets__flask-5014", "src/flask", ("tests/test_basic.py",)),
    ("psf__requests-1142", "requests", ("test_requests.py", "tests/test_requests.py")),
    ("pydata__xarray-2905", "xarray", ("xarray/tests/test_dataarray.py",)),
    ("pylint-dev__pylint-4551", "pylint", ("tests/test_functional.py", "tests/test_lint.py")),
    ("pytest-dev__pytest-10051", "src", ("testing/test_assertion.py",)),
    ("scikit-learn__scikit-learn-10297", "sklearn", ("sklearn/metrics/tests",)),
    ("sphinx-doc__sphinx-10323", "sphinx", ("tests/test_build_html.py",)),
    ("sympy__sympy-11618", "sympy", ("sympy/core/tests",)),
)
LIMITS = {"container_network": "none", "cpu_quota": 2, "memory_limit": "4g"}


def regression_targets(task, patch, fail_ids):
    changed = [
        line.split(" b/", 1)[1]
        for line in patch.splitlines()
        if line.startswith("diff --git ") and " b/" in line
    ]
    if task.startswith("django__"):
        return sorted({p.split("/")[1] for p in changed if p.startswith("tests/")})
    named = {str(t).split("::")[0] for t in fail_ids if str(t).split("::")[0].endswith(".py")}
    return sorted(named or {p for p in changed if p.endswith(".py") and "test" in Path(p).name})


def freeze(dataset, output, *, command):
    from tokenmoe_collect.launcher import image_name

    out = Output(
        output, kind="experiment", identity={"scope": "real-test-build-streams-v1"}, command=command
    )
    try:
        rows = load_benchmark(
            dataset,
            [s[0] for s in SUITES],
            columns=[
                "instance_id",
                "repo",
                "base_commit",
                "FAIL_TO_PASS",
                "test_patch",
            ],
        )
        prefix, jobs, checks = runtime_command("podman"), [], []
        for row, (_, package, candidates) in zip(rows, SUITES):
            task, image = row["instance_id"], image_name(row)
            image_id = subprocess.check_output(
                [*prefix, "image", "inspect", image, "--format", "{{.Id}}"], text=True
            ).strip()
            metadata_script = (
                "import json,os,subprocess,sys;paths=json.loads(sys.argv[1]);"
                "print(json.dumps({'paths':{p:os.path.exists(p) for p in paths},"
                "'commit':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),"
                "'base_object':subprocess.check_output(['git','cat-file','-t',sys.argv[2]]).decode().strip()}))"
            )
            metadata = json.loads(
                subprocess.check_output(
                    [
                        *prefix,
                        "run",
                        *container_run_args(LIMITS),
                        "-w",
                        "/testbed",
                        "--entrypoint",
                        "python3",
                        image,
                        "-c",
                        metadata_script,
                        json.dumps([package, *candidates]),
                        row["base_commit"],
                    ],
                    text=True,
                    timeout=60,
                )
            )
            checks.append(
                {
                    "task": task,
                    "image_id": image_id,
                    "metadata": metadata,
                    "benchmark_base_commit": row["base_commit"],
                }
            )
            out.write("metadata-checks.json", checks)
            if metadata["base_object"] != "commit" or not metadata["paths"][package]:
                raise ValueError(f"Cached image differs from benchmark base/layout: {task}")
            focus = next((p for p in candidates if metadata["paths"][p]), None)
            if focus is None:
                raise ValueError(f"No preregistered focused suite exists: {task}")
            fail_ids = json.loads(row["FAIL_TO_PASS"])
            regression = regression_targets(task, row["test_patch"], fail_ids)
            if task.startswith("django__"):

                def test(paths):
                    return "python tests/runtests.py --verbosity=2 --parallel=1 " + shlex.join(
                        p.removeprefix("tests/").removesuffix(".py").replace("/", ".")
                        for p in paths
                    )
            else:

                def test(paths):
                    return "python -m pytest -vv --tb=short " + shlex.join(paths)

            bodies = {"full": test([]), "focused": test([focus]), "regression": test(regression)}
            if not regression:
                raise ValueError(f"No official regression test path: {task}")
            bodies["compile"] = "python -m compileall -q " + shlex.quote(package)
            for kind, body in bodies.items():
                jobs.append(
                    {
                        "job_id": content_id(
                            {"protocol": "long-tools-v1", "task": task, "kind": kind}
                        ),
                        "task": task,
                        "kind": kind,
                        "image": image,
                        "image_id": image_id,
                        "base_commit": row["base_commit"],
                        "test_patch": row["test_patch"],
                        "command": "source /opt/miniconda3/bin/activate testbed && " + body,
                        "timeout_seconds": {
                            "full": 180,
                            "focused": 300,
                            "regression": 120,
                            "compile": 90,
                        }[kind],
                    }
                )
        protocol = {
            "schema_version": 1,
            "kind": "experiment",
            "identity": out.record["identity"],
            "protocol_id": "real-test-build-long-tools-v1",
            "status": "preregistered-unexecuted",
            "dataset": str(Path(dataset).resolve()),
            "dataset_sha256": sha256_file(dataset),
            "workload_origin": "authored commands on explored SWE-bench base images with official test patches",
            "solution_patch_used": False,
            "workspace_setup": "reset tracked files to benchmark base; preserve installed/untracked build artifacts; apply only official test_patch",
            "confirmation_tasks_used": False,
            "runtime": "podman",
            "limits": LIMITS,
            "concurrency": 2,
            "job_order": "ascending SHA256-derived job ID",
            "jobs": sorted(jobs, key=lambda j: j["job_id"]),
            "coverage_minima": {
                "task_groups": 10,
                "tools": 36,
                "gt5_seconds": 16,
                "gt30_seconds": 8,
                "gt90_seconds": 3,
                "completed_gt30_seconds": 3,
                "early_lead_gt0_5_seconds": 16,
                "early_lead_gt5_seconds": 8,
                "multi_chunk": 16,
                "zero_chunk": 4,
                "success": 6,
                "failure": 3,
                "timeout": 3,
            },
            "censoring": "timeouts use the configured deadline as a conservative lower bound, excluding cleanup delay",
            "scope": "real tool streams; authored workflow, no LLM decisions or agent task success evidence",
        }
        out.write("metadata-checks.json", checks)
        out.write("protocol.json", protocol)
        out.finish(evidence="long-tool-preregistration", corpus_collected=False)
    except BaseException as exc:
        out.fail(exc)
        raise
    return out.path


def validate_case(root, job):
    root = Path(root)
    case = json.loads((root / "case.json").read_text())
    if case["status"] != "complete":
        return None
    if any(case[key] != job[key] for key in ("job_id", "task", "image_id", "command")):
        raise ValueError("Tool case differs from its frozen job")
    timing = case["timing"]
    started, finished = timing["started_at"], timing["finished_at"]
    if finished < started:
        raise ValueError("Tool completion predates start")
    raw = (root / timing["output_ref"]).resolve()
    if not raw.is_relative_to(root.resolve()) or raw.stat().st_size != timing["output_bytes_raw"]:
        raise ValueError("Tool output path/byte count mismatch")
    producer = (root / case["producer"]).resolve()
    if not producer.is_relative_to(root.resolve()):
        raise ValueError("Tool producer is outside its immutable case")
    metadata = json.loads((producer / "producer.json").read_text())
    if metadata["clock_domain_id"] != case["preflight"]["boot_id"]:
        raise ValueError("Tool producer clock differs from its container")
    chunks = list(read_jsonl(producer / "tool_output_chunks.jsonl"))
    offset, observed, first = 0, started, None
    for index, chunk in enumerate(chunks):
        if (
            chunk["tool_call_id"] != case["tool_id"]
            or chunk["chunk_index"] != index
            or chunk["byte_offset"] != offset
            or chunk["byte_length"] <= 0
            or not observed <= chunk["chunk_time"] <= finished
        ):
            raise ValueError("Invalid tool chunk order, range or observation time")
        offset += chunk["byte_length"]
        observed = chunk["chunk_time"]
        first = chunk["chunk_time"] if first is None else first
    if offset != raw.stat().st_size:
        raise ValueError("Tool chunks do not cover raw output")
    status = json.loads((producer / "control_export_status.json").read_text())
    if not status["flush_complete"] or status["dropped"] or status["export_errors"]:
        raise ValueError("Tool control export is incomplete")
    seconds = (finished - started) / 1e9
    censored = bool(timing["timed_out"])
    bound = min(seconds, job["timeout_seconds"]) if censored else seconds
    return {
        "task": job["task"],
        "case_root": str(root.resolve()),
        "clock_domain_id": metadata["clock_domain_id"],
        "source_hashes": {
            str(p.relative_to(root.resolve())): sha256_file(p)
            for p in (
                root.resolve() / "case.json",
                raw,
                producer / "producer.json",
                producer / "tool_output_chunks.jsonl",
                producer / "control_export_status.json",
                producer / "control_events.jsonl",
            )
        },
        "job_id": job["job_id"],
        "seconds": bound,
        "observed_seconds": seconds,
        "censored": censored,
        "chunks": len(chunks),
        "early_lead": None if first is None else max(0, bound - (first - started) / 1e9),
        "outcome": "timeout" if censored else "success" if case["returncode"] == 0 else "failure",
    }


def coverage(records, minima):
    values = {"task_groups": len({r["task"] for r in records}), "tools": len(records)}
    for threshold in (5, 30, 90):
        values[f"gt{threshold}_seconds"] = sum(r["seconds"] > threshold for r in records)
    values["completed_gt30_seconds"] = sum(r["seconds"] > 30 and not r["censored"] for r in records)
    for name, threshold in (("early_lead_gt0_5_seconds", 0.5), ("early_lead_gt5_seconds", 5)):
        values[name] = sum(
            r["early_lead"] is not None and r["early_lead"] > threshold for r in records
        )
    values.update(
        multi_chunk=sum(r["chunks"] >= 2 for r in records),
        zero_chunk=sum(r["chunks"] == 0 for r in records),
    )
    values.update(
        {
            state: sum(r["outcome"] == state for r in records)
            for state in ("success", "failure", "timeout")
        }
    )
    return {
        "actual": values,
        "minima": minima,
        "passed": all(values[key] >= count for key, count in minima.items()),
        "unmet": {
            key: {"actual": values[key], "minimum": count}
            for key, count in minima.items()
            if values[key] < count
        },
    }


def collect(protocol_path, output, *, command):
    from tokenmoe_collect.minisweagent_adapter import TracedEnvironment
    from tokenmoe_collect.records import RecordStore

    protocol = read_manifest(protocol_path, "experiment")
    out = Output(output, kind="experiment", identity=protocol["identity"], command=command)
    frozen_hash = sha256_file(protocol_path)
    out.write("protocol.json", protocol)
    stop, lock, active = threading.Event(), threading.Lock(), {}
    prefix = runtime_command(protocol["runtime"])

    def cleanup(sid):
        with lock:
            env = active.pop(sid, None)
        if env is not None:
            try:
                env.cleanup()
            except BaseException:
                with lock:
                    active[sid] = env
                raise

    def one(job):
        directory = out.path / "jobs" / job["job_id"]
        directory.mkdir(parents=True, exist_ok=False)
        sid, tid = new_id("ses"), new_id("tc")
        case = {**job, "status": "setup-failed", "tool_id": tid, "host_load": os.getloadavg()}
        try:
            if stop.is_set():
                raise InterruptedError("Corpus collection cancelled")
            actual = subprocess.check_output(
                [*prefix, "image", "inspect", job["image"], "--format", "{{.Id}}"], text=True
            ).strip()
            if actual != job["image_id"]:
                raise ValueError("Frozen task image changed")
            with RecordStore(directory, control_events=True) as store:
                case["producer"] = str(store.directory.relative_to(directory))
                options = copy.deepcopy(minisweagent_config()["environment"])
                options.pop("environment_class", None)
                options.update(
                    image=job["image"],
                    pull_timeout=60,
                    timeout=job["timeout_seconds"],
                    run_args=container_run_args(protocol["limits"]),
                )
                options.setdefault("env", {}).update(
                    CUDA_VISIBLE_DEVICES="",
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MPLBACKEND="Agg",
                )
                env = TracedEnvironment(
                    store=store, session_id=sid, runtime=protocol["runtime"], **options
                )
                with lock:
                    active[sid] = env
                try:
                    if stop.is_set():
                        raise InterruptedError("Corpus collection cancelled")
                    import tokenmoe_collect.container_probe as probe_source

                    probe = json.loads(
                        subprocess.check_output(
                            [
                                *prefix,
                                "exec",
                                env.container_id,
                                "python3",
                                "-c",
                                Path(probe_source.__file__).read_text(),
                            ],
                            text=True,
                            timeout=30,
                        )
                    )
                    validate_limits(protocol["limits"], probe)
                    if probe["boot_id"] != store.clock["clock_domain_id"]:
                        raise ValueError("Tool container and collector clocks differ")
                    case["preflight"] = probe
                    subprocess.run(
                        [
                            *prefix,
                            "exec",
                            "-w",
                            "/testbed",
                            env.container_id,
                            "git",
                            "reset",
                            "--hard",
                            job["base_commit"],
                        ],
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
                    subprocess.run(
                        [
                            *prefix,
                            "exec",
                            "-i",
                            "-w",
                            "/testbed",
                            env.container_id,
                            "git",
                            "apply",
                            "--whitespace=nowarn",
                            "-",
                        ],
                        input=job["test_patch"].encode(),
                        check=True,
                        capture_output=True,
                        timeout=30,
                    )
                    emit(
                        store,
                        "session_started",
                        session_id=sid,
                        benchmark_item_id=job["task"],
                        identity=protocol["identity"],
                        authored_workflow=True,
                    )
                    emit(
                        store,
                        "tools_issued",
                        session_id=sid,
                        dependencies=[tid],
                        dependency_mode="sequential",
                        tools=[
                            {"tool_id": tid, "command": job["command"], "command_truncated": False}
                        ],
                    )
                    result, timing = env.executor.execute(
                        job["command"], tid, job["timeout_seconds"]
                    )
                    case.update(
                        status="cancelled" if stop.is_set() else "complete",
                        timing=timing,
                        returncode=result["returncode"],
                        exception=result.get("extra"),
                    )
                    emit(store, "session_ended", session_id=sid, status="completed")
                finally:
                    cleanup(sid)
        except BaseException as exc:
            case["status"] = "failed"
            case["error"] = f"{type(exc).__name__}: {exc}"
        write_json(directory / "case.json", case)
        return job

    def interrupt(*_):
        raise InterruptedError("Corpus collection interrupted")

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=protocol["concurrency"])
    try:
        futures = [executor.submit(one, job) for job in protocol["jobs"]]
        completed = []
        for future in concurrent.futures.as_completed(futures):
            job = future.result()
            completed.append(job["job_id"])
            out.write("progress.json", {"completed": completed, "total": len(protocol["jobs"])})
        records = [validate_case(out.path / "jobs" / j["job_id"], j) for j in protocol["jobs"]]
        records = [r for r in records if r is not None]
        checked = coverage(records, protocol["coverage_minima"])
        out.write("coverage.json", checked)
        out.write("tools.json", records)
        if active:
            raise RuntimeError("Tool collection has unreleased owned containers")
        if (
            frozen_hash != sha256_file(protocol_path)
            or source_version()["files"] != out.record["source"]["files"]
        ):
            raise ValueError("Protocol or collector source changed during collection")
        out.finish(
            evidence="real-test-build-tool-corpus",
            coverage_passed=checked["passed"],
            agent_policy_gain_measured=False,
            protocol_sha256=frozen_hash,
        )
    except BaseException as exc:
        stop.set()
        with lock:
            identifiers = list(active)
        cleanup_errors = []
        for sid in identifiers:
            try:
                cleanup(sid)
            except Exception as error:
                cleanup_errors.append(str(error))
        out.record["cleanup_errors"] = cleanup_errors
        out.fail(exc)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return out.path


def report(directory, protocol_path, source_snapshot, output, *, command):
    directory = Path(directory).resolve()
    record = read_manifest(directory / "manifest.json", "experiment")
    protocol = read_manifest(protocol_path, "experiment")
    if record["identity"] != protocol["identity"]:
        raise ValueError("Corpus and protocol identities differ")
    if sha256_file(directory / "protocol.json") != sha256_file(protocol_path):
        raise ValueError("Corpus protocol differs from the frozen source")
    if record["status"] != "complete" and record.get("error") != (
        "RuntimeError: Tool collection has unreleased owned containers"
    ):
        raise ValueError("Corpus has an unsupported collection failure")
    actual_source = source_version(Path(source_snapshot))
    if (
        actual_source["files"] != record["source"]["files"]
        or actual_source["fork_commit"] != record["source"]["fork_commit"]
    ):
        raise ValueError("Collector source snapshot changed")
    names = sorted(
        {
            name
            for error in record.get("cleanup_errors", [])
            for name in re.findall(r"tokenmoe-ses_[a-zA-Z0-9_]+", error)
        }
    )
    if record.get("cleanup_errors") and not names:
        raise ValueError("Cleanup failures have no verifiable owned container identity")
    if names and protocol["runtime"] != "podman":
        raise ValueError("Cleanup reconciliation currently requires Podman")
    for name in names:
        exists = subprocess.run(
            [*runtime_command("podman"), "container", "exists", name],
            capture_output=True,
            timeout=10,
        )
        if exists.returncode != 1:
            raise ValueError(f"Owned container is still present or runtime unavailable: {name}")
    previous = {r["job_id"]: r for r in json.loads((directory / "tools.json").read_text())}
    records, incomplete = [], []
    for job in protocol["jobs"]:
        case_root = directory / "jobs" / job["job_id"]
        row = validate_case(case_root, job)
        if row is None:
            case = json.loads((case_root / "case.json").read_text())
            incomplete.append(
                {
                    "job_id": job["job_id"],
                    "task": job["task"],
                    "source": str(case_root),
                    "error": case.get("error"),
                }
            )
            continue
        if row["source_hashes"] != previous[row["job_id"]]["source_hashes"]:
            raise ValueError("Tool observations changed after original validation")
        records.append(row)
    checked = coverage(records, protocol["coverage_minima"])
    out = Output(output, kind="result", identity=protocol["identity"], command=command)
    out.write("protocol.json", protocol)
    out.write("tools.json", records)
    out.write("coverage.json", checked)
    out.write(
        "validation.json",
        {
            "source": str(directory),
            "source_status": record["status"],
            "source_manifest_sha256": sha256_file(directory / "manifest.json"),
            "protocol_sha256": sha256_file(protocol_path),
            "source_snapshot_verified": str(source_snapshot),
            "reconciled_absent_containers": names,
            "incomplete_captures": incomplete,
            "selection": "all complete individually validated original captures; failed capture retained and excluded",
        },
    )
    lines = [
        "# Validated real test/build streams",
        "",
        f"Valid captures: {len(records)}. Incomplete captures retained: {len(incomplete)}.",
        f"Original collection status: {record['status']}. Source snapshot and raw hashes verified.",
        "",
        "| Coverage metric | Actual | Frozen minimum |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {key} | {checked['actual'][key]} | {minimum} |"
        for key, minimum in checked["minima"].items()
    )
    lines.extend(
        [
            "",
            "The incomplete capture is excluded from model fitting; completed test failures remain included.",
            "These authored tool workflows provide no solved-agent-task or prefetch-speedup evidence.",
        ]
    )
    (out.path / "report.md").write_text("\n".join(lines) + "\n")
    out.finish(
        evidence="validated-real-test-build-corpus",
        coverage_passed=checked["passed"],
        valid_tools=len(records),
        incomplete_tools=len(incomplete),
        agent_policy_gain_measured=False,
    )
    return out.path
