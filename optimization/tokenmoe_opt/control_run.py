"""Single-engine CPU controller loop over append-only local observation files."""

import json
import signal
import threading
import time
from pathlib import Path

from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.clock import now
from tokenmoe_collect.jsonl import JsonlWriter

from .arrival import Survival
from .arrival_eval import duration_observations
from .artifacts import Output, input_files, read_manifest
from .controller import ResidencyController, deliver, load_predictor


class Tails:
    def __init__(self):
        self.offsets = {}

    def read(self, paths):
        for path in sorted(paths):
            with path.open("rb") as stream:
                if path.stat().st_size < self.offsets.get(path, 0):
                    raise RuntimeError(f"Append-only observation file was truncated: {path}")
                stream.seek(self.offsets.get(path, 0))
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    self.offsets[path] = stream.tell()
                    yield json.loads(line)


def run(
    predictor_path,
    inputs_path,
    cost_path,
    agent_run,
    observations,
    workers,
    output,
    *,
    command,
    duration=60,
    policy="previous+tool",
    interval=0.1,
):
    if duration <= 0 or not 0.02 <= interval <= 1:
        raise ValueError("Positive controller duration and a 0.02..1 second interval are required")
    lifetime_wall, lifetime_cpu = time.monotonic_ns(), time.process_time_ns()
    predictor = load_predictor(predictor_path)
    inputs = read_manifest(inputs_path, "input")
    cost = read_manifest(cost_path, "cost-profile")
    if predictor.identity != inputs["identity"] or predictor.identity != cost["identity"]:
        raise ValueError("Controller input/predictor/cost model identities differ")
    if "training_tasks" not in predictor.artifact:
        raise ValueError("Regenerate predictor artifacts with training-task provenance")
    roots = [Path(root) for root in inputs["sources"]]
    if input_files(roots) != inputs["sources"]:
        raise ValueError("Duration training sources changed")
    durations = [
        record
        for _, record in duration_observations(
            roots, label_version=predictor.artifact["label_version"]
        )
        if record.task in predictor.artifact["training_tasks"]
    ]
    controller = ResidencyController(
        predictor, Survival(durations), cost["controller"], policy=policy
    )
    out = Output(output, kind="experiment", identity=predictor.identity, command=command)
    out.write(
        "inputs.json",
        {
            "source_hashes": {
                str(p.resolve()): sha256_file(p) for p in (predictor_path, inputs_path, cost_path)
            },
            "duration_training_tasks": sorted({d.task for d in durations}),
            "policy": policy,
            "interval_seconds": interval,
            "duration_seconds": duration,
        },
    )
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    tails, last, snapshots = Tails(), None, []
    engine_id, decisions, sends = None, 0, 0
    ready = False
    started = time.monotonic()

    def send(decision):
        return deliver(decision, snapshots, workers)

    def cancel():
        controller.epoch += 1
        return {
            "schema_version": 1,
            "epoch": controller.epoch,
            "clock_domain_id": controller.sessions.clock_domain,
            "created_at": now(),
            "expires_at": now() + 100_000_000,
            "desired": {},
            "transfers": [],
            "reason": "withdraw stale predictions; existing DMA remains tracked",
        }

    try:
        with (
            JsonlWriter(out.path / "plans.jsonl") as writer,
            JsonlWriter(out.path / "cycles.jsonl") as cycles,
        ):
            while not stop.is_set() and time.monotonic() - started < duration:
                cycle = time.monotonic_ns()
                cpu = time.process_time_ns()
                snapshots = [
                    json.loads(path.read_text())
                    for path in sorted(Path(workers).glob("rank-*.state.json"))
                ]
                if snapshots:
                    ids = {s["engine_id"] for s in snapshots}
                    if len(ids) != 1 or (engine_id is not None and engine_id not in ids):
                        raise RuntimeError(
                            "Worker engine lifetime changed; start a separate controller run"
                        )
                    engine_id = ids.pop()
                    controller.epoch = max(controller.epoch, *(s["plan_epoch"] for s in snapshots))
                for event in tails.read(Path(agent_run).glob("raw/harness/*/control_events.jsonl")):
                    controller.event(event)
                if engine_id is not None:
                    for event in tails.read(
                        (Path(observations) / engine_id).glob("token-readiness.jsonl")
                    ):
                        controller.engine_event(event)
                    for event in tails.read(
                        (Path(observations) / engine_id).glob("cache-rank-0.jsonl")
                    ):
                        controller.feedback.counts(event, now())
                decision = controller.decide(snapshots)
                if (
                    not ready
                    and not controller.engine_uncertain
                    and not controller.feedback.unreliable
                    and controller.snapshots_ready(snapshots, decision_at=now())
                ):
                    out.write("ready.json", {"engine_id": engine_id, "ready_at": now()})
                    ready = True
                if decision is not None:
                    decisions += 1
                    last = decision
                elif last is not None and (
                    controller.engine_uncertain
                    or controller.feedback.unreliable
                    or any(
                        source["session"] not in controller.sessions.states
                        or not controller.sessions.decision(source["session"], decision_at=now())[
                            "predictive_demand"
                        ]
                        for source in last["sources"]
                    )
                ):
                    decision, last = cancel(), None
                if decision is not None and snapshots:
                    delivered = send(decision)
                    sends += len(delivered)
                    writer.write(
                        {
                            **decision,
                            "delivered_ranks": delivered,
                            "controller_wall_ns": time.monotonic_ns() - cycle,
                        }
                    )
                finished = time.monotonic_ns()
                cycles.write(
                    {
                        "started_at": cycle,
                        "finished_at": finished,
                        "clock_domain_id": controller.sessions.clock_domain,
                        "cpu_ns": time.process_time_ns() - cpu,
                        "wall_ns": finished - cycle,
                        "planned": decision is not None,
                        "running_requests": len(controller.running),
                        "waiting_requests": len(controller.waiting),
                    }
                )
                stop.wait(max(0, interval - (time.monotonic_ns() - cycle) / 1e9))
        out.finish(
            evidence="live-controller-diagnostic",
            decisions=decisions,
            renewals=0,
            active_plan_policy="original lease expires; fresh idle prediction required to renew",
            delivered_rank_plans=sends,
            rejected_training_sessions=sorted(controller.rejected_sessions),
            process_cpu_seconds=(time.process_time_ns() - lifetime_cpu) / 1e9,
            process_wall_seconds=(time.monotonic_ns() - lifetime_wall) / 1e9,
            policy_comparison_status="unqualified: held-out controlled service evaluation outstanding",
        )
        return out.path
    except Exception as exc:
        out.fail(exc)
        raise
    finally:
        if snapshots:
            send(cancel())
