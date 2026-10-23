"""Causal shared residency decisions and bounded local worker delivery."""

import json
import socket
from collections import defaultdict
from pathlib import Path

import numpy as np
from tokenmoe_collect.clock import now

from .artifacts import read_manifest
from .data import PHASES, Sample
from .features import command_categories
from .planner import Demand, Ledger, aggregate, plan
from .predictors import History, Priors
from .session import SessionState


class FrozenPredictor:
    def __init__(self, artifact):
        self.artifact = artifact
        self.identity = artifact["identity"]
        parameters = artifact["parameters"]
        self.prior = Priors.__new__(Priors)
        self.prior.min_support = parameters["min_support"]
        self.prior.global_hist = np.asarray(parameters["global_hist"], dtype=np.int64)
        self.prior.training_requests = parameters["training_requests"]
        self.prior.tables, self.prior.support = {}, defaultdict(int)
        for row in parameters["tables"]:
            key = (row["kind"], tuple(row["key"]) if row["kind"] == "tool" else row["key"])
            self.prior.tables[key] = np.asarray(row["hist"], dtype=np.int64)
            self.prior.support[key] = row["support"]
        self.history = History(alpha=artifact["history_alpha"], ewma_alpha=artifact["ewma_alpha"])
        shape = (3, len(self.identity["layers"]), self.identity["experts"])
        if self.prior.global_hist.shape != shape or (self.prior.global_hist < 0).any():
            raise ValueError("Predictor shape differs from its model identity")

    def matches(self, identity):
        return all(
            identity.get(key) == self.identity.get(key)
            for key in (
                "model_id",
                "weight_revision",
                "tokenizer_hash",
                "template_id",
                "template_version",
            )
        )

    def features(self, state, decision_at):
        categories = set()
        for tool_id in state.get("dependencies", []):
            tool = state["tools"].get(tool_id, {})
            if tool.get("command_truncated") or "command" not in tool:
                categories.add("unknown")
            else:
                categories.update(
                    command_categories(tool["command"], self.artifact["label_version"])
                )
        return {
            "stage": "first" if state.get("request_count", 0) == 0 else "subsequent",
            "tool_key": tuple(sorted(categories)),
            "decision_at": decision_at,
        }

    def volume(self, target, policy):
        kind = "tool" if policy.endswith("+tool") else "stage" if "stage" in policy else "global"
        _, provenance = self.prior.distribution(target, kind)
        key = (kind, target.features["tool_key"] if kind == "tool" else target.features["stage"])
        table = (
            self.prior.tables[key] if provenance["source"] != "global" else self.prior.global_hist
        )
        support = (
            self.prior.support[key]
            if provenance["source"] != "global"
            else len(self.prior.training_requests)
        )
        volume = table.sum(axis=-1)[:, 0] / self.identity["top_k"] / support
        previous = self.history.previous.get(target.session)
        if previous is not None and (policy == "previous" or "+" in policy):
            old = previous.hist.sum(axis=-1)[:, 0] / self.identity["top_k"]
            alpha = 1.0 if policy == "previous" else self.history.alpha
            volume = np.where(old > 0, alpha * old + (1 - alpha) * volume, volume)
        return volume


class FeedbackHistory:
    """Canonical-rank compact counts joined with observed request completion."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.requests, self.aliases = {}, {}
        self.client_requests = {}
        self.unreliable = set()
        self.sequences = {}
        self.completed = {}

    def client(self, event, receipt):
        rid = event.get("request_id")
        if event["kind"] == "request_created":
            self.client_requests[rid] = {
                "session": event["session_id"],
                "decision": event["source_at"],
            }
        elif event["kind"] == "request_completed" and rid in self.client_requests:
            self.client_requests[rid]["observed"] = receipt
        self._advance()

    def engine(self, event, receipt):
        rid = event.get("request_id")
        if event["kind"] == "request_received":
            self.aliases[rid] = event.get("llm_request_id")
        elif event["kind"] == "request_finished":
            self.requests.setdefault(rid, {})["finished"] = {**event, "receipt": receipt}
        self._advance()

    def counts(self, event, receipt):
        if event.get("rank") != 0:
            return
        stream = event["engine_id"]
        previous = self.sequences.get(stream, 0)
        if event["sequence"] <= previous:
            return
        if event["sequence"] != previous + 1:
            self.unreliable.add(stream)
        self.sequences[stream] = event["sequence"]
        if event["kind"] != "feedback":
            return
        rid, layer, phase = event["request_id"], event["layer"], event["phase"]
        if self.aliases.get(rid, event.get("llm_request_id")) in self.completed:
            return
        if phase == "recompute":
            return
        row = self.requests.setdefault(rid, {})
        row.setdefault("engine", stream)
        row.setdefault("hist", np.zeros_like(self.predictor.prior.global_hist))
        row.setdefault("decode_positions", defaultdict(int))
        row["receipt"] = receipt
        alias = event.get("llm_request_id")
        if alias:
            self.aliases[rid] = alias
        index = PHASES.index(phase)
        counts = np.asarray(event["counts"], dtype=np.int64)
        if counts.shape != (self.predictor.identity["experts"],) or (counts < 0).any():
            raise ValueError("Malformed compact feedback histogram")
        row["hist"][index, layer] += counts
        if phase == "decode":
            positions = event["end"] - event["start"]
            if row["decode_positions"][layer] + positions <= 32:
                row["hist"][2, layer] += counts
            row["decode_positions"][layer] += positions
        self._advance()

    def _advance(self):
        for rid, row in list(self.requests.items()):
            alias = self.aliases.get(rid)
            client = self.client_requests.get(alias)
            if (
                alias in self.completed
                or "finished" not in row
                or "hist" not in row
                or not client
                or "observed" not in client
            ):
                continue
            if row["engine"] in self.unreliable:
                continue
            hist = row["hist"]
            totals = hist.sum(axis=-1)
            expected_decode = (
                max(0, row["finished"]["num_output_tokens"] - 1) * self.predictor.identity["top_k"]
            )
            if not np.all(totals[1] == expected_decode) or not np.all(totals == totals[:, :1]):
                continue
            sample = Sample(
                "live",
                client["session"],
                alias,
                client["decision"],
                max(client["observed"], row["receipt"], row["finished"]["receipt"]),
                hist.copy(),
                np.zeros_like(hist),
                {},
                Path("/live"),
                {},
            )
            self.predictor.history.defer(sample)
            self.completed[alias] = None
            if len(self.completed) > 4096:
                del self.completed[next(iter(self.completed))]
            self.client_requests.pop(alias, None)
            self.aliases.pop(rid, None)
            del self.requests[rid]


class ResidencyController:
    def __init__(self, predictor, durations, cost_profile, *, policy="previous+tool", horizon=0.5):
        if policy not in {
            "global",
            "stage",
            "previous",
            "ewma",
            "previous+global",
            "previous+stage",
            "previous+tool",
        }:
            raise ValueError("Runtime policy must be deployable from compact observations")
        if not cost_profile.get("measured"):
            raise ValueError("A measured copy/compute cost profile is required")
        if (
            not 0 < horizon <= 5
            or cost_profile.get("decode_token_seconds", 0) <= 0
            or cost_profile.get("prefill_seconds", -1) < 0
        ):
            raise ValueError("Finite horizon and measured phase service times are required")
        self.predictor, self.durations, self.cost = predictor, durations, cost_profile
        self.policy, self.horizon = policy, horizon
        self.sessions = SessionState()
        self.feedback = FeedbackHistory(predictor)
        self.running, self.waiting = set(), set()
        self.epoch = 0
        self.residency_epoch = {}
        self.rejected_sessions = set()
        self.engine_sequences = {}
        self.engine_uncertain = set()

    def event(self, event, *, receipt=None):
        receipt = now() if receipt is None else receipt
        if self.sessions.receive(event, received_at=receipt):
            if event["kind"] == "session_started" and event.get(
                "benchmark_item_id"
            ) in self.predictor.artifact.get("training_tasks", []):
                self.rejected_sessions.add(event["session_id"])
            self.feedback.client(event, receipt)

    def engine_event(self, event, *, receipt=None):
        receipt = now() if receipt is None else receipt
        if event["clock_domain_id"] != self.sessions.clock_domain or event["source_at"] > receipt:
            raise ValueError("Engine event is not available in the controller clock")
        engine, sequence = event["engine_id"], event["sequence"]
        previous = self.engine_sequences.get(engine, 0)
        if sequence <= previous:
            return
        if sequence != previous + 1:
            self.engine_uncertain.add(engine)
        self.engine_sequences[engine] = sequence
        if event["kind"] == "schedule":
            self.running, self.waiting = set(event["running"]), set(event["waiting"])
            self.engine_uncertain.discard(engine)
        elif event["kind"] == "request_received":
            self.waiting.add(event["request_id"])
        elif event["kind"] == "request_finished":
            self.running.discard(event["request_id"])
            self.waiting.discard(event["request_id"])
        self.feedback.engine(event, receipt)

    def _arrival(self, state, horizon, decision_at):
        tools = [state["tools"].get(t, {}) for t in state.get("dependencies", [])]
        if not tools or all(t.get("status") == "completed" for t in tools):
            return 1.0
        remaining = [t for t in tools if t.get("status") != "completed"]
        # Sequential queues reserve an equal portion of the horizon per unresolved
        # dependency, a conservative declared bound on completion of the sum.
        window = (
            horizon / len(remaining) if state.get("dependency_mode") == "sequential" else horizon
        )
        probability = 1.0
        for tool in remaining:
            categories = command_categories(
                tool.get("command", ""), self.predictor.artifact["label_version"]
            )
            elapsed = max(0, (decision_at - tool.get("started_at", decision_at)) / 1e9)
            prediction = self.durations.predict("+".join(categories), elapsed=elapsed)
            if not prediction["prefetch_supported"]:
                return 0.0
            probability *= max(
                (cdf for time, cdf in prediction["cdf"] if time <= window), default=0
            )
        return probability

    def snapshots_ready(self, snapshots, *, decision_at):
        if not snapshots:
            return False
        for snapshot in snapshots:
            if (
                snapshot["clock_domain_id"] != self.sessions.clock_domain
                or not 0 <= decision_at - snapshot["source_at"] <= 500_000_000
                or not snapshot["consistent_idle_snapshot"]
            ):
                return False
            if snapshot["model_hash"] != self.cost["model_hash"]:
                raise ValueError("Measured costs refer to another model/backend")
        if {s["device_uuid"] for s in snapshots} != set(self.cost["device_uuids"]):
            raise ValueError("Measured costs refer to another TP/device group")
        return True

    def decide(self, snapshots, *, decision_at=None):
        decision_at = now() if decision_at is None else decision_at
        if self.running or self.waiting or self.engine_uncertain or self.feedback.unreliable:
            return None
        if not self.snapshots_ready(snapshots, decision_at=decision_at):
            return None
        identity = self.predictor.identity
        self.epoch += 1
        resident, pinned, inflight = {}, defaultdict(set), set()
        rank_bytes, free = {}, {}
        for snapshot in snapshots:
            rank, size = snapshot["rank"], snapshot["bytes_per_expert"]
            rank_bytes[rank], free[rank] = size, 0
            for layer, cache in enumerate(snapshot["caches"]):
                ready = {s["expert"] for s in cache["slots"] if s["state"] in {"READY", "IN_USE"}}
                resident[layer] = ready if layer not in resident else resident[layer] & ready
                for slot in cache["slots"]:
                    if slot["state"] in {"LOADING", "IN_USE"}:
                        pinned[layer].add(slot["expert"])
                    else:
                        free[rank] += size
                    if slot["state"] == "LOADING":
                        inflight.add((rank, layer, slot["expert"]))
            free[rank] = min(free[rank], snapshot["speculative_bytes_available"])
        for layer, experts in resident.items():
            for expert in experts:
                self.residency_epoch.setdefault((layer, expert), self.epoch)
        self.residency_epoch = {
            key: value
            for key, value in self.residency_epoch.items()
            if key[1] in resident.get(key[0], ())
        }
        demands = []
        shape = (len(identity["layers"]), identity["experts"])
        for sid, state in self.sessions.states.items():
            if sid in self.rejected_sessions:
                continue
            decision = self.sessions.decision(sid, decision_at=decision_at)
            if not decision["predictive_demand"] or not self.predictor.matches(
                state.get("identity") or {}
            ):
                continue
            if state.get("status") not in {"active", "waiting_tools"}:
                continue
            features = self.predictor.features(state, decision_at)
            target = Sample(
                "live",
                sid,
                f"future-{sid}",
                decision_at,
                decision_at,
                np.zeros_like(self.predictor.prior.global_hist),
                np.zeros_like(self.predictor.prior.global_hist),
                features,
                Path("/live"),
                {},
            )
            dist, _ = self.predictor.history.predict(target, self.predictor.prior, self.policy)
            volume = self.predictor.volume(target, self.policy)
            for phase in (0, 1):
                lead = (
                    self.horizon
                    if phase == 0
                    else max(0, self.horizon - self.cost["prefill_seconds"])
                )
                tokens = (
                    float(volume[phase])
                    if phase == 0
                    else min(float(volume[phase]), 32, lead / self.cost["decode_token_seconds"])
                )
                demands.append(
                    Demand(
                        sid,
                        identity["id"],
                        dist[phase],
                        tokens,
                        self._arrival(state, lead, decision_at),
                        1,
                        "waiting_tools",
                    )
                )
        expected, sources = aggregate(
            demands, identity=identity["id"], shape=shape, top_k=identity["top_k"]
        )
        if not expected.any():
            return None
        limit = sum(s["speculative_bytes_available"] for s in snapshots)
        ledger = Ledger(
            free,
            int(self.cost["shared_bandwidth_bytes_second"] * self.horizon),
            limit,
            sum(s["inflight_limit"] for s in snapshots),
            rank_bytes,
            inflight=inflight,
        )
        cost = {**self.cost, "identity": identity["id"]}
        capacity = len(snapshots[0]["caches"][0]["slots"])
        result = plan(
            expected,
            identity=identity["id"],
            epoch=self.epoch,
            now_ns=decision_at,
            expiry_ns=decision_at + 5_000_000_000,
            capacity=capacity,
            resident=resident,
            pinned=pinned,
            demand={},
            residency_epoch=self.residency_epoch,
            ledger=ledger,
            cost_profile=cost,
            sources=sources,
        )
        result["visible_sequences"] = dict(self.sessions.sequences)
        result["arrival_assumption"] = (
            "independent durations; sequential equal-horizon sufficient condition"
        )
        return result


def deliver(decision, snapshots, directory):
    delivered = []
    for snapshot in snapshots:
        packet = {
            key: decision[key]
            for key in (
                "schema_version",
                "epoch",
                "clock_domain_id",
                "expires_at",
                "desired",
                "transfers",
            )
        }
        packet.update(
            engine_id=snapshot["engine_id"],
            model_hash=snapshot["model_hash"],
            backend_hash=snapshot["backend_hash"],
        )
        payload = json.dumps(packet).encode()
        if len(payload) > 65536:
            raise ValueError("Plan exceeds the bounded worker message size")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
                sender.setblocking(False)
                sender.sendto(payload, str(Path(directory) / f"rank-{snapshot['rank']}.sock"))
            delivered.append(snapshot["rank"])
        except (BlockingIOError, FileNotFoundError, ConnectionRefusedError):
            pass
    return delivered


def load_predictor(path):
    return FrozenPredictor(read_manifest(path, "predictor"))
