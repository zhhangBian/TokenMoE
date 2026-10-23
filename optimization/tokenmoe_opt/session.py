"""Receipt-time event state with conservative gaps and dependency joins."""

import copy
from collections import deque

from tokenmoe_collect.clock import clock_metadata, now
from tokenmoe_collect.events import project


class SessionState:
    def __init__(self, *, clock_domain=None):
        self.clock_domain = clock_domain or clock_metadata()["clock_domain_id"]
        self.states, self.sequences, self.owners = {}, {}, {}
        self.uncertain = set()
        self.evidence = deque(maxlen=4096)
        self.latest = {}

    def receive(self, event, *, received_at=None):
        receipt = now() if received_at is None else received_at
        if event.get("schema_version") != 1 or event.get("clock_domain_id") != self.clock_domain:
            raise ValueError("Unsupported event version or incomparable clock domain")
        if event["source_at"] > receipt:
            raise ValueError("Event observation is later than controller receipt")
        producer, sequence = event["producer_id"], event["sequence"]
        row = {**event, "receipt_at": receipt, "available_at": max(receipt, event["source_at"])}
        previous = self.sequences.get(producer, 0)
        if sequence <= previous:
            row["disposition"] = "duplicate_or_stale"
            self.evidence.append(row)
            return False
        if sequence != previous + 1 or event.get("dropped_before"):
            self.uncertain.add(producer)
        if "snapshot" in event:
            if event.get("snapshot_at_sequence") != sequence:
                raise ValueError("Snapshot references a different event sequence")
            for sid in list(self.states):
                if self.owners.get(sid) == producer and sid not in event["snapshot"]:
                    del self.states[sid]
            for sid, state in event["snapshot"].items():
                old = self.states.get(sid, {})
                if not old.get("terminal"):
                    self.states[sid] = copy.deepcopy(state)
                self.owners[sid] = producer
            self.uncertain.discard(producer)
        sid = event.get("session_id")
        if sid is not None:
            self.owners[sid] = producer
            project(self.states, event["kind"], event)
        self.sequences[producer] = sequence
        row["disposition"] = "conservative_gap" if producer in self.uncertain else "applied"
        self.evidence.append(row)
        self.latest[producer] = row
        return True

    def decision(self, session_id, *, decision_at):
        state = self.states.get(session_id)
        if state is None:
            return {"predictive_demand": False, "reason": "unknown_session"}
        producer = self.owners[session_id]
        latest = self.latest.get(producer)
        if latest is None or latest["available_at"] > decision_at:
            raise ValueError(
                "Decision precedes an applied controller receipt; replay events chronologically"
            )
        if state["terminal"] or state.get("unknown") or producer in self.uncertain:
            return {"predictive_demand": False, "reason": "terminal_or_event_gap"}
        dependencies = state.get("dependencies", [])
        joined = all(state["tools"].get(t, {}).get("status") == "completed" for t in dependencies)
        return {
            "predictive_demand": True,
            "dependencies_complete": joined,
            "can_submit": joined and state.get("status") in {"waiting_tools", "created"},
            "gpu_executing": False,
            "scheduler_state": "must be observed separately",
            "visible_sequence": self.sequences[producer],
            "available_at": latest["available_at"],
            "state": copy.deepcopy(state),
        }
