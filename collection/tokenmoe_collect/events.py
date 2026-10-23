"""Optional bounded control export; authoritative capture remains independent."""

import copy
import json
import queue
import threading

from .clock import clock_metadata, now
from .ids import new_id
from .jsonl import JsonlWriter


def emit(store, kind, **fields):
    sink = getattr(store, "event_sink", None)
    if sink is not None:
        sink.emit(kind, **fields)


def project(states, kind, fields):
    sid = fields.get("session_id")
    if sid is None:
        return
    state = states.setdefault(
        sid,
        {"terminal": False, "tools": {}, "dependencies": [], "unknown": kind != "session_started"},
    )
    if state["terminal"]:
        return
    if kind == "session_started":
        state.update(identity=fields.get("identity"), status="active", unknown=False,
                     task=fields.get("benchmark_item_id"))
    elif kind in {"session_ended", "session_cancelled"}:
        state.update(
            terminal=True, status=fields.get("status", "cancelled"), tools={}, dependencies=[]
        )
    elif kind == "request_created":
        if state.get("request") != fields["request_id"]:
            state["request_count"] = state.get("request_count", 0) + 1
        dependencies = fields.get("dependencies", [])
        state.update(request=fields["request_id"], status="created", dependencies=dependencies)
        state["tools"] = {
            key: value for key, value in state["tools"].items() if key in dependencies
        }
    elif kind == "request_sent":
        state["status"] = "queued"
    elif kind == "request_completed":
        state["status"] = "awaiting_actions"
    elif kind == "tools_issued":
        state.update(
            status="waiting_tools",
            dependencies=fields["dependencies"],
            dependency_mode=fields.get("dependency_mode", "unknown"),
        )
        for tool in fields["dependencies"]:
            state["tools"].setdefault(tool, {"status": "pending", "observed_bytes": 0})
        for tool in fields.get("tools", []):
            state["tools"][tool["tool_id"]].update(tool)
    elif kind.startswith("tool_"):
        tool = state["tools"].setdefault(fields["tool_id"], {"observed_bytes": 0})
        if kind == "tool_started":
            tool.update(status="running", started_at=fields["source_at"])
        elif kind == "tool_output":
            tool["observed_bytes"] = max(
                tool["observed_bytes"], fields["byte_offset"] + fields["byte_length"]
            )
        elif kind == "tool_completed":
            tool.update(
                status="completed", outcome=fields.get("outcome"), finished_at=fields["source_at"]
            )


class BoundedEventSink:
    def __init__(self, path, *, capacity=256, max_sessions=512, writer=None):
        if capacity < 1 or max_sessions < 1:
            raise ValueError("Event capacity and session limit must be positive")
        self.clock = clock_metadata()
        self.producer = new_id("evtproducer")
        self.queue = queue.Queue(maxsize=capacity)
        self.max_sessions = max_sessions
        self.states = {}
        self.lock = threading.Lock()
        self.sequence = 0
        self.dropped = self.export_errors = 0
        self.pending_loss = 0
        self.closed = False
        self.stop = threading.Event()
        self.file = JsonlWriter(path) if writer is None else None
        self.write = writer or self.file.write
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="tokenmoe-control-export"
        )
        self.thread.start()

    def emit(self, kind, *, source_at=None, **fields):
        with self.lock:
            if self.closed:
                return False
            self.sequence += 1
            source_at = now() if source_at is None else source_at
            project(self.states, kind, {**fields, "source_at": source_at})
            overflow = len(self.states) > self.max_sessions
            if overflow:
                # Missing sessions cause conservative demand suppression at the consumer.
                self.states.pop(next(iter(self.states)))
                self.pending_loss += 1
            event = {
                "schema_version": 1,
                "event_id": new_id("evt"),
                "producer_id": self.producer,
                "sequence": self.sequence,
                "kind": kind,
                "source_at": source_at,
                "clock_domain_id": self.clock["clock_domain_id"],
                **copy.deepcopy(fields),
                "dropped_before": self.pending_loss,
            }
            if self.pending_loss:
                event["snapshot"] = copy.deepcopy(self.states)
                event["snapshot_at_sequence"] = self.sequence
            if len(json.dumps(event).encode()) > 65536:
                self.dropped += 1
                self.pending_loss += 1
                return False
            try:
                self.queue.put_nowait(event)
            except queue.Full:
                self.dropped += 1
                self.pending_loss += 1
                return False
            self.pending_loss = 0
            return True

    def _run(self):
        try:
            while not self.stop.is_set() or not self.queue.empty():
                try:
                    event = self.queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                try:
                    self.write(event)
                except Exception:
                    with self.lock:
                        self.export_errors += 1
                        self.pending_loss += 1
            # Snapshot time describes shutdown, never earlier prediction availability.
            with self.lock:
                terminal = {
                    "schema_version": 1,
                    "event_id": new_id("evt"),
                    "producer_id": self.producer,
                    "sequence": self.sequence + 1,
                    "kind": "snapshot",
                    "source_at": now(),
                    "clock_domain_id": self.clock["clock_domain_id"],
                    "snapshot": copy.deepcopy(self.states),
                    "dropped_before": self.pending_loss,
                    "snapshot_at_sequence": self.sequence + 1,
                    "total_dropped": self.dropped,
                    "export_errors": self.export_errors,
                }
            try:
                self.write(terminal)
            except Exception:
                self.export_errors += 1
        finally:
            if self.file is not None:
                self.file.close()

    def close(self, timeout=1.0):
        with self.lock:
            self.closed = True
        self.stop.set()
        self.thread.join(timeout)
        return {
            "dropped": self.dropped,
            "export_errors": self.export_errors,
            "flush_complete": not self.thread.is_alive(),
        }
