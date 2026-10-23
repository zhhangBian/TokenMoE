import threading
import time

import pytest
from tokenmoe_collect.clock import clock_metadata
from tokenmoe_collect.events import BoundedEventSink
from tokenmoe_collect.executor import StreamingExecutor
from tokenmoe_collect.jsonl import read_jsonl
from tokenmoe_collect.records import RecordStore

from tokenmoe_opt.arrival import Duration, Survival, parallel_join_probability
from tokenmoe_opt.session import SessionState


def test_streaming_observations_are_unchanged_and_ranges_reference_only_observed_bytes(tmp_path):
    with RecordStore(tmp_path, control_events=True) as store:
        store.event_sink.emit("session_started", session_id="s", identity={"model": "q"})
        executor = StreamingExecutor(store, runtime="local", cwd=str(tmp_path), session_id="s")
        result, timing = executor.execute("printf 'abc\\r\\ndef'; printf 'ghi' >&2", "t", 3)
    assert result["output"] == "abc\ndefghi"
    raw = (tmp_path / timing["output_ref"]).read_bytes()
    assert raw == b"abc\r\ndefghi"
    events = list(read_jsonl(store.directory / "control_events.jsonl"))
    chunks = [event for event in events if event["kind"] == "tool_output"]
    assert sum(event["byte_length"] for event in chunks) == len(raw)
    assert all(event["byte_offset"] + event["byte_length"] <= len(raw) for event in chunks)
    assert events[-1]["kind"] == "snapshot"
    assert events[-1]["snapshot"]["s"]["tools"]["t"]["observed_bytes"] == len(raw)


def test_stalled_export_is_bounded_and_recovers_terminal_state(tmp_path):
    blocked, release = threading.Event(), threading.Event()
    written = []

    def slow_writer(event):
        if not blocked.is_set():
            blocked.set()
            release.wait(3)
        written.append(event)

    sink = BoundedEventSink(tmp_path / "unused", capacity=1, writer=slow_writer)
    sink.emit("session_started", session_id="s", identity={"model": "q"})
    assert blocked.wait(1)
    began = time.monotonic()
    for _ in range(100):
        sink.emit("tool_started", session_id="s", tool_id="t")
    sink.emit("session_ended", session_id="s", status="cancelled")
    assert time.monotonic() - began < 0.5
    status = sink.close(timeout=0.01)
    assert status["dropped"] > 0 and not status["flush_complete"]
    release.set()
    sink.thread.join(2)
    assert written[-1]["snapshot"]["s"]["terminal"]
    assert written[-1]["total_dropped"] > 0
    state = SessionState()
    for event in written:
        state.receive(event)
    assert not state.decision("s", decision_at=time.monotonic_ns())["predictive_demand"]


def event(sequence, kind, **fields):
    return {
        "schema_version": 1,
        "event_id": f"e{sequence}",
        "producer_id": "p",
        "sequence": sequence,
        "source_at": sequence,
        "clock_domain_id": clock_metadata()["clock_domain_id"],
        "kind": kind,
        "session_id": "s",
        **fields,
    }


@pytest.mark.parametrize("dependency_mode", ["sequential", "parallel"])
def test_dependency_join_requires_every_tool_and_cancellation_is_terminal(dependency_mode):
    state = SessionState()
    state.receive(event(1, "session_started"), received_at=2)
    issued = event(2, "tools_issued", dependencies=["a", "b"], dependency_mode=dependency_mode)
    state.receive(issued, received_at=3)
    assert not state.receive(issued, received_at=4)
    state.receive(event(3, "tool_completed", tool_id="a"), received_at=5)
    decision = state.decision("s", decision_at=5)
    assert not decision["dependencies_complete"] and not decision["can_submit"]
    state.receive(event(4, "tool_completed", tool_id="b"), received_at=6)
    assert state.decision("s", decision_at=6)["can_submit"]
    state.receive(event(5, "session_cancelled"), received_at=7)
    state.receive(event(6, "tool_started", tool_id="a"), received_at=8)
    assert not state.decision("s", decision_at=8)["predictive_demand"]


def test_event_gap_suppresses_predictions_until_observed_snapshot():
    state = SessionState()
    state.receive(event(1, "session_started"), received_at=2)
    state.receive(event(3, "tool_started", tool_id="a"), received_at=4)
    assert not state.decision("s", decision_at=4)["predictive_demand"]
    state.receive(
        event(
            4,
            "snapshot",
            snapshot_at_sequence=4,
            snapshot={"s": {"terminal": False, "tools": {}, "dependencies": []}},
        ),
        received_at=5,
    )
    assert state.decision("s", decision_at=5)["predictive_demand"]
    with pytest.raises(ValueError, match="precedes"):
        state.decision("s", decision_at=3)
    with pytest.raises(ValueError, match="clock"):
        state.receive({**event(5, "tool_started"), "clock_domain_id": "other"})


def test_forgotten_or_unobserved_session_cannot_reactivate_from_progress():
    state = SessionState()
    state.receive(event(1, "session_started"), received_at=2)
    state.receive(event(2, "session_cancelled"), received_at=3)
    state.receive(event(3, "snapshot", snapshot={}, snapshot_at_sequence=3), received_at=4)
    state.receive(event(4, "tool_started", tool_id="old"), received_at=5)
    assert not state.decision("s", decision_at=5)["predictive_demand"]


def test_survival_conditions_on_elapsed_time_and_preserves_censored_tail():
    model = Survival(
        [
            Duration("train", "test", 1),
            Duration("train", "test", 3),
            Duration("train", "test", 5, censored=True),
        ],
        min_support=2,
    )
    initial = model.predict("test")
    assert initial["p50"] == 3 and initial["p90"] is None
    assert initial["unresolved_tail_probability"] == pytest.approx(1 / 3)
    conditioned = model.predict("unseen", elapsed=2)
    assert conditioned["fallback"] and conditioned["p50"] == 1
    probability = parallel_join_probability([conditioned, conditioned], 1)["probability"]
    assert probability == pytest.approx(0.25)
    assert Survival([]).predict("unseen")["p50"] is None
    assert not Survival([]).predict("unseen")["prefetch_supported"]
