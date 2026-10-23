import json
import socket

import numpy as np
import pytest
from tokenmoe_collect.clock import clock_metadata

from tokenmoe_opt.arrival import Survival
from tokenmoe_opt.control_run import Tails
from tokenmoe_opt.controller import FeedbackHistory, FrozenPredictor, ResidencyController, deliver
from tokenmoe_opt.data import Sample


def predictor():
    identity = {
        "id": "q",
        "model_id": "q",
        "weight_revision": "w",
        "tokenizer_hash": "t",
        "template_id": "template",
        "template_version": "v",
        "layers": [0, 1],
        "experts": 4,
        "top_k": 2,
    }
    hist = np.zeros((3, 2, 4), dtype=int)
    hist[..., 2:] = 10
    return FrozenPredictor(
        {
            "identity": identity,
            "history_alpha": 0.5,
            "ewma_alpha": 0.5,
            "label_version": "shell-command-v2",
            "parameters": {
                "global_hist": hist.tolist(),
                "tables": [],
                "min_support": 8,
                "training_requests": [f"train-{i}" for i in range(10)],
            },
        }
    )


def snapshot():
    return {
        "rank": 0,
        "source_at": 90,
        "clock_domain_id": clock_metadata()["clock_domain_id"],
        "model_hash": "model",
        "backend_hash": "backend",
        "engine_id": "engine",
        "device_uuid": "gpu",
        "consistent_idle_snapshot": True,
        "bytes_per_expert": 10,
        "speculative_bytes_available": 100,
        "inflight_limit": 4,
        "caches": [{"slots": [{"expert": i, "state": "READY"} for i in (0, 1)]} for _ in range(2)],
    }


def controller():
    return ResidencyController(
        predictor(),
        Survival([]),
        {
            "measured": True,
            "model_hash": "model",
            "device_uuids": ["gpu"],
            "copy_seconds": 0.001,
            "demand_wait_seconds": 0.5,
            "prefill_seconds": 0.1,
            "decode_token_seconds": 0.1,
            "shared_bandwidth_bytes_second": 1000,
            "version": "test-only",
        },
    )


def session_event(c, sequence, kind, **fields):
    return {
        "schema_version": 1,
        "event_id": f"e{sequence}",
        "producer_id": "p",
        "sequence": sequence,
        "kind": kind,
        "session_id": "s",
        "source_at": sequence,
        "clock_domain_id": c.sessions.clock_domain,
        **fields,
    }


@pytest.mark.parametrize("advertised", [False, True])
def test_received_features_drive_shared_plan_and_worker_delivery(tmp_path, advertised):
    c = controller()
    c.event(session_event(c, 1, "session_started", identity=c.predictor.identity), receipt=10)
    decision = None
    for time in (100, 200, 300):
        decision = c.decide([snapshot()], decision_at=time)
    assert decision["transfers"] and decision["visible_sequences"] == {"p": 1}
    assert len(decision["desired"]["0"]) == 2
    state = snapshot()
    address = tmp_path / ("endpoint.sock" if advertised else "rank-0.sock")
    if advertised:
        state["mailbox_socket"] = str(address)
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
        receiver.bind(str(address))
        receiver.settimeout(1)
        assert deliver(
            decision, [state], tmp_path / ("long-" * 40) if advertised else tmp_path
        ) == [0]
        packet = json.loads(receiver.recv(65536))
    assert packet["engine_id"] == "engine" and packet["model_hash"] == "model"
    assert packet["transfers"] == decision["transfers"]


def test_actual_scheduler_work_and_stale_snapshots_suppress_speculation():
    c = controller()
    c.event(session_event(c, 1, "session_started", identity=c.predictor.identity), receipt=10)
    assert c.decide([{**snapshot(), "source_at": 0}], decision_at=600_000_000) is None
    c.running.add("actual-request")
    assert c.decide([snapshot()], decision_at=100) is None


def test_residency_age_advances_while_no_session_predicts_work():
    c = controller()
    for time in (100, 200):
        assert c.decide([snapshot()], decision_at=time) is None
    c.event(session_event(c, 1, "session_started", identity=c.predictor.identity), receipt=250)
    assert c.decide([snapshot()], decision_at=300)["transfers"]


def test_feedback_loss_stops_new_speculative_plans():
    c = controller()
    c.event(session_event(c, 1, "session_started", identity=c.predictor.identity), receipt=10)
    c.feedback.unreliable.add("engine")
    assert c.decide([snapshot()], decision_at=100) is None
    c.running.clear()
    c.event(session_event(c, 2, "session_cancelled"), receipt=20)
    assert c.decide([snapshot()], decision_at=100) is None


def test_controller_rejects_different_model_costs_and_future_receipts():
    c = controller()
    c.event(session_event(c, 1, "session_started", identity=c.predictor.identity), receipt=200)
    with pytest.raises(ValueError, match="precedes"):
        c.decide([snapshot()], decision_at=100)
    with pytest.raises(ValueError, match="another model"):
        c.decide([{**snapshot(), "model_hash": "other"}], decision_at=300)
    with pytest.raises(ValueError, match="another cache execution"):
        c.decide(
            [
                {
                    **snapshot(),
                    "cache_execution": {
                        "prefill_execution": "expert-major",
                        "max_route_pairs": 256,
                    },
                }
            ],
            decision_at=300,
        )


def test_expert_major_prefetch_cannot_reuse_token_transfer_benefits():
    c = controller()
    with pytest.raises(ValueError, match="transfer-benefit"):
        ResidencyController(
            c.predictor,
            c.durations,
            {**c.cost, "cache_execution": {"prefill_execution": "expert-major"}},
        )


def test_compact_feedback_is_deduplicated_and_not_observed_before_receipt(tmp_path):
    p = predictor()
    feedback = FeedbackHistory(p)
    feedback.client(
        {"kind": "request_created", "request_id": "r", "session_id": "s", "source_at": 1}, 1
    )
    for sequence, (layer, phase) in enumerate(
        [(0, "new_prefill"), (1, "new_prefill"), (0, "decode"), (1, "decode")], 1
    ):
        row = {
            "kind": "feedback",
            "engine_id": "engine",
            "rank": 0,
            "request_id": "vllm-r",
            "llm_request_id": "r",
            "layer": layer,
            "phase": phase,
            "sequence": sequence,
            "counts": [0, 1, 1, 0],
            "start": 0,
            "end": 1,
        }
        feedback.counts(row, 10 + sequence)
    feedback.counts(row, 19)
    feedback.client({"kind": "request_completed", "request_id": "r"}, 20)
    feedback.engine(
        {"kind": "request_finished", "request_id": "vllm-r", "num_output_tokens": 2}, 21
    )
    target = Sample(
        "live",
        "s",
        "next",
        19,
        30,
        np.zeros((3, 2, 4)),
        np.zeros((3, 2, 4)),
        {"stage": "subsequent", "tool_key": ()},
        tmp_path,
        {},
    )
    assert p.history.predict(target, p.prior, "previous")[1]["history_request"] is None
    target.decision = 22
    assert p.history.predict(target, p.prior, "previous")[1]["history_request"] == "r"


def test_append_only_tail_preserves_partial_lines_and_rejects_truncation(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"sequence":1}\n{"sequence":')
    tails = Tails()
    assert list(tails.read([path])) == [{"sequence": 1}]
    assert list(tails.read([path])) == []
    with path.open("ab") as stream:
        stream.write(b"2}\n")
    assert list(tails.read([path])) == [{"sequence": 2}]
    path.write_bytes(b"{}\n")
    with pytest.raises(RuntimeError, match="truncated"):
        list(tails.read([path]))
