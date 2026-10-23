import numpy as np
import pytest

from tokenmoe_opt.planner import Demand, Ledger, aggregate, plan


def test_shared_demand_uses_volume_arrival_and_actual_service_probability():
    dist = np.array([[0.75, 0.25]])
    expected, sources = aggregate(
        [
            Demand("a", "q", dist, 10, 1, 1, "running"),
            Demand("b", "q", dist, 20, 0.5, 0.2, "queued"),
            Demand("cancelled", "q", dist, 999, 1, 1, "running", terminal=True),
        ],
        identity="q",
        shape=(1, 2),
        top_k=2,
    )
    np.testing.assert_allclose(expected, [[18, 6]])
    assert [s["weight"] for s in sources] == [10, 2]
    with pytest.raises(ValueError, match="identity"):
        aggregate(
            [Demand("a", "other", dist, 1, 1, 1, "running")], identity="q", shape=(1, 2), top_k=2
        )


def ledger(**kwargs):
    return Ledger(
        rank_free_bytes={0: 40, 1: 40},
        shared_link_bytes=80,
        speculative_byte_limit=40,
        speculative_inflight_limit=4,
        expert_bytes={0: 10, 1: 10},
        **kwargs,
    )


def test_rank_and_link_reservations_are_atomic_and_shared_experts_are_coalesced():
    resources = ledger(kv_reserved_bytes=10)
    assert resources.reserve(0, 1)[0]
    assert resources.reserved_bytes == 20
    assert resources.reserve(0, 1) == (True, "joined_existing_transfer")
    assert resources.reserved_bytes == 20
    resources.rank_free_bytes[1] = 0
    before = resources.rank_free_bytes.copy()
    assert resources.reserve(0, 2) == (False, "per_rank_memory")
    assert resources.rank_free_bytes == before


def make_plan(**overrides):
    args = dict(
        identity="q",
        epoch=5,
        now_ns=10,
        expiry_ns=100,
        capacity=1,
        resident={0: {0}},
        pinned={},
        demand={},
        residency_epoch={(0, 0): 1},
        ledger=ledger(),
        cost_profile={
            "identity": "q",
            "version": "test-only-costs",
            "measured": True,
            "copy_seconds": 0.01,
            "demand_wait_seconds": 1.0,
        },
        max_churn=1,
    )
    args.update(overrides)
    return plan(np.array([[0.01, 10.0]]), **args)


def test_cost_planner_accounts_for_displacement_churn_and_demand_priority():
    result = make_plan()
    assert result["desired"] == {"0": [1]}
    assert result["decisions"][0]["displaced"] == 0
    assert result["reserved_rank_bytes"] == 20
    protected = make_plan(pinned={0: {0}})
    assert protected["desired"] == {"0": [0]} and not protected["transfers"]
    assert not make_plan(max_churn=0)["transfers"]
    assert not make_plan(residency_epoch={(0, 0): 5})["transfers"]
    with pytest.raises(ValueError, match="microbatch"):
        make_plan(demand={0: {0, 1}})
    with pytest.raises(ValueError, match="measured"):
        make_plan(cost_profile={"measured": False})


def test_shared_budget_prefers_greater_gain_across_layers():
    result = plan(
        np.array([[0.01, 0.1], [0.01, 10.0]]),
        identity="q",
        epoch=5,
        now_ns=10,
        expiry_ns=100,
        capacity=1,
        resident={0: {0}, 1: {0}},
        pinned={},
        demand={},
        residency_epoch={(0, 0): 1, (1, 0): 1},
        ledger=ledger(),
        cost_profile={
            "identity": "q",
            "version": "test-only",
            "measured": True,
            "copy_seconds": 0.01,
            "demand_wait_seconds": 1,
        },
        max_churn=1,
    )
    assert result["transfers"] == [{"layer": 1, "expert": 1}]


def test_full_inflight_ledger_still_allows_joining_an_existing_transfer():
    resources = ledger(inflight={(0, 1, 1), (1, 1, 1)})
    resources.speculative_inflight_limit = 2
    result = plan(
        np.array([[0.01, 10.0], [0.01, 1.0]]),
        identity="q",
        epoch=5,
        now_ns=10,
        expiry_ns=100,
        capacity=1,
        resident={0: {0}, 1: {0}},
        pinned={},
        demand={},
        residency_epoch={(0, 0): 1, (1, 0): 1},
        ledger=resources,
        cost_profile={
            "identity": "q",
            "version": "test-only",
            "measured": True,
            "copy_seconds": 0.01,
            "demand_wait_seconds": 1,
        },
    )
    assert result["transfers"] == [{"layer": 1, "expert": 1}]
    assert result["reserved_rank_bytes"] == 0
    assert result["decisions"][0]["reason"] == "joined_existing_transfer"
