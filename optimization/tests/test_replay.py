import pytest

from tokenmoe_opt.replay import CopyCost, replay, token_microbatches
from tokenmoe_opt.replay_cache import CacheModel
from tokenmoe_opt.timing import simulate


def plan(time=0, *, expert=0, epoch=1, expiry=5):
    return {
        "kind": "prefetch",
        "time_seconds": time,
        "epoch": epoch,
        "expires_at_seconds": expiry,
        "desired": {"0": [expert]},
        "transfers": [{"layer": 0, "expert": expert}],
    }


def demand(time, expert):
    return {"time_seconds": time, "layer": 0, "tokens": [[expert]]}


def test_inflight_prefetch_is_joined_once_and_expiry_does_not_cancel_dma():
    result = replay(
        [plan(), demand(0.5, 0)], capacity=1, cost=CopyCost(1, 1), timeline_kind="neutral"
    )
    assert result["bytes"] == 1 and result["joined_inflight_demands"] == 1
    assert result["serialized_readiness_wait_seconds"] == 0.5
    result = replay(
        [plan(expiry=0.1), demand(0.2, 1)], capacity=1, cost=CopyCost(1, 1), timeline_kind="neutral"
    )
    assert result["bytes"] == 2 and result["finish_seconds"] == 2
    assert result["evicted_unused_prefetch_bytes"] == 1
    assert result["slots"][0]["expert"] == 1 and result["slots"][0]["generation"] == 2


def test_prefetch_cannot_use_miss_delayed_timestamps_or_gpu_busy_intervals():
    with pytest.raises(ValueError, match="containing demand stalls"):
        replay([plan()], capacity=1, cost=CopyCost(1, 1))
    result = replay(
        [demand(0, 0), plan(0.5, expert=1)],
        capacity=1,
        cost=CopyCost(1, 1),
        timeline_kind="neutral",
    )
    assert result["bytes"] == 1 and result["busy_plan_suppression"] == 1


def test_tp_shards_share_link_and_each_copy_is_coalesced():
    result = replay(
        [demand(0, 0), demand(1, 0)],
        capacity=1,
        cost=CopyCost(1, 10),
        rank_costs={0: CopyCost(2, 10), 1: CopyCost(1, 10)},
    )
    assert result["bytes"] == 3 and result["unique_demand_hits"] == 2
    assert result["serialized_readiness_wait_seconds"] == pytest.approx(0.3)


def test_initial_residency_and_inflight_readiness_are_explicit():
    snapshot = {
        "kind": "snapshot",
        "time_seconds": 0,
        "slots": [
            {
                "rank": 0,
                "layer": 0,
                "slot": 0,
                "expert": 0,
                "generation": 2,
                "state": "LOADING",
                "ready_at_seconds": 1,
            }
        ],
    }
    result = replay([snapshot, demand(0.5, 0)], capacity=1, cost=CopyCost(1, 1))
    assert result["bytes"] == 0 and result["initial_inflight_bytes"] == 1
    assert result["serialized_readiness_wait_seconds"] == 0.5
    del snapshot["slots"][0]["ready_at_seconds"]
    with pytest.raises(ValueError, match="authoritative readiness"):
        replay([snapshot], capacity=1, cost=CopyCost(1, 1))


def test_prefetch_rate_limits_survive_new_epochs_and_leave_partial_copies_visible():
    cache = CacheModel(2, {0: CopyCost(1, 10)}, byte_rate=1)
    cache.prefetch(plan(expert=0, expiry=4), 0)
    cache.prefetch(plan(expert=1, epoch=2, expiry=4), 0.2)
    state = cache.result(0.9)
    assert state["bytes"] == 1 and state["pending_speculative_shards_at_end"] == 1
    state = cache.result(1.05)
    assert state["bytes"] == 2 and state["inflight_bytes_at_end"] == 1


def test_prefill_token_limit_and_prepartitioned_capacity_are_enforced():
    assert len(token_microbatches([[0]] * 65, 1)) == 2
    with pytest.raises(ValueError, match="without token routes"):
        replay(
            [{"time_seconds": 0, "layer": 0, "experts": [0], "partition_source_capacity": 2}],
            capacity=1,
            cost=CopyCost(1, 1),
        )


def test_model_stalls_shift_tools_dependency_joins_and_request_returns():
    nodes = [
        {
            "id": "first",
            "kind": "model",
            "duration_seconds": 0,
            "layers": [{"layer": 0, "tokens": [[0]]}],
        },
        {"id": "tool-a", "kind": "tool", "dependencies": ["first"], "duration_seconds": 2},
        {"id": "tool-b", "kind": "tool", "dependencies": ["first"], "duration_seconds": 3},
        {"id": "join", "kind": "join", "dependencies": ["tool-a", "tool-b"]},
        {
            "id": "second",
            "kind": "model",
            "duration_seconds": 0,
            "dependencies": ["join"],
            "layers": [{"layer": 0, "tokens": [[0]]}],
        },
        {"id": "return", "kind": "return", "dependencies": ["second"], "duration_seconds": 0.1},
    ]
    result = simulate(nodes, capacity=1, costs={0: CopyCost(1, 1)})
    assert result["nodes"]["tool-a"]["started"] == 1
    assert result["nodes"]["second"]["started"] == 4
    assert result["nodes"]["return"]["finished"] == 4.1
    assert not result["calibrated_timing"]
    nodes[1]["release_at_seconds"] = 0.1
    with pytest.raises(ValueError, match="absolute observation"):
        simulate(nodes, capacity=1, costs={0: CopyCost(1, 1)})


def test_causal_timing_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        simulate(
            [{"id": "a", "kind": "join", "dependencies": ["a"]}],
            capacity=1,
            costs={0: CopyCost(1, 1)},
        )


def test_resident_lfu_honors_initial_frequency_and_incomplete_snapshots_fail():
    slots = [
        {
            "rank": 0,
            "layer": 0,
            "slot": i,
            "expert": i,
            "generation": 1,
            "state": "READY",
            "last_use": i + 1,
            "frequency": 10 if i == 0 else 0,
        }
        for i in range(2)
    ]
    records = [
        {"kind": "snapshot", "time_seconds": 0, "slots": slots},
        demand(0.5, 2),
        demand(2, 0),
    ]
    result = replay(records, capacity=2, cost=CopyCost(1, 1), policy="lfu")
    assert result["bytes"] == 1 and result["unique_demand_hits"] == 1
    records[0] = {**records[0], "slots": slots[:1]}
    with pytest.raises(ValueError, match="complete slot snapshots"):
        replay(records, capacity=2, cost=CopyCost(1, 1))


def test_ready_model_work_precedes_speculative_admission_at_the_same_time():
    nodes = [
        {
            "id": "plan",
            "kind": "prefetch",
            "epoch": 1,
            "lease_seconds": 2,
            "desired": {"0": [1]},
            "transfers": [{"layer": 0, "expert": 1}],
        },
        {
            "id": "model",
            "kind": "model",
            "duration_seconds": 0,
            "layers": [{"layer": 0, "tokens": [[0]]}],
        },
    ]
    result = simulate(nodes, capacity=1, costs={0: CopyCost(1, 1)})
    assert result["cache"]["bytes"] == 1
    assert result["cache"]["busy_plan_suppression"] == 1


def test_measured_service_references_are_bound_to_actual_token_and_layer_work():
    native = {
        "layers": [0],
        "requests": [
            {
                "computed_tokens": 1,
                "model_service_seconds": 2,
                "input_tokens": 1,
                "output_tokens": 1,
            }
        ],
    }
    node = {
        "id": "model",
        "kind": "model",
        "service_profile_index": 0,
        "input_tokens": 1,
        "output_tokens": 1,
        "layers": [{"layer": 0, "tokens": [[0]]}],
    }
    result = simulate([node], capacity=1, costs={0: CopyCost(1, 1)}, native_service=native)
    assert result["nodes"]["model"]["finished"] == 3
    node["layers"][0]["tokens"].append([0])
    with pytest.raises(ValueError, match="token/layer shape"):
        simulate([node], capacity=1, costs={0: CopyCost(1, 1)}, native_service=native)
