import copy

from tokenmoe_opt.validation import validate_cache_events, validate_features


def events():
    fields = {
        "engine_id": "e",
        "rank": 0,
        "layer": 0,
        "schema_version": 1,
        "clock_domain_id": "clock",
    }
    rows = [
        {
            "kind": "initial_snapshot",
            "slots": [{"slot": 0, "generation": 0, "expert": None, "state": "FREE"}],
            "inflight": [],
        },
        {
            "kind": "load_start",
            "expert": 1,
            "slot": 0,
            "generation": 1,
            "transfer_id": "0:0:1",
            "bytes": 1024,
        },
        {
            "kind": "load_complete",
            "expert": 1,
            "slot": 0,
            "generation": 1,
            "transfer_id": "0:0:1",
            "device_duration_ns": 100,
        },
        {"kind": "use", "expert": 1, "slot": 0, "generation": 1},
        {"kind": "kernel_complete", "expert": 1, "slot": 0, "generation": 1},
        {"kind": "eviction", "expert": 1, "slot": 0, "generation": 1},
    ]
    return [
        {
            **fields,
            "sequence": i + 1,
            "event_id": f"ev{i}",
            "source_at": i * 10,
            "exported_at": i * 10 + 5,
            **row,
        }
        for i, row in enumerate(rows)
    ]


def status(**changes):
    return {"flush_complete": True, "dropped": 0, "errors": 0, **changes}


def test_slot_reconstruction_requires_matching_generations_and_readiness():
    valid = events()
    result = validate_cache_events(valid, status=status(), expected_layers=[0])
    assert result["passed"] and result["exact_replay_available"]
    assert result["bytes_started"] == result["bytes_completed"] == 1024
    stale = copy.deepcopy(valid)
    stale[2]["generation"] = 2
    assert not validate_cache_events(stale, status=status())["passed"]
    early = [row for row in valid if row["kind"] != "load_complete"]
    for i, row in enumerate(early):
        row["sequence"] = i + 1
    assert not validate_cache_events(early, status=status())["passed"]


def test_cache_checkpoint_contains_only_observed_residency_and_replacement_history():
    result = validate_cache_events(events(), status=status(), checkpoint_at=35)
    slot = result["checkpoint"]["slots"][0]
    assert slot["state"] == "IN_USE" and slot["expert"] == 1
    assert slot["frequency"] == 1 and slot["last_use"] == 1
    assert result["checkpoint"]["previously_loaded"] == [{"rank": 0, "layer": 0, "expert": 1}]


def test_loss_and_crash_keep_inference_evidence_but_remove_exact_replay_coverage():
    result = validate_cache_events(events(), status=status(dropped=1))
    assert result["passed"] and not result["exact_replay_available"]
    partial = validate_cache_events(events()[:2])
    assert partial["passed"] and not partial["exact_replay_available"]
    assert any("incomplete transfers" in reason for reason in partial["incomplete_reasons"])


def test_compact_observation_never_qualifies_exact_cache_replay():
    records = events()
    records[0]["kind"] = "observer_config"
    records[0]["mode"] = "compact"
    assert not validate_cache_events(records[:1], status=status())["exact_replay_available"]


def test_serialized_events_may_have_delayed_independent_observation_times():
    rows = events()
    delayed = {**rows[3], "kind": "demand", "expert": 2, "source_at": 15, "event_id": "delayed"}
    delayed.pop("slot")
    rows.insert(4, delayed)
    for sequence, row in enumerate(rows, 1):
        row["sequence"] = sequence
    assert validate_cache_events(rows, status=status())["passed"]
    rows[2]["source_at"] = 1
    assert not validate_cache_events(rows, status=status())["passed"]


def test_cache_memory_reconstruction_checks_layout_and_rank_scratch_budget():
    rows = events()
    rows[0]["memory"] = {
        "host_pageable_bytes": 128 * 1024,
        "resident_bytes": 1024,
        "host_pinned_staging_bytes": 1024,
        "mapping_bytes": 1024,
        "gpu_staging_bytes": 0,
    }
    rows.append(
        {
            **rows[-1],
            "kind": "observer_config",
            "event_id": "config",
            "sequence": 7,
            "mode": "detailed",
            "budget_bytes_per_rank": 2112,
            "scratch_bytes": 64,
        }
    )
    assert validate_cache_events(rows, status=status())["passed"]
    rows[-1]["budget_bytes_per_rank"] -= 1
    assert not validate_cache_events(rows, status=status())["passed"]
    del rows[0]["memory"]["resident_bytes"]
    assert "invalid byte accounting" in " ".join(
        validate_cache_events(rows, status=status())["errors"]
    )


def test_feature_validator_rejects_receipt_and_future_history_leaks():
    record = {
        "request": "r",
        "features": {
            "clock_domain_id": "clock",
            "decision_at": 10,
            "availability": [
                {"tool_call_id": "t", "source_observed_at": 2, "receipt_at": 8, "available_at": 8}
            ],
        },
        "provenance": {"previous": {"deployable": True, "history_observed_at": 9}},
    }
    assert validate_features([record])["passed"]
    record["features"]["availability"][0]["receipt_at"] = 11
    record["provenance"]["previous"]["history_observed_at"] = 12
    assert len(validate_features([record])["errors"]) == 2
