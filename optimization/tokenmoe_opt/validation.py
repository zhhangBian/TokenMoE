"""Optimization-only causal and slot-generation checks; no collection writes."""

from collections import defaultdict


def validate_features(records):
    errors = []
    for record in records:
        features = record["features"]
        decision = features["decision_at"]
        if not features.get("clock_domain_id"):
            errors.append(f"{record['request']}: missing decision clock")
        for feature in features["availability"]:
            if feature.get("clock_domain_id", features.get("clock_domain_id")) != features.get(
                "clock_domain_id"
            ):
                errors.append(f"{record['request']}: incomparable feature clock")
            available = max(feature["source_observed_at"], feature["receipt_at"])
            if feature["available_at"] != available or available > decision:
                errors.append(f"{record['request']}: unavailable feature {feature['tool_call_id']}")
            if feature.get("outcome_observed_at", decision) > decision:
                errors.append(f"{record['request']}: future tool outcome")
        for policy, prediction in record["provenance"].items():
            observed = prediction.get("history_observed_at")
            if prediction.get("deployable") and observed is not None and observed > decision:
                errors.append(f"{record['request']}/{policy}: future history")
    return {"passed": not errors, "errors": errors}


def validate_cache_events(records, *, status=None, expected_layers=None):
    slots, transfers, sequences, clocks = {}, {}, {}, {}
    engine_clocks = {}
    budgets, memory = {}, defaultdict(dict)
    uncertain, initialized, event_ids = set(), defaultdict(set), set()
    errors, reasons = [], []
    bytes_started = bytes_completed = 0
    for event in records:
        engine, rank, layer = event["engine_id"], event.get("rank"), event.get("layer")
        stream = (engine, rank)
        sequence, kind = event["sequence"], event["kind"]

        def fail(reason):
            errors.append(f"{engine}/rank-{rank}/event-{sequence}: {reason}")

        if event.get("schema_version") != 1:
            fail("unsupported schema")
        event_id = event["event_id"]
        if event_id in event_ids:
            fail("duplicate event ID")
        event_ids.add(event_id)
        if sequence <= sequences.get(stream, 0):
            fail("non-increasing sequence")
        if sequence != sequences.get(stream, 0) + 1:
            uncertain.add(stream)
            reasons.append(f"{stream}: event gap before {sequence}")
        sequences[stream] = sequence
        clock = event["clock_domain_id"]
        if stream in clocks and clocks[stream] != clock:
            fail("incomparable clocks")
        clocks[stream] = clock
        if engine in engine_clocks and engine_clocks[engine] != clock:
            fail("TP ranks have incomparable clocks")
        engine_clocks[engine] = clock
        source = event.get("source_at")
        if source is None or source < 0:
            fail("missing or negative source observation time")
        if source is not None and event.get("exported_at", source) < source:
            fail("export predates source observation")
        if kind in {"initial_snapshot", "snapshot"}:
            initialized[stream].add(layer)
            slot_ids = [slot["slot"] for slot in event["slots"]]
            if sorted(slot_ids) != list(range(len(slot_ids))):
                fail("snapshot slot identities are not unique and contiguous")
            if "memory" in event:
                values = event["memory"]
                required = {
                    "resident_bytes",
                    "host_pageable_bytes",
                    "host_pinned_staging_bytes",
                    "mapping_bytes",
                    "gpu_staging_bytes",
                }
                valid = required <= values.keys() and all(
                    type(value) is int and value >= 0 for value in values.values()
                )
                if not valid:
                    fail("invalid byte accounting")
                else:
                    expert_bytes, remainder = divmod(values["host_pageable_bytes"], 128)
                    if (
                        remainder
                        or not expert_bytes
                        or values["resident_bytes"] != expert_bytes * len(slot_ids)
                        or values["host_pinned_staging_bytes"] != expert_bytes
                        or values["mapping_bytes"] != 128 * 8
                    ):
                        fail("resident/staging/mapping byte accounting differs from expert layout")
                    memory[stream][layer] = values
            for slot in event["slots"]:
                slots[(*stream, layer, slot["slot"])] = {
                    **slot,
                    "observed_at": source if source is not None else 0,
                }
            for transfer in event["inflight"]:
                transfer_id = f"{layer}:{transfer['slot']}:{transfer['generation']}"
                transfers[(*stream, layer, transfer_id)] = transfer
            continue
        if kind == "observer_config" and event.get("mode") != "detailed":
            reasons.append("Compact feedback does not export a cache transfer lifecycle")
        if kind == "observer_config" and "budget_bytes_per_rank" in event:
            budgets[stream] = event
        if kind == "feedback":
            counts = event["counts"]
            if len(counts) != 128 or any(c < 0 for c in counts):
                fail("invalid compact histogram")
            if sum(counts) != (event["end"] - event["start"]) * 8:
                fail("feedback lookup count differs from actual token span")
            if event["phase"] not in {"new_prefill", "decode", "recompute"}:
                fail("unknown feedback phase")
            continue
        if "expert" in event and not 0 <= event["expert"] < 128:
            fail("invalid logical expert")
        if kind in {"demand", "observer_config"} or stream in uncertain:
            continue
        if "slot" not in event:
            continue
        key = (*stream, layer, event["slot"])
        state = slots.get(key)
        if state is None:
            fail("slot used without initialization snapshot")
            continue
        generation, expert = event["generation"], event["expert"]
        if source is not None and source < state.get("observed_at", 0):
            fail("slot lifecycle observation predates its causal predecessor")
        state["observed_at"] = source if source is not None else 0
        if kind == "load_start":
            if state["state"] != "FREE" or generation != state["generation"] + 1:
                fail("load overwrites an occupied slot or invalid generation")
            state.update(state="LOADING", expert=expert, generation=generation)
            tid = (*stream, layer, event["transfer_id"])
            if tid in transfers or event["bytes"] <= 0:
                fail("duplicate transfer or invalid rank-local bytes")
            transfers[tid] = event
            bytes_started += event["bytes"]
            continue
        if state["generation"] != generation or state["expert"] != expert:
            fail("stale slot generation or logical expert")
            continue
        if kind == "load_complete":
            transfer = transfers.pop((*stream, layer, event["transfer_id"]), None)
            if transfer is None or state["state"] != "LOADING":
                fail("copy completed without matching loading reservation")
            elif transfer["generation"] != generation:
                fail("stale copy published into a reused slot")
            else:
                bytes_completed += transfer.get("bytes", 0)
            if event["device_duration_ns"] < 0:
                fail("negative device duration")
            state["state"] = "READY"
        elif kind == "use":
            if state["state"] not in {"READY", "IN_USE"}:
                fail("expert used before copy readiness")
            state["state"] = "IN_USE"
        elif kind == "kernel_complete":
            if state["state"] != "IN_USE":
                fail("kernel completion without expert use")
            state["state"] = "READY"
        elif kind == "eviction":
            if state["state"] != "READY":
                fail("eviction of a loading or in-use slot")
            state.update(state="FREE", expert=None)
    if status is None or not status.get("flush_complete"):
        reasons.append("No complete export status; shutdown/crash coverage is unknown")
    if status and (status.get("dropped") or status.get("errors")):
        reasons.append("Export reports loss or writer errors")
    if (
        status
        and len(sequences) == 1
        and status.get("last_sequence", next(iter(sequences.values())))
        != next(iter(sequences.values()))
    ):
        reasons.append("Trace end differs from completed export status")
    if transfers:
        reasons.append(f"{len(transfers)} incomplete transfers")
    if any(s["state"] == "IN_USE" for s in slots.values()):
        reasons.append("Open kernel use at trace boundary")
    if expected_layers is not None:
        for stream, layers in initialized.items():
            if layers != set(expected_layers):
                reasons.append(f"{stream}: missing layer snapshots")
    if not initialized:
        reasons.append("No initial residency snapshots")
    for stream, layers in memory.items():
        if stream in budgets:
            required = (
                sum(
                    m[key]
                    for m in layers.values()
                    for key in ("resident_bytes", "mapping_bytes", "gpu_staging_bytes")
                )
                + budgets[stream]["scratch_bytes"]
            )
            if required > budgets[stream]["budget_bytes_per_rank"]:
                errors.append(f"{stream}: expert memory exceeds per-rank budget")
    return {
        "passed": not errors,
        "exact_replay_available": not errors and not reasons,
        "errors": errors,
        "incomplete_reasons": reasons,
        "bytes_started": bytes_started,
        "bytes_completed": bytes_completed,
    }
