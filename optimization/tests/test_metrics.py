import json

import pytest

from tokenmoe_opt.agent_compare import preconditions
from tokenmoe_opt.agent_metrics import measure, task_uncertainty
from tokenmoe_opt.artifacts import manifest
from tokenmoe_opt.costs import fit
from tokenmoe_opt.metrics import Transfers, resource_metrics, token_metrics


def test_pilot_comparison_preserves_missing_repeats_sources_and_export_gates():
    base = dict(
        identity="q",
        configuration="c",
        runtime_sources="code",
        fork_files_hashed=True,
        devices=["gpu"],
        tasks=["task"],
        concurrency=1,
        protocol="p",
        side_channel_complete=True,
        source_stable=True,
        slots=64,
    )
    rows = [
        {**base, "policy": policy, "repetition": i}
        for policy in ("full", "previous+tool")
        for i in range(3)
    ]
    checks = preconditions(rows, 3)
    assert all(passed for key, passed in checks.items() if key != "rotated_order_matrix_verified")
    assert not checks["rotated_order_matrix_verified"]
    rows[0]["side_channel_complete"] = False
    rows[1]["runtime_sources"] = "different"
    checks = preconditions(rows[:-1], 3)
    assert not checks["complete_repetitions"]
    assert not checks["complete_side_channels"] and not checks["same_runtime_sources"]


def test_token_metrics_preserve_batch_resolution_and_separate_endpoint_tpot():
    fields = {"request_id": "r", "clock_domain_id": "clock"}
    events = [
        {**fields, "kind": "request_received", "source_at": 100},
        {
            **fields,
            "kind": "token_readiness",
            "source_at": 200,
            "processing_batch": 1,
            "token_indices": [0],
        },
        {
            **fields,
            "kind": "token_readiness",
            "source_at": 500,
            "processing_batch": 2,
            "token_indices": [1, 2],
        },
        {
            **fields,
            "kind": "request_finished",
            "source_at": 600,
            "num_output_tokens": 3,
            "status": "FINISHED_STOPPED",
        },
    ]
    result = token_metrics(events)[0]
    assert result["engine_ttft_seconds"] == 100 / 1e9
    assert result["endpoint_average_tpot_seconds"] == 150 / 1e9
    assert result["observed_itl_seconds"] is None
    assert result["network_delivery"] is None


def test_missing_initial_token_is_not_fabricated_as_ttft():
    result = token_metrics(
        [
            {
                "request_id": "r",
                "clock_domain_id": "clock",
                "kind": "token_readiness",
                "source_at": 500,
                "processing_batch": 2,
                "token_indices": [2],
            }
        ]
    )[0]
    assert result["engine_ttft_seconds"] is None
    assert not result["complete_token_coverage"]


@pytest.mark.parametrize("field", ["kv_cache_memory_bytes", "enforce_eager", "async_scheduling"])
def test_copy_cost_fitting_rejects_confounded_execution_profiles(tmp_path, field):
    keys = (
        "model tensor_parallel_size dtype max_model_len max_num_seqs "
        "max_num_batched_tokens kv_cache_memory_bytes enforce_eager async_scheduling "
        "enable_prefix_caching kernel_config"
    ).split()
    native = {"source_stable": True, "trials": [{"tokens": [1]}], "config": dict.fromkeys(keys, 1)}
    for name in ("native", "cache"):
        path = tmp_path / name
        path.mkdir()
        record = {
            **native,
            "config": {**native["config"], **({field: 2} if name == "cache" else {})},
        }
        (path / "result.json").write_text(json.dumps(record))
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps(manifest("input", {"model": "q"})))
    with pytest.raises(ValueError, match=field):
        fit(tmp_path / "native", tmp_path / "cache", inputs, tmp_path / "out", command=["test"])
    assert not (tmp_path / "out").exists()


@pytest.fixture
def agent_trial(tmp_path):
    from tokenmoe_collect.jsonl import write_json

    root = tmp_path / "agent"
    producer = root / "raw/harness/p"
    producer.mkdir(parents=True)
    write_json(producer / "producer.json", {"clock_domain_id": "clock"})
    tables = {
        "application_runs": [
            {"application_run_id": "a", "benchmark_item_id": "task-a"},
            {"application_run_id": "b", "benchmark_item_id": "task-b"},
        ],
        "sessions": [
            {
                "session_id": "s",
                "application_run_id": "a",
                "created_at": 100,
                "started_at": 110,
                "finished_at": 900,
                "final_status": "succeeded",
                "exit_reason": "Submitted",
            },
            {
                "session_id": "f",
                "application_run_id": "b",
                "created_at": 100,
                "started_at": 120,
                "finished_at": 1000,
                "final_status": "timeout",
                "exit_reason": "TimeExceeded",
            },
        ],
        "llm_requests": [
            {
                "llm_request_id": "r",
                "session_id": "s",
                "request_created_at": 300,
                "response_received_at": 800,
                "prev_tool_call_ids": ["x", "y"],
                "request_status": "succeeded",
                "finish_reason": "length",
            },
            {
                "llm_request_id": "failed",
                "session_id": "f",
                "request_created_at": 350,
                "response_received_at": None,
                "prev_tool_call_ids": [],
                "request_status": "transport_error",
            },
        ],
        "tool_calls": [
            {"tool_call_id": "x", "session_id": "s", "finished_at": 150},
            {"tool_call_id": "y", "session_id": "s", "finished_at": 250},
        ],
    }
    for name, records in tables.items():
        (producer / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    events = []
    for rid, received, first, last, count in [("warm", 10, 20, 20, 1), ("r", 320, 400, 600, 2)]:
        fields = {"clock_domain_id": "clock", "request_id": rid}
        events.extend(
            [
                {
                    **fields,
                    "kind": "request_received",
                    "source_at": received,
                    "llm_request_id": rid,
                },
                {
                    **fields,
                    "kind": "token_readiness",
                    "source_at": first,
                    "processing_batch": rid + "-first",
                    "token_indices": [0],
                },
            ]
        )
        if count == 2:
            events.append(
                {
                    **fields,
                    "kind": "token_readiness",
                    "source_at": last,
                    "processing_batch": rid + "-last",
                    "token_indices": [1],
                }
            )
        events.append(
            {
                **fields,
                "kind": "request_finished",
                "source_at": last + 50,
                "num_output_tokens": count,
                "status": "FINISHED_STOPPED",
            }
        )
    observed = tmp_path / "observations/e"
    observed.mkdir(parents=True)
    (observed / "token-readiness.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
    write_json(
        tmp_path / "boundaries.json",
        {"warmup_request_ids": ["warm"], "measured_started_at": 100, "measured_finished_at": 1000},
    )
    return tmp_path


def test_agent_metrics_exclude_warmup_and_keep_failures_and_actual_dependency_join(agent_trial):
    result = measure(agent_trial)
    assert len(result["requests"]) == 2
    row = result["requests"][0]
    assert row["engine_ttft_seconds"] == 80 / 1e9
    assert row["tools_to_complete_response_seconds"] == 550 / 1e9
    assert row["tools_to_first_visible_token_seconds"] == 550 / 1e9
    assert result["outcome_counts"] == {"succeeded": 1, "timeout": 1}
    assert result["unknown_output_token_requests"] == 1 and result["truncated_requests"] == 1
    assert result["observed_output_tokens_per_second"] == pytest.approx(2 / (900 / 1e9))
    assert result["benchmark_solved_tasks_per_second"] is None
    assert result["side_channel_export_complete"] is False


def test_agent_metrics_reject_mixed_warmup_and_incomparable_clocks(agent_trial):
    from tokenmoe_collect.jsonl import write_json

    boundary = agent_trial / "boundaries.json"
    value = json.loads(boundary.read_text())
    write_json(boundary, {**value, "warmup_request_ids": ["r"]})
    with pytest.raises(ValueError, match="Mixed warmup"):
        measure(agent_trial)
    write_json(boundary, value)
    write_json(agent_trial / "agent/raw/harness/p/producer.json", {"clock_domain_id": "other"})
    with pytest.raises(ValueError, match="clocks differ"):
        measure(agent_trial)


def test_tokens_emitted_after_measurement_do_not_inflate_throughput(agent_trial):
    path = agent_trial / "agent/raw/harness/p/llm_requests.jsonl"
    requests = [json.loads(line) for line in path.read_text().splitlines()]
    requests[0].update(request_status="transport_error", response_received_at=None)
    path.write_text("".join(json.dumps(r) + "\n" for r in requests))
    path = agent_trial / "observations/e/token-readiness.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[-1].update(source_at=1150, num_output_tokens=3)
    events.insert(
        -1,
        {
            "clock_domain_id": "clock",
            "request_id": "r",
            "kind": "token_readiness",
            "source_at": 1100,
            "processing_batch": "late",
            "token_indices": [2],
        },
    )
    path.write_text("".join(json.dumps(r) + "\n" for r in events))
    result = measure(agent_trial)
    assert result["requests"][0]["observed_output_tokens"] == 3
    assert result["requests"][0]["output_tokens_in_measurement"] == 2
    assert result["observed_output_tokens_per_second"] == pytest.approx(2 / (900 / 1e9))


def test_metric_extraction_rejects_new_observation_files_during_read(agent_trial, monkeypatch):
    import tokenmoe_opt.agent_metrics as module

    def changed(directory):
        (directory / "late.json").write_text("{}")
        return {}

    monkeypatch.setattr(module, "measure", changed)
    (agent_trial / "manifest.json").write_text(json.dumps({"identity": {"test": "fixture"}}))
    output = agent_trial.parent / (agent_trial.name + "-metrics")
    with pytest.raises(ValueError, match="changed during"):
        module.extract(agent_trial, output, command=["test"])
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_latency_uncertainty_weights_whole_task_groups():
    result = task_uncertainty(
        [{"task": "a", "latency": 1}] * 10 + [{"task": "b", "latency": 100}], "latency"
    )
    assert result["task_macro_mean"] == 50.5 and result["tasks"] == 2
    assert result["task_ci95"] == [1, 100]


def test_transfer_windows_exclude_warmup_copies_and_do_not_sum_individual_miss_waits():
    metrics = Transfers(window=(100, 300))
    common = {"layer": 0, "slot": 0, "generation": 1}
    for row in [
        {"kind": "load_start", "source_at": 90, "bytes": 10, "trigger": "prefetch"},
        {"kind": "load_complete", "source_at": 110, "device_duration_ns": 20},
        {
            "kind": "load_start",
            "source_at": 120,
            "bytes": 20,
            "trigger": "prefetch",
            "generation": 2,
        },
        {"kind": "miss", "source_at": 200, "demand_at": 100},
        {"kind": "miss", "source_at": 250, "demand_at": 100},
        {"kind": "demand_ready", "source_at": 250, "demand_at": 100, "duration_ns": 150},
        {"kind": "eviction", "source_at": 301, "generation": 2},
    ]:
        metrics.observe({**common, **row})
    assert metrics.started == 20 and metrics.device_ns == 0
    assert metrics.recovery_ns == 150
    assert metrics.evicted_unused == 0 and sum(metrics.prefetched.values()) == 20


def test_resource_peak_uses_simultaneous_samples_and_actual_measurement_window(tmp_path):
    from tokenmoe_collect.jsonl import write_json

    write_json(tmp_path / "peaks.json", {"metadata": {"devices": [{"index": 2}, {"index": 3}]}})
    records = [
        {"time": 0, "phase": "warmup", "gpu_mem_used": [100, 10], "process_rss": {"p": 3}},
        {"time": 100, "phase": "measured", "gpu_mem_used": [20, 100], "process_rss": {"p": 5}},
        {"time": 200, "phase": "measured", "gpu_mem_used": [70, 70], "process_rss": {"p": 4}},
        {"time": 400, "phase": "shutdown", "gpu_mem_used": [999, 999], "process_rss": {"p": 9}},
    ]
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    memory = resource_metrics(tmp_path, window=(100, 300))
    assert memory["measured"]["device_bytes"] == {"2": 70, "3": 100}
    assert memory["measured"]["aggregate_device_bytes"] == 140
    assert memory["measured"]["process_tree_rss"] == 5
