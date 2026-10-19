"""Local collection integrity and recovery checks without GPUs or downloads."""

import copy
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_pipeline import finalized_run as finalized_run
from tokenizers import Tokenizer

from tokenmoe_collect.admission import rejection_fields
from tokenmoe_collect.align import align_prompt
from tokenmoe_collect.assets import inspect_dataset, inspect_model, restore_asset
from tokenmoe_collect.client import ChatClient
from tokenmoe_collect.finalize import write_table
from tokenmoe_collect.jsonl import read_jsonl, write_json
from tokenmoe_collect.local import LocalCollection, health_snapshot, make_plan
from tokenmoe_collect.records import RecordStore
from tokenmoe_collect.report import (
    artifact_checksums,
    classify_units,
    distribution,
    summarize_runtime,
)
from tokenmoe_collect.validate import validate

COLLECTION = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(
        f"local_{name}", COLLECTION / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_model(root):
    root.mkdir(parents=True)
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        write_json(root / name, {})
    write_json(
        root / "model.safetensors.index.json", {"weight_map": {"tensor": "model-1.safetensors"}}
    )
    header = json.dumps({"tensor": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    shard = struct.pack("<Q", len(header)) + header + b"\0" * 4
    (root / "model-1.safetensors").write_bytes(shard)
    return shard


def test_asset_checks_find_missing_and_corrupt_shards(tmp_path):
    tiny_model(tmp_path / "model")
    assert inspect_model(tmp_path / "model")["valid"]
    shard = tmp_path / "model/model-1.safetensors"
    shard.unlink()
    assert inspect_model(shard.parent)["missing"] == [shard.name]
    shard.write_bytes(b"damaged")
    assert inspect_model(shard.parent)["damaged"] == [shard.name]
    assert not inspect_dataset(tmp_path / "missing-data")["valid"]


def test_restore_reuses_complete_assets_and_scopes_missing_download(tmp_path, monkeypatch):
    root = tmp_path / "model"
    shard = tiny_model(root)
    download = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    job = {
        "repo_id": "Qwen/Qwen3-30B-A3B",
        "repo_type": "model",
        "local_dir": str(root),
        "revision": "pinned",
    }
    assert restore_asset(job)["valid"]
    download.assert_not_called()
    (root / "model-1.safetensors").unlink()
    download.side_effect = lambda **kwargs: (root / "model-1.safetensors").write_bytes(shard)
    assert restore_asset(job)["valid"]
    assert download.call_args.kwargs["allow_patterns"] == ["model-1.safetensors"]
    assert download.call_args.kwargs["repo_id"] == "Qwen/Qwen3-30B-A3B"


def test_qwen_configuration_is_bounded_and_dry_run_has_no_writes(tmp_path, capsys):
    args = [
        "--profile",
        "local-qwen3-runtime-v1",
        "--model-root",
        str(tmp_path / "models"),
        "--dataset-dir",
        str(tmp_path / "dataset"),
        "--output-dir",
        str(tmp_path / "configs"),
        "--run-root",
        str(tmp_path / "runs"),
    ]
    prepare = script("prepare_configs")
    prepare.main([*args, "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert not (tmp_path / "configs").exists()
    configs = plan["configurations"]
    assert len(configs) == 3
    assert sorted(x["instance_count"] for x in configs.values() if "instance_count" in x) == [3, 20]
    prepare.main(args)
    output = tmp_path / "configs"
    primary = output / "runs/qwen3-30b-a3b-full.yaml"
    pilot = output / "runs/qwen3-30b-a3b-pilot.yaml"
    model = output / "models/qwen3-30b-a3b.yaml"
    local = make_plan(primary, pilot, model)
    assert len(local["units"]) == 40
    assert local["primary"]["concurrency_matrix"] == [1, 4]
    assert local["model"]["serve"]["max_model_len"] == 32768
    assert "hf_overrides" not in local["model"]["serve"]


def test_proven_template_omissions_cannot_hide_lost_payloads():
    tokenizer = Tokenizer.from_file("/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B/tokenizer.json")
    messages = [
        {"role": "assistant", "reasoning": "omitted reasoning", "content": "visible"},
        {"role": "tool", "content": "lost payload"},
    ]
    tokens = tokenizer.encode("visible", add_special_tokens=False).ids
    provenance = [
        {"segment_type": "agent_prior_output"},
        {"segment_type": "tool_output", "source_tool_call_id": "tc_x"},
    ]

    def digest(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()

    evidence = {
        "full_render_matches_engine": True,
        "ablated_render_unchanged": True,
        "engine_prompt_ids_sha256": digest(tokens),
        "messages_sha256": digest(messages),
        "template_sha256": "template",
        "tokenizer_sha256": "tokenizer",
    }
    _, segments, metrics = align_prompt(
        tokenizer,
        tokens,
        messages,
        provenance,
        "req_x",
        template_omissions={(0, "reasoning"): evidence, (1, "content"): evidence},
    )
    assert metrics["template_omitted"] == 1 and metrics["failed"] == 2
    lost = next(x for x in segments if x.get("source_tool_call_id") == "tc_x")
    assert lost["alignment_error"] == "message_not_found"
    wrong = {**evidence, "engine_prompt_ids_sha256": "different"}
    _, _, metrics = align_prompt(
        tokenizer,
        tokens,
        messages,
        provenance,
        "req_x",
        template_omissions={(0, "reasoning"): wrong},
    )
    assert metrics["template_omitted"] == 0


def test_failure_sessions_are_valid_but_missing_routing_requires_recollection(finalized_run):
    root, _, _ = finalized_run
    path = root / "runtime/sessions.jsonl"
    session = next(read_jsonl(path))
    session.update(final_status="timeout", exit_reason="LimitsExceeded")
    path.write_text(json.dumps(session) + "\n")
    assert validate(root)["passed"]
    assert classify_units(root)[0]["valid"]
    request = next(read_jsonl(root / "runtime/llm_requests.jsonl"))
    (root / request["routing_ref"]).unlink()
    validate(root)
    unit = classify_units(root)[0]
    assert not unit["valid"] and "routing_missing" in unit["invalid_reasons"]


def test_startup_only_session_does_not_count_as_coverage(finalized_run):
    root, _, _ = finalized_run
    session = next(read_jsonl(root / "runtime/sessions.jsonl"))
    session["session_id"] = "ses_no_inference"
    session.update(final_status="failed", exit_reason="SetupError")
    with (root / "runtime/sessions.jsonl").open("a") as output:
        output.write(json.dumps(session) + "\n")
    by_session = {x["session_id"]: x for x in classify_units(root)}
    assert "no_real_inference" in by_session["ses_no_inference"]["invalid_reasons"]


def test_local_summary_defaults_to_zero_submission_threshold(finalized_run):
    root, _, _ = finalized_run
    config = json.loads((root / "run_config.json").read_text())
    write_json(root / "run_config.json", {**config, "collection_profile": "local-qwen3-runtime-v1"})
    write_json(root / "run_summary.json", {"sessions": [{"status": "failed"}]})
    report = script("summarize_runs").summarize(root)
    assert report["automated_checks_passed"]
    assert report["runs"][0]["success_rate"] == 0
    assert not script("summarize_runs").summarize(root, min_success_rate=0.5)[
        "automated_checks_passed"
    ]


def test_runtime_report_counts_first_computations_and_retains_real_outcomes(finalized_run):
    root, _, _ = finalized_run
    report = summarize_runtime(root)
    expected = sum(
        len(np.load(root / r["routing_ref"])["token_positions"])
        for r in read_jsonl(root / "runtime/llm_requests.jsonl")
    )
    assert report["counts"]["first_computed_tokens"] == expected
    total_lookups = sum(np.asarray(x).sum() for x in report["expert_lookup_counts"].values())
    assert total_lookups == expected * 2 * 8
    assert report["tool_outcomes"]["timeout"] == 1
    assert report["g1_task_groups"] == ["test__repo-1"]
    assert report["units"][0]["valid"]
    assert distribution([])["p95"] is None
    result = artifact_checksums(root)
    assert result["files"] > result["unique_inodes"]


def test_recovery_preserves_completed_units_and_protocol(tmp_path, monkeypatch):
    plan = {
        "profile": "local-qwen3-runtime-v1",
        "protocol_id": "original",
        "primary": {"concurrency_matrix": [1], "base_seed": 42},
        "model_config": str(tmp_path / "model.yaml"),
        "units": [
            {
                "benchmark_item_id": "done",
                "concurrency": 1,
                "base_seed": 42,
                "protocol_id": "original",
                "status": "valid",
                "attempts": [],
                "artifact": "existing",
            },
            {
                "benchmark_item_id": "missing",
                "concurrency": 1,
                "base_seed": 42,
                "protocol_id": "original",
                "status": "pending",
                "attempts": [],
            },
        ],
    }
    collection = LocalCollection(
        tmp_path / "experiment", plan, fork_venv=tmp_path, devices=["1", "2"]
    )
    monkeypatch.setattr(collection, "gate", lambda: tmp_path / "gate.json")
    observed = []

    def failed_run(label, args):
        config = json.loads(Path(args[args.index("--config") + 1]).read_text())
        observed.append(config["instances"])
        return 1

    monkeypatch.setattr(collection, "execute", failed_run)
    with pytest.raises(RuntimeError, match="invalid units"):
        collection.run_stage("collect")
    assert observed == [["missing"]]
    assert collection.state["units"][0]["artifact"] == "existing"
    assert collection.state["units"][0]["status"] == "valid"
    assert collection.state["units"][1]["status"] == "needs_recollection"
    assert len(list(read_jsonl(collection.root / "recovery_events.jsonl"))) == 1
    changed = copy.deepcopy(plan)
    changed["protocol_id"] = "different"
    with pytest.raises(ValueError, match="Protocol/configuration changed"):
        LocalCollection(collection.root, changed, fork_venv=tmp_path, devices=["1", "2"])
    assert health_snapshot(collection.root)["valid_units"] == 1


def test_partial_prefix_holes_and_recompute_do_not_fabricate_demand(tmp_path):
    root = tmp_path / "partial"
    runtime = root / "runtime"
    runtime.mkdir(parents=True)

    def table(name, records):
        (runtime / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))

    application = {
        "application_run_id": "run_partial",
        "benchmark_item_id": "partial-task",
        "run_finished_at": 40,
    }
    session = {
        "session_id": "ses_partial",
        "application_run_id": "run_partial",
        "final_status": "timeout",
        "exit_reason": "TimeExceeded",
        "started_at": 1,
        "finished_at": 40,
    }
    request = {
        "llm_request_id": "req_partial",
        "vllm_request_id": "vllm-partial",
        "session_id": "ses_partial",
        "application_run_id": "run_partial",
        "step_index": 0,
        "attempt_id": 0,
        "prev_tool_call_ids": [],
        "request_status": "succeeded",
        "model_profile_id": "profile",
        "num_prompt_tokens": 8,
        "num_output_tokens": 2,
        "num_cached_tokens": 3,
        "num_preemptions": 1,
        "row_start": 3,
        "row_end": 9,
        "routing_complete": False,
        "engine_finish_status": "FINISHED_ABORTED",
        "routing_ref": "runtime/partial.npz",
        "prompt_ref": "runtime/prompt.txt",
        "request_created_at": 2,
        "request_sent_at": 3,
        "response_received_at": 35,
        "engine_received_at": 4,
        "first_scheduled_at": 10,
        "first_token_at": 28,
        "inference_finished_at": 34,
    }
    for name, records in [
        ("application_runs", [application]),
        ("sessions", [session]),
        ("llm_requests", [request]),
        ("tool_calls", []),
    ]:
        table(name, records)
    positions = np.array([3, 4, 7, 8], dtype=np.int32)
    np.savez(
        runtime / "partial.npz",
        token_ids=np.arange(10, dtype=np.int32),
        row_start=3,
        row_end=9,
        routing_complete=False,
        token_positions=positions,
        experts=np.broadcast_to(np.arange(8, dtype=np.uint8), (4, 2, 8)),
        step_index=np.array([0, 0, 2, 3], dtype=np.int32),
        layer_ids=[0, 1],
    )
    steps = []
    for i, (phase, start, end) in enumerate(
        [("new_prefill", 3, 5), ("recompute", 3, 5), ("new_prefill", 7, 8), ("decode", 8, 9)]
    ):
        stamp = 10 + i * 5
        steps.append(
            {
                "step_id": f"eng_partial:{i}",
                "engine_instance_id": "eng_partial",
                "step_index": i,
                "t_sched": stamp,
                "step_started_at": stamp + 1,
                "step_finished_at": stamp + 2,
                "t_end": stamp + 3,
                "num_tokens_total": end - start,
                "num_running": 1,
                "num_waiting": 0,
                "entries": [
                    {
                        "llm_request_id": "req_partial",
                        "vllm_request_id": "vllm-partial",
                        "phase": phase,
                        "token_start": start,
                        "token_end": end,
                    }
                ],
            }
        )
    (runtime / "prompt.txt").write_text("prompt")
    segments = [
        {
            "llm_request_id": "req_partial",
            "segment_type": "task",
            "char_start": 0,
            "char_end": 6,
            "token_start": 0,
            "token_end": 8,
            "alignment_error": None,
            "message_index": 0,
            "message_field": "content",
        }
    ]
    for name, records in [
        ("engine_steps", steps),
        ("prompt_segments", segments),
        ("tool_output_chunks", []),
        ("host_load", []),
    ]:
        pq.write_table(pa.Table.from_pylist(records), runtime / f"{name}.parquet")
    (root / "static").mkdir()
    profile = {
        "model_profile_id": "profile",
        "moe_layer_ids": [0, 1],
        "top_k": 8,
        "num_routed_experts": 128,
    }
    (root / "static/model_profiles.jsonl").write_text(json.dumps(profile) + "\n")
    (root / "static/experiment_configs.jsonl").write_text(
        json.dumps({"engine_config_id": "config", "concurrency": {"num_concurrent_sessions": 1}})
        + "\n"
    )
    write_json(
        root / "static/equivalence/config.json",
        {"engine_config_id": "config", "num_prompts": 50, "passed": True},
    )
    write_json(root / "alignment.json", {"location_rate": 1.0, "rendered_location_rate": 1.0})
    assert validate(root)["passed"]
    report = summarize_runtime(root)
    assert report["counts"]["first_computed_tokens"] == 4
    assert report["counts"]["new_prefill_tokens"] == 3
    assert report["counts"]["decode_tokens"] == 1
    assert report["counts"]["recompute_tokens"] == 2
    assert sum(np.asarray(x).sum() for x in report["expert_lookup_counts"].values()) == 4 * 2 * 8
    assert report["seconds"]["tool_first_output"]["p50"] is None


def test_early_failure_allocates_a_new_attempt_without_overwriting_logs(tmp_path):
    plan = {"protocol_id": "original", "primary": {}, "units": []}
    collector = LocalCollection(tmp_path, plan, fork_venv=tmp_path, devices=["1", "2"])
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/pilot-01.log").write_text("early failure")
    parent = tmp_path / "pilot"
    parent.mkdir()
    assert collector.next_attempt(parent, "pilot") == 2
    assert (tmp_path / "logs/pilot-01.log").read_text() == "early failure"


CONTEXT_REJECTION = {
    "error": {
        "message": "This model's maximum context length is 32768 tokens.",
        "type": "BadRequestError",
        "param": "input_tokens",
        "code": 400,
    }
}


def test_http_errors_preserve_the_actual_rejection_response(tmp_path):
    with RecordStore(tmp_path) as store:
        client = ChatClient(
            base_url="http://test.invalid/v1",
            model="fake",
            store=store,
            session_id="ses_test",
            application_run_id="run_test",
            benchmark_item_id="item",
            base_seed=42,
            tools=[],
            sampling={"max_tokens": 4096},
            retries=0,
        )
        client.http.close()
        client.http = httpx.Client(
            base_url="http://test.invalid/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json=CONTEXT_REJECTION)
            ),
        )
        try:
            with pytest.raises(httpx.HTTPStatusError):
                client.query(
                    [{"role": "user", "content": "long input"}], [{"segment_type": "task"}], 0, []
                )
        finally:
            client.close()
        raw = next(read_jsonl(store.directory / "llm_requests.jsonl"))
        assert raw["http_status_code"] == 400
        assert raw["http_error_response"] == CONTEXT_REJECTION
        assert rejection_fields(raw)["rejection_evidence_source"] == "observed_http_response"


def test_context_replay_evidence_must_match_the_original_input():
    messages = [{"role": "user", "content": "original"}]
    model = {"name": "fake", "sampling": {"max_tokens": 4096}}
    raw = {
        "llm_request_id": "req_rejected",
        "sampling_seed": 42,
        "request_status": "request_error",
        "error": "400 Bad Request",
    }
    replay = {
        "source": "reproduced_rejection",
        "original_request_id": "req_rejected",
        "probe_request_id": "req_probe",
        "engine_config_id": "config",
        "http_status_code": 400,
        "response_body": CONTEXT_REJECTION,
        "request_body": {
            **model["sampling"],
            "model": "fake",
            "messages": messages,
            "tools": [],
            "seed": 42,
            "stream": False,
        },
    }
    assert not rejection_fields(
        raw, replay=replay, messages=messages, model=model, tools=[], engine_config_id="config"
    )["original_response_body_available"]
    changed = copy.deepcopy(replay)
    changed["request_body"]["messages"][0]["content"] = "different"
    with pytest.raises(ValueError, match="does not match"):
        rejection_fields(
            raw, replay=changed, messages=messages, model=model, tools=[], engine_config_id="config"
        )


def test_pre_engine_context_rejection_has_no_fabricated_routing(finalized_run):
    root, _, _ = finalized_run
    original = list(read_jsonl(root / "runtime/llm_requests.jsonl"))
    rejected = {
        "llm_request_id": "req_rejected",
        "session_id": original[-1]["session_id"],
        "application_run_id": original[-1]["application_run_id"],
        "step_index": 3,
        "attempt_id": 0,
        "prev_tool_call_ids": [],
        "request_status": "request_error",
        "prompt_ref": None,
        "routing_ref": None,
        **rejection_fields({"http_status_code": 400, "http_error_response": CONTEXT_REJECTION}),
    }
    path = root / "runtime/llm_requests.jsonl"
    with path.open("a") as output:
        output.write(json.dumps(rejected) + "\n")
    assert not validate(root)["passed"]
    experiment_path = root / "static/experiment_configs.jsonl"
    experiment = next(read_jsonl(experiment_path))
    experiment["collection_profile"] = "local-qwen3-runtime-v1"
    experiment_path.write_text(json.dumps(experiment) + "\n")
    assert validate(root)["passed"]
    assert classify_units(root)[0]["valid"]
    bad = {**rejected, "engine_received_at": 1}
    path.write_text("".join(json.dumps(row) + "\n" for row in [*original, bad]))
    assert not validate(root)["passed"]


def test_parquet_preserves_optional_evidence_fields_in_later_rows(tmp_path):
    path = tmp_path / "segments.parquet"
    write_table(
        path,
        [
            {"segment_type": "other", "alignment_error": None},
            {
                "segment_type": "agent_prior_output",
                "alignment_error": "template_omitted",
                "alignment_status": "template_omitted",
                "omission_evidence_json": '{"verified": true}',
            },
        ],
    )
    rows = pq.read_table(path).to_pylist()
    assert rows[1]["alignment_status"] == "template_omitted"
    assert json.loads(rows[1]["omission_evidence_json"])["verified"]
