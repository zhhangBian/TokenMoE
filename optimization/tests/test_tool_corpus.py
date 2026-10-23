import json

import pytest
from tokenmoe_collect.jsonl import write_json
from tokenmoe_collect.static import load_benchmark

from tokenmoe_opt.tool_corpus import coverage, regression_targets, validate_case


def test_regression_targets_use_real_runner_labels_for_data_only_and_bare_test_patches():
    django = "diff --git a/tests/validators/invalid_urls.txt b/tests/validators/invalid_urls.txt"
    assert regression_targets("django__task", django, ["test_x (module.Class)"]) == ["validators"]
    sympy = "diff --git a/sympy/core/tests/test_expr.py b/sympy/core/tests/test_expr.py"
    assert regression_targets("sympy__task", sympy, ["test_issue_1"]) == [
        "sympy/core/tests/test_expr.py"
    ]


def test_benchmark_projection_excludes_solution_and_preserves_requested_groups(tmp_path):
    path = tmp_path / "benchmark.json"
    write_json(path, [{"instance_id": "t", "test_patch": "tests", "patch": "solution"}])
    assert load_benchmark(path, ["t"], columns=["instance_id", "test_patch"]) == [
        {"instance_id": "t", "test_patch": "tests"}
    ]
    with pytest.raises(ValueError, match="instance_id"):
        load_benchmark(path, ["t"], columns=["test_patch"])


@pytest.fixture
def case(tmp_path):
    producer = tmp_path / "raw/harness/fixture"
    producer.mkdir(parents=True)
    (producer / "output.out").write_bytes(b"abcd")
    job = {
        "job_id": "j",
        "task": "task",
        "image_id": "fixture",
        "command": "synthetic-unit-fixture",
        "timeout_seconds": 60,
    }
    write_json(
        tmp_path / "case.json",
        {
            **job,
            "status": "complete",
            "producer": "raw/harness/fixture",
            "tool_id": "tool",
            "preflight": {"boot_id": "clock"},
            "returncode": -1,
            "timing": {
                "started_at": 0,
                "finished_at": 61_000_000_000,
                "timed_out": True,
                "output_ref": "raw/harness/fixture/output.out",
                "output_bytes_raw": 4,
            },
        },
    )
    write_json(producer / "producer.json", {"clock_domain_id": "clock"})
    write_json(
        producer / "control_export_status.json",
        {"flush_complete": True, "dropped": 0, "export_errors": 0},
    )
    (producer / "control_events.jsonl").write_text("")
    chunks = [
        {
            "tool_call_id": "tool",
            "chunk_index": 0,
            "byte_offset": 0,
            "byte_length": 2,
            "chunk_time": 1_000_000_000,
        },
        {
            "tool_call_id": "tool",
            "chunk_index": 1,
            "byte_offset": 2,
            "byte_length": 2,
            "chunk_time": 59_000_000_000,
        },
    ]
    (producer / "tool_output_chunks.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in chunks)
    )
    return tmp_path, job, producer


def test_tool_censoring_excludes_cleanup_delay_and_does_not_count_as_completion(case):
    root, job, _ = case
    result = validate_case(root, job)
    assert result["observed_seconds"] == 61 and result["seconds"] == 60
    assert result["early_lead"] == 59 and result["censored"]
    checked = coverage([result] * 10, {"task_groups": 2, "completed_gt30_seconds": 1})
    assert not checked["passed"] and checked["actual"]["task_groups"] == 1
    assert checked["actual"]["completed_gt30_seconds"] == 0


@pytest.mark.parametrize("invalid", ["gap", "future", "clock", "loss"])
def test_tool_stream_validation_rejects_missing_or_unavailable_observations(case, invalid):
    root, job, producer = case
    if invalid in {"gap", "future"}:
        path = producer / "tool_output_chunks.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["byte_offset" if invalid == "gap" else "chunk_time"] = (
            3 if invalid == "gap" else 62_000_000_000
        )
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif invalid == "clock":
        write_json(producer / "producer.json", {"clock_domain_id": "other"})
    else:
        write_json(
            producer / "control_export_status.json",
            {"flush_complete": True, "dropped": 1, "export_errors": 0},
        )
    with pytest.raises(ValueError):
        validate_case(root, job)
