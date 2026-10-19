"""Recognize evidenced context rejections that never entered the model engine."""

import copy
import json
from pathlib import Path

from .assets import sha256_file

EMPTY_ENGINE_FIELDS = {
    key: None
    for key in [
        "engine_received_at",
        "first_scheduled_at",
        "first_token_at",
        "inference_finished_at",
        "num_prompt_tokens",
        "num_output_tokens",
        "num_cached_tokens",
        "num_preemptions",
        "engine_finish_status",
        "row_start",
        "row_end",
        "routing_complete",
    ]
}


def is_context_rejection(status, response):
    if status != 400 or not isinstance(response, dict):
        return False
    error = response.get("error")
    return (
        isinstance(error, dict)
        and error.get("type") == "BadRequestError"
        and error.get("param") in {"input_tokens", "input_text"}
        and isinstance(error.get("message"), str)
        and "maximum context length" in error["message"].lower()
    )


def rejection_fields(
    raw, *, replay=None, messages=None, model=None, tools=None, engine_config_id=None
):
    if is_context_rejection(raw.get("http_status_code"), raw.get("http_error_response")):
        return {
            **EMPTY_ENGINE_FIELDS,
            "engine_admission_status": "rejected_before_engine",
            "http_status_code": raw["http_status_code"],
            "rejection_response": raw["http_error_response"],
            "rejection_evidence_source": "observed_http_response",
            "original_response_body_available": True,
        }
    if not replay or messages is None:
        return None
    # Recovery evidence is disclosed as a replay, never presented as a saved original response.
    expected = {
        **model["sampling"],
        "model": model["name"],
        "messages": messages,
        "tools": tools,
        "seed": raw["sampling_seed"],
        "stream": False,
    }
    actual = copy.deepcopy(replay["request_body"])
    actual.pop("vllm_xargs", None)
    if not (
        replay.get("source") == "reproduced_rejection"
        and replay.get("original_request_id") == raw["llm_request_id"]
        and replay.get("engine_config_id") == engine_config_id
        and actual == expected
        and raw.get("request_status") == "request_error"
        and "400 Bad Request" in raw.get("error", "")
        and is_context_rejection(replay.get("http_status_code"), replay.get("response_body"))
    ):
        raise ValueError("Rejection replay does not match the actual failed request and engine")
    return {
        **EMPTY_ENGINE_FIELDS,
        "engine_admission_status": "rejected_before_engine",
        "http_status_code": 400,
        "rejection_response": replay["response_body"],
        "rejection_evidence_source": "reproduced_rejection",
        "original_response_body_available": False,
        "rejection_probe_request_id": replay["probe_request_id"],
    }


def attach_replay_evidence(run_dir, evidence_paths):
    """Repair derived admission annotations without changing any raw producer record."""
    import shutil

    from .jsonl import read_jsonl, write_json

    root = Path(run_dir)
    if (
        json.loads((root / "run_config.json").read_text()).get("collection_profile")
        != "local-qwen3-runtime-v1"
    ):
        raise ValueError("Historical rejection repair is restricted to the local Qwen profile")
    evidence = {
        json.loads(Path(p).read_text())["original_request_id"]: (
            Path(p),
            json.loads(Path(p).read_text()),
        )
        for p in evidence_paths
    }
    raw = sorted(
        [r for p in (root / "raw/harness").glob("*/llm_requests.jsonl") for r in read_jsonl(p)],
        key=lambda r: (r["request_created_at"], r["llm_request_id"]),
    )
    model = json.loads((root / "model_config.json").read_text())
    template = json.loads(next((root / "static/role_templates").rglob("config.json")).read_text())
    engine = json.loads((root / "static/engine_meta.json").read_text())
    engine_requests = {
        r["llm_request_id"]
        for r in read_jsonl(root / "raw/engine" / engine["engine_instance_id"] / "requests.jsonl")
    }
    messages, patches = {}, {}
    for row in raw:
        rid = row["llm_request_id"]
        base = messages[row["messages_base_request_id"]] if row["messages_mode"] == "delta" else []
        messages[rid] = base + row["messages_delta"]
        if rid in evidence:
            path, replay = evidence[rid]
            if rid in engine_requests or replay["probe_request_id"] in engine_requests:
                raise ValueError("A rejection or its replay unexpectedly has an engine record")
            patches[rid] = rejection_fields(
                row,
                replay=replay,
                messages=messages[rid],
                model=model,
                tools=template["tools"],
                engine_config_id=engine["engine_config_id"],
            )
            patches[rid].update(
                request_status=row["request_status"],
                rejection_evidence_ref=str(path.resolve()),
                rejection_evidence_sha256=sha256_file(path),
            )
    if set(evidence) != set(patches):
        raise ValueError("Rejection evidence refers to unknown request IDs")
    source = root / "runtime/llm_requests.jsonl"
    rows = list(read_jsonl(source))
    backup = root / "diagnostics/before-admission-repair"
    backup.mkdir(parents=True, exist_ok=False)
    for name in [
        "runtime/llm_requests.jsonl",
        "validation.json",
        "reports/runtime_summary.json",
        "reports/runtime_summary.md",
    ]:
        path = root / name
        if path.exists():
            target = backup / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for row in rows:
        if row["llm_request_id"] in patches:
            if row.get("routing_ref") or row.get("engine_received_at") is not None:
                raise ValueError("Cannot annotate an admitted request as rejected")
            row.update(patches[row["llm_request_id"]])
    temporary = source.with_suffix(".admission-repair.jsonl")
    with temporary.open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(source)
    write_json(
        backup / "repair.json",
        {
            "raw_data_changed": False,
            "request_ids": sorted(patches),
            "evidence": [str(p) for p in evidence_paths],
        },
    )
    return sorted(patches)
