"""Single-host, single-loop validation of collection/data_collect.md rules 1–14."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .jsonl import read_jsonl, write_json


def validate(run_dir):
    root = Path(run_dir)
    runtime = root / "runtime"
    rules = {
        str(i): {"status": "passed", "violation_count": 0, "examples": []} for i in range(1, 15)
    }
    for i in [10, 13, 14]:
        rules[str(i)]["status"] = "not_applicable"

    def fail(rule, identifier):
        result = rules[str(rule)]
        result["status"] = "failed"
        result["violation_count"] += 1
        if len(result["examples"]) < 20:
            result["examples"].append(str(identifier))

    tables = {
        name: list(read_jsonl(runtime / f"{name}.jsonl"))
        for name in ["application_runs", "sessions", "llm_requests", "tool_calls"]
    }
    for name in ["engine_steps", "prompt_segments", "tool_output_chunks", "host_load"]:
        tables[name] = pq.read_table(runtime / f"{name}.parquet").to_pylist()
    runs = {row["application_run_id"]: row for row in tables["application_runs"]}
    sessions = {row["session_id"]: row for row in tables["sessions"]}
    requests = {row["llm_request_id"]: row for row in tables["llm_requests"]}
    tools = {row["tool_call_id"]: row for row in tables["tool_calls"]}
    profiles = {
        row["model_profile_id"]: row for row in read_jsonl(root / "static/model_profiles.jsonl")
    }
    seen = set()
    for name, field, prefix in [
        ("application_runs", "application_run_id", "run_"),
        ("sessions", "session_id", "ses_"),
        ("llm_requests", "llm_request_id", "req_"),
        ("tool_calls", "tool_call_id", "tc_"),
        ("engine_steps", "step_id", "eng_"),
    ]:
        for row in tables[name]:
            identifier = row[field]
            if not identifier.startswith(prefix) or identifier in seen:
                fail(1, identifier)
            seen.add(identifier)
    for row in requests.values():
        if row["session_id"] not in sessions or row["application_run_id"] not in runs:
            fail(1, row["llm_request_id"])
    for row in sessions.values():
        if row["application_run_id"] not in runs:
            fail(1, row["session_id"])
    grouped = defaultdict(lambda: defaultdict(list))
    for row in requests.values():
        grouped[row["session_id"]][row["step_index"]].append(row["attempt_id"])
    for sid, steps in grouped.items():
        indices = sorted(steps)
        if indices != list(range(len(indices))) or any(
            sorted(values) != list(range(len(values))) for values in steps.values()
        ):
            fail(2, sid)
    for tcid, tool in tools.items():
        consumers = tool.get("llm_request_ids_consuming")
        if consumers is None:
            consumers = (
                [tool["llm_request_id_consuming"]] if tool.get("llm_request_id_consuming") else []
            )
        for rid in consumers:
            request = requests.get(rid)
            if (
                request is None
                or tcid not in request["prev_tool_call_ids"]
                or tool["finished_at"] > request["request_created_at"]
            ):
                fail(3, tcid)
        issuing = requests.get(tool["llm_request_id_issuing"])
        if (
            issuing is None
            or tool.get("issued_at") is None
            or tool["issued_at"] != issuing.get("inference_finished_at")
        ):
            fail(4, tcid)
    for rid, request in requests.items():
        for tcid in request["prev_tool_call_ids"]:
            tool = tools.get(tcid)
            consumers = (
                []
                if tool is None
                else tool.get("llm_request_ids_consuming", [tool.get("llm_request_id_consuming")])
            )
            if tool is None or rid not in consumers:
                fail(3, rid)
    chunks = defaultdict(list)
    for chunk in tables["tool_output_chunks"]:
        chunks[chunk["tool_call_id"]].append(chunk)
        if chunk["tool_call_id"] not in tools:
            fail(5, chunk["tool_call_id"])
    for tcid, tool in tools.items():
        offset, previous_time = 0, tool["started_at"]
        invalid = tool["finished_at"] < tool["started_at"]
        for index, chunk in enumerate(sorted(chunks[tcid], key=lambda c: c["chunk_index"])):
            invalid |= (
                chunk["chunk_index"] != index
                or chunk["byte_offset"] != offset
                or chunk["byte_length"] <= 0
            )
            invalid |= not (previous_time <= chunk["chunk_time"] <= tool["finished_at"])
            offset += chunk["byte_length"]
            previous_time = chunk["chunk_time"]
        path = root / tool["output_ref"]
        invalid |= not path.exists() or offset != tool["output_bytes_raw"]
        if path.exists():
            invalid |= offset != path.stat().st_size
        if invalid:
            fail(5, tcid)
    request_entries = defaultdict(list)
    previous_dispatch = {}
    for step in tables["engine_steps"]:
        eng, sid = step["engine_instance_id"], step["step_id"]
        invalid = step["step_started_at"] <= previous_dispatch.get(eng, -1)
        invalid |= not (
            step["t_sched"] <= step["step_started_at"] <= step["step_finished_at"] <= step["t_end"]
        )
        previous_dispatch[eng] = step["step_started_at"]
        local = defaultdict(list)
        total = 0
        for entry in step["entries"]:
            a, b = entry["token_start"], entry["token_end"]
            total += b - a
            invalid |= (
                a < 0 or b <= a or entry["phase"] not in {"new_prefill", "decode", "recompute"}
            )
            local[entry["vllm_request_id"]].append((a, b))
            if entry["llm_request_id"] is not None:
                request_entries[entry["llm_request_id"]].append((step["step_index"], entry))
        for intervals in local.values():
            intervals.sort()
            invalid |= any(a < previous[1] for previous, (a, _) in zip(intervals, intervals[1:]))
        if invalid or total != step["num_tokens_total"]:
            fail(8, sid)
    for rid, request in requests.items():
        try:
            if request.get("request_status") == "engine_record_missing" or not request.get(
                "routing_ref"
            ):
                raise ValueError("Missing engine record or routing")
            profile = profiles[request["model_profile_id"]]
            with np.load(root / request["routing_ref"], allow_pickle=False) as data:
                positions = data["token_positions"]
                experts = data["experts"]
                indices = data["step_index"]
                start, end = int(data["row_start"]), int(data["row_end"])
                prompt = request["num_prompt_tokens"]
                total = prompt + request["num_output_tokens"]
                normal = request["engine_finish_status"] in {
                    "FINISHED_STOPPED",
                    "FINISHED_LENGTH_CAPPED",
                    "FINISHED_REPETITION",
                }
                valid = 0 <= start <= prompt and start <= end <= total
                valid &= start == request["num_cached_tokens"] and end == request["row_end"]
                valid &= experts.shape == (
                    len(positions),
                    len(profile["moe_layer_ids"]),
                    profile["top_k"],
                )
                valid &= len(data["token_ids"]) == total and indices.shape == positions.shape
                valid &= (
                    positions.dtype == np.int32
                    and data["token_ids"].dtype == np.int32
                    and indices.dtype == np.int32
                )
                valid &= experts.dtype == (
                    np.uint8 if profile["num_routed_experts"] <= 256 else np.uint16
                )
                valid &= np.array_equal(data["layer_ids"], profile["moe_layer_ids"])
                valid &= bool(np.all((positions >= start) & (positions < end))) and bool(
                    np.all(np.diff(positions) > 0)
                )
                valid &= (
                    (positions[0] == start and positions[-1] + 1 == end)
                    if len(positions)
                    else start == end
                )
                valid &= bool(data["routing_complete"]) == request["routing_complete"]
                valid &= bool(np.all(experts < profile["num_routed_experts"]))
                valid &= (
                    (bool(data["routing_complete"]) and end == total - 1)
                    if normal
                    else not bool(data["routing_complete"])
                )
                if not valid:
                    fail(6, rid)
                first = {}
                covered = set(range(start))
                routing_valid = True
                for step_index, entry in sorted(
                    request_entries[rid], key=lambda x: (x[0], x[1]["token_start"])
                ):
                    for position in range(entry["token_start"], entry["token_end"]):
                        phase = entry["phase"]
                        if phase == "recompute":
                            routing_valid &= position in covered
                        else:
                            routing_valid &= position not in covered and phase == (
                                "new_prefill" if position < prompt else "decode"
                            )
                            first[position] = step_index
                            covered.add(position)
                routing_valid &= sorted(first) == positions.tolist()
                routing_valid &= all(
                    first.get(int(p)) == int(s) for p, s in zip(positions, indices)
                )
                if not routing_valid:
                    fail(7, rid)
        except (KeyError, ValueError, OSError, TypeError, IndexError):
            fail(6, rid)
    by_request = defaultdict(list)
    for segment in tables["prompt_segments"]:
        by_request[segment["llm_request_id"]].append(segment)
    for rid, request in requests.items():
        if not request.get("prompt_ref"):
            fail(9, rid)
            continue
        text = (root / request["prompt_ref"]).read_text()
        valid, cursor = True, 0
        located = []
        for segment in by_request[rid]:
            if segment["char_start"] is None:
                valid &= bool(segment.get("alignment_error")) and segment["char_end"] is None
            else:
                located.append(segment)
        for segment in sorted(located, key=lambda s: s["char_start"]):
            valid &= segment["char_start"] == cursor and segment["char_end"] > cursor
            cursor = segment["char_end"]
        if not valid or cursor != len(text):
            fail(9, rid)
    for sid, session in sessions.items():
        run = runs.get(session["application_run_id"])
        if (
            run is None
            or session["final_status"] not in {"succeeded", "failed", "cancelled", "timeout"}
            or session["finished_at"] is None
            or session["finished_at"] > run["run_finished_at"]
        ):
            fail(11, sid)
    for experiment in read_jsonl(root / "static/experiment_configs.jsonl"):
        identifier = experiment["engine_config_id"]
        path = root / "static/equivalence" / f"{identifier}.json"
        if not path.exists():
            fail(12, identifier)
        else:
            record = json.loads(path.read_text())
            if (
                record.get("engine_config_id") != identifier
                or not record.get("passed")
                or record.get("num_prompts") != 50
            ):
                fail(12, identifier)
    report = {"passed": all(row["status"] != "failed" for row in rules.values()), "rules": rules}
    write_json(root / "validation.json", report)
    return report
