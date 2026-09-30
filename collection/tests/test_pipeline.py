import copy
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from tokenizers import Tokenizer

from tokenmoe_collect.align import align_prompt
from tokenmoe_collect.clock import clock_metadata, now
from tokenmoe_collect.finalize import finalize
from tokenmoe_collect.jsonl import read_jsonl, write_json
from tokenmoe_collect.launcher import run
from tokenmoe_collect.validate import validate

MODEL = Path("/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B")


class FakeHost:
    def sample(self):
        return dict(
            load_avg_1m=0,
            cpu_util=0,
            mem_util=0,
            gpu_util=None,
            gpu_mem_used=None,
            net_rx_bytes=0,
            net_tx_bytes=0,
        )

    def close(self):
        pass


@pytest.fixture
def finalized_run(tmp_path):
    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    engine = tmp_path / "engine/eng_fixture"
    (engine / "routing").mkdir(parents=True)
    meta = {
        "raw_format_version": 1,
        "engine_instance_id": "eng_fixture",
        "engine_config_id": "fixture-config",
        "engine_config": {"tensor_parallel": 2},
        **clock_metadata(),
    }
    write_json(engine / "engine_meta.json", meta)
    write_json(
        engine / "layer_map.json",
        [{"layer_id": 0, "capture_path": "router"}, {"layer_id": 1, "capture_path": "router"}],
    )
    (engine / "requests.jsonl").touch()
    (engine / "steps.jsonl").touch()
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(calls)
            calls.append(body)
            commands = [
                "printf 'hello\\n'; printf 'world\\n' >&2",
                "printf partial; sleep 2",
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo patch",
            ]
            response = {
                "role": "assistant",
                "content": f"action {index}",
                "reasoning": "思考",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": json.dumps({"command": commands[index]}),
                        },
                    }
                ],
            }
            prompt = "\n".join(
                (m.get("reasoning") or "") + "\n" + (m.get("content") or "")
                for m in body["messages"]
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False).ids
            output_tokens = tokenizer.encode("result", add_special_tokens=False).ids
            tokens = prompt_tokens + output_tokens
            rid = body["vllm_xargs"]["tokenmoe_llm_request_id"]
            rows = len(tokens) - 1
            experts = np.broadcast_to(np.arange(8, dtype=np.uint8), (rows, 2, 8)).copy()
            np.savez(
                engine / f"routing/{rid}.npz",
                token_ids=np.asarray(tokens, dtype=np.int32),
                row_start=0,
                row_end=rows,
                routing_complete=True,
                token_positions=np.arange(rows, dtype=np.int32),
                experts=experts,
                step_index=np.full(rows, index, dtype=np.int32),
                layer_ids=np.array([0, 1]),
            )
            received, scheduled, ready, ended = now(), now(), now(), now()
            record = {
                "llm_request_id": rid,
                "vllm_request_id": rid + "-vllm",
                "engine_instance_id": "eng_fixture",
                "engine_received_at": received,
                "first_scheduled_at": scheduled,
                "first_token_at": ready,
                "inference_finished_at": ready,
                "num_prompt_tokens": len(prompt_tokens),
                "num_output_tokens": len(output_tokens),
                "num_cached_tokens": 0,
                "num_preemptions": 0,
                "sampling_seed": body["seed"],
                "engine_finish_status": "FINISHED_STOPPED",
                "row_start": 0,
                "row_end": rows,
                "routing_complete": True,
                "routing_file": f"routing/{rid}.npz",
            }
            entries = [[rid, rid + "-vllm", "new_prefill", 0, len(prompt_tokens)]]
            if rows > len(prompt_tokens):
                entries.append([rid, rid + "-vllm", "decode", len(prompt_tokens), rows])
            with (engine / "requests.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            with (engine / "steps.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "step_index": index,
                            "t_sched": received,
                            "step_started_at": scheduled,
                            "step_finished_at": ready,
                            "t_end": ended,
                            "num_tokens_total": rows,
                            "num_running": 1,
                            "num_waiting": 0,
                            "kv_cache_usage": 0.1,
                            "entries": entries,
                        }
                    )
                    + "\n"
                )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"choices": [{"message": response, "finish_reason": "tool_calls"}]}
                ).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = {
        "name": "fake",
        "model_path": str(MODEL),
        "sampling": {"max_tokens": 256, "temperature": 0},
    }
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(model))
    (tmp_path / "items.json").write_text(
        json.dumps(
            [
                {
                    "instance_id": "test__repo-1",
                    "repo": "test/repo",
                    "problem_statement": "Print hello, handle a timeout, then submit.",
                }
            ]
        )
    )
    config = {
        "model_config": "model.yaml",
        "server_url": f"http://127.0.0.1:{server.server_port}/v1",
        "dataset_path": str(tmp_path / "items.json"),
        "instances": ["test__repo-1"],
        "num_concurrent_sessions": 1,
        "base_seed": 42,
        "step_limit": 4,
        "time_limit_seconds": 30,
        "runtime": "local",
        "local_cwd": str(tmp_path),
        "tool_timeout": 1,
        "output_dir": str(tmp_path / "run"),
        "engine_dir": str(engine),
    }
    (tmp_path / "run.yaml").write_text(yaml.safe_dump(config))
    try:
        root = run(tmp_path / "run.yaml", sampler_source=FakeHost())
        equivalence = root / "static/equivalence/fixture-config.json"
        write_json(
            equivalence, {"engine_config_id": "fixture-config", "passed": True, "num_prompts": 50}
        )
        report = finalize(root)
        assert report["passed"], report
        yield root, calls, config
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_real_agent_fake_transport_roundtrip(finalized_run):
    root, calls, config = finalized_run
    sessions = list(read_jsonl(root / "runtime/sessions.jsonl"))
    assert len(sessions) == 1 and sessions[0]["final_status"] == "succeeded"
    requests = list(read_jsonl(root / "runtime/llm_requests.jsonl"))
    tools = list(read_jsonl(root / "runtime/tool_calls.jsonl"))
    assert len(requests) == len(tools) == 3
    assert calls[1]["messages"][2]["reasoning"] == "思考"
    assert tools[1]["timed_out"] and tools[1]["output_bytes_raw"] == 7
    assert tools[1]["output_bytes_seen"] > 0
    for tool in tools:
        assert tool["issued_at"] is not None
        raw = next((root / "raw/harness").glob(f"*/tool_outputs/{tool['tool_call_id']}.out"))
        assert os.path.samefile(raw, root / tool["output_ref"])
    assert pq.read_table(root / "runtime/host_load.parquet").num_rows >= 1
    assert pq.read_table(root / "runtime/engine_steps.parquet").num_rows == 3
    assert json.loads((root / "alignment.json").read_text())["location_rate"] == 1.0
    with pytest.raises(FileExistsError):
        run(root.parent / "run.yaml", sampler_source=FakeHost())


def test_long_output_has_two_payload_segments_and_unicode_offsets():
    from jinja2 import Template

    from tokenmoe_collect.minisweagent_adapter import payload_ranges
    from tokenmoe_collect.static import minisweagent_config

    tokenizer = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    template = minisweagent_config()["model"]["observation_template"]
    output = {"output": "中文" + ("a" * 12000) + "尾部", "returncode": 0, "exception_info": ""}
    text = Template(template).render(output=output)
    spans = payload_ranges(template, output, {}, text)
    ids = tokenizer.encode("header\n" + text + "\nfooter", add_special_tokens=False).ids
    prompt, segments, metrics = align_prompt(
        tokenizer,
        ids,
        [{"role": "tool", "content": text}],
        [
            {
                "segment_type": "tool_output",
                "source_tool_call_id": "tc_test",
                "payload_ranges": spans,
            }
        ],
        "req_test",
    )
    payloads = [s for s in segments if s["segment_type"] == "tool_output"]
    assert len(payloads) == 2 and metrics["failed"] == 0
    assert [prompt[s["char_start"] : s["char_end"]] for s in payloads] == [
        output["output"][:5000],
        output["output"][-5000:],
    ]
    assert segments[0]["char_start"] == 0 and segments[-1]["char_end"] == len(prompt)
    _, failed, _ = align_prompt(
        tokenizer,
        ids,
        [{"role": "user", "content": "missing marker"}],
        [{"segment_type": "task"}],
        "req_missing",
    )
    assert any(s["alignment_error"] for s in failed)


@pytest.mark.parametrize("rule", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12])
def test_validator_detects_each_applicable_rule(finalized_run, rule):
    root, _, _ = finalized_run

    def mutate_jsonl(name, change):
        path = root / f"runtime/{name}.jsonl"
        rows = list(read_jsonl(path))
        change(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def mutate_parquet(name, change):
        import pyarrow as pa

        path = root / f"runtime/{name}.parquet"
        rows = pq.read_table(path).to_pylist()
        change(rows)
        pq.write_table(pa.Table.from_pylist(rows), path)

    if rule == 1:
        mutate_jsonl("sessions", lambda rows: rows.append(copy.deepcopy(rows[0])))
    elif rule == 2:
        mutate_jsonl("llm_requests", lambda rows: rows[0].update(step_index=99))
    elif rule == 3:
        mutate_jsonl("tool_calls", lambda rows: rows[0].update(finished_at=10**30))
    elif rule == 4:
        mutate_jsonl("tool_calls", lambda rows: rows[0].update(issued_at=0))
    elif rule == 5:
        mutate_parquet("tool_output_chunks", lambda rows: rows[0].update(byte_offset=5))
    elif rule == 6:
        mutate_jsonl("llm_requests", lambda rows: rows[0].update(num_prompt_tokens=999999))
    elif rule == 7:
        mutate_parquet("engine_steps", lambda rows: rows[0]["entries"][0].update(phase="recompute"))
    elif rule == 8:
        mutate_parquet("engine_steps", lambda rows: rows[0].update(num_tokens_total=0))
    elif rule == 9:
        mutate_parquet("prompt_segments", lambda rows: rows[0].update(char_start=1))
    elif rule == 11:
        mutate_jsonl("sessions", lambda rows: rows[0].update(final_status="running"))
    elif rule == 12:
        (root / "static/equivalence/fixture-config.json").unlink()
    report = validate(root)
    assert report["rules"][str(rule)]["status"] == "failed"
    assert report["rules"][str(rule)]["examples"]


def test_validator_accepts_one_tool_result_consumed_by_multiple_attempts(finalized_run):
    root, _, _ = finalized_run
    requests_path = root / "runtime/llm_requests.jsonl"
    requests = list(read_jsonl(requests_path))
    # Exercise the association check independently of the unchanged engine arrays.
    requests[2]["prev_tool_call_ids"].append(requests[1]["prev_tool_call_ids"][0])
    tcid = requests[1]["prev_tool_call_ids"][0]
    requests_path.write_text("".join(json.dumps(row) + "\n" for row in requests))
    tools_path = root / "runtime/tool_calls.jsonl"
    tools = list(read_jsonl(tools_path))
    tool = next(t for t in tools if t["tool_call_id"] == tcid)
    tool["llm_request_ids_consuming"].append(requests[2]["llm_request_id"])
    tools_path.write_text("".join(json.dumps(row) + "\n" for row in tools))
    assert validate(root)["rules"]["3"]["status"] == "passed"
