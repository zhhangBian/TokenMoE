import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tokenmoe_collect.client import ChatClient
from tokenmoe_collect.executor import StreamingExecutor
from tokenmoe_collect.hostload import HostSampler
from tokenmoe_collect.ids import content_id, new_id, sampling_seed
from tokenmoe_collect.jsonl import JsonlWriter, read_jsonl
from tokenmoe_collect.records import RecordStore


class CoreTests(unittest.TestCase):
    def test_content_hash_is_order_independent_and_seeds_are_stable(self):
        self.assertEqual(content_id({"a": 1, "b": 2}), content_id({"b": 2, "a": 1}))
        self.assertEqual(sampling_seed(3, "task", 4, 0), sampling_seed(3, "task", 4, 0))
        self.assertNotEqual(sampling_seed(3, "task", 4, 0), sampling_seed(3, "task", 4, 1))
        self.assertEqual(len({new_id("req") for _ in range(1000)}), 1000)

    def test_concurrent_jsonl_writes_remain_complete_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            with JsonlWriter(path) as writer:
                threads = [
                    threading.Thread(
                        target=lambda i=i: [writer.write({"id": i, "value": j}) for j in range(30)]
                    )
                    for i in range(4)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            self.assertEqual(len(list(read_jsonl(path))), 120)

    def test_http_retries_use_distinct_ids_and_preserve_reasoning(self):
        bodies = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                bodies.append(body)
                if len(bodies) == 1:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "x",
                                        "reasoning": "思考",
                                    },
                                    "finish_reason": "stop",
                                }
                            ]
                        }
                    ).encode()
                )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, RecordStore(tmp) as store:
                client = ChatClient(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="fake",
                    store=store,
                    session_id="ses_test",
                    application_run_id="run_test",
                    benchmark_item_id="item",
                    base_seed=7,
                    tools=[],
                    sampling={"temperature": 0},
                    retries=1,
                )
                try:
                    messages = [{"role": "system", "content": "test"}]
                    payload, _ = client.query(messages, [{"segment_type": "system"}], 0, [])
                    messages.append(payload["choices"][0]["message"])
                    client.query(
                        messages,
                        [{"segment_type": "system"}, {"segment_type": "agent_prior_output"}],
                        1,
                        [],
                    )
                finally:
                    client.close()
                records = list(read_jsonl(store.directory / "llm_requests.jsonl"))
                self.assertEqual([r["attempt_id"] for r in records], [0, 1, 0])
                self.assertEqual([r["step_index"] for r in records], [0, 0, 1])
                self.assertEqual(len({r["llm_request_id"] for r in records}), 3)
                self.assertEqual(records[1]["messages_delta"], [])
                self.assertEqual(len(records[2]["messages_delta"]), 1)
                self.assertEqual(bodies[2]["messages"][1]["reasoning"], "思考")
                for body, record in zip(bodies, records):
                    self.assertEqual(
                        body["vllm_xargs"]["tokenmoe_llm_request_id"], record["llm_request_id"]
                    )
                    self.assertEqual(body["seed"], record["sampling_seed"])
                    self.assertLessEqual(record["request_created_at"], record["request_sent_at"])
                    self.assertLessEqual(record["request_sent_at"], record["response_received_at"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_tee_preserves_merged_bytes_and_timed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp, RecordStore(tmp) as store:
            executor = StreamingExecutor(store, runtime="local", cwd=tmp)
            output, timing = executor.execute(
                "printf 'one\\r\\n'; sleep 0.12; printf 'two\\n' >&2; sleep 0.12; printf 'three\\n'",
                "tc_test",
                5,
            )
            raw = (store.outputs / "tc_test.out").read_bytes()
            self.assertEqual(raw, b"one\r\ntwo\nthree\n")
            self.assertEqual(output["output"], "one\ntwo\nthree\n")
            self.assertEqual(output["returncode"], 0)
            self.assertEqual(timing["output_bytes_raw"], len(raw))
            chunks = list(read_jsonl(store.directory / "tool_output_chunks.jsonl"))
            self.assertGreaterEqual(len(chunks), 3)
            offset = 0
            for i, chunk in enumerate(chunks):
                self.assertEqual(chunk["chunk_index"], i)
                self.assertEqual(chunk["byte_offset"], offset)
                self.assertLessEqual(timing["started_at"], chunk["chunk_time"])
                self.assertLessEqual(chunk["chunk_time"], timing["finished_at"])
                offset += chunk["byte_length"]
            self.assertEqual(offset, len(raw))
            self.assertEqual(executor.active, 0)

    def test_timeout_preserves_partial_output_and_kills_children(self):
        with tempfile.TemporaryDirectory() as tmp, RecordStore(tmp) as store:
            executor = StreamingExecutor(store, runtime="local", cwd=tmp)
            start = time.monotonic()
            output, timing = executor.execute("printf partial; sleep 30", "tc_timeout", 0.2)
            self.assertLess(time.monotonic() - start, 3)
            self.assertTrue(timing["timed_out"])
            self.assertEqual(output["output"], "partial")
            self.assertEqual(output["returncode"], -1)
            self.assertEqual(output["extra"]["exception_type"], "TimeoutExpired")
            self.assertEqual((store.outputs / "tc_timeout.out").read_bytes(), b"partial")

    def test_sampler_writes_repeated_rows_with_null_gpu_metrics(self):
        class FakeSource:
            closed = False

            def sample(self):
                return {
                    "load_avg_1m": 1,
                    "cpu_util": 2,
                    "mem_util": 3,
                    "gpu_util": None,
                    "gpu_mem_used": None,
                    "net_rx_bytes": 4,
                    "net_tx_bytes": 5,
                }

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp, RecordStore(tmp) as store:
            source = FakeSource()
            sampler = HostSampler(store, lambda: 2, source, interval=0.02).start()
            time.sleep(0.09)
            sampler.close()
            rows = list(read_jsonl(store.directory / "host_load.jsonl"))
            self.assertGreaterEqual(len(rows), 3)
            self.assertTrue(source.closed)
            self.assertTrue(
                all(
                    row["gpu_util"] is None and row["concurrent_tool_processes"] == 2
                    for row in rows
                )
            )
            self.assertEqual([row["time"] for row in rows], sorted(row["time"] for row in rows))


if __name__ == "__main__":
    unittest.main()
