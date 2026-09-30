"""Chat-completions transport with per-attempt identity and message deltas."""

import copy

import httpx

from .clock import now
from .ids import new_id, sampling_seed


class ChatClient:
    def __init__(
        self,
        *,
        base_url,
        model,
        store,
        session_id,
        application_run_id,
        benchmark_item_id,
        base_seed,
        tools,
        sampling,
        timeout=600,
        retries=2,
    ):
        self.http = httpx.Client(base_url=base_url.rstrip("/") + "/", timeout=timeout)
        self.model = model
        self.store = store
        self.session_id = session_id
        self.application_run_id = application_run_id
        self.benchmark_item_id = benchmark_item_id
        self.base_seed = base_seed
        self.tools = tools
        self.sampling = sampling
        self.retries = retries
        self.previous_messages = []
        self.previous_request_id = None

    def query(self, messages, provenance, step_index, prev_tool_call_ids):
        if len(messages) != len(provenance):
            raise ValueError("Every message needs provenance")
        for attempt_id in range(self.retries + 1):
            created = now()
            identifier = new_id("req")
            seed = sampling_seed(self.base_seed, self.benchmark_item_id, step_index, attempt_id)
            extension = messages[: len(self.previous_messages)] == self.previous_messages
            base_count = len(self.previous_messages) if extension else 0
            record = {
                "llm_request_id": identifier,
                "application_run_id": self.application_run_id,
                "session_id": self.session_id,
                "step_index": step_index,
                "attempt_id": attempt_id,
                "prev_tool_call_ids": list(prev_tool_call_ids),
                "request_created_at": created,
                "request_sent_at": None,
                "response_received_at": None,
                "sampling_seed": seed,
                "messages_mode": "delta" if extension and self.previous_request_id else "full",
                "messages_base_request_id": self.previous_request_id if extension else None,
                "messages_base_count": base_count,
                "messages_delta": copy.deepcopy(messages[base_count:]),
                "message_provenance": copy.deepcopy(provenance[base_count:]),
                "assistant_response": None,
                "finish_reason": None,
                "request_status": "pending",
            }
            self.previous_messages = copy.deepcopy(messages)
            self.previous_request_id = identifier
            body = {
                **self.sampling,
                "model": self.model,
                "messages": messages,
                "tools": self.tools,
                "seed": seed,
                "stream": False,
                "vllm_xargs": {"tokenmoe_llm_request_id": identifier},
            }
            try:
                record["request_sent_at"] = now()
                response = self.http.post("chat/completions", json=body)
                record["response_received_at"] = now()
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                message = choice["message"]
                # Some OpenAI-compatible endpoints use the older key.
                if not message.get("reasoning") and message.get("reasoning_content"):
                    message["reasoning"] = message["reasoning_content"]
                record.update(
                    assistant_response=message,
                    finish_reason=choice["finish_reason"],
                    request_status="succeeded",
                    usage=payload.get("usage"),
                )
            except Exception as exc:
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, httpx.HTTPStatusError)
                    and (exc.response.status_code == 429 or exc.response.status_code >= 500)
                )
                record.update(
                    request_status="transport_error" if retryable else "request_error",
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.store.write("llm_requests", record)
                if retryable and attempt_id < self.retries:
                    continue
                raise
            self.store.write("llm_requests", record)
            return payload, record
        raise AssertionError("Unreachable")

    def close(self):
        self.http.close()
