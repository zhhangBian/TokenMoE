"""The mini-swe-agent 2.4.6 coupling, verified against the installed source."""

import copy
import logging
import os
import re
import subprocess
from types import SimpleNamespace

from jinja2 import StrictUndefined, Template
from minisweagent.environments.docker import DockerEnvironment, DockerEnvironmentConfig
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL as BASH_TOOL,
)
from minisweagent.models.utils.actions_toolcall import (
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)

from .clock import now
from .executor import StreamingExecutor
from .ids import new_id


def payload_ranges(template, output, template_vars, expected):
    """Instrument Jinja string conversions without changing branches or slicing."""
    nonce = new_id("seg")
    start, end = f"\ue000{nonce}:start\ue001", f"\ue000{nonce}:end\ue001"

    class Payload(str):
        def __str__(self):
            return start + super().__str__() + end

        def __getitem__(self, key):
            return Payload(super().__getitem__(key))

    marked = Template(template, undefined=StrictUndefined).render(
        output={**output, "output": Payload(output["output"])}, **(template_vars or {})
    )
    clean, spans, cursor = "", [], 0
    pattern = re.compile(re.escape(start) + "(.*?)" + re.escape(end), re.DOTALL)
    for match in pattern.finditer(marked):
        clean += marked[cursor : match.start()]
        a = len(clean)
        clean += match.group(1)
        spans.append([a, len(clean)])
        cursor = match.end()
    clean += marked[cursor:]
    if clean != expected:
        raise ValueError(
            "Observation template transforms payload text; provenance cannot be verified"
        )
    return spans


class TracedModel:
    def __init__(self, client, environment, model_config):
        self.client = client
        self.environment = environment
        self.config = model_config
        self.step_index = 0
        self.last_record = None

    def format_message(self, **kwargs):
        return kwargs

    def query(self, messages, **kwargs):
        prepared, provenance = [], []
        appended_tools = []
        for message in messages:
            prepared.append({k: v for k, v in message.items() if k != "extra"})
            extra = message.get("extra", {})
            role = message["role"]
            default_type = (
                "system"
                if role == "system"
                else "agent_prior_output"
                if role == "assistant"
                else "task"
                if len(prepared) == 2
                else "harness_scaffold"
            )
            source = extra.get(
                "tokenmoe_provenance",
                {
                    "segment_type": default_type,
                    "source_tool_call_id": None,
                    "produced_at": self.environment.started_at,
                    "available_at": self.environment.started_at,
                    "payload_ranges": [],
                },
            )
            provenance.append(source)
        previous_count = len(self.client.previous_messages)
        extension = prepared[:previous_count] == self.client.previous_messages
        for source in provenance[previous_count if extension else 0 :]:
            tcid = source.get("source_tool_call_id")
            if tcid and tcid not in appended_tools:
                appended_tools.append(tcid)
        step = self.step_index
        self.step_index += 1
        payload, record = self.client.query(prepared, provenance, step, appended_tools)
        self.last_record = record
        choice = payload["choices"][0]
        assistant = copy.deepcopy(choice["message"])
        calls = [
            SimpleNamespace(id=c["id"], function=SimpleNamespace(**c["function"]))
            for c in assistant.get("tool_calls") or []
        ]
        try:
            actions = parse_toolcall_actions(
                calls,
                format_error_template=self.config["format_error_template"],
                template_kwargs={"finish_reason": choice["finish_reason"]},
            )
        except FormatError as exc:
            exc.messages[0].setdefault("extra", {}).update(cost=0.0, response=payload)
            raise
        parsed_at = now()
        for index, action in enumerate(actions):
            action.update(
                trace_tool_call_id=new_id("tc"),
                llm_request_id_issuing=record["llm_request_id"],
                parsed_at=parsed_at,
                call_index_within_request=index,
            )
        assistant["extra"] = {
            "actions": actions,
            "cost": 0.0,
            "response": payload,
            "tokenmoe_provenance": {
                "segment_type": "agent_prior_output",
                "source_tool_call_id": None,
                "payload_ranges": [],
                "produced_at": record["response_received_at"],
                "available_at": record["response_received_at"],
            },
        }
        return assistant

    def format_observation_messages(self, message, outputs, template_vars=None):
        actions = message.get("extra", {}).get("actions", [])
        result = format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config["observation_template"],
            template_vars=template_vars,
        )
        for action, output, observation in zip(actions, outputs, result):
            tcid = action["trace_tool_call_id"]
            record = self.environment.pending[tcid]
            ranges = payload_ranges(
                self.config["observation_template"], output, template_vars, observation["content"]
            )
            observation["extra"]["tokenmoe_provenance"] = {
                "segment_type": "tool_output",
                "source_tool_call_id": tcid,
                "payload_ranges": ranges,
                "produced_at": record["finished_at"],
                "available_at": record["finished_at"],
            }
            self.environment.observe(tcid, observation["content"])
        return result

    def get_template_vars(self, **kwargs):
        return self.config | kwargs

    def serialize(self):
        return {"info": {"config": {"model": self.config, "model_type": f"{__name__}.TracedModel"}}}


class TracedEnvironment(DockerEnvironment):
    def __init__(self, *, store, session_id, runtime="podman", create_workdir=False, **kwargs):
        self.store = store
        self.session_id = session_id
        self.runtime = runtime
        self.create_workdir = create_workdir
        self.pending = {}
        self.started_at = now()
        self.logger = logging.getLogger(__name__)
        self.container_id = None
        self.config = DockerEnvironmentConfig(executable=runtime, **kwargs)
        if runtime == "local":
            self.container_id = "local"
        else:
            self._start_container()
        forwarded = {key: os.environ[key] for key in self.config.forward_env if key in os.environ}
        self.executor = StreamingExecutor(
            store,
            runtime,
            self.container_id,
            self.config.cwd,
            forwarded | self.config.env,
            self.config.interpreter,
        )

    def _start_container(self):
        self.container_id = "tokenmoe-" + new_id("ses")
        command = [
            self.runtime,
            "run",
            "-d",
            "--name",
            self.container_id,
            "-w",
            "/" if self.create_workdir else self.config.cwd,
            *self.config.run_args,
            self.config.image,
            "sleep",
            self.config.container_timeout,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.pull_timeout,
            )
            if self.create_workdir:
                subprocess.run(
                    [self.runtime, "exec", self.container_id, "mkdir", "-p", "--", self.config.cwd],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
        except BaseException as exc:
            self.cleanup()
            if isinstance(exc, subprocess.CalledProcessError):
                raise RuntimeError(f"Container setup failed: {exc.stderr}") from exc
            raise

    def execute(self, action, cwd="", *, timeout=None):
        tcid = action["trace_tool_call_id"]
        command = action.get("command", "")
        args_path = self.store.directory / "tool_args" / f"{tcid}.txt"
        args_path.parent.mkdir(exist_ok=True)
        args_path.write_text(command, encoding="utf-8")
        output, timing = self.executor.execute(
            command, tcid, timeout or self.config.timeout, cwd=cwd or None
        )
        self.pending[tcid] = {
            "tool_call_id": tcid,
            "session_id": self.session_id,
            "llm_request_id_issuing": action["llm_request_id_issuing"],
            "llm_request_ids_consuming": [],
            "call_index_within_request": action["call_index_within_request"],
            "tool_name": "bash",
            "tool_call_api_id": action.get("tool_call_id"),
            "tool_args_ref": str(args_path.relative_to(self.store.run_dir)),
            "issued_at": None,
            "parsed_at": action["parsed_at"],
            "output_bytes_seen": 0,
            "spawned_session_id": None,
            **timing,
        }
        try:
            self._check_finished(output)
        except Submitted as exc:
            self.observe(tcid, exc.messages[0]["content"])
            raise
        return output

    def observe(self, tool_call_id, text):
        record = self.pending.pop(tool_call_id)
        record["output_bytes_seen"] = len(text.encode("utf-8"))
        self.store.write("tool_calls", record)

    def cleanup(self):
        for record in list(getattr(self, "pending", {}).values()):
            self.store.write("tool_calls", record)
        self.pending = {}
        container_id = getattr(self, "container_id", None)
        if container_id is not None and self.runtime != "local":
            subprocess.run(
                [self.runtime, "rm", "-f", container_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.container_id = None

    def __del__(self):
        # The launcher owns cleanup, including failures; destructors cannot report it.
        pass
