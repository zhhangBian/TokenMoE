#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from typing import Any

from dataset_adapters.common import (
    CLAIM_SCOPE_REAL_AGENT,
    PromptBlock,
    first_text,
    iter_raw_records,
    make_record,
    run_converter,
)


SOURCE_DATASET = "nebius/SWE-agent-trajectories"
OUTPUT_NAME = "swe_agent_prompt_workloads.jsonl"
MAX_EVENTS_PER_TRAJECTORY = 8

# Version of the action-sequence heuristics that derive trajectory_phase,
# tool_type, and event_outcome. Recorded in the conversion manifest so
# downstream analysis can attribute results to the exact derivation rules.
HEURISTIC_VERSION = "swe_agent.metadata_heuristics.v1"

TRAJECTORY_PHASES = (
    "issue_understanding",
    "edit",
    "test",
    "debug",
    "finalize",
)
TOOL_TYPES = ("read_file", "search", "edit", "run_test", "inspect_error", "other")
EVENT_OUTCOMES = ("test_failed", "patch_applied", "error_observed", "none")

_FENCED_BLOCK_RE = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)

_EDIT_COMMANDS = {"edit", "create", "insert", "str_replace", "apply_patch"}
_SEARCH_COMMANDS = {
    "search_file",
    "search_dir",
    "find_file",
    "find",
    "grep",
    "ls",
    "search",
}
_READ_COMMANDS = {
    "open",
    "cat",
    "goto",
    "scroll_up",
    "scroll_down",
    "view",
    "less",
    "head",
    "tail",
}


def _event_list(raw: dict[str, Any]) -> list[Any]:
    for key in ("trajectory", "history", "events", "steps", "messages", "conversation"):
        value = raw.get(key)
        if isinstance(value, list) and value:
            return value
    return []


def _event_text(event: Any) -> str:
    if isinstance(event, dict):
        for key in ("text", "content", "message", "thought", "action", "observation", "tool_output"):
            value = event.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return json.dumps(event, ensure_ascii=False)
    return str(event)


def _event_kind(event: Any) -> str:
    """Classify a raw trajectory event as system / action / observation."""

    if not isinstance(event, dict):
        return "action"
    marker = str(event.get("role", event.get("type", event.get("kind", "")))).lower()
    if marker == "system":
        return "system"
    if (
        marker in ("user", "human", "tool")
        or "observation" in marker
        or "output" in marker
    ):
        return "observation"
    return "action"


def _extract_command(text: str) -> str | None:
    """Return the command from the last fenced block of an agent action."""

    blocks = _FENCED_BLOCK_RE.findall(text)
    if not blocks:
        return None
    command = blocks[-1].strip()
    return command or None


def _tool_type_from_command(command: str | None) -> str:
    if not command:
        return "other"
    lines = command.splitlines()
    first_line = lines[0].strip() if lines else ""
    tokens = first_line.split()
    word = tokens[0].lower() if tokens else ""
    lower_cmd = command.lower()
    if word in _EDIT_COMMANDS:
        return "edit"
    if word in _SEARCH_COMMANDS:
        return "search"
    if word in _READ_COMMANDS:
        return "read_file"
    if word in ("pytest", "tox") or "pytest" in lower_cmd or "unittest" in lower_cmd:
        return "run_test"
    if word in ("python", "python3", "bash", "sh") and (
        "test" in lower_cmd or "reproduce" in lower_cmd
    ):
        return "run_test"
    return "other"


def _event_outcome(text: str, last_action_tool: str | None) -> str:
    lower = text[:4000].lower()
    if last_action_tool == "run_test" and (
        "failed" in lower or "failure" in lower or "assertionerror" in lower
    ):
        return "test_failed"
    if "traceback" in lower or "syntaxerror" in lower or "error" in lower:
        return "error_observed"
    if last_action_tool == "edit" and (
        "file updated" in lower or "edit applied" in lower or "updated" in lower
    ):
        return "patch_applied"
    return "none"


def _trajectory_phase(
    *,
    event_idx: int,
    num_events: int,
    tool_type: str | None,
    command: str | None,
    prev_outcome: str,
    seen_edit: bool,
) -> str:
    first_word = ""
    if command:
        tokens = command.split()
        first_word = tokens[0].lower() if tokens else ""
    if first_word == "submit" or event_idx == num_events - 1:
        return "finalize"
    if prev_outcome in ("test_failed", "error_observed"):
        return "debug"
    if tool_type == "run_test":
        return "test"
    if tool_type == "edit":
        return "edit"
    if not seen_edit:
        return "issue_understanding"
    return "debug"


def _issue_from_events(events: list[Any]) -> str | None:
    """The first observation before any agent action is the issue statement."""

    for event in events:
        kind = _event_kind(event)
        if kind == "action":
            return None
        if kind == "observation":
            return _event_text(event)
    return None


def _context_blocks(raw: dict[str, Any], issue_from_trajectory: str | None) -> list[PromptBlock]:
    blocks = [PromptBlock("system", "SWE-agent trajectory prompt.")]
    issue, _ = first_text(
        raw,
        ("problem_statement", "issue", "task", "instruction", "prompt", "repo_problem_statement"),
    )
    repo, _ = first_text(raw, ("repo", "repository", "instance_id"))
    if repo:
        blocks.append(PromptBlock("code_context", f"Repository: {repo}"))
    issue = issue or issue_from_trajectory
    if issue:
        blocks.append(PromptBlock("code_context", issue))
    return blocks


def _step_events(events: list[Any]) -> list[Any]:
    """Drop the system prompt and leading issue statement; keep agent steps."""

    steps: list[Any] = []
    saw_action = False
    for event in events:
        kind = _event_kind(event)
        if kind == "system":
            continue
        if kind == "observation" and not saw_action:
            # Leading issue/context observation: lifted into context blocks.
            continue
        if kind == "action":
            saw_action = True
        steps.append(event)
    return steps


def convert(source_path, limit: int, repo_id: str):
    records = []
    unavailable_fields: list[str] = []
    any_dag = False
    for source_index, _, raw in iter_raw_records(source_path, limit=None):
        group_id, group_field = first_text(
            raw,
            ("trajectory_id", "instance_id", "id", "task_id", "repo"),
        )
        if group_field is None:
            unavailable_fields.append("source_group_id")
        base_group_id = group_id or str(source_index)
        group_id = f"{base_group_id}:{source_index}"
        timestamp, timestamp_field = first_text(raw, ("timestamp", "created_at", "start_time"))
        if timestamp_field is None:
            unavailable_fields.append("timestamp")
        events = _event_list(raw)
        issue_from_trajectory = _issue_from_events(events)
        context_blocks = _context_blocks(raw, issue_from_trajectory)
        steps = _step_events(events)[:MAX_EVENTS_PER_TRAJECTORY]
        if not steps:
            issue, issue_field = first_text(
                raw,
                ("problem_statement", "issue", "task", "instruction", "prompt"),
            )
            issue = issue or issue_from_trajectory
            if issue_field is None and issue is None:
                unavailable_fields.append("trajectory")
                continue
            request_id = f"swe-agent-{source_index:06d}-000"
            records.append(
                make_record(
                    request_id=request_id,
                    workflow="swe-agent-trajectory",
                    blocks=context_blocks + [PromptBlock("trajectory_event", issue or "")],
                    source_dataset=repo_id,
                    source_index=source_index,
                    source_group_id=group_id,
                    timestamp=timestamp,
                    claim_scope=CLAIM_SCOPE_REAL_AGENT,
                    role="debugger",
                    phase="reflect",
                    graph_node_type="trajectory_singleton",
                    trajectory_phase="issue_understanding",
                    unavailable_fields=[
                        "dependency_edges",
                        "ready_times",
                        "event_order",
                        "event_outcome",
                        "tool_type",
                        "dag_depth",
                        "group_local_step_index",
                        "on_critical_path",
                    ],
                    dag_available=False,
                )
            )
        else:
            previous_request_id: str | None = None
            prev_outcome = "none"
            seen_edit = False
            last_action_tool: str | None = None
            num_steps = len(steps)
            for event_idx, event in enumerate(steps):
                kind = _event_kind(event)
                text = _event_text(event)
                command = None
                event_outcome: str | None = None
                if kind == "action":
                    command = _extract_command(text)
                    tool_type = _tool_type_from_command(command)
                    role, phase, block_type = "tool_caller", "act", "trajectory_event"
                    last_action_tool = tool_type
                    if tool_type == "edit":
                        seen_edit = True
                else:  # observation
                    event_outcome = _event_outcome(text, last_action_tool)
                    role, phase, block_type = "tester", "observe", "tool_result"
                    if event_outcome in ("test_failed", "error_observed"):
                        tool_type = "inspect_error"
                    else:
                        tool_type = last_action_tool or "other"
                trajectory_phase = _trajectory_phase(
                    event_idx=event_idx,
                    num_events=num_steps,
                    tool_type=tool_type,
                    command=command,
                    prev_outcome=prev_outcome,
                    seen_edit=seen_edit,
                )
                if kind == "observation":
                    prev_outcome = event_outcome or "none"
                request_id = f"swe-agent-{source_index:06d}-{event_idx:03d}"
                dependencies = [previous_request_id] if previous_request_id else []
                dependency_edges = (
                    [(previous_request_id, request_id)] if previous_request_id else []
                )
                blocks = context_blocks + [PromptBlock(block_type, text)]
                records.append(
                    make_record(
                        request_id=request_id,
                        workflow="swe-agent-trajectory",
                        blocks=blocks,
                        source_dataset=repo_id,
                        source_index=f"{source_index}:{event_idx}",
                        source_group_id=group_id,
                        timestamp=timestamp,
                        claim_scope=CLAIM_SCOPE_REAL_AGENT,
                        role=role,
                        phase=phase,
                        graph_node_type=block_type,
                        tool_type=tool_type,
                        ready_time=float(event_idx),
                        dependencies=dependencies,
                        dependency_edges=dependency_edges,
                        dag_available=True,
                        trajectory_phase=trajectory_phase,
                        event_outcome=event_outcome,
                        dag_depth=event_idx,
                        group_local_step_index=event_idx,
                        # Linear per-trajectory chain: every step is on the
                        # critical path of its trajectory DAG.
                        on_critical_path=True,
                    )
                )
                previous_request_id = request_id
                any_dag = True
                if len(records) >= limit:
                    return records, unavailable_fields, any_dag
        if len(records) >= limit:
            return records[:limit], unavailable_fields, any_dag
    if not any_dag:
        unavailable_fields.extend(["dependency_edges", "ready_times"])
    return records[:limit], unavailable_fields, any_dag


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SWE-agent trajectories to TokenMoE prompt workloads.")
    run_converter(
        parser=parser,
        adapter_name="swe_agent",
        source_dataset=SOURCE_DATASET,
        output_name=OUTPUT_NAME,
        claim_scope=CLAIM_SCOPE_REAL_AGENT,
        convert_fn=convert,
        manifest_extra={"metadata_heuristic_version": HEURISTIC_VERSION},
    )


if __name__ == "__main__":
    main()
