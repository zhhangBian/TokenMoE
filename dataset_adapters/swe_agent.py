#!/usr/bin/env python3
"""Convert SWE-agent trajectories into actual pre-generation request prompts."""

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
MAX_REQUESTS_PER_TRAJECTORY = 8
HEURISTIC_VERSION = "swe_agent.pre_admission.v2"

_FENCED_BLOCK = re.compile(r"```(?:\w+)?\n?(.*?)```", re.DOTALL)


def _events(raw: dict[str, Any]) -> list[Any]:
    for key in ("trajectory", "history", "events", "steps", "messages", "conversation"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
    return []


def _text(event: Any) -> str:
    if not isinstance(event, dict):
        return str(event)
    for key in (
        "text",
        "content",
        "message",
        "thought",
        "action",
        "observation",
        "tool_output",
    ):
        value = event.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return json.dumps(event, ensure_ascii=False)


def _kind(event: Any) -> str:
    if not isinstance(event, dict):
        return "action"
    marker = str(event.get("role", event.get("type", event.get("kind", "")))).lower()
    if marker == "system":
        return "system"
    if marker in {"user", "human", "tool"} or any(
        word in marker for word in ("observation", "output")
    ):
        return "observation"
    return "action"


def _tool_type(action: str) -> str:
    matches = _FENCED_BLOCK.findall(action)
    command = matches[-1].strip().lower() if matches else ""
    first = command.split(maxsplit=1)[0] if command else ""
    if first in {"edit", "create", "insert", "str_replace", "apply_patch"}:
        return "edit"
    if first in {"grep", "find", "search", "ls", "search_file", "search_dir"}:
        return "search"
    if first in {"open", "cat", "view", "head", "tail", "less"}:
        return "read_file"
    if "pytest" in command or "unittest" in command or " test" in command:
        return "run_test"
    if first == "submit":
        return "submit"
    return "other"


def _outcome(observation: str, previous_tool: str | None) -> str:
    text = observation[:4000].lower()
    if previous_tool == "run_test" and any(
        marker in text for marker in ("failed", "failure", "assertionerror")
    ):
        return "test_failed"
    if any(marker in text for marker in ("traceback", "syntaxerror", "error:")):
        return "error_observed"
    if previous_tool == "edit" and any(
        marker in text for marker in ("updated", "applied", "done")
    ):
        return "patch_applied"
    return "none"


def _phase(
    *, previous_tool: str | None, previous_outcome: str, has_edited: bool
) -> str:
    if previous_outcome in {"test_failed", "error_observed"}:
        return "debug"
    if previous_tool == "edit":
        return "test"
    if not has_edited:
        return "issue_understanding"
    return "edit"


def _context(raw: dict[str, Any], events: list[Any]) -> tuple[list[PromptBlock], int]:
    blocks: list[PromptBlock] = []
    cursor = 0
    if events and _kind(events[0]) == "system":
        blocks.append(PromptBlock("system", _text(events[0])))
        cursor = 1
    problem, _ = first_text(
        raw,
        (
            "problem_statement",
            "issue",
            "task",
            "instruction",
            "prompt",
            "repo_problem_statement",
        ),
    )
    repository, _ = first_text(raw, ("repo", "repository", "instance_id"))
    if repository:
        blocks.append(PromptBlock("code_context", f"Repository: {repository}"))
    if problem:
        blocks.append(PromptBlock("user_message", problem))
    elif cursor < len(events) and _kind(events[cursor]) == "observation":
        blocks.append(PromptBlock("user_message", _text(events[cursor])))
        cursor += 1
    return blocks, cursor


def convert(source_path, limit: int, repo_id: str):
    records = []
    unavailable = ["next_action_tool_type", "on_critical_path"]
    for source_index, _, raw in iter_raw_records(source_path):
        group_id, group_field = first_text(
            raw, ("trajectory_id", "instance_id", "id", "task_id", "repo")
        )
        if group_field is None:
            unavailable.append("source_group_id")
        group = f"{group_id or source_index}:{source_index}"
        timestamp, timestamp_field = first_text(
            raw, ("timestamp", "created_at", "start_time")
        )
        if timestamp_field is None:
            unavailable.append("timestamp")
        events = _events(raw)
        context, cursor = _context(raw, events)
        history: list[PromptBlock] = []
        previous_request: str | None = None
        previous_tool: str | None = None
        previous_outcome = "none"
        has_edited = False
        request_index = 0
        for event in events[cursor:]:
            kind = _kind(event)
            text = _text(event).strip()
            if not text or kind == "system":
                continue
            if kind == "observation":
                previous_outcome = _outcome(text, previous_tool)
                history.append(PromptBlock("tool_result", text))
                continue

            # An action is the target model output. The corresponding request
            # contains only context and events that existed before this action.
            if request_index < MAX_REQUESTS_PER_TRAJECTORY and context + history:
                phase = _phase(
                    previous_tool=previous_tool,
                    previous_outcome=previous_outcome,
                    has_edited=has_edited,
                )
                request_id = f"swe-agent-{source_index:06d}-{request_index:03d}"
                records.append(
                    make_record(
                        request_id=request_id,
                        workflow="swe-agent-trajectory",
                        blocks=context + history,
                        source_dataset=repo_id,
                        source_index=f"{source_index}:{request_index}",
                        source_group_id=group,
                        timestamp=timestamp,
                        claim_scope=CLAIM_SCOPE_REAL_AGENT,
                        role="solver",
                        phase="act",
                        graph_node_type="agent_llm_request",
                        dependencies=(
                            () if previous_request is None else (previous_request,)
                        ),
                        trajectory_phase=phase,
                        event_outcome=previous_outcome,
                        dag_depth=request_index,
                        group_local_step_index=request_index,
                    )
                )
                previous_request = request_id
                request_index += 1
                if len(records) >= limit:
                    return records, unavailable
            previous_tool = _tool_type(text)
            has_edited = has_edited or previous_tool == "edit"
            history.append(PromptBlock("trajectory_event", text))
    return records, unavailable


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWE-agent trajectories to pre-generation prompts."
    )
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
