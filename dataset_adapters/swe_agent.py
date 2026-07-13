#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _event_list(raw: dict[str, Any]) -> list[Any]:
    for key in ("trajectory", "history", "events", "steps", "messages", "conversation"):
        value = raw.get(key)
        if isinstance(value, list) and value:
            return value
    return []


def _event_text(event: Any) -> str:
    if isinstance(event, dict):
        for key in ("content", "message", "thought", "action", "observation", "tool_output", "text"):
            value = event.get(key)
            if value is not None and str(value).strip():
                return str(value)
        return json.dumps(event, ensure_ascii=False)
    return str(event)


def _role_phase(event: Any) -> tuple[str, str, str | None, str]:
    if not isinstance(event, dict):
        return "assistant", "act", None, "trajectory_event"
    event_type = str(event.get("type", event.get("role", event.get("kind", "")))).lower()
    action = str(event.get("action", event.get("tool", ""))).lower()
    if "observation" in event_type or "output" in event_type:
        return "tester", "observe", "shell", "tool_result"
    if "thought" in event_type or "reason" in event_type:
        return "debugger", "reflect", None, "agent_thought"
    if "action" in event_type or action:
        return "tool_caller", "act", action or "shell", "tool_call"
    if "assistant" in event_type:
        return "assistant", "summarize", None, "agent_message"
    return "debugger", "reflect", None, "trajectory_event"


def _context_blocks(raw: dict[str, Any]) -> list[PromptBlock]:
    blocks = [PromptBlock("system", "SWE-agent trajectory prompt.")]
    issue, _ = first_text(
        raw,
        ("problem_statement", "issue", "task", "instruction", "prompt", "repo_problem_statement"),
    )
    repo, _ = first_text(raw, ("repo", "repository", "instance_id"))
    if repo:
        blocks.append(PromptBlock("code_context", f"Repository: {repo}"))
    if issue:
        blocks.append(PromptBlock("issue", issue))
    return blocks


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
        context_blocks = _context_blocks(raw)
        if not events:
            issue, issue_field = first_text(
                raw,
                ("problem_statement", "issue", "task", "instruction", "prompt"),
            )
            if issue_field is None:
                unavailable_fields.append("trajectory")
                continue
            request_id = f"swe-agent-{source_index:06d}-000"
            records.append(
                make_record(
                    request_id=request_id,
                    workflow="swe-agent-trajectory",
                    blocks=context_blocks + [PromptBlock("agent_thought", issue or "")],
                    source_dataset=repo_id,
                    source_index=source_index,
                    source_group_id=group_id,
                    timestamp=timestamp,
                    claim_scope=CLAIM_SCOPE_REAL_AGENT,
                    role="debugger",
                    phase="reflect",
                    graph_node_type="trajectory_singleton",
                    unavailable_fields=["dependency_edges", "ready_times", "event_order"],
                    dag_available=False,
                )
            )
        else:
            previous_request_id: str | None = None
            for event_idx, event in enumerate(events[:MAX_EVENTS_PER_TRAJECTORY]):
                role, phase, tool_type, block_type = _role_phase(event)
                request_id = f"swe-agent-{source_index:06d}-{event_idx:03d}"
                dependencies = [previous_request_id] if previous_request_id else []
                dependency_edges = (
                    [(previous_request_id, request_id)] if previous_request_id else []
                )
                blocks = context_blocks + [
                    PromptBlock(block_type, _event_text(event)),
                ]
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
    )


if __name__ == "__main__":
    main()
