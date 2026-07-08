from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from tokenmoe.schema import AgentNodeMeta, WorkloadRecord


@dataclass(frozen=True)
class NodeTemplate:
    suffix: str
    role: str
    phase: str
    graph_node_type: str
    tool_type: str | None
    prompt_block_types: list[str]
    prompt_template: str
    dependencies: list[str]
    criticality: float = 1.0


WORKFLOW_TEMPLATES: dict[str, list[NodeTemplate]] = {
    "planner-coder-tester": [
        NodeTemplate(
            "plan",
            "planner",
            "plan",
            "root",
            None,
            ["system", "instruction", "shared_context"],
            "You are the planner. Break down this coding task into concrete steps: {task}",
            [],
            1.2,
        ),
        NodeTemplate(
            "code",
            "coder",
            "act",
            "worker",
            "python",
            ["system", "instruction", "private_memory", "code_context"],
            "Implement the highest priority step for this repository task: {task}\nPlan: {hint}",
            ["plan"],
            1.0,
        ),
        NodeTemplate(
            "test",
            "tester",
            "verify",
            "worker",
            "pytest",
            ["system", "tool_result", "code_context"],
            "Design focused tests for the implementation of: {task}\nPatch summary: {hint}",
            ["code"],
            1.1,
        ),
        NodeTemplate(
            "critic",
            "critic",
            "reflect",
            "join",
            None,
            ["system", "shared_context", "tool_result"],
            "Review the plan, code, and tests for hidden correctness issues in: {task}",
            ["test"],
            0.9,
        ),
    ],
    "search-summarize": [
        NodeTemplate(
            "plan",
            "planner",
            "plan",
            "root",
            None,
            ["system", "instruction"],
            "Plan a concise web research strategy for: {task}",
            [],
            1.0,
        ),
        NodeTemplate(
            "search",
            "searcher",
            "act",
            "tool_call",
            "web_search",
            ["system", "tool_schema", "query"],
            "Select search queries and source filters for: {task}",
            ["plan"],
            0.9,
        ),
        NodeTemplate(
            "summarize",
            "summarizer",
            "summarize",
            "worker",
            None,
            ["system", "tool_result", "shared_context"],
            "Summarize the search findings and cite uncertainty for: {task}\nFindings: {hint}",
            ["search"],
            1.2,
        ),
    ],
    "tool-use": [
        NodeTemplate(
            "inspect",
            "tool_caller",
            "observe",
            "tool_call",
            "shell",
            ["system", "tool_schema", "private_memory"],
            "Inspect the local environment before solving: {task}",
            [],
            0.9,
        ),
        NodeTemplate(
            "act",
            "tool_caller",
            "act",
            "tool_call",
            "python",
            ["system", "tool_result", "instruction"],
            "Choose and execute a lightweight tool action for: {task}\nObservation: {hint}",
            ["inspect"],
            1.0,
        ),
        NodeTemplate(
            "explain",
            "assistant",
            "summarize",
            "join",
            None,
            ["system", "tool_result", "answer"],
            "Explain the tool result and next step for: {task}",
            ["act"],
            0.8,
        ),
    ],
    "multi-agent-discussion": [
        NodeTemplate(
            "moderator",
            "moderator",
            "plan",
            "root",
            None,
            ["system", "shared_context"],
            "Frame a structured multi-agent discussion about: {task}",
            [],
            1.0,
        ),
        NodeTemplate(
            "analyst_a",
            "analyst",
            "argue",
            "worker",
            None,
            ["system", "private_memory", "shared_context"],
            "Argue for a conservative implementation of: {task}",
            ["moderator"],
            0.9,
        ),
        NodeTemplate(
            "analyst_b",
            "analyst",
            "argue",
            "worker",
            None,
            ["system", "private_memory", "shared_context"],
            "Argue for a performance-focused implementation of: {task}",
            ["moderator"],
            0.9,
        ),
        NodeTemplate(
            "merge",
            "merger",
            "summarize",
            "join",
            None,
            ["system", "shared_context", "answer"],
            "Merge the discussion into an actionable decision for: {task}",
            ["analyst_a", "analyst_b"],
            1.1,
        ),
    ],
    "swe-agent-code-repair": [
        NodeTemplate(
            "reproduce",
            "tester",
            "observe",
            "tool_call",
            "pytest",
            ["system", "issue", "tool_result"],
            "Reproduce this bug and identify the failing behavior: {task}",
            [],
            1.2,
        ),
        NodeTemplate(
            "localize",
            "debugger",
            "reflect",
            "worker",
            None,
            ["system", "code_context", "tool_result"],
            "Localize the most likely code path for this issue: {task}\nFailure: {hint}",
            ["reproduce"],
            1.1,
        ),
        NodeTemplate(
            "patch",
            "coder",
            "act",
            "worker",
            "python",
            ["system", "code_context", "private_memory"],
            "Patch the localized bug with minimal changes: {task}",
            ["localize"],
            1.0,
        ),
        NodeTemplate(
            "regress",
            "tester",
            "verify",
            "worker",
            "pytest",
            ["system", "tool_result", "code_context"],
            "Run regression-oriented tests for the patch: {task}",
            ["patch"],
            1.3,
        ),
    ],
}


TASK_BANK = [
    "fix a race in streaming request cleanup",
    "add validation for a JSONL trace field",
    "summarize new benchmark results for a MoE router",
    "repair a failing unit test around retry behavior",
    "compare two expert placement policies under bursty load",
    "inspect a latency regression in batched decoding",
    "write a migration note for a changed scheduler option",
    "debug an off-by-one token accounting issue",
    "prepare a reproducible experiment for a small MoE model",
    "explain why a fallback path preserved output semantics",
]


def generate_workloads(
    *,
    per_workflow: int = 12,
    seed: int = 7,
    workflows: list[str] | None = None,
) -> list[WorkloadRecord]:
    rng = random.Random(seed)
    selected_workflows = workflows or list(WORKFLOW_TEMPLATES)
    records: list[WorkloadRecord] = []
    ready_counter = itertools.count()

    for workflow in selected_workflows:
        templates = WORKFLOW_TEMPLATES[workflow]
        for run_idx in range(per_workflow):
            task = rng.choice(TASK_BANK)
            hint = rng.choice(
                [
                    "focus on correctness before micro-optimizations",
                    "prefer direct evidence from traces",
                    "keep the request graph dependencies unchanged",
                    "make fallback behavior explicit",
                    "include enough metadata to replay the decision",
                ]
            )
            prefix = f"{workflow}-{run_idx:03d}"
            suffix_to_request: dict[str, str] = {}
            for node in templates:
                request_id = f"{prefix}-{node.suffix}"
                suffix_to_request[node.suffix] = request_id
                prompt = node.prompt_template.format(task=task, hint=hint)
                meta = AgentNodeMeta(
                    request_id=request_id,
                    agent_id=f"{workflow}:{node.role}:{run_idx % 4}",
                    role=node.role,
                    phase=node.phase,
                    tool_type=node.tool_type,
                    graph_node_type=node.graph_node_type,
                    prompt_block_types=list(node.prompt_block_types),
                    ready_time=float(next(ready_counter)),
                    deadline=None,
                    run_probability=1.0,
                    criticality=node.criticality,
                )
                records.append(
                    WorkloadRecord(
                        request_id=request_id,
                        workflow=workflow,
                        prompt=prompt,
                        meta=meta,
                        dependencies=[
                            suffix_to_request[dep]
                            for dep in node.dependencies
                            if dep in suffix_to_request
                        ],
                        source="synthetic-agent-template",
                        expected_output_tokens=16,
                    )
                )
    return records
