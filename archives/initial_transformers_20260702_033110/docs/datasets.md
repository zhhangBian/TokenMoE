# TokenMoE Models and Workloads

## Model Tiers

| Tier | Models | Use |
| --- | --- | --- |
| Small | `TitanML/tiny-mixtral` | Local real-router experiments; 2 MoE layers, 8 experts, top-2 routing. |
| Target | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen3-30B-A3B`, `deepseek-ai/DeepSeek-V2-Lite-Chat` | Larger vLLM evaluation when model cache, GPU memory, and full vLLM dependencies are available. |
| Trace-only | Deterministic schema-compatible trace generation | CI/replay fallback when external model resources are unavailable. |

Run `MODEL_SET=small DRY_RUN=1 scripts/download_models.sh` to inspect model
downloads without fetching files.

## Workload Schema

Each JSONL row is a normalized agent request:

```json
{
  "request_id": "planner-coder-tester-000-plan",
  "workflow": "planner-coder-tester",
  "prompt": "You are the planner...",
  "meta": {
    "request_id": "planner-coder-tester-000-plan",
    "agent_id": "planner-coder-tester:planner:0",
    "role": "planner",
    "phase": "plan",
    "tool_type": null,
    "graph_node_type": "root",
    "prompt_block_types": ["system", "instruction", "shared_context"],
    "ready_time": 0.0,
    "deadline": null,
    "run_probability": 1.0,
    "criticality": 1.2
  },
  "dependencies": [],
  "source": "synthetic-agent-template",
  "expected_output_tokens": 16
}
```

Required `AgentNodeMeta` fields are `request_id`, `agent_id`, `role`, `phase`,
`tool_type`, `graph_node_type`, and `prompt_block_types`. `tool_type` may be
null for non-tool requests.

## Workload Families

| Workflow | Roles | Metadata mapping |
| --- | --- | --- |
| planner-coder-tester | planner, coder, tester, critic | `role` from node function; `phase` as plan/act/verify/reflect; `tool_type` python or pytest for tool-backed nodes; `prompt_block_types` includes code context and tool results where applicable. |
| search-summarize | planner, searcher, summarizer | Search node uses `tool_type=web_search`; summarizer includes `tool_result` and shared context. |
| tool-use | tool_caller, assistant | Tool nodes use shell/python `tool_type`; final assistant node has no tool. |
| multi-agent-discussion | moderator, analyst, merger | Parallel analyst nodes depend on moderator; merger depends on both analysts. |
| SWE-agent/code repair | tester, debugger, coder | Reproduce/localize/patch/regress phases map to observe/reflect/act/verify. |

The default generator produces all five workflows and preserves dependencies so
scheduler replay can enforce legal ready-node choices.

## External Dataset Fallback

The prototype does not require external datasets. If future SWE-bench, tool-use,
or web-agent traces are unavailable, `scripts/download_workloads.py` still
generates schema-valid synthetic agent records with the same metadata fields.
