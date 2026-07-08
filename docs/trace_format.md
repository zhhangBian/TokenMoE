# TokenMoE Trace Format

Trace files are JSONL: one JSON object per request. They are also convertible to
parquet-compatible one-row-per-request-layer records.

## Request Record

```json
{
  "schema_version": "tokenmoe.trace.v1",
  "request_id": "planner-coder-tester-000-plan",
  "metadata": {
    "request_id": "planner-coder-tester-000-plan",
    "agent_id": "planner-coder-tester:planner:0",
    "role": "planner",
    "phase": "plan",
    "tool_type": null,
    "graph_node_type": "root",
    "prompt_block_types": ["system", "instruction", "shared_context"]
  },
  "model_id": "TitanML/tiny-mixtral",
  "backend": "transformers-router-logits",
  "prompt": "You are the planner...",
  "prompt_token_count": 27,
  "output_token_count": 0,
  "token_ids": [1, 887, 460],
  "generated_token_ids": null,
  "layers": []
}
```

## Layer Record

Each layer entry stores selected expert IDs for every token:

```json
{
  "layer_id": 0,
  "selected_experts": [[7, 1], [2, 5], [2, 5]],
  "router_scores": [[0.1392, 0.1296], [0.1461, 0.1287], [0.1520, 0.1200]],
  "active_expert_histogram": {"1": 1, "2": 2, "5": 2, "7": 1}
}
```

Shapes:

- `selected_experts`: `[num_tokens, top_k]`
- `router_scores`: `[num_tokens, top_k]`, optional and aligned with
  `selected_experts`
- full request array from vLLM: `[num_tokens, num_layers, top_k]`

## vLLM Compatibility

vLLM returns `CompletionOutput.routed_experts` when
`enable_return_routed_experts=True`. That array is `[seq_len, layer_num, top_k]`
and contains logical expert IDs captured before EPLB physical replica mapping.
`scripts/collect_traces.py --backend vllm` consumes that array directly in
environments with a complete vLLM install.

## Parquet-Compatible Rows

The parquet writer stores one row per `(request_id, layer_id)` with JSON strings
for nested fields:

- `request_id`, `model_id`, `backend`
- `agent_id`, `workflow_role`, `workflow_phase`, `graph_node_type`
- `prompt_block_types`, `metadata_json`
- `layer_id`, `token_count`, `top_k`
- `selected_experts_json`, `router_scores_json`,
  `active_expert_histogram_json`

This layout works with pyarrow and pandas while preserving the original nested
JSONL trace as the source of truth.
