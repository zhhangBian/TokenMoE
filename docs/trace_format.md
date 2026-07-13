# TokenMoE Trace Format

Current-stage trace files are JSONL with `schema_version =
"tokenmoe.trace.v2"`. v1 readers remain for compatibility, but analysis rejects
mixed v1/v2 inputs and rejects v1 for current-stage validation.

## Request Record

```json
{
  "schema_version": "tokenmoe.trace.v2",
  "request_id": "sharegpt-000001",
  "metadata": {
    "request_id": "sharegpt-000001",
    "agent_id": "sharegpt-chat:assistant:conv-1",
    "role": "assistant",
    "phase": "chat",
    "tool_type": null,
    "graph_node_type": "conversation",
    "prompt_block_types": ["system", "user_message", "assistant_message"]
  },
  "model_id": "/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B",
  "backend": "vllm-routed-experts",
  "routing_scope": "prompt_only",
  "backend_fallback_used": false,
  "prompt_routing_start": 0,
  "decode_routing_excluded": true,
  "prompt_token_count": 128,
  "output_token_count": 1,
  "moe_layer_ids": [0, 1, 2],
  "router_top_k": 8,
  "num_experts": 128,
  "prompt_segments": [],
  "layers": []
}
```

## Prompt Segments

Adapters write character spans. The collector maps them to token spans with the
same tokenizer path used by vLLM and compares token IDs with
`RequestOutput.prompt_token_ids`.

```json
{
  "segment_id": "instruction-001",
  "block_type": "instruction",
  "segment_position": 1,
  "char_start": 17,
  "char_end": 82,
  "token_start": 4,
  "token_end": 23,
  "alignment_status": "aligned",
  "alignment_error": null
}
```

If alignment fails, segment-level metrics are unavailable for that record
instead of silently treating the block as aligned.

## Layer Record

Each layer stores only MoE layers listed in `moe_layer_ids`:

```json
{
  "layer_id": 0,
  "selected_experts": [[7, 1], [2, 5]],
  "router_scores": null,
  "active_expert_histogram": {"1": 1, "2": 1, "5": 1, "7": 1}
}
```

Shapes:

- `selected_experts`: `[prompt_tokens, router_top_k]`
- full vLLM capture before filtering: `[tokens, hidden_layers, router_top_k]`

Dense/non-MoE layer slots are filtered or marked unavailable. They must never
be interpreted as expert `0` activity.

## Validation

`scripts/analyze_traces.py` rejects:

- non-`vllm-routed-experts` backends
- fallback traces
- mixed schema versions
- non-prompt-only routing
- missing `moe_layer_ids`
- layer IDs not present in `moe_layer_ids`
- segment token spans that extend beyond prompt token count
