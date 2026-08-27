# Trace format

TokenMoE accepts one trace format: `tokenmoe.trace.v3`, captured by the local
vLLM fork with backend identifier `vllm-routed-experts`.

Each JSONL record contains one prompt admission:

```json
{
  "schema_version": "tokenmoe.trace.v3",
  "backend": "vllm-routed-experts",
  "request_id": "request-42",
  "model_id": "/path/to/moe-model",
  "prompt": "...",
  "prompt_token_ids": [1, 2, 3],
  "generated_token_ids": [4],
  "moe_layer_ids": [1, 3],
  "router_top_k": 2,
  "num_experts": 64,
  "router_score_semantics": "model_type:softmax_topk_renormalized",
  "router_scores_unavailable_reason": null,
  "metadata": {},
  "prompt_segments": [],
  "layers": []
}
```

## Layer data

Each layer entry has the real model layer ID and arrays shaped
`[prompt_tokens, router_top_k]`:

```json
{
  "layer_id": 3,
  "selected_experts": [[4, 9], [9, 12]],
  "router_scores": [[0.63, 0.37], [0.54, 0.46]]
}
```

`layers` must exactly match `moe_layer_ids`. Expert IDs must lie in
`[0, num_experts)`. Scores must be finite, non-negative, and aligned with IDs.
If scores were intentionally disabled or unavailable, every layer stores
`router_scores: null` and the request-level unavailability reason is required.

## Prompt segments

Workload character spans are aligned against the same tokenizer IDs returned by
vLLM. A successful segment stores `token_start` and exclusive `token_end`. An
unaligned segment stores null token bounds and a non-empty `alignment_error`;
evaluation skips it rather than assigning routing from another span.

## Validation boundary

The reader rejects:

- legacy or mixed schema versions;
- non-vLLM backends;
- mixed models in one analysis file;
- missing MoE layer metadata;
- invalid Expert IDs or array shapes;
- partially available router scores;
- silent segment-alignment failures.

Trace capture is observational. Separate capture-on/off token-equivalence tests
are required before using a new model family or parallel configuration.
