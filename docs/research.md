# TokenMoE Research and vLLM Positioning

## Problem Summary

TokenMoE targets MoE expert state, not KV cache reuse or agent-graph
speculation. MoE serving learns selected experts only after the router runs,
but offload, prefetch, batching, and expert-parallel placement need lead time.
Agent workloads expose metadata before model execution: role, phase, graph node,
tool type, and prompt block layout. The contribution point is to test whether
that metadata predicts per-layer expert working sets without changing router
semantics or model outputs.

The correctness boundary is strict: the real MoE router still executes. A
prediction can only affect performance-side decisions such as prefetch,
residency, scheduling, or replayed EPLB demand. Missing metadata or low
confidence falls back to baseline behavior.

## vLLM Insertion Points

- Request metadata: `vllm/vllm/v1/engine/__init__.py:EngineCoreRequest` carries
  request IDs, trace headers, sampling params, and arrival time. For the first
  prototype, `scripts/collect_traces.py` keeps a sidecar
  `request_id -> AgentNodeMeta`, avoiding invasive request-struct changes.
- Request state: `vllm/vllm/v1/request.py:Request.from_engine_core_request`
  is the future place to preserve metadata inside the scheduler if server-side
  trace writing is needed.
- Scheduler: `vllm/vllm/v1/core/sched/scheduler.py` owns `waiting`,
  `skipped_waiting`, and `running` queues. The replay prototype models legal
  ready-node scheduling before considering online scheduling changes.
- Routed expert capture: `vllm/vllm/model_executor/layers/fused_moe/router/base_router.py`
  captures logical `topk_ids` before EPLB maps them to physical replicas.
- Worker capture buffer: `vllm/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`
  stores per-token, per-layer top-k expert IDs.
- Output path: `vllm/vllm/v1/outputs.py`, `gpu_model_runner.py`, and
  `output_processor.py` carry routed expert arrays to `CompletionOutput`.
- EPLB: `vllm/vllm/distributed/eplb/eplb_state.py`,
  `vllm/vllm/config/parallel.py`, and `GPUModelRunner.eplb_step()` are the
  integration points for future proactive demand, but this change uses replay
  first.
- Expert parallel and DeepEP: `vllm/vllm/distributed/device_communicators/all2all.py`
  and fused MoE `prepare_finalize/deepep_*` paths are communication-sensitive
  and should not be modified until replay shows a useful signal.

## Related Work Boundary

| Work | Signal | Optimization object | TokenMoE difference |
| --- | --- | --- | --- |
| MoE-Infinity | Request-level expert activation traces | Expert cache/offload/prefetch | TokenMoE uses pre-router agent metadata as an earlier signal. |
| ProMoE | Intermediate/request-internal evidence | Proactive expert caching | TokenMoE predicts at request admission, before hidden states exist. |
| FineMoE/fMoE | Expert pattern and semantic hints | Fine-grained expert offloading | TokenMoE uses runtime agent fields and ready-node scheduling, not only prompt semantics. |
| ExpertFlow | Routing-path predictor plus token scheduling | Expert cache and token scheduling | TokenMoE schedules agent nodes/batches within dependency constraints. |
| Local Routing Consistency | Empirical routing locality | Model/offload suitability | TokenMoE measures locality by agent role, phase, and graph node. |
| ReMoE | Router fine-tuning | Higher expert reuse | TokenMoE does not train or change router outputs. |
| Speculative MoE | Speculative expert use/prefetch | Latency hiding | TokenMoE keeps speculative compute out of scope; prefetch miss only wastes bandwidth. |
| vLLM EPLB | Recent expert load windows | Expert replica balancing | TokenMoE provides future demand from agent metadata for replay. |
| DeepEP | Efficient all-to-all kernels | Expert-parallel communication | TokenMoE can reduce fanout/burstiness before dispatch, but does not replace DeepEP. |
| SGLang MoE | Serving runtime MoE/EP paths | Runtime scheduling/dispatch | TokenMoE prototype remains vLLM-oriented while preserving simulator fallback. |

## Real vs Simulator Scope

Real in this prototype:

- Workload records with required `AgentNodeMeta` fields.
- Tiny Mixtral router-logit tracing through Transformers.
- vLLM-compatible `routed_experts` ingestion path.
- JSONL and parquet-compatible trace persistence.
- RouteSig online update and predictor evaluation.

Simulator/replay in this prototype:

- Expert prefetch/residency hit, waste, bandwidth, and stall-reduction proxy.
- Expert-overlap-aware scheduler replay over legal ready nodes.
- Proactive EPLB replay comparing moving-average demand with future demand.

Deferred runtime integration:

- Online vLLM scheduler reordering.
- Real expert-weight offload/prefetch hooks.
- EPLB policy modification and migration scheduling.
- Router score capture inside fused vLLM kernels. The prototype captures scores
  from Transformers router logits and keeps vLLM score capture optional.
