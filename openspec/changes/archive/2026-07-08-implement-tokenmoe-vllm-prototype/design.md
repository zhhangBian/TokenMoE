## Context

TokenMoE is a vLLM-oriented prototype for optimizing MoE expert state in agent workloads. The working hypothesis is that agent metadata visible before routing, such as role, phase, graph node type, tool type, and prompt block types, can predict per-layer expert working sets early enough to help prefetch, scheduling, and replica placement.

The current implementation plan is documentation- and trace-first. The first implementation stage must locate vLLM insertion points and build a runnable trace/replay path before changing scheduler, EPLB, or expert placement behavior. Correctness is non-negotiable: the real MoE router still executes, model outputs remain exact, and prediction errors only affect performance.

## Goals / Non-Goals

**Goals:**

- Define a phased vLLM implementation plan from research to trace collection, locality analysis, RouteSig, and replay/simulator prototypes.
- Capture `AgentNodeMeta` alongside per-layer MoE router top-k traces.
- Use existing vLLM routed-expert capture where possible before adding new hooks.
- Define RouteSig lookup, fallback, and online update behavior.
- Evaluate whether agent metadata has explanatory power before online prefetch, scheduling, or EPLB changes.
- Provide clear fallback paths when a module cannot be safely integrated into vLLM.

**Non-Goals:**

- Do not optimize KV cache as the TokenMoE contribution.
- Do not speculate or reorder agent graph dependencies.
- Do not change router selection, logits, or model output semantics.
- Do not require router fine-tuning.
- Do not begin broad vLLM refactors before trace evidence exists.

## Decisions

### Decision: Trace-first before runtime optimization

TokenMoE will first collect routed-expert traces and agent metadata, then analyze locality and simulate downstream policies. This avoids committing to expert prefetch, scheduler, or EPLB changes before validating the core assumption.

Alternative considered: implement online scheduler/EPLB hooks immediately. This is rejected because those paths are distributed-performance sensitive and require evidence that agent-conditioned predictions beat simple baselines.

### Decision: Reuse vLLM routed-experts capture for selected expert IDs

The vLLM checkout already contains a routed-experts capture path around MoE routers. The initial prototype should use or minimally extend this path for logical expert IDs rather than instrumenting fused kernels directly.

Alternative considered: add a separate tracing path inside each MoE kernel. This is rejected because vLLM has multiple MoE router/kernel variants, and router-level capture is more general.

### Decision: Add router scores only after ID traces work

Selected expert IDs are enough for first locality metrics, RouteSig histograms, Top-M hit rate, and scheduling fanout analysis. Router scores require additional capture buffers and D2H traffic, so they should be added after the ID path is validated.

Alternative considered: make scores mandatory in the first trace. This is rejected because it increases risk and is not needed to answer whether agent-router locality exists.

### Decision: Sidecar metadata first, request-struct metadata later

For the first benchmark, the driver can keep `request_id -> AgentNodeMeta` and merge metadata with vLLM outputs. If server-side trace writing is needed later, add metadata fields to vLLM request structs.

Alternative considered: patch every request path up front. This is rejected because offline benchmark-side metadata is enough to produce phase 3 traces and reduces initial vLLM surface area.

### Decision: Simulate prefetch, scheduling, and EPLB before online integration

Expert offload/prefetch availability varies by environment, scheduler changes are invasive, and EPLB weight movement affects distributed correctness. The first TokenMoE implementation should provide replay/simulator outputs for hit rate, fanout, and tail-latency estimates, then graduate only validated pieces into vLLM.

Alternative considered: require real expert offload and EPLB integration immediately. This is rejected because it would block prototype progress on environment-specific capabilities.

## Risks / Trade-offs

- Agent metadata may not explain router choices -> add prompt block type and lightweight prompt embeddings; confidence-gate RouteSig; fall back to global or normal vLLM behavior.
- Different MoE models may have weak or inconsistent routing locality -> report model suitability and avoid overgeneralizing from one model family.
- Router-score capture may add overhead -> make it optional and sampleable.
- Scheduler replay gains may disappear online due waiting penalties -> model waiting penalty and dependency constraints before online integration.
- EPLB proactive placement may be outweighed by migration cost -> keep first implementation as replay simulation and compare against moving-average EPLB.
- Trace buffers can be large -> start with selected expert IDs, configurable max tokens, JSONL output, and optional parquet conversion.

## Migration Plan

1. Create OpenSpec artifacts that define the implementation contract.
2. Implement phase 1 and phase 2 documentation/scripts without vLLM behavior changes.
3. Build the phase 3 trace benchmark using existing vLLM routed-experts return.
4. Add minimal vLLM metadata and router-score hooks only after the benchmark path is working.
5. Implement analysis and simulators as independent modules.
6. Integrate online vLLM optimizations only when replay results justify the risk.

Rollback is straightforward for early phases because changes are additive. Runtime patches must be gated by config flags and disabled by default.
