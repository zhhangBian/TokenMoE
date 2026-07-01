## Why

MoE serving has a timing problem: the system learns which experts are needed only after the router runs, but expert state movement, batching, and replica placement decisions need lead time. Agent workloads expose request-level structure before model execution, so TokenMoE should test whether agent role, phase, graph node, and prompt block metadata can predict per-layer expert working sets without changing router semantics or model outputs.

## What Changes

- Add a trace-first TokenMoE prototype plan for vLLM that records agent metadata and per-layer MoE router top-k behavior.
- Define RouteSig as a serving-side statistical abstraction for agent-conditioned expert working-set prediction with hierarchical fallback and online updates.
- Add dataset and model preparation workflows for agent-style workloads and MoE trace collection, with small-model and trace-only fallbacks.
- Add analysis and simulator paths for locality, expert prefetch/residency, overlap-aware scheduling, and proactive EPLB demand.
- Defer invasive vLLM scheduling, expert offload, and EPLB changes until trace evidence and replay simulators identify the viable integration points.
- Do not change model output semantics: real routers still execute, and prediction errors only affect performance decisions.

## Capabilities

### New Capabilities
- `tokenmoe-tracing`: Capture AgentNodeMeta and MoE routed-expert traces from vLLM requests in JSONL or parquet-ready records.
- `tokenmoe-route-signature`: Maintain and query agent-conditioned RouteSignature statistics with hierarchical fallback and online updates.
- `tokenmoe-evaluation-resources`: Prepare MoE models, agent workloads, locality analysis, and replay/simulation inputs for end-to-end evaluation.
- `tokenmoe-system-optimization`: Prototype expert prefetch/residency, expert-overlap-aware scheduling, and proactive expert replica placement using real hooks where available and simulators otherwise.

### Modified Capabilities

None.

## Impact

- vLLM request path: `EngineCoreRequest`, `Request`, scheduler outputs, GPU model runner cached request state, OpenAI/offline entrypoints.
- vLLM MoE path: `FusedMoE`, `BaseRouter.select_experts`, routed-experts capture, router top-k scores, EPLB logical/physical expert mapping.
- vLLM scheduler/EPLB path: V1 scheduler ready/running queues, `RoutedExpertsManager`, `EplbState`, `DefaultEplbPolicy`, and expert-load windows.
- New project files: research/design/dataset/trace docs, model/workload download scripts, trace collection script, analysis scripts, replay/simulator prototype, and reproducibility README.
- External dependencies: Hugging Face Hub/Datasets, optional parquet writer (`pyarrow`), plotting (`matplotlib`/`seaborn`), and vLLM MoE model checkpoints.
