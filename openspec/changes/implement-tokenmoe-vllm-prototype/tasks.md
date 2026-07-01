## 1. Research and vLLM Positioning

- [ ] 1.1 Read `idea.md` and summarize TokenMoE problem, boundaries, and contribution points.
- [ ] 1.2 Locate vLLM fused MoE router, routed-expert capture, expert parallel, EPLB, scheduler, and request metadata paths.
- [ ] 1.3 Compare MoE-Infinity, ProMoE, FineMoE/fMoE, ExpertFlow, Local Routing Consistency, ReMoE, Speculative MoE, vLLM EPLB, DeepEP, and SGLang MoE.
- [ ] 1.4 Write `docs/research.md` with related-work differences, concrete vLLM insertion points, and real-vs-simulator scope.

## 2. Models and Workloads

- [ ] 2.1 Select runnable MoE models with small-model, target-model, and trace-only tiers.
- [ ] 2.2 Define normalized agent workload JSONL schema with required `AgentNodeMeta` fields.
- [ ] 2.3 Implement `scripts/download_models.sh` with configurable model set, cache path, HF token, and dry-run behavior.
- [ ] 2.4 Implement `scripts/download_workloads.py` for planner-coder-tester, search-summarize, tool-use, multi-agent discussion, and SWE-agent/code repair workflows.
- [ ] 2.5 Write `docs/datasets.md` describing workload sources, fields, and metadata mapping.

## 3. vLLM Trace Prototype

- [ ] 3.1 Implement sidecar `request_id -> AgentNodeMeta` support in the trace benchmark.
- [ ] 3.2 Use existing vLLM routed-experts capture to collect selected expert IDs per token and layer.
- [ ] 3.3 Add optional router top-k score capture after selected expert ID tracing works.
- [ ] 3.4 Persist traces as JSONL and parquet-compatible records with active expert histograms.
- [ ] 3.5 Implement `scripts/run_trace_collection.sh` and a minimal benchmark that runs agent requests and produces traces.
- [ ] 3.6 Write `docs/trace_format.md` with schema, shapes, and examples.

## 4. Agent-Router Locality Analysis

- [ ] 4.1 Implement Agent Expert Overlap, Cross-Agent Divergence, Route Entropy, Top-M Hit Rate, and Layer Sensitivity metrics.
- [ ] 4.2 Implement baseline predictors: global expert frequency, per-request/history LRU, and sequence-level history.
- [ ] 4.3 Implement RouteSig predictor accuracy evaluation with hierarchical metadata keys.
- [ ] 4.4 Generate `analysis/locality_report.md` and plots.
- [ ] 4.5 If metadata has weak explanatory power, add prompt block type and lightweight embedding experiments before runtime optimization.

## 5. RouteSig Prototype

- [ ] 5.1 Implement `RouteSignature` data structure with expert probabilities, entropy, confidence, miss cost, and sample count.
- [ ] 5.2 Implement lookup fallback from agent-specific keys to global statistics.
- [ ] 5.3 Implement online update from routed expert traces.
- [ ] 5.4 Add tests or validation scripts for cold-start, fallback, and confidence behavior.

## 6. System Optimization Prototypes

- [ ] 6.1 Implement expert prefetch/residency simulator with hit rate, wasted prefetch, bandwidth, and estimated stall reduction outputs.
- [ ] 6.2 Implement offline expert-overlap-aware scheduler replay over legal ready agent nodes.
- [ ] 6.3 Implement proactive EPLB replay simulator comparing moving-average demand and TokenMoE predicted future demand.
- [ ] 6.4 Gate all optimizations by RouteSig confidence and provide baseline fallback behavior.

## 7. End-to-End Evaluation and Reproducibility

- [ ] 7.1 Evaluate whether agent-router locality exists across selected models and workloads.
- [ ] 7.2 Report RouteSig top-M hit rate against baselines.
- [ ] 7.3 Report prefetch lead time and simulated miss stall reduction.
- [ ] 7.4 Report batching effects on active expert fanout and per-expert token count.
- [ ] 7.5 Report proactive placement effects on p95/p99 latency or simulated all-to-all tail.
- [ ] 7.6 Document prediction error fallback behavior.
- [ ] 7.7 Write `README.md` with reproduction commands and known environment gaps.
