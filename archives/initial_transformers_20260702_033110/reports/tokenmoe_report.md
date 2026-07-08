# TokenMoE vLLM Prototype Report

## Executive Summary

This prototype validates a trace-first TokenMoE path: agent metadata is joined
with real MoE router traces, RouteSig predicts per-layer expert working sets,
and prefetch/scheduling/EPLB policies are evaluated in replay before any
correctness-sensitive vLLM runtime changes.

Trace records: **80**. Model backend: **transformers-router-logits**.

## Agent-Router Locality

![Role-conditioned expert distribution](../analysis/figures/role_expert_heatmap.png)

The heatmap shows that roles and phases do not activate experts uniformly.
This is the signal RouteSig exploits before request admission.

## Predictor Results

![Top-M predictor comparison](../analysis/figures/topm_hit_rate.png)

RouteSig best hit rate: **0.969**.
Global-frequency best hit rate: **0.957**.
The delta is the first-order evidence for or against agent-conditioned
expert prediction on this workload/model pair.

## Layer Sensitivity

![Layer sensitivity](../analysis/figures/layer_sensitivity.png)

Layer sensitivity identifies where metadata is useful. Runtime integration
should prioritize high-delta MoE layers and leave low-confidence layers on the
baseline path.

## System Replay

![Simulator summary](../analysis/figures/simulator_summary.png)

- Prefetch hit rate: **0.861** with **0.021** wasted-prefetch rate.
- Estimated stall reduction proxy: **160.36 ms** over the replay window.
- Expert-overlap scheduler fanout: baseline **13.48**, TokenMoE **14.24**.
- EPLB p95 tail proxy: moving-average **584.96**, TokenMoE **546.43**.

## Correctness Boundary

All prototype optimizations are confidence-gated and replay-only. Real router
outputs remain the source of truth. When metadata is missing, confidence is low,
or a prediction misses, the runtime falls back to normal vLLM routing/dispatch.
