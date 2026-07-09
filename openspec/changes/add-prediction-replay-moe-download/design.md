## Context

The repository already has real MoE router-logit traces from Tiny Mixtral,
RouteSig and baseline predictor evaluation, and replay simulators for prefetch,
scheduling, and EPLB. The current scheduler replay is still optimistic: it uses
the real active experts from each trace as the scheduling decision signal. That
is useful as an upper bound, but it is not the prediction path described in the
updated research notes.

The vLLM/Qwen end-to-end experiment used
`/home/youwei/bzh/model/Qwen/Qwen2.5-7B-Instruct`, which is a dense Qwen2
model. The sidecar correctly disabled routed expert capture, but the next
experiment needs a MoE model available under `/home/youwei/bzh/model/`.
`/home/youwei/bzh/model/download.py` currently downloads a fixed dense Qwen
list and hardcodes a Hugging Face token. For this local experiment, the token
will remain hardcoded at the user's request, but the script must not print or
expose it in dry-run output or logs.

This change prepares the next implementation pass without changing model output
semantics. Prediction affects offline replay and future system-side actions
only. Real router traces remain the ground truth for scoring and online updates.

## Goals / Non-Goals

**Goals:**

- Add a reusable prediction interface that returns per-layer top-M expert sets
  from RouteSig and baseline predictors before router execution.
- Convert scheduler DAG replay from true-active-expert decisions to
  prediction-based decisions.
- Add metrics that separate predictor quality, oracle upper bound, and actual
  replay outcome.
- Add temporal and spatial locality measurements needed by the updated notes.
- Update `/home/youwei/bzh/model/download.py` so MoE target downloads are
  profile-based, dry-runnable, and use `https://hf-mirror.com`.

**Non-Goals:**

- Do not modify vLLM's live scheduler in this change.
- Do not add real expert weight prefetch/offload hooks in this change.
- Do not change router choices, token dispatch correctness, prompts, sampling
  parameters, or model weights.
- Do not download large MoE models during unit tests.
- Do not train a GNN, RL policy, or dense neural predictor in the first
  implementation pass.

## Decisions

1. Prediction API first, learned models later.

   Add a small data structure, for example `PredictedExpertSet`, that contains
   `request_id`, `layer_id`, `experts`, `confidence`, `source`, and optional
   `scores`. RouteSig, global frequency, request LRU, sequence history, and
   oracle adapters can all produce the same shape. This keeps simulators from
   depending on internal predictor classes.

   Alternative considered: pass `RouteSigStore` directly into every simulator.
   That would keep the implementation short but make baseline and oracle
   comparisons harder to report consistently.

2. Scheduler replay must decide from predictions and score from truth.

   For each replay step, build the legal ready set from workload dependencies.
   Candidate batches are scored with predicted per-layer top-M sets, waiting
   penalty, and optional confidence gating. After a batch is selected, compute
   actual fanout and per-expert token counts from real traces. This preserves
   the correctness boundary and makes the replay result interpretable.

   Alternative considered: keep the current true-active-expert replay and only
   relabel it as oracle. The implementation should still keep an oracle mode,
   but it must not be the default TokenMoE replay result.

3. Locality metrics should be derived from existing traces first.

   Reuse distance, phase-transition overlap, temporal Jaccard, per-layer fanout,
   and per-expert token count can be computed from existing JSONL traces and
   workload metadata. Temporal metrics should be reported as analysis features
   but should not drive first-pass replay decisions.

   Alternative considered: add a learned temporal embedding model immediately.
   The current trace volume is too small for that to be the first useful step.

4. Model download must target real MoE experiments.

   Replace the fixed dense model list in `/home/youwei/bzh/model/download.py`
   with profiles including `dense-qwen` and `moe-target`. `moe-target` is the
   primary profile and should include real target MoE candidates rather than
   smoke-only models:

   - `Qwen/Qwen3-30B-A3B`
   - `mistralai/Mixtral-8x7B-Instruct-v0.1`
   - `deepseek-ai/DeepSeek-V2-Lite-Chat`

   The script should support `--profile`, `--model`, `--dry-run`,
   `--local-root`, and `--endpoint`. The default endpoint is
   `https://hf-mirror.com`. The local experiment keeps the hardcoded token
   behavior requested by the user, but token values must not appear in logs.

   Alternative considered: make `moe-small` the default for smoke testing. The
   user rejected smoke-only experiments as not meaningful for this stage.

5. Real MoE execution should use vLLM.

   The first real MoE path should use vLLM routed expert capture rather than
   Transformers router logits. Transformers tracing can remain as a fallback
   for existing tests and trace compatibility, but implementation should
   prioritize a vLLM model capability check and routed expert capture path.

6. Tests should not depend on network or large models.

   Unit tests should cover predictor output shape, dependency-safe replay,
   fallback behavior, and download plan construction. Download execution should
   be tested in dry-run or with a mocked `snapshot_download`.

## Risks / Trade-offs

- Prediction replay may show less benefit than oracle replay -> Report oracle,
  baseline, and predictor results separately instead of hiding the gap.
- RouteSig may be low confidence for cold keys -> Keep hierarchical fallback
  and confidence gating; report gated fraction.
- Scheduler overlap can increase waiting time -> Include waiting penalty,
  fairness guard, and waiting/deadline metrics.
- Large MoE model downloads may exceed disk or GPU resources -> Keep dry-run
  output and make model IDs visible before download starts.
- Hardcoded credentials are unsafe -> This is accepted as a local experiment
  constraint; do not print the token and do not represent it as a reusable
  public-script pattern.
- Model availability or mirror behavior may change -> Keep model IDs
  configurable and report skipped or failed downloads clearly.
- vLLM MoE support may differ by model -> Add capability checks and clear
  fallback/error messages before running routed capture.

## Migration Plan

1. Add prediction result structures and adapters while preserving existing
   predictor evaluation behavior.
2. Add prediction-based scheduler replay beside the current replay, then make
   the reporting distinguish baseline, RouteSig prediction, and oracle modes.
3. Add locality metrics and reports derived from existing traces.
4. Refactor `/home/youwei/bzh/model/download.py` into a profile-based CLI with
   `moe-target`, `https://hf-mirror.com`, dry-run, and non-logging hardcoded
   token behavior.
5. Update docs and tests. Existing reproduction commands should keep working.

Rollback is straightforward because all replay changes are offline and additive.
If a new replay path is incorrect, reports can fall back to the existing
baseline and oracle replay functions while keeping trace collection unchanged.

## Confirmed Decisions

- Default experimental profile: `moe-target`.
- Download endpoint: `https://hf-mirror.com`.
- Token policy: hardcoded local token retained for this experiment; never log it.
- First real execution backend: vLLM routed expert capture.
- Prediction policy: statistical RouteSig drives replay decisions; temporal
  metrics are computed and reported but do not drive first-pass scheduling.
