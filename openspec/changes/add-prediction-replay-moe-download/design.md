## Context

TokenMoE's research goal is to use information visible before MoE router
execution, especially agent role, phase, graph position, and prompt-block
structure, to predict expert working sets for system-side decisions. The current
prototype is a useful trace-first baseline, but it still has three gaps:

1. Real vLLM evidence used a dense Qwen model, while MoE routing evidence came
   from Tiny Mixtral through Transformers router logits.
2. Scheduler replay currently uses true active experts as a TokenMoE decision
   proxy, which is an oracle upper bound rather than a prediction path.
3. The current RouteSig abstraction is request/layer oriented and does not yet
   model prompt block or token-segment locality.

This change tightens the stage around prompt-only vLLM MoE traces and
block-level prediction replay. It deliberately does not implement online expert
prefetch, vLLM scheduler changes, or EPLB replica placement. Those are next
stage runtime integrations that should consume the evidence and interfaces
produced here.

## Goals / Non-Goals

**Goals:**

- Collect prompt-only routed expert traces with vLLM on real local MoE models.
- Download external datasets through the existing dataset download script by
  editing only `DATASET_LIST`.
- Convert each external dataset into its own normalized workload JSONL using a
  dedicated dataset adapter folder outside core `tokenmoe/` modules.
- Preserve prompt block character spans in normalized workloads and map them to
  token spans during trace collection.
- Preserve source provenance, claim scope, group IDs, source indices, timestamps
  where available, and license/access/redaction metadata.
- Extract dependency edges and ready times from real agent trajectory datasets
  where available, especially SWE-agent-derived workloads.
- Introduce explicit workload/trace v2 schemas and reject mixed v1/v2 inputs in
  analysis.
- Persist model MoE-layer IDs and ignore or mark non-MoE layers unavailable in
  trace writing and metrics.
- Require every predictor to emit normalized `expert_weights` in addition to
  ranked expert IDs.
- Add segment-aware trace, prediction, and RouteSig structures.
- Evaluate simple baselines and RouteSig using time-ordered, online
  predict-before-update evaluation.
- Replace scheduler replay's oracle decision signal with predicted expert
  demand aggregated from segment-level predictions.
- Report prediction quality, fallback behavior, block-level locality, scheduler
  replay metrics, and oracle gaps.

**Non-Goals:**

- Do not modify vLLM's live scheduler in this change.
- Do not add online expert weight prefetch, offload, or cache hooks in this
  change.
- Do not implement proactive EPLB replica placement in this change.
- Do not keep Transformers router-logit collection as an active experiment path.
- Do not require semantic embedding, GNN, RL, contextual bandit, or learned
  dense predictor baselines in this change.
- Do not make mixed multi-dataset workload files a requirement.
- Do not treat mock, synthetic, deterministic, or smoke traces as completion
  evidence.
- Do not use non-agent chat or domain-instruction datasets as evidence for the
  agent-DAG metadata claim.
- Do not include prefetch or EPLB simulator outputs in the main analysis/report
  path for this change unless they are clearly archived and excluded.

## Decisions

1. Real trace collection uses vLLM routed-experts only.

   `scripts/collect_traces.py` should collect routed experts through vLLM with
   `enable_return_routed_experts=True`. The active user-facing backend should no
   longer include Transformers router-logit collection. Small synthetic fixtures
   may remain in tests for pure formatting or algorithm regressions, but they
   cannot satisfy validation tasks.

   The collector may generate the minimum output token needed to drive vLLM, but
   it must retain only prompt-token routing for this stage. If vLLM returns a
   combined prompt/decode routed array, the collector truncates routing to
   `prompt_token_count` and records that decode routing was excluded.

   vLLM routed-experts buffers are allocated across hidden layers and can leave
   non-MoE layers as zero-filled slots. The collector must derive `moe_layer_ids`
   from the model implementation/config or a verified vLLM capability probe,
   persist them in the trace metadata, and filter or mark non-MoE layers
   unavailable before trace writing and evaluation. A dense or unavailable layer
   must never be interpreted as expert `0` activity.

2. Model coverage is staged by research value, not by current free GPU memory.

   The primary model is `/home/youwei/bzh/model/Qwen/Qwen3-30B-A3B`, because it
   has many experts and a larger top-k than Mixtral-style models. The required
   cross-model validation is
   `/home/youwei/bzh/model/deepseek-ai/DeepSeek-V2-Lite-Chat` on ShareGPT and
   SWE-agent trajectories. Mixtral remains optional compatibility validation.
   Runtime commands should perform capability/resource checks and fail clearly
   when a model cannot run, but current transient GPU occupancy is not a spec
   constraint.

   The validation commands need concrete vLLM launch profiles. Before running,
   they must assert that the selected model is MoE, routed-experts return is
   enabled, pipeline parallelism is disabled, context parallelism is disabled,
   KV transfer/connectors are disabled, and TP/EP, dtype, max model length, and
   GPU memory settings are explicit. If a profile cannot satisfy these
   constraints, validation fails instead of falling back to another backend.

3. Dataset download is constrained to a single list edit.

   `/home/youwei/bzh/dataset/download_dataset.py` already defines `BASE_DIR`,
   endpoint behavior, token usage, retry logic, and `DATASET_LIST`. This change
   may edit only `DATASET_LIST`, setting it to:

   ```python
   DATASET_LIST = [
       "anon8231489123/ShareGPT_Vicuna_unfiltered",
       "lmsys/lmsys-chat-1m",
       "nebius/SWE-agent-trajectories",
       "nvidia/OpenCodeInstruct",
       "nvidia/OpenMathInstruct-2",
   ]
   ```

   If a gated dataset such as LMSYS requires prior access approval, the command
   should fail visibly rather than silently replacing it with a different
   dataset.

4. Dataset adapters are isolated from core TokenMoE logic.

   Add a dedicated adapter folder, for example `dataset_adapters/`, containing
   one converter per dataset plus shared adapter utilities. Core modules under
   `tokenmoe/` consume only normalized workload JSONL and do not parse raw
   ShareGPT, LMSYS, SWE-agent, OpenCodeInstruct, or OpenMathInstruct formats.

   Each adapter outputs one dataset-specific file under a local artifact root,
   for example `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/`:

   - `sharegpt_prompt_workloads.jsonl`
   - `lmsys_prompt_workloads.jsonl`
   - `swe_agent_prompt_workloads.jsonl`
   - `opencode_prompt_workloads.jsonl`
   - `openmath_prompt_workloads.jsonl`

   A manifest records source path, output path, sample count, field mapping
   version, conversion command, source index fields, group ID fields, timestamp
   fields, license/access status, and redaction status. Mixed workload
   generation is out of scope. Prompt-bearing normalized workloads and traces
   should not be written under tracked repository `data/` paths.

   Adapters also assign a claim scope:

   - `real_agent_metadata`: SWE-agent trajectories, where trajectory and tool
     events can support agent metadata claims.
   - `chat_prompt_only`: ShareGPT and LMSYS, which support chat/prompt block
     locality but not agent-DAG claims.
   - `domain_instruction`: OpenCodeInstruct and OpenMathInstruct-2, which
     support code/math block locality but not agent-DAG claims.

   Reports must present these scopes separately and must not use non-agent
   datasets as evidence for the central agent-DAG metadata claim.

   For `real_agent_metadata` outputs, the adapter must emit dependency edges and
   ready times whenever the source trajectory provides enough event ordering to
   reconstruct them. Scheduler replay can make agent-DAG claims only for
   workload files with valid dependency information. Datasets without dependency
   graphs are prediction/locality inputs and, at most, prompt/domain locality
   replay inputs.

5. Prompt block spans are first-class experiment data.

   Adapters construct prompt text from semantically typed blocks and record
   character spans for each block. The collector maps those character spans to
   token spans using tokenizer offsets, then slices the vLLM routed-experts array
   to derive segment-level routed traces.

   A prompt block span should include at least:

   - `segment_id`
   - `block_type`
   - `segment_position`
   - `char_start`
   - `char_end`
   - `token_start`
   - `token_end`
   - alignment status or failure reason

   If a dataset record cannot be reliably aligned, the collector marks segment
   metrics unavailable for that record rather than quietly mixing it into normal
   block-level results.

   Alignment must use the same tokenizer settings as vLLM prompt processing.
   After mapping character spans to token spans, the collector compares the
   aligned token IDs with `RequestOutput.prompt_token_ids`. If tokenizer
   normalization, special tokens, BOS handling, or chat templates make the IDs
   inconsistent, the whole record is segment-unavailable for block-level
   metrics.

6. RouteSig predicts per segment and layer.

   Add a shared prediction shape such as `PredictedExpertSet`:

   ```text
   request_id
   segment_id
   layer_id
   expert_ids
   expert_weights
   confidence
   source
   fallback_key
   scores
   token_segment = "prompt"
   block_type
   segment_position
   token_start
   token_end
   ```

   `expert_weights` are normalized probabilities or demand weights over the
   listed expert IDs. Count-based predictors normalize historical counts.
   Predictors that only know a ranked Top-M set use an explicit uniform-weight
   rule. Oracle weights are derived from true routed counts only in oracle mode
   and never before a prediction-mode scheduling decision.

   RouteSig fallback order is:

   ```text
   (agent_id, role, phase, block_type, segment_position)
     -> (role, phase, block_type, segment_position)
     -> (role, phase, block_type)
     -> (role, block_type)
     -> (block_type)
     -> (role, phase)
     -> (role)
     -> global
   ```

   This keeps the first implementation statistical and explainable. Semantic
   nearest-neighbor, embeddings, graph predictors, and learned rerankers are
   future extensions.

7. Segment predictions aggregate by token count for replay.

   Scheduler replay needs request and batch-level expert-token-label demand,
   while RouteSig predicts segment/layer experts. Aggregate by segment
   prompt-token count and the model router top-k:

   ```text
   Demand(v, l, e) =
     sum over segments s in prompt(v):
       token_count(s) * router_top_k(model) *
       weight(e | role, phase, block_type(s), position(s), l)
   ```

   `weight(...)` is the normalized `expert_weights` value emitted by the
   predictor. Demand units are expected expert-token labels, matching the fact
   that true routing contributes one label per selected expert per token. Batch
   demand is the sum of request demand. Replay decisions use this predicted
   demand. True routed experts are used only after a batch has been selected to
   score actual fanout, token density, hit rate, and oracle gap.

8. Top-M is a model-relative budget.

   The primary Top-M result uses `2 * router_top_k`. Reports also include budget
   curves at `1x`, `1.5x`, `2x`, and `3x` router top-k. For the confirmed
   models, this means:

   - Qwen3 top-k 8: 8, 12, 16, 24
   - DeepSeek top-k 6: 6, 9, 12, 18
   - Mixtral top-k 2: 2, 3, 4, 6

9. Evaluation is online and time ordered.

   For every `(dataset, model)` pair, traces are split in original order by
   group before request or segment expansion: conversation for ShareGPT/LMSYS,
   trajectory or workflow for SWE-agent, and source item for code/math
   instruction datasets. The first 70% of groups are used for training and the
   remaining 30% for evaluation. During evaluation, each predictor must predict
   before it sees the current trace, score against true vLLM routed experts,
   then update from that trace. If timestamps exist, they define order;
   otherwise adapters preserve and report `source_index` order.

   Baselines in scope are global frequency, request LRU, sequence history,
   temporal window frequency with default `W=64` and optional `W=16/64/256`
   sensitivity, RouteSig, and oracle future demand as an upper bound only.

10. Scheduler replay optimizes a constrained MoE execution proxy.

    At each replay step, construct the legal ready set from workload
    dependencies. Candidate batches are scored from predicted demand using:

    - predicted active expert fanout cost
    - predicted low token-density cost
    - added waiting cost
    - low-confidence penalty or gating
    - starvation guard

    Hard constraints include no dependency violations, batch size limits, and
    maximum delay bounds. The report must include MoE execution proxy metrics,
    scheduling cost, policy comparison, and failure/fallback fractions.

    If a workload lacks dependency edges or ready times, scheduler replay must
    label the result as non-agent prompt/domain locality replay or mark DAG
    scheduling unavailable. It must not report agent-DAG scheduling gains from
    independent-request batching.

## Risks / Trade-offs

- vLLM routed-experts capture can fail for a model or environment -> Fail with a
  clear capability/resource error; do not silently substitute Transformers or
  synthetic traces as experiment evidence.
- vLLM routed-experts arrays may contain zero-filled non-MoE layers -> Persist
  `moe_layer_ids`, filter dense layers, and reject metrics that treat non-MoE
  layers as active expert IDs.
- Prompt-block alignment can be imperfect -> Record alignment status and report
  missing segment metrics explicitly.
- Tokenizer offsets may not match vLLM prompt tokens -> Verify aligned token IDs
  against vLLM `prompt_token_ids` and mark the record segment-unavailable on
  mismatch.
- Dataset fields may not contain real agent metadata -> Adapters must map the
  best available role/phase/block fields, record source provenance and claim
  scope, and prevent chat/domain datasets from supporting agent-DAG claims.
- Real agent replay can lack valid DAG fields -> Require dependencies and
  ready times for SWE-agent-derived scheduling replay and mark replay
  unavailable when they cannot be reconstructed.
- RouteSig may underperform temporal baselines on some datasets -> Report
  baseline, RouteSig, and oracle separately rather than hiding negative results.
- Schema changes can invalidate existing traces -> Use explicit workload/trace
  v2 schema versions, keep v1 readers only for compatibility, and reject mixed
  v1/v2 analysis inputs.
- External prompts can have privacy or license constraints -> Store normalized
  workloads and prompt-bearing traces outside tracked repo paths and include
  license/access/redaction fields in manifests.
- Longer prompt blocks can dominate demand aggregation -> This is intended for
  system-cost replay, but reports must also include per-segment locality so the
  source of demand is visible.
- Demand can be incomparable across model top-k values -> Define demand units as
  expert-token labels and scale segment demand by router top-k.
- Legacy reports can still contain prefetch/EPLB sections -> Remove those from
  main analysis/report outputs for this change or label them archived/excluded.
- Real-model validation is resource-heavy -> Keep per-dataset default at 256
  prompt records and make commands resumable where practical.

## Migration Plan

1. Update OpenSpec artifacts to remove prefetch/EPLB requirements from this
   change and lock the prompt-only vLLM/block-level replay scope.
2. Update the dataset download list with the confirmed dataset IDs and no other
   script changes.
3. Add isolated dataset adapters and per-dataset normalized workload outputs
   under `/home/youwei/bzh/dataset/tokenmoe_artifacts/`.
4. Extend workload and trace schemas to v2 for prompt block spans, token spans,
   source provenance, claim scope, MoE-layer IDs, and prompt-only routing flags.
5. Replace the active trace collector with a vLLM routed-experts prompt-only
   path and remove Transformers backend support from the active CLI.
6. Add vLLM routed-experts preflight profiles and hard validation checks.
7. Implement segment-aware RouteSig and simple baselines using the shared
   prediction shape with normalized expert weights.
8. Implement prediction-based scheduler replay using expert-token-label demand,
   dependency/ready-time checks, and oracle-only scoring.
9. Remove or archive/exclude prefetch and EPLB outputs from main
   analysis/report commands for this change.
10. Update reports/docs to present per-dataset/model prediction and scheduler
   replay results.
11. Validate on the required Qwen3 and DeepSeek real-model runs with hard
    rejection of legacy/fallback backends.

Rollback is straightforward because the change is offline and additive around
data preparation, trace capture, prediction, replay, and reporting. If the new
vLLM trace path fails, the change should remain incomplete rather than passing
through a mock or Transformers fallback.

## Next Stage

After this change establishes real vLLM prompt-routing prediction and
prediction-based scheduler replay, the next stage should implement runtime
TokenMoE actions:

- online expert prefetch/offload and expert residency management
- vLLM scheduler modification for expert-overlap-aware batching
- proactive EPLB replica placement from predicted future demand
- semantic/block embedding rerankers, graph predictors, or learned action
  policies
