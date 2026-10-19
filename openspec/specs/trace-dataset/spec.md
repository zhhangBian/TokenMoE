# Trace Dataset

## Purpose
Define the static records, finalized dataset layout, prompt alignment, validation, equivalence gate, and storage estimates for TokenMoE trace data.

## Requirements

### Requirement: Static records
Before any session starts, the launcher SHALL write the static records of §3:
- the experiment config, carrying the engine config id read from the server's `engine_meta.json`;
- the model profile;
- the role template, with `role_type = generalist`, the system prompt file and the tool schema file;
- the benchmark items.

`benchmark_items` SHALL use the repository name as `task_type` and the dataset's difficulty label as `difficulty`. Static IDs SHALL be content hashes, so identical content yields identical IDs.

#### Scenario: Template change yields a new version
- **WHEN** the system prompt of the mini-swe-agent config is edited and a new run starts
- **THEN** the role template has the same `agent_template_id` and a different `version_hash`

#### Scenario: Model profile from the layer map
- **WHEN** static records are written for a model
- **THEN** `moe_layer_ids` in the model profile equals the layer list in the server's `layer_map.json`, and `layer_latency_profile_ref` is null

### Requirement: Finalize into the storage layout
`python -m tokenmoe_collect finalize <run_dir>` SHALL convert the harness and engine raw files into the §6 layout:
- jsonl for application runs, sessions, requests and tool calls;
- parquet for prompt segments, tool output chunks, engine steps and host load;
- `tool_outputs/<tool_call_id>.out`;
- `routing/<model_profile_id>/<llm_request_id>.npz`;
- `prompts/<llm_request_id>.prompt.txt` and `prompts/<llm_request_id>.output.txt`: the decoded engine prompt and output tokens, special tokens kept.

`prompt_ref` and `output_ref` SHALL point to these files. Segment character ranges SHALL index into the prompt file. Routing and tool output files SHALL be hardlinked from the raw files, and copied only when a hardlink is impossible.

It SHALL join harness and engine records on `llm_request_id`. It SHALL fill:
- `issued_at` from the issuing request's `inference_finished_at`;
- `llm_request_ids_consuming` from `prev_tool_call_ids`;
- the engine-side lifecycle fields of each request.

It SHALL keep the raw files, and it SHALL run the validator at the end.

For the local Qwen profile, an evidenced context-budget rejection before engine admission SHALL retain its failed harness request with `engine_admission_status = rejected_before_engine`, the HTTP status and rejection evidence, null engine lifecycle/token fields, and null prompt/output/routing references. No engine prompt or routing rows SHALL be fabricated. Actual error responses and exact-input replay evidence SHALL be distinguished explicitly; replay verification SHALL match model/messages/tools/sampling/seed and engine configuration and require no engine record for the original or probe. Unknown missing engine records SHALL retain the normal failure behavior.

#### Scenario: Issued time filled
- **WHEN** request `req_a` finishes at t=100 and its response contains tool call `tc_b`
- **THEN** `tc_b.issued_at` equals 100 after finalize

#### Scenario: Prompt text matches segments
- **WHEN** a finalized request has a `tool_output` segment with character range `[a, b)`
- **THEN** characters `[a, b)` of `prompts/<llm_request_id>.prompt.txt` contain that tool call's payload as passed to the agent

#### Scenario: Large files stored once
- **WHEN** finalize runs on a run directory that sits on one filesystem
- **THEN** every routing and tool output file in the layout shares its inode with the corresponding raw file

#### Scenario: Missing engine record
- **WHEN** a harness request has no matching engine request record and no evidenced pre-engine context rejection
- **THEN** finalize keeps the harness record, marks the request `engine_record_missing`, and validation reports it

#### Scenario: Context rejection before engine admission
- **WHEN** the API rejects a local request for its context budget before engine admission and verified rejection evidence exists
- **THEN** finalize preserves the failed request and raw inputs with explicit rejection metadata and null engine artifacts, retaining earlier computed traces without inventing routing

### Requirement: Prompt segments

Finalize SHALL build the prompt segments of each engine-admitted request as follows:

1. Decode the engine's prompt token IDs.
2. Locate the harness messages in order inside the decoded text.
3. Split tool-output payloads out of their wrapping templates, using the harness provenance offsets.
4. Give every other character range the type `other`.

Each segment SHALL carry character and token ranges. A message that cannot be located SHALL produce a segment with null ranges and a non-null `alignment_error`. Located segments SHALL tile the prompt without overlap.

Consumers SHALL preserve original line endings when reading prompt artifacts for character-range checks. A CRLF pair SHALL count as two characters in these ranges.

For local Qwen collection, a field omitted by the actual chat template SHALL remain in the raw harness messages and SHALL have null prompt/token ranges, a non-null `alignment_error`, and an explicit `template_omitted` classification. This classification SHALL require reproducible evidence from the effective template/input and actual engine prompt token IDs. Missing fields without that evidence SHALL remain unexpected alignment failures. Tool payload loss SHALL NOT be classified as an allowed omission.

The existing `location_rate` definition SHALL remain unchanged. Local `alignment.json` SHALL additionally report omitted-field counts, unexpected failures and `rendered_location_rate = located / (message_fields - template_omitted)` with its numerator and denominator; zero denominator SHALL be reported as null with a reason.

Parquet export SHALL preserve optional `alignment_status` and `omission_evidence_json` fields even when they occur only in later rows. Rows without those fields SHALL use null values without dropping the columns or their evidence.

#### Scenario: Tool output segment
- **WHEN** a request contains the observation of tool call `tc_x`
- **THEN** one or more payload segments have `segment_type = tool_output`, `source_tool_call_id = tc_x`, and a token range inside the request's new-prefill range, or inside the cached range if the prefix was cached

#### Scenario: Alignment failure
- **WHEN** the chat template rewrote a message so that none of its text variants appear in the decoded prompt
- **THEN** that message's segment has null ranges and a non-null `alignment_error`, and any unlocated actual prompt characters remain covered by an `other` segment

#### Scenario: CRLF in a benchmark prompt
- **WHEN** an engine prompt contains CRLF pairs and segment ranges tile the original decoded text
- **THEN** validation reads those pairs unchanged and accepts the correctly covered prompt without shortening it through newline normalization

#### Scenario: Proven omission of historical reasoning
- **WHEN** a fixed-input reproduction and the actual engine token IDs show that the effective template omitted a historical reasoning field
- **THEN** the field is classified as template_omitted with retained raw text and evidence, has no fabricated token range, and both original and rendered-field location rates are reported

#### Scenario: Unexplained missing tool payload
- **WHEN** a tool observation cannot be found in the actual prompt and no supported rendering transformation explains it
- **THEN** it remains an unexpected alignment failure, and cannot be removed from the local location-rate denominator as a template omission

#### Scenario: Optional evidence appears after ordinary prompt rows
- **WHEN** an initial ordinary segment has no omission metadata and a later segment contains proven template-omission evidence
- **THEN** the exported Parquet table retains the later classification and evidence, enabling the same rendered-field alignment rate before and after serialization

### Requirement: Validation
`python -m tokenmoe_collect validate <run_dir>` SHALL check the rules of `collection/data_collect.md` §7 that apply to single-loop, single-host runs: rules 1–9, 11 and 12, with rules 6–9 in their revised form. Rules 6 and 7 SHALL join rows through `token_positions`, including holes from later cache hits. Aborted/error requests with `routing_complete = false` SHALL be checked against their actual computed positions; complete generations SHALL still require `row_end = T-1`. It SHALL report rules 10, 13 and 14 as `not_applicable`. It SHALL write `validation.json` with one entry per rule: status, violation count, and up to 20 example IDs. The exit code SHALL be non-zero if any applicable rule fails.


For evidenced local pre-engine context rejections, rules 6 and 7 SHALL verify structured HTTP 400 context-error evidence, null engine lifecycle/routing fields, and the absence of engine-step entries. Rule 9 SHALL require no fabricated engine prompt or segments. These cases SHALL NOT mask missing routing for admitted/computed requests or unknown engine-record loss. Sessions without any actual computed routing SHALL NOT count as collected task coverage.

#### Scenario: Routing shape violation
- **WHEN** a contiguous routing file has fewer rows than its computed token_positions and matching step entries
- **THEN** rule 6 fails and names that `llm_request_id`

#### Scenario: Recompute allowed
- **WHEN** a preempted request has recompute entries in addition to its new_prefill entries
- **THEN** rule 7 passes, provided that every new_prefill and decode token appears exactly once

#### Scenario: Missing equivalence record
- **WHEN** no passing equivalence record exists for the run's `engine_config_id`
- **THEN** rule 12 fails

#### Scenario: Rejection metadata contradicts engine execution
- **WHEN** a request marked rejected_before_engine has engine lifecycle timestamps, routing, or computed step entries
- **THEN** validation fails the applicable integrity rule instead of accepting the rejection annotation

### Requirement: Equivalence gate
`python -m tokenmoe_collect equiv` SHALL send a fixed set of 50 first-step prompts sequentially, greedily, with `max_tokens` 256. It SHALL do so to four server runs with the same serve config: capture off, off, on, on. It SHALL pass when all three conditions hold:
- the on/off token-sequence disagreement count is no greater than the off/off count plus one;
- every captured row of a bound layer holds K distinct IDs in `[0, E)`;
- the two capture-on runs produce identical routing wherever their tokens agree.

It SHALL write the result to `static/equivalence/<engine_config_id>.json`.

#### Scenario: Gate fails on divergence
- **WHEN** capture-on outputs diverge on 6 prompts while off/off diverge on 1
- **THEN** the gate fails and lists the diverging prompts

### Requirement: Storage estimate
`python -m tokenmoe_collect estimate` SHALL compute the expected storage per session and per run from:
- the model facts (L, K, E);
- the assumed or measured tokens computed per session;
- the context-growth assumptions.

The collection README SHALL state the estimate for every configured model.

#### Scenario: Measured estimate
- **WHEN** `estimate` is run on a finalized pilot directory
- **THEN** it reports the measured bytes per category next to the formula-based prediction

### Requirement: Spec consistency
`collection/data_collect.md` SHALL describe the records, fields, layout and rules that the code emits and checks. The changes C1–C7, the removal of router scores, and the write-backs listed in the proposal SHALL be applied in the same change as the code.

#### Scenario: Spec describes routing rows
- **WHEN** a reader looks up §4.8 in `collection/data_collect.md`
- **THEN** it states that routing rows cover `[num_cached_tokens, T-1)`, explains `row_start` and `step_index`, and gives the expert ID dtype rule

#### Scenario: No score fields remain
- **WHEN** a reader searches `collection/data_collect.md` for router scores
- **THEN** the spec states that scores are not recorded, and no record, rule or estimate refers to a score array

### Requirement: Local trace acceptance without task-success gating

Acceptance for `local-qwen3-runtime-v1` SHALL apply structural validation rules 1–9, 11, and 12, marking rules 10, 13, and 14 not_applicable. Submission rates or business-test outcomes SHALL NOT override failures of applicable rules. The local submission-rate threshold SHALL default to 0, benchmark_score SHALL remain null, and actual task states, exit reasons, and explicitly configured thresholds SHALL be reported honestly.

Valid local data SHALL include formal capture-equivalence evidence matching the effective engine_config and code/weight versions, complete raw/finalized files, traceable tool observations, and rendered_location_rate ≥99%. Reports SHALL disclose the original location_rate, template-omission counts, and supporting evidence. Traces with missing equivalence evidence or structural corruption SHALL remain diagnostic data for the executing agent to repair and recollect; passing records SHALL NOT be fabricated.

Local recorder overhead SHALL be measured through capture-off/on comparisons with a fixed workload, recording hardware, parameters, sample size, repetitions, and measurement scope. Overhead above 5% SHALL be recorded for optimization without independently invalidating structurally sound local data. The existing formal Hopper requirement of less than 5% SHALL remain in force.

#### Scenario: No successful submissions
- **WHEN** every task in a batch fails or exhausts its limits but each planned unit has a valid trace and actual terminal state
- **THEN** local mode accepts the collected batch and reports a submission rate of 0 without requiring SWE-bench scores or recollection to improve that rate

#### Scenario: Equivalence is absent
- **WHEN** a structurally sound trace lacks a valid equivalence record for the effective serving configuration
- **THEN** rule 12 still fails, the trace remains diagnostic data, and the executing agent completes equivalence verification before determining acceptance without granting an exemption

#### Scenario: Recorder overhead exceeds the formal target
- **WHEN** local data meets structural and equivalence requirements and measured overhead is 8%
- **THEN** the report records measurement conditions and the 8% result, and the executing agent investigates and performs bounded optimization without deleting or rejecting valid traces solely because of that percentage

### Requirement: Runtime data summary

The summary SHALL provide machine-readable JSON and a readable report for each concurrency setting, including at least:

- Planned/valid/recollection unit counts, unique task counts, actual task states and exit_reason, fault categories, profile/configuration, and version provenance.
- Session, request/attempt, tool, engine-step, and chunk counts; prompt/output/cached tokens; first-computed tokens; and recompute tokens.
- Per-layer new_prefill/decode expert-lookup histograms joined through token_positions/step_index, excluding duplicate computation and cache-skipped ranges, and checking K=8 against the actual layer map.
- Raw/seen tool bytes, chunk integrity, coverage linking tools to subsequent requests and new-prefill payloads, and counts of first requests and tool-category/outcome transition samples.
- Tool duration, time to first output, silence intervals, tool-completion-to-consuming-request intervals, queueing/first-token/generation intervals, distributions, and p50/p95.
- Actual session/request overlap, batch composition by concurrency setting, host load, and missing GPU telemetry.
- Alignment, validation, equivalence, overhead, storage deduplicated by inode, and artifact checksums.

Missing endpoints, zero denominators, and unobservable values SHALL be reported as null with a reason rather than fabricated numbers. Engine-step wall time and request generation intervals SHALL NOT be described as per-layer kernel time or measured per-token latency. The report SHALL NOT claim that this collection measures expert-cache benefits, offload stalls, or end-to-end acceleration.

#### Scenario: Cached prompt and preemption
- **WHEN** a request hits a cached prefix and some computed positions are recomputed after preemption
- **THEN** histograms include only that request's first-computation routing rows, cached/recompute work is reported separately, and no expert demand is fabricated for the cached prefix

#### Scenario: Tool emits no output
- **WHEN** a tool has no output chunks in the actual trace
- **THEN** output bytes are reported as 0, time to first output is null with a no-output explanation, and tool duration still uses actual start/end timestamps

### Requirement: Versioned labels and limited generalization claims

Offline tool-category, outcome-category, and transition labels SHALL be derived from real commands, outputs, and states using versioned rules and stored under `derived/labels_v<N>/`. Uncertain categories SHALL use unknown. Features used at a prediction time SHALL have verifiable event times or available_at values. Final exit codes, final outcomes, and future logs SHALL be marked as supervision information and SHALL NOT be treated as inputs available when a tool call is issued.

Local reports SHALL state the single generalist role, bash tool interface, and limited task scale. G1 splits SHALL group by benchmark item; all requests, N1/N4 executions, pilot runs, and recovery copies for the same task SHALL belong to one group without random request-level leakage. G2/G3 and unseen-tool generalization SHALL be identified as outside this collection's coverage.

#### Scenario: Same task under both concurrency settings
- **WHEN** the same SWE-bench item appears in N1, N4, and recovery runs
- **THEN** the downstream G1 grouping manifest assigns all copies to the same task group without placing different copies on both sides of the training/test split

#### Scenario: Failure outcome is known only at completion
- **WHEN** a pytest failure can be determined only after the tool finishes
- **THEN** the outcome is retained as a supervision label with provenance and cannot appear among inputs available when the tool was issued

### Requirement: Local documentation consistency

Implementation SHALL document the new profile, fields, operational files, collection matrix, acceptance metrics, and recovery rules in `collection/data_collect.md` §1, §3.1, §4.1–4.4, §5, §6, §7.1, §8.2, §9, and §10. README/HANDOFF/STATUS SHALL provide executable Qwen commands, actual run results, and valid directories without describing historical reports or unexecuted steps as newly completed work. `openspec/config.yaml` SHALL reflect local hardware discovery and execution ownership while preserving other collection constraints.

#### Scenario: Implementation adds recovery metadata
- **WHEN** recovery_events or template-omission fields enter actual outputs
- **THEN** the same change updates the authoritative specification, layout documentation, and validation semantics so consumers can find each field's meaning and provenance
