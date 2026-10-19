## MODIFIED Requirements

### Requirement: Prompt segments

Finalize SHALL build the prompt segments of each request as follows:

1. Decode the engine's prompt token IDs.
2. Locate the harness messages in order inside the decoded text.
3. Split tool-output payloads out of their wrapping templates, using the harness provenance offsets.
4. Give every other character range the type `other`.

Each segment SHALL carry character and token ranges. A message that cannot be located SHALL produce a segment with null ranges and a non-null `alignment_error`. Located segments SHALL tile the prompt without overlap.

For local Qwen collection, a field omitted by the actual chat template SHALL remain in the raw harness messages and SHALL have null prompt/token ranges, a non-null `alignment_error`, and an explicit `template_omitted` classification. This classification SHALL require reproducible evidence from the effective template/input and actual engine prompt token IDs. Missing fields without that evidence SHALL remain unexpected alignment failures. Tool payload loss SHALL NOT be classified as an allowed omission.

The existing `location_rate` definition SHALL remain unchanged. Local `alignment.json` SHALL additionally report omitted-field counts, unexpected failures and `rendered_location_rate = located / (message_fields - template_omitted)` with its numerator and denominator; zero denominator SHALL be reported as null with a reason.

#### Scenario: Tool output segment
- **WHEN** a request contains the observation of tool call `tc_x`
- **THEN** one or more payload segments have `segment_type = tool_output`, `source_tool_call_id = tc_x`, and a token range inside the request's new-prefill range, or inside the cached range if the prefix was cached

#### Scenario: Alignment failure
- **WHEN** the chat template rewrote a message so that none of its text variants appear in the decoded prompt
- **THEN** that message's segment has null ranges and a non-null `alignment_error`, and any unlocated actual prompt characters remain covered by an `other` segment

#### Scenario: Proven omission of historical reasoning
- **WHEN** a fixed-input reproduction and the actual engine token IDs show that the effective template omitted a historical reasoning field
- **THEN** the field is classified as template_omitted with retained raw text and evidence, has no fabricated token range, and both original and rendered-field location rates are reported

#### Scenario: Unexplained missing tool payload
- **WHEN** a tool observation cannot be found in the actual prompt and no supported rendering transformation explains it
- **THEN** it remains an unexpected alignment failure, and cannot be removed from the local location-rate denominator as a template omission

## ADDED Requirements

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
