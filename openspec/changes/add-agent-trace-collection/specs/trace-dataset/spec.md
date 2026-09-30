## ADDED Requirements

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
- `llm_request_id_consuming` from `prev_tool_call_ids`;
- the engine-side lifecycle fields of each request.

It SHALL keep the raw files, and it SHALL run the validator at the end.

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
- **WHEN** a harness request has no matching engine request record
- **THEN** finalize keeps the harness record, marks the request `engine_record_missing`, and validation reports it

### Requirement: Prompt segments
Finalize SHALL build the prompt segments of each request as follows:
1. Decode the engine's prompt token IDs.
2. Locate the harness messages in order inside the decoded text.
3. Split tool-output payloads out of their wrapping templates, using the harness provenance offsets.
4. Give every other character range the type `other`.

Each segment SHALL carry character and token ranges. A message that cannot be located SHALL produce a segment with null ranges and a non-null `alignment_error`. Located segments SHALL tile the prompt without overlap.

#### Scenario: Tool output segment
- **WHEN** a request contains the observation of tool call `tc_x`
- **THEN** exactly one segment has `segment_type = tool_output`, `source_tool_call_id = tc_x`, and a token range inside the request's new-prefill range, or inside the cached range if the prefix was cached

#### Scenario: Alignment failure
- **WHEN** the chat template rewrote a message so that none of its text variants appear in the decoded prompt
- **THEN** that message's segment has null ranges and a non-null `alignment_error`, and its text is covered by an `other` segment

### Requirement: Validation
`python -m tokenmoe_collect validate <run_dir>` SHALL check the rules of `collection/data_collect.md` §7 that apply to single-loop, single-host runs: rules 1–9, 11 and 12, with rules 6–9 in their revised form. It SHALL report rules 10, 13 and 14 as `not_applicable`. It SHALL write `validation.json` with one entry per rule: status, violation count, and up to 20 example IDs. The exit code SHALL be non-zero if any applicable rule fails.

#### Scenario: Routing shape violation
- **WHEN** a routing file has fewer rows than `T - 1 - num_cached_tokens`
- **THEN** rule 6 fails and names that `llm_request_id`

#### Scenario: Recompute allowed
- **WHEN** a preempted request has recompute entries in addition to its new_prefill entries
- **THEN** rule 7 passes, provided that every new_prefill and decode token appears exactly once

#### Scenario: Missing equivalence record
- **WHEN** no passing equivalence record exists for the run's `engine_config_id`
- **THEN** rule 12 fails

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
