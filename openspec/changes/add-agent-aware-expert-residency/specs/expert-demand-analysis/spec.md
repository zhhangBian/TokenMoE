## ADDED Requirements

### Requirement: Validated phase-specific demand inputs

The analysis pipeline SHALL consume validated finalized traces and preserve their source hashes. It SHALL derive demand from actual `token_positions`, logical expert IDs, layer IDs, request metadata, and engine-step attribution. It SHALL distinguish new prefill, decode, and recomputation; exclude cached gaps and the final uncomputed sampled token from first-computed demand; preserve valid partial requests; and assign no routing to pre-engine rejections. Empty targets SHALL produce null metrics with reasons.

#### Scenario: Prefix hits and an evidenced rejection
- **WHEN** a run contains cached prompt positions, one partially computed request, and a request rejected before engine admission
- **THEN** demand includes only the observed computations in the appropriate phases, preserves partial coverage, and includes no fabricated routing for cached positions or the rejection

### Requirement: Task-grouped evaluation and tuning

The evaluator SHALL persist task-grouped splits that keep every N1/N4, pilot, recovery, and duplicate copy of a benchmark task in the same group. Predictors, label-rule choices, backoff thresholds, smoothing, and mixture weights SHALL be fitted without held-out task data. Tuning SHALL use an inner training-only split. Evaluation SHALL identify frozen-prior and causal online-adaptation protocols separately, including whether state can be shared between test sessions. Tasks used to choose the design SHALL be labeled exploratory rather than untouched confirmation data.

#### Scenario: A task has several executions
- **WHEN** the same benchmark item occurs in primary N1, primary N4, and a recovered pilot execution
- **THEN** all copies share one outer split and none contributes to the prior or tuning used to evaluate that held-out task

#### Scenario: Online adaptation receives a later observation
- **WHEN** an evaluated session finishes a request after an earlier residency decision
- **THEN** that request updates only subsequent predictions under the declared adaptation protocol and cannot retroactively improve the earlier decision

### Requirement: Causal feature availability

Every prediction SHALL have a decision time, visible event sequence, feature version, and model/template identity. Features SHALL be available no earlier than both their observation and controller receipt in a comparable clock domain. Historical receipt-time proxies SHALL be documented. Tool result classes, future output bytes, future graph branches, and the target request's routes SHALL NOT be used before they become available. First-request, no-preceding-tool, multiple-tool, and unknown-category states SHALL be represented independently.

#### Scenario: A tool eventually fails
- **WHEN** a prediction is made at tool issuance and the tool reports failure later
- **THEN** the final exit status is absent from the deployable prediction features and is usable only as a later observation or explicitly labeled hindsight supervision

#### Scenario: A later request has no tool result
- **WHEN** a noninitial request follows an agent formatting retry without an intervening tool call
- **THEN** it is represented as a subsequent request with no preceding tool, rather than as a first request

### Requirement: Strong demand baselines and reproducible predictions

The evaluator SHALL implement global, first/subsequent-stage, previous-request, request-EWMA, previous+global, previous+stage, and previous+tool predictors, with explicit cold-start and sparse-key backoff. It SHALL retain a recent-token-window diagnostic. Predictors SHALL output per-layer, per-phase nonnegative normalized expert distributions, token-volume estimates where used, sample support, and fallback provenance. Models and weight/template revisions SHALL NOT share routing tables implicitly. Tie-breaking and all parameters SHALL be recorded.

#### Scenario: Unseen tool category
- **WHEN** a category has insufficient training support
- **THEN** the predictor selects the configured coarser prior, records that fallback and its support, and emits a valid distribution without reading held-out target routes

### Requirement: Separate coverage metrics and labeled oracles

Analysis SHALL report fixed-set lookup coverage at expert budgets 16/32/64/96/128 when applicable, separately for new prefill, full decode, and the first 32 computed decode positions. It SHALL report lookup-weighted and task-macro results, task-level uncertainty, counts, layer variation, initial/subsequent requests, and tool-switch subgroups. Tool-conditioned mixtures SHALL be compared against both global-smoothed and stage-smoothed history. Lookup coverage SHALL NOT be labeled cache hit rate, avoided stall, TTFT, or speedup.

The per-request fixed-set oracle SHALL select from that request phase's actual histogram and be labeled hindsight. A dynamic replacement oracle, if implemented, SHALL declare its own capacity, transfer, and future-information assumptions and SHALL NOT be presented as the same bound.

#### Scenario: A predictor covers more lookups
- **WHEN** a predictor improves fixed-set decode coverage without any measured or simulated transfer execution
- **THEN** the report attributes only a coverage gain and leaves latency and speedup unmeasured

### Requirement: Reproducible analysis delivery

An analysis run SHALL save its input manifest, source/code versions, command, split groups, labels, predictor parameters, metric definitions, and machine-readable and readable results in a new output directory. It SHALL reproduce the exploratory comparison described in the design before interpreting extensions, recording differences rather than forcing rounded values to match. Source captures SHALL remain unchanged. Negative or heterogeneous feature increments SHALL be retained.

#### Scenario: Packaged results differ from exploratory results
- **WHEN** corrected label semantics or tie-breaking changes a reported percentage
- **THEN** the run preserves both definitions, records the reason and affected comparison, and does not silently overwrite the historical observation
