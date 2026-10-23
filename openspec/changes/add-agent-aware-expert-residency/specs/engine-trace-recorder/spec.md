## MODIFIED Requirements

### Requirement: Recorder enablement

The vLLM fork SHALL create the full trace recorder only when the environment variable `TOKENMOE_TRACE_DIR` is set. Each engine lifetime SHALL write into its own subdirectory `<TOKENMOE_TRACE_DIR>/<engine_instance_id>/`. When `TOKENMOE_TRACE_DIR` is unset and expert caching is disabled, the fork SHALL behave exactly like upstream v0.30.0 with the same flags. Explicit expert-cache enablement SHALL be independent of full trace recording and SHALL preserve the OpenAI-compatible response contract; compact feedback or cache telemetry SHALL not implicitly enable full routing capture.

#### Scenario: Recorder disabled
- **WHEN** the engine starts without `TOKENMOE_TRACE_DIR` and with expert caching disabled
- **THEN** no recorder is created, no trace files are written, and API responses match upstream behavior

#### Scenario: Recorder enabled
- **WHEN** the engine starts with `TOKENMOE_TRACE_DIR=/x` and `--enable-return-routed-experts`
- **THEN** the directory `/x/<engine_instance_id>/` exists and contains `engine_meta.json` before the first request is accepted

#### Scenario: Cache enabled without full capture
- **WHEN** the engine enables a supported expert cache without setting `TOKENMOE_TRACE_DIR`
- **THEN** cache execution and selected compact telemetry work without full routing files or a routed-experts API payload

## ADDED Requirements

### Requirement: Versioned cache and transfer telemetry

Optimization telemetry SHALL record schema version, engine/rank identity, policy and plan epochs, ordered event sequence, layer/logical expert, slot generation, transfer ID, trigger, bytes, source/destination tiers, and relevant request/step references. It SHALL distinguish demand, hit, miss, load start/completion/failure, eviction, and plan cancellation. Shared prefetch SHALL allow null request attribution. A demand timestamp SHALL precede any miss-induced wait. Detailed event streams SHALL begin from complete initialization/load events or a snapshot of resident slots, generations, and in-flight transfers, permit residency reconstruction, and identify dropped records.

#### Scenario: One prefetch serves several requests
- **WHEN** an aggregate plan loads one expert shard used by multiple requests
- **THEN** telemetry records one rank-local transfer with aggregate-plan attribution and links later demands to it without inventing an exclusive owning request

### Requirement: Measured intervals and independent observation modes

Telemetry SHALL distinguish host observation times, actual device-copy intervals, and exposed readiness waits, documenting clock conversion or null unavailable values. Detailed tracing and compact counters SHALL have independent enablement and measured overhead. File I/O SHALL use bounded asynchronous export with explicit loss/error accounting and shutdown flush. Full-recorder startup restrictions and first-computation routing semantics SHALL remain unchanged.

#### Scenario: Asynchronous transfer is enqueued early
- **WHEN** a host enqueues a copy before the device begins transferring
- **THEN** telemetry distinguishes enqueue time from measured device duration and does not report queue delay as device bandwidth or as hidden compute

### Requirement: Token timing observations

Live optimization measurement SHALL offer token-readiness records containing request ID, output-token index, engine step, host observation time, clock domain, observation point, and processing batch identity. Network delivery timestamps SHALL be recorded only when separately observed. Tokens exposed in one processing batch SHALL retain that shared timing resolution. Endpoint-derived average TPOT SHALL be distinguished from measured inter-token distributions. This observation mode SHALL be independent of full routing capture and SHALL not change agent-visible messages or tool behavior.

#### Scenario: Several tokens are exposed together
- **WHEN** one output-processing event exposes several token indices
- **THEN** all retain the observed batch identity and time, and the report does not interpolate fictitious individual emission timestamps
