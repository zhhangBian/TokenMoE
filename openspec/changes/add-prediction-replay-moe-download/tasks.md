## 1. Discuss Implementation Decisions

- [x] 1.1 Confirm the first MoE model profile for `/home/youwei/bzh/model/download.py`: `moe-target`
- [x] 1.2 Confirm endpoint and token policy: use `https://hf-mirror.com` and hardcoded local token without logging it
- [x] 1.3 Confirm the first real MoE run should use vLLM routed expert capture
- [x] 1.4 Confirm prediction policy: statistical RouteSig decisions plus temporal metrics reporting

## 2. Prediction Interface

- [ ] 2.1 Add a shared prediction result data structure for request ID, layer ID, expert IDs, confidence, source, fallback key, and optional scores
- [ ] 2.2 Add RouteSig prediction adapter that emits the shared prediction result shape
- [ ] 2.3 Add global frequency, request LRU, sequence history, and oracle adapters using the same prediction result shape
- [ ] 2.4 Add unit tests for prediction shape, fallback key reporting, and low-confidence fallback behavior

## 3. Prediction-Based DAG Replay

- [ ] 3.1 Refactor scheduler replay so legal ready-node construction remains dependency-safe
- [ ] 3.2 Implement prediction-based batch scoring from per-layer top-M expert sets, waiting penalty, and confidence
- [ ] 3.3 Preserve an oracle replay mode that uses true routed experts only for upper-bound comparison
- [ ] 3.4 Report baseline, RouteSig prediction, and oracle replay metrics separately
- [ ] 3.5 Add tests proving scheduler decisions do not violate dependencies and do not use true routed experts in prediction mode

## 4. Locality Metrics and Reporting

- [ ] 4.1 Compute temporal locality metrics from ordered traces, including reuse distance and phase-transition overlap where data is available
- [ ] 4.2 Compute spatial locality metrics including per-layer fanout, per-expert token count, and batch expert overlap
- [ ] 4.3 Add locality metrics to `analysis/locality_metrics.json` and markdown reports with unavailable metrics marked explicitly
- [ ] 4.4 Add tests for locality metrics on synthetic traces with known reuse and overlap patterns

## 5. MoE Model Download Preparation

- [ ] 5.1 Refactor `/home/youwei/bzh/model/download.py` into an argparse CLI with profile, model, dry-run, local-root, and endpoint options
- [ ] 5.2 Add `moe-target` as the primary model profile with Qwen3-30B-A3B, Mixtral-8x7B-Instruct, and DeepSeek-V2-Lite-Chat candidates
- [ ] 5.3 Preserve the local hardcoded token for this experiment while ensuring it is never printed in logs or dry-run output
- [ ] 5.4 Add dry-run output that prints planned repo IDs, local paths, endpoint, and profile without downloading
- [ ] 5.5 Add mocked or dry-run tests for download plan construction without network access

## 6. Documentation and Validation

- [ ] 6.1 Update README and dataset/model documentation with prediction replay and MoE download commands
- [ ] 6.2 Update reports to state clearly when results are replay, prediction, oracle, or real vLLM execution
- [ ] 6.3 Run `pytest -q tests`
- [ ] 6.4 Run the trace analysis command on existing traces and inspect generated metrics for baseline/prediction/oracle separation
- [ ] 6.5 Run `/home/youwei/bzh/model/download.py --dry-run` with the confirmed MoE profile
