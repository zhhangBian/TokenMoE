## 1. Protocol and local entry points

- [x] 1.1 Update `collection/data_collect.md` with the local profile, separate acceptance criteria, recovery provenance, and new operational files. Correct outdated hardware and execution-ownership guidance in `openspec/config.yaml`. The executing agent owns the implementation and run tasks below; specification delivery does not start them.
- [x] 1.2 Add separate local Qwen model/run configurations fixing the 3-task pilot, 20-task N1/N4 lists, sampling, and budgets. Preserve the original 131072/YaRN debug configuration and remote-model configurations.
- [x] 1.3 Extend `prepare_configs.py` with explicit Qwen/profile selection and local templates. Make dry-run output include tasks, concurrency, parameters, and absolute paths without generating a 500-task local run.
- [x] 1.4 Extend existing download entry points with Qwen support and asset verification, restoring only missing or damaged selected model/benchmark assets while retaining existing authentication sources and excluding credentials from output.

## 2. Preflight, provenance, and recovery

- [x] 2.1 Implement local preflight and the source manifest: verify shard indexes/headers, tokenizer/dataset hashes, task IDs, dependencies/imports/virtual environments, main-repository and fork commits/patches, actual GPU mappings, and available disk space.
- [x] 2.2 Reuse the container probe to check the available entry point, shared host boot ID, real task-image startup, and 2 CPU/4 GiB kernel-enforced limits, producing actionable missing-resource and failure logs.
- [x] 2.3 Add collection plans/manifests tracking valid data, required recollection, and blocked units by task×concurrency×seed×protocol. Require data verification independently of directory existence or submission state.
- [x] 2.4 Add recovery events linking old/new IDs, directories, fault evidence, repair versions, and verification results. Allocate new sessions for server restarts/task reruns, preserving raw failure data and skipping already valid units.
- [x] 2.5 Integrate the local workflow with existing stage entry points for smoke, equiv, pilot, N1/N4, finalize, and reporting. Restrict abnormal-exit cleanup to this run's servers, workers, and containers.

## 3. Data correctness and reporting

- [x] 3.1 Use fixed multi-turn reasoning/tool inputs to inspect actual HTTP messages, template adaptation, engine token IDs, and prompt alignment. Repair reproducible transport/alignment defects; classify omissions as template_omitted only with evidence from the template and actual tokens.
- [x] 3.2 Preserve the original location_rate and add template-omission evidence, unexpected-failure counts, and rendered_location_rate. Exclude tool-payload loss from omission exemptions and apply the local ≥99% rendered-field threshold.
- [x] 3.3 Default the local submission-rate threshold to 0 and report task outcomes, structural validation, equivalence, and coverage separately. Accept valid failure/limit traces and evidenced pre-engine context rejections with null engine artifacts; preserve actual HTTP error bodies and disclose exact-input replay recovery. Schedule recollection for missing real computation or corrupted data.
- [x] 3.4 Implement runtime reporting for phase-specific token/routing coverage, per-layer new_prefill/decode lookups, caching/recompute, tool streams/consumption links, duration and queueing distributions, actual concurrency, and host load. Report missing values as null with reasons.
- [x] 3.5 Add versioned tool-category, outcome, and transition labels with available_at and rule provenance. Produce a task-grouped G1 manifest and declare G2/G3 outside the collected scope.
- [x] 3.6 Add bounded capture-off/on overhead measurements, storage accounting deduplicated by inode, and artifact hash reports. Record and optimize local overhead above 5% without using it alone to reject local data.

## 4. Targeted verification

- [x] 4.1 Use temporary local assets to cover complete/missing shards, missing datasets, unknown tasks, Qwen-only selection, and dry runs without downloads or collection writes.
- [x] 4.2 Verify classification of valid failure sessions, startup failures without requests, and submissions with missing routing. Simulate a mid-run server failure and confirm that recovery recollects only missing units with complete provenance.
- [x] 4.3 Verify reasoning preservation, proven template omissions versus unexpected losses, raw/rendered alignment rates, and the rule disallowing tool-payload exemptions.
- [x] 4.4 Check statistics, missing values, integrity failures, and rejection of invalid equivalence records with prefix-hit, recompute, empty-output, and partial-routing samples. Confirm task-level grouping across N1/N4/recovery.
- [x] 4.5 Run affected harness checks. If the fork is repaired, run corresponding CPU recorder checks and verify the Python-only constraint and version provenance.

## 5. Real local execution and repairs

- [x] 5.1 Recheck devices, virtual environments, and containers; create a new experiment directory and fixed plan. Download/pull required assets only when confirmed missing or damaged, preserving actual preflight results and commands.
- [x] 5.2 Run real Qwen smoke requests and local tool tasks, checking 48 layers, 8-of-128 uint8 routing, decode, chunks, prompts, and finalize. Repair collection faults while retaining original failure records.
- [x] 5.3 Run the 50-prompt off/off/on/on equivalence protocol for the actual collection configuration. Diagnose and repair failures before rerunning without fabricating or weakening the gate.
- [x] 5.4 Run the 3-task Verified pilot and check data structure, reasoning, tool output, and alignment. Continue despite task-test failures or low submission rates.
- [x] 5.5 Fix a working configuration and execute the 20-task N1 run, preserving actual terminal states and raw/finalized data. Repair specific faults and recollect invalid units.
- [x] 5.6 Execute the 20-task N4 run with identical tasks and protocol, verifying actual overlap/batch composition and recovering failed units. Record concurrency/budget changes under a separate protocol without claiming the original plan is complete.
- [x] 5.7 Measure recorder overhead and perform bounded parameter/storage optimization based on observed bottlenecks. Rerun matching equivalence and necessary pilot checks after configuration/capture changes, preserving valid data from prior versions.

## 6. Data delivery

- [x] 6.1 Run finalize, validation, and local acceptance on delivery directories. Verify all 40 planned units, routing/chunk/tool links, and raw failure traces, and produce valid-data and recollection manifests.
- [x] 6.2 Generate runtime_summary.json/md, G1 groups, provenance, storage reports, and artifact checksums. Disclose tool/outcome coverage gaps, repairs, overhead, and outstanding work.
- [x] 6.3 Update `collection/README.md`, `HANDOFF.md`, and `COLLECTION_STATUS.md` with local commands and measured status, and run strict OpenSpec validation. Provide directly usable absolute data paths without marking unexecuted work complete.
