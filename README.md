# TokenMoE

TokenMoE is a research project on predicting Mixture-of-Experts (MoE) expert working sets before routing begins. It studies whether information already available to an agent runtime, such as node role, execution phase, tool state, and prompt structure, can complement short-term routing history.

The current repository provides the measurement and evaluation foundation for that question. It does not yet claim end-to-end serving acceleration.

## Research question

An MoE router reveals expert demand only after a token reaches each sparse
layer. This is late for systems that page, prefetch, place, or replicate expert
weights. Agent runtimes know the request's execution context earlier.

TokenMoE asks whether that context provides a useful prior over the routed
experts while preserving one non-negotiable rule: the model router remains
authoritative and model outputs must not change.

The project currently evaluates three explainable predictors:

- global expert frequency;
- recent routing history;
- RouteSig, a hierarchical metadata-conditioned routing distribution.

Comparisons use predict-before-update evaluation, source-group-preserving time
splits, model-relative expert budgets, and paired bootstrap confidence
intervals.

## Repository structure

```text
tokenmoe/           schemas, vLLM traces, RouteSig, predictors, evaluation
dataset_adapters/   converters from public corpora to admission requests
scripts/            trace collection and offline evaluation entry points
tests/              fast correctness and leakage-boundary tests
docs/               data, trace, model, and report documentation
TokenMoE-paper/     compact paper draft without unverified results
vllm/               TokenMoE vLLM fork as a Git submodule
idea.md             research idea and system-design direction
openspec/           current behavioral specifications
```

Datasets, traces, model weights, logs, and generated analyses are external
artifacts and are not versioned in this repository.

## Installation

Python 3.10 or newer is required.

```bash
git submodule update --init vllm
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,datasets]'
```

Trace collection additionally requires building the local `vllm/` fork for
the machine's CUDA and PyTorch versions.

## Data preparation

Use environment variables rather than repository-relative data paths:

```bash
export TOKENMOE_DATASET_ROOT=/path/to/datasets
export TOKENMOE_ARTIFACT_ROOT=/path/to/tokenmoe_artifacts
PYTHONPATH=. python -m dataset_adapters.convert_all --limit 1000 \
  --dataset-root "$TOKENMOE_DATASET_ROOT" \
  --output-dir "$TOKENMOE_ARTIFACT_ROOT/workloads"
```

Each converter writes a workload JSONL and a provenance manifest. A workload
record represents a prompt immediately before target-model generation. Target
answers and the current agent action are excluded from the prompt.

See [docs/datasets.md](docs/datasets.md) for supported sources and the external
artifact layout.

## Collecting vLLM traces

The collector accepts only routed-expert output from the TokenMoE vLLM fork and
writes the current `tokenmoe.trace.v3` format.

```bash
export TOKENMOE_ROOT="$(pwd)"
export TOKENMOE_MODEL=/path/to/a/supported-moe-model

PYTHONPATH="$TOKENMOE_ROOT/vllm:$TOKENMOE_ROOT" \
python scripts/collect_traces.py \
  --workload "$TOKENMOE_ARTIFACT_ROOT/workloads/swe_agent_prompt_workloads.jsonl" \
  --model "$TOKENMOE_MODEL" \
  --output "$TOKENMOE_ARTIFACT_ROOT/traces/model_swe_agent.jsonl" \
  --env-report "$TOKENMOE_ARTIFACT_ROOT/logs/model_swe_agent.json" \
  --limit 1000 \
  --batch-size 16 \
  --tensor-parallel-size 2 \
  --expert-parallel on \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
```

Router scores are enabled by default. Use `--no-router-scores` only when an
ID-only trace is intentional; the trace will record the explicit reason for
missing scores. `--resume` validates an existing JSONL file and appends only
missing request IDs.

## Evaluation

```bash
PYTHONPATH=. python scripts/analyze_traces.py \
  --traces "$TOKENMOE_ARTIFACT_ROOT/traces/model_swe_agent.jsonl" \
  --output "$TOKENMOE_ARTIFACT_ROOT/analysis/model_swe_agent.json"
```

One analysis file must contain a single model and one strict trace schema. The
output contains both group-preserving time splits, all expert budgets, weighted
coverage when router scores exist, and confidence intervals against global and
temporal baselines.

## Validation and claim boundary

```bash
pytest -q
openspec validate --all --strict
```

Passing unit tests establishes schema, alignment, online-update, and data
leakage boundaries. It does not establish the research hypothesis. A TokenMoE
systems claim additionally requires real MoE traces from multiple models,
statistically supported gains over temporal history, capture-on/off output
equivalence, and measured runtime integration.

The current implementation deliberately contains no prefetch, scheduler, or
expert-placement simulator. Those components should be implemented only after
the predictive-signal gate is supported by real traces.
