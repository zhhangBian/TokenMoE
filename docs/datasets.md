# Datasets and artifacts

TokenMoE keeps source datasets and generated artifacts outside the Git
repository. Set these roots for the local machine:

```bash
export TOKENMOE_DATASET_ROOT=/path/to/datasets
export TOKENMOE_ARTIFACT_ROOT=/path/to/tokenmoe_artifacts
```

The local workspace currently stores the public routing-trace corpus at:

```text
/home/youwei/bzh/dataset/MoE_expert_selection_trace
```

That corpus is useful for model- and domain-level routing characterization. It
does not contain TokenMoE agent metadata and must not be treated as evidence for
the agent-conditioned hypothesis.

## Supported sources

| Adapter | Source | Claim scope |
| --- | --- | --- |
| `sharegpt` | `anon8231489123/ShareGPT_Vicuna_unfiltered` | chat prompt |
| `lmsys` | `lmsys/lmsys-chat-1m` | chat prompt |
| `swe_agent` | `nebius/SWE-agent-trajectories` | agent trajectory |
| `opencode` | `nvidia/OpenCodeInstruct` | domain instruction |
| `openmath` | `nvidia/OpenMathInstruct-2` | domain instruction |

Converters emit prompts representing the state before target-model generation.
For chat and instruction corpora, target answers are removed. For SWE-agent,
one workload record is emitted before each action and contains only prior
actions and observations.

## Artifact layout

```text
$TOKENMOE_ARTIFACT_ROOT/
  workloads/     normalized admission-request JSONL
  manifests/     dataset provenance and conversion semantics
  traces/        strict vLLM routed-expert JSONL
  analysis/      generated evaluation JSON
  logs/          collection environment reports
```

Artifacts may contain raw prompts and must remain local unless the source
dataset license and redaction policy permit redistribution.

## Workload contract

The only accepted workload version is `tokenmoe.workload.v2`. Every record
contains:

- one request ID and matching `AgentNodeMeta`;
- the exact prompt passed to vLLM;
- typed prompt blocks with character spans;
- source dataset, source item, source group, and claim scope;
- dependencies only when they correspond to earlier LLM requests.

Missing values are represented as JSON `null`, not string sentinels or inferred
defaults.
