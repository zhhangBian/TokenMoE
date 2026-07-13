# TokenMoE External Datasets

## Download

Use the existing local script:

```bash
python /home/youwei/bzh/dataset/download_dataset.py
```

Only `DATASET_LIST` is changed by this repository work. It targets:

- `anon8231489123/ShareGPT_Vicuna_unfiltered`
- `lmsys/lmsys-chat-1m`
- `nebius/SWE-agent-trajectories`
- `nvidia/OpenCodeInstruct`
- `nvidia/OpenMathInstruct-2`

If a dataset is gated, the failure remains visible in the download log and the
dataset is not replaced.

## Conversion

Converters live in `dataset_adapters/`, outside core `tokenmoe/` runtime code.
Run all converters:

```bash
PYTHONPATH=. python -m dataset_adapters.convert_all --limit 256
```

Per-dataset outputs:

- `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/sharegpt_prompt_workloads.jsonl`
- `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/lmsys_prompt_workloads.jsonl`
- `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/swe_agent_prompt_workloads.jsonl`
- `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/opencode_prompt_workloads.jsonl`
- `/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads/openmath_prompt_workloads.jsonl`

Manifests are written under
`/home/youwei/bzh/dataset/tokenmoe_artifacts/manifests/` and record source
paths, output paths, sample counts, mapping version, command, unavailable
fields, license/access status, redaction status, claim scope, and DAG
availability.

## Workload v2

Each JSONL row includes:

- `schema_version = "tokenmoe.workload.v2"`
- `prompt`
- `prompt_segments` with `segment_id`, `block_type`, `segment_position`,
  `char_start`, `char_end`, and alignment status fields
- `source_dataset`, `source_index`, `source_group_id`, and timestamp when
  available
- `claim_scope`: `real_agent_metadata`, `chat_prompt_only`, or
  `domain_instruction`
- `dependencies`, `dependency_edges`, `dag_available`, and `meta.ready_time`
  when reconstructable

Only SWE-agent trajectory workloads can support real agent-DAG scheduler claims.
ShareGPT/LMSYS support chat prompt locality, and OpenCode/OpenMath support
domain-instruction locality.
