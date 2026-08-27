"""Shared, provenance-preserving dataset conversion utilities."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from tokenmoe.schema import (
    CLAIM_SCOPE_CHAT,
    CLAIM_SCOPE_DOMAIN,
    CLAIM_SCOPE_REAL_AGENT,
    AgentNodeMeta,
    PromptSegment,
    WorkloadRecord,
    write_workload_jsonl,
)


DATASET_ROOT = Path(os.environ.get("TOKENMOE_DATASET_ROOT", "/home/youwei/bzh/dataset"))
ARTIFACT_ROOT = Path(
    os.environ.get(
        "TOKENMOE_ARTIFACT_ROOT", "/home/youwei/bzh/dataset/tokenmoe_artifacts"
    )
)
WORKLOAD_OUTPUT_DIR = ARTIFACT_ROOT / "workloads"
MANIFEST_DIR = ARTIFACT_ROOT / "manifests"
FIELD_MAPPING_VERSION = "tokenmoe.admission-request.v2"


@dataclass(frozen=True, slots=True)
class PromptBlock:
    block_type: str
    text: str


def dataset_dir(repo_id: str, dataset_root: Path = DATASET_ROOT) -> Path:
    return dataset_root / repo_id


def candidate_data_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    suffixes = (".json", ".jsonl", ".jsonl.gz", ".parquet")
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.name.endswith(suffixes)
        ),
        key=lambda item: (
            "train" not in item.name.lower(),
            len(str(item)),
            str(item),
        ),
    )


def _iter_json(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
        return
    with opener(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    if isinstance(value, list):
        yield from (item for item in value if isinstance(item, dict))
    elif isinstance(value, dict):
        for key in ("data", "train", "records", "examples"):
            if isinstance(value.get(key), list):
                yield from (item for item in value[key] if isinstance(item, dict))
                return
        yield value


def _iter_parquet(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("install TokenMoE with the 'datasets' extra") from exc
    yield from (
        row for row in parquet.read_table(path).to_pylist() if isinstance(row, dict)
    )


def iter_raw_records(
    source_path: Path, limit: int | None = None
) -> Iterator[tuple[int, Path, dict[str, Any]]]:
    index = 0
    for path in candidate_data_files(source_path):
        records = _iter_parquet(path) if path.suffix == ".parquet" else _iter_json(path)
        for record in records:
            yield index, path, record
            index += 1
            if limit is not None and index >= limit:
                return


def text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = (
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, dict))
        else str(value)
    ).strip()
    return text or None


def first_text(
    data: dict[str, Any], keys: Iterable[str]
) -> tuple[str | None, str | None]:
    for key in keys:
        if text := text_or_none(data.get(key)):
            return text, key
    return None, None


_BLOCK_PREFIXES = {
    "user_message": "User:\n",
    "assistant_message": "Assistant:\n",
    "tool_result": "Tool:\n",
    "trajectory_event": "Assistant:\n",
}


def build_prompt(
    blocks: Iterable[PromptBlock],
) -> tuple[str, tuple[PromptSegment, ...]]:
    prompt = ""
    segments: list[PromptSegment] = []
    for position, block in enumerate(block for block in blocks if block.text.strip()):
        if prompt:
            prompt += "\n\n"
        rendered = _BLOCK_PREFIXES.get(block.block_type, "") + block.text.strip()
        start = len(prompt)
        prompt += rendered
        segments.append(
            PromptSegment(
                segment_id=f"{block.block_type}-{position:03d}",
                block_type=block.block_type,
                position=position,
                char_start=start,
                char_end=len(prompt),
            )
        )
    if not segments:
        raise ValueError("cannot build an empty prompt")
    return prompt, tuple(segments)


def make_record(
    *,
    request_id: str,
    workflow: str,
    blocks: Iterable[PromptBlock],
    source_dataset: str,
    source_index: int | str,
    source_group_id: str,
    claim_scope: str,
    role: str,
    phase: str,
    graph_node_type: str = "request",
    tool_type: str | None = None,
    timestamp: str | float | int | None = None,
    dependencies: Iterable[str] = (),
    trajectory_phase: str | None = None,
    event_outcome: str | None = None,
    dag_depth: int | None = None,
    group_local_step_index: int | None = None,
    on_critical_path: bool | None = None,
) -> WorkloadRecord:
    prompt, segments = build_prompt(blocks)
    meta = AgentNodeMeta(
        request_id=request_id,
        agent_id=f"{workflow}:{role}:{source_group_id}",
        role=role,
        phase=phase,
        graph_node_type=graph_node_type,
        prompt_block_types=tuple(segment.block_type for segment in segments),
        tool_type=tool_type,
        trajectory_phase=trajectory_phase,
        event_outcome=event_outcome,
        dag_depth=dag_depth,
        group_local_step_index=group_local_step_index,
        on_critical_path=on_critical_path,
    )
    return WorkloadRecord(
        request_id=request_id,
        workflow=workflow,
        prompt=prompt,
        meta=meta,
        prompt_segments=segments,
        source_dataset=source_dataset,
        source_index=source_index,
        source_group_id=source_group_id,
        claim_scope=claim_scope,
        timestamp=timestamp,
        dependencies=tuple(dependencies),
    )


def write_manifest(
    *,
    adapter_name: str,
    source_dataset: str,
    source_path: Path,
    output_path: Path,
    sample_count: int,
    unavailable_fields: Iterable[str],
    claim_scope: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    stem = output_path.stem.replace("_prompt_workloads", "")
    path = MANIFEST_DIR / f"{stem}_manifest.json"
    payload = {
        "adapter": adapter_name,
        "source_dataset": source_dataset,
        "source_dataset_path": str(source_path),
        "output_workload_path": str(output_path),
        "sample_count": sample_count,
        "field_mapping_version": FIELD_MAPPING_VERSION,
        "admission_semantics": "prompt_before_target_model_generation",
        "conversion_command": [sys.executable, *sys.argv],
        "unavailable_source_fields": sorted(set(unavailable_fields)),
        "claim_scope": claim_scope,
        "license_access_status": "source_dataset_terms_required",
        "redaction_status": "raw_prompt_preserved_local_artifact_only",
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_converter(
    *,
    parser: argparse.ArgumentParser,
    adapter_name: str,
    source_dataset: str,
    output_name: str,
    claim_scope: str,
    convert_fn: Any,
    manifest_extra: dict[str, Any] | None = None,
) -> None:
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--source")
    parser.add_argument("--output", default=str(WORKLOAD_OUTPUT_DIR / output_name))
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--repo-id", default=source_dataset)
    args = parser.parse_args()
    source = (
        Path(args.source)
        if args.source
        else dataset_dir(args.repo_id, Path(args.dataset_root))
    )
    records, unavailable = convert_fn(source, args.limit, args.repo_id)
    output = Path(args.output)
    write_workload_jsonl(records, output)
    manifest = write_manifest(
        adapter_name=adapter_name,
        source_dataset=args.repo_id,
        source_path=source,
        output_path=output,
        sample_count=len(records),
        unavailable_fields=unavailable,
        claim_scope=claim_scope,
        extra=manifest_extra,
    )
    print(f"wrote {len(records)} workload records to {output}")
    print(f"wrote manifest to {manifest}")


__all__ = [
    "CLAIM_SCOPE_CHAT",
    "CLAIM_SCOPE_DOMAIN",
    "CLAIM_SCOPE_REAL_AGENT",
    "PromptBlock",
    "first_text",
    "iter_raw_records",
    "make_record",
    "run_converter",
]
