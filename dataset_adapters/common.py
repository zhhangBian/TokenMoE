from __future__ import annotations

import argparse
import gzip
import hashlib
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
    WORKLOAD_SCHEMA_V2,
    AgentNodeMeta,
    PromptSegment,
    WorkloadRecord,
)


DATASET_ROOT = Path(os.environ.get("TOKENMOE_DATASET_ROOT", "/home/youwei/bzh/dataset"))
ARTIFACT_ROOT = Path(
    os.environ.get(
        "TOKENMOE_ARTIFACT_ROOT",
        "/home/youwei/bzh/dataset/tokenmoe_artifacts",
    )
)
WORKLOAD_OUTPUT_DIR = ARTIFACT_ROOT / "workloads"
MANIFEST_DIR = ARTIFACT_ROOT / "manifests"
FIELD_MAPPING_VERSION = "tokenmoe.external-adapter.v1"

JSON_SUFFIXES = {".json", ".jsonl", ".gz"}
PARQUET_SUFFIXES = {".parquet"}


@dataclass(frozen=True)
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
    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.name not in {".gitattributes"}
        and (
            item.suffix in JSON_SUFFIXES
            or item.suffix in PARQUET_SUFFIXES
            or item.name.endswith(".jsonl.gz")
        )
    ]
    files.sort(key=lambda item: (0 if "train" in item.name.lower() else 1, len(str(item)), str(item)))
    return files


def iter_json_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    if path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz"):
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for key in ("data", "train", "records", "examples"):
            value = data.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        yield data


def iter_parquet_records(path: Path) -> Iterator[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(f"pyarrow is required to read parquet dataset file {path}") from exc
    table = pq.read_table(path)
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def iter_raw_records(source_path: Path, limit: int | None = None) -> Iterator[tuple[int, Path, dict[str, Any]]]:
    count = 0
    for file_path in candidate_data_files(source_path):
        if file_path.suffix in PARQUET_SUFFIXES:
            iterator = iter_parquet_records(file_path)
        elif file_path.suffix in JSON_SUFFIXES or file_path.name.endswith(".jsonl.gz"):
            iterator = iter_json_records(file_path)
        else:
            continue
        for record in iterator:
            yield count, file_path, record
            count += 1
            if limit is not None and count >= limit:
                return


def text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = text.strip()
    return text or None


def first_text(data: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, str | None]:
    for key in keys:
        value = text_or_none(data.get(key))
        if value:
            return value, key
    return None, None


def build_prompt(blocks: list[PromptBlock]) -> tuple[str, list[PromptSegment]]:
    prompt = ""
    segments: list[PromptSegment] = []
    position = 0
    for block in blocks:
        text = block.text.strip()
        if not text:
            continue
        rendered = f"[{block.block_type}]\n{text}\n"
        start = len(prompt)
        prompt += rendered
        end = len(prompt)
        segments.append(
            PromptSegment(
                segment_id=f"{block.block_type}-{position:03d}",
                block_type=block.block_type,
                segment_position=position,
                char_start=start,
                char_end=end,
                alignment_status="char_span_only",
                text_sha1=hashlib.sha1(rendered.encode("utf-8")).hexdigest()[:16],
            )
        )
        position += 1
    return prompt, segments


def make_record(
    *,
    request_id: str,
    workflow: str,
    blocks: list[PromptBlock],
    source_dataset: str,
    source_index: int | str,
    source_group_id: str,
    claim_scope: str,
    role: str,
    phase: str,
    graph_node_type: str = "request",
    tool_type: str | None = None,
    ready_time: float = 0.0,
    timestamp: str | float | int | None = None,
    dependencies: list[str] | None = None,
    dependency_edges: list[tuple[str, str]] | None = None,
    unavailable_fields: list[str] | None = None,
    dag_available: bool = False,
) -> WorkloadRecord:
    prompt, segments = build_prompt(blocks)
    block_types = [segment.block_type for segment in segments] or ["prompt"]
    meta = AgentNodeMeta(
        request_id=request_id,
        agent_id=f"{workflow}:{role}:{source_group_id}",
        role=role,
        phase=phase,
        tool_type=tool_type,
        graph_node_type=graph_node_type,
        prompt_block_types=block_types,
        ready_time=ready_time,
    )
    return WorkloadRecord(
        request_id=request_id,
        workflow=workflow,
        prompt=prompt,
        meta=meta,
        dependencies=list(dependencies or []),
        source=source_dataset,
        expected_output_tokens=1,
        schema_version=WORKLOAD_SCHEMA_V2,
        prompt_segments=segments,
        source_dataset=source_dataset,
        source_index=source_index,
        source_group_id=source_group_id,
        timestamp=timestamp,
        claim_scope=claim_scope,
        unavailable_fields=list(unavailable_fields or []),
        dag_available=dag_available,
        dependency_edges=list(dependency_edges or []),
    )


def write_jsonl(records: Iterable[WorkloadRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_manifest(
    *,
    adapter_name: str,
    source_dataset: str,
    source_path: Path,
    output_path: Path,
    sample_count: int,
    conversion_command: list[str],
    unavailable_fields: list[str],
    claim_scope: str,
    dag_available: bool,
    license_access_status: str = "source_dataset_terms_required",
    redaction_status: str = "raw_prompt_preserved_local_artifact_only",
) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / f"{adapter_name}_manifest.json"
    payload = {
        "adapter": adapter_name,
        "source_dataset": source_dataset,
        "source_dataset_path": str(source_path),
        "output_workload_path": str(output_path),
        "sample_count": sample_count,
        "field_mapping_version": FIELD_MAPPING_VERSION,
        "conversion_command": conversion_command,
        "unavailable_source_fields": sorted(set(unavailable_fields)),
        "claim_scope": claim_scope,
        "dag_available": dag_available,
        "license_access_status": license_access_status,
        "redaction_status": redaction_status,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def add_common_args(parser: argparse.ArgumentParser, repo_id: str, output_name: str) -> None:
    parser.add_argument("--dataset-root", default=str(DATASET_ROOT))
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", default=str(WORKLOAD_OUTPUT_DIR / output_name))
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--repo-id", default=repo_id)


def run_converter(
    *,
    parser: argparse.ArgumentParser,
    adapter_name: str,
    source_dataset: str,
    output_name: str,
    claim_scope: str,
    convert_fn: Any,
) -> None:
    add_common_args(parser, source_dataset, output_name)
    args = parser.parse_args()
    source_path = Path(args.source) if args.source else dataset_dir(args.repo_id, Path(args.dataset_root))
    output_path = Path(args.output)
    records, unavailable_fields, dag_available = convert_fn(source_path, args.limit, args.repo_id)
    sample_count = write_jsonl(records, output_path)
    manifest = write_manifest(
        adapter_name=adapter_name,
        source_dataset=args.repo_id,
        source_path=source_path,
        output_path=output_path,
        sample_count=sample_count,
        conversion_command=[sys.executable, *sys.argv],
        unavailable_fields=unavailable_fields,
        claim_scope=claim_scope,
        dag_available=dag_available,
    )
    print(f"wrote {sample_count} workload records to {output_path}")
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
