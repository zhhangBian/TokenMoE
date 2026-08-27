"""Canonical request schema for TokenMoE trace collection."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


WORKLOAD_SCHEMA = "tokenmoe.workload.v2"

CLAIM_SCOPE_REAL_AGENT = "real_agent_metadata"
CLAIM_SCOPE_CHAT = "chat_prompt_only"
CLAIM_SCOPE_DOMAIN = "domain_instruction"
CLAIM_SCOPES = frozenset({CLAIM_SCOPE_REAL_AGENT, CLAIM_SCOPE_CHAT, CLAIM_SCOPE_DOMAIN})


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_text(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class AgentNodeMeta:
    """Metadata known before an LLM request reaches the MoE router."""

    request_id: str
    agent_id: str
    role: str
    phase: str
    graph_node_type: str
    prompt_block_types: tuple[str, ...]
    tool_type: str | None = None
    trajectory_phase: str | None = None
    event_outcome: str | None = None
    dag_depth: int | None = None
    group_local_step_index: int | None = None
    on_critical_path: bool | None = None

    def __post_init__(self) -> None:
        for name in ("request_id", "agent_id", "role", "phase", "graph_node_type"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if not self.prompt_block_types or any(
            not item for item in self.prompt_block_types
        ):
            raise ValueError("prompt_block_types must contain non-empty strings")
        for name in ("dag_depth", "group_local_step_index"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AgentNodeMeta":
        block_types = data.get("prompt_block_types")
        if not isinstance(block_types, list) or not all(
            isinstance(item, str) and item for item in block_types
        ):
            raise ValueError("prompt_block_types must be a non-empty list[str]")
        critical_path = data.get("on_critical_path")
        if critical_path is not None and not isinstance(critical_path, bool):
            raise TypeError("on_critical_path must be bool or null")
        return cls(
            request_id=_required_text(data, "request_id"),
            agent_id=_required_text(data, "agent_id"),
            role=_required_text(data, "role"),
            phase=_required_text(data, "phase"),
            graph_node_type=_required_text(data, "graph_node_type"),
            prompt_block_types=tuple(block_types),
            tool_type=_optional_text(data.get("tool_type"), "tool_type"),
            trajectory_phase=_optional_text(
                data.get("trajectory_phase"), "trajectory_phase"
            ),
            event_outcome=_optional_text(data.get("event_outcome"), "event_outcome"),
            dag_depth=None if data.get("dag_depth") is None else int(data["dag_depth"]),
            group_local_step_index=(
                None
                if data.get("group_local_step_index") is None
                else int(data["group_local_step_index"])
            ),
            on_critical_path=critical_path,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "role": self.role,
            "phase": self.phase,
            "graph_node_type": self.graph_node_type,
            "prompt_block_types": list(self.prompt_block_types),
            "tool_type": self.tool_type,
            "trajectory_phase": self.trajectory_phase,
            "event_outcome": self.event_outcome,
            "dag_depth": self.dag_depth,
            "group_local_step_index": self.group_local_step_index,
            "on_critical_path": self.on_critical_path,
        }

    @property
    def block_type_key(self) -> str:
        return "+".join(self.prompt_block_types)


@dataclass(frozen=True, slots=True)
class PromptSegment:
    """A typed prompt block and its character/token span."""

    segment_id: str
    block_type: str
    position: int
    char_start: int
    char_end: int
    token_start: int | None = None
    token_end: int | None = None
    alignment_error: str | None = None

    def validate(self, prompt: str, token_count: int | None = None) -> None:
        if not self.segment_id or not self.block_type:
            raise ValueError("segment_id and block_type must be non-empty")
        if self.position < 0:
            raise ValueError("segment position must be non-negative")
        if not 0 <= self.char_start < self.char_end <= len(prompt):
            raise ValueError(f"invalid character span for {self.segment_id}")
        if (self.token_start is None) != (self.token_end is None):
            raise ValueError(f"partial token span for {self.segment_id}")
        if self.token_start is not None:
            assert self.token_end is not None
            upper = token_count if token_count is not None else self.token_end
            if not 0 <= self.token_start < self.token_end <= upper:
                raise ValueError(f"invalid token span for {self.segment_id}")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PromptSegment":
        return cls(
            segment_id=_required_text(data, "segment_id"),
            block_type=_required_text(data, "block_type"),
            position=int(data["position"]),
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
            token_start=(
                None if data.get("token_start") is None else int(data["token_start"])
            ),
            token_end=None if data.get("token_end") is None else int(data["token_end"]),
            alignment_error=_optional_text(
                data.get("alignment_error"), "alignment_error"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "block_type": self.block_type,
            "position": self.position,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_start": self.token_start,
            "token_end": self.token_end,
            "alignment_error": self.alignment_error,
        }

    @property
    def token_count(self) -> int:
        if self.token_start is None or self.token_end is None:
            return 0
        return self.token_end - self.token_start


@dataclass(frozen=True, slots=True)
class WorkloadRecord:
    """One actual LLM admission request before tokenization by vLLM."""

    request_id: str
    workflow: str
    prompt: str
    meta: AgentNodeMeta
    prompt_segments: tuple[PromptSegment, ...]
    source_dataset: str
    source_index: int | str
    source_group_id: str
    claim_scope: str
    timestamp: str | float | int | None = None
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    schema_version: str = WORKLOAD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != WORKLOAD_SCHEMA:
            raise ValueError(
                f"unsupported workload schema {self.schema_version!r}; "
                f"expected {WORKLOAD_SCHEMA!r}"
            )
        if self.request_id != self.meta.request_id:
            raise ValueError("record and metadata request IDs must match")
        if not self.prompt:
            raise ValueError("prompt must be non-empty")
        if not self.prompt_segments:
            raise ValueError("prompt_segments must be non-empty")
        if self.claim_scope not in CLAIM_SCOPES:
            raise ValueError(f"unsupported claim_scope {self.claim_scope!r}")
        for segment in self.prompt_segments:
            segment.validate(self.prompt)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "WorkloadRecord":
        schema = data.get("schema_version")
        if schema != WORKLOAD_SCHEMA:
            raise ValueError(
                f"unsupported workload schema {schema!r}; expected {WORKLOAD_SCHEMA!r}"
            )
        prompt = _required_text(data, "prompt")
        segments = tuple(
            PromptSegment.from_mapping(item) for item in data["prompt_segments"]
        )
        record = cls(
            request_id=_required_text(data, "request_id"),
            workflow=_required_text(data, "workflow"),
            prompt=prompt,
            meta=AgentNodeMeta.from_mapping(data["meta"]),
            prompt_segments=segments,
            source_dataset=_required_text(data, "source_dataset"),
            source_index=data["source_index"],
            source_group_id=_required_text(data, "source_group_id"),
            claim_scope=_required_text(data, "claim_scope"),
            timestamp=data.get("timestamp"),
            dependencies=tuple(str(item) for item in data.get("dependencies", [])),
            schema_version=schema,
        )
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "workflow": self.workflow,
            "prompt": self.prompt,
            "meta": self.meta.to_dict(),
            "prompt_segments": [item.to_dict() for item in self.prompt_segments],
            "source_dataset": self.source_dataset,
            "source_index": self.source_index,
            "source_group_id": self.source_group_id,
            "claim_scope": self.claim_scope,
            "timestamp": self.timestamp,
            "dependencies": list(self.dependencies),
        }


def iter_workload_jsonl(path: str | Path) -> Iterator[WorkloadRecord]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield WorkloadRecord.from_mapping(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid workload record at line {line_number}: {exc}"
                ) from exc


def read_workload_jsonl(path: str | Path) -> list[WorkloadRecord]:
    return list(iter_workload_jsonl(path))


def write_workload_jsonl(records: Iterable[WorkloadRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
