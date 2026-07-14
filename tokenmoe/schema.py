from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


WORKLOAD_SCHEMA_V1 = "tokenmoe.workload.v1"
WORKLOAD_SCHEMA_V2 = "tokenmoe.workload.v2"

CLAIM_SCOPE_REAL_AGENT = "real_agent_metadata"
CLAIM_SCOPE_CHAT = "chat_prompt_only"
CLAIM_SCOPE_DOMAIN = "domain_instruction"
CLAIM_SCOPE_SYNTHETIC = "synthetic"

REQUIRED_AGENT_META_FIELDS = (
    "request_id",
    "agent_id",
    "role",
    "phase",
    "tool_type",
    "graph_node_type",
    "prompt_block_types",
)


@dataclass(frozen=True)
class AgentNodeMeta:
    """Metadata visible before an agent LLM request reaches the MoE router."""

    request_id: str
    agent_id: str
    role: str
    phase: str
    tool_type: str | None
    graph_node_type: str
    prompt_block_types: list[str]
    ready_time: float = 0.0
    deadline: float | None = None
    run_probability: float = 1.0
    criticality: float = 1.0
    # Enriched agent metadata (heuristic-derived for SWE-agent; None when the
    # source dataset cannot provide them).
    trajectory_phase: str | None = None
    event_outcome: str | None = None
    dag_depth: int | None = None
    group_local_step_index: int | None = None
    on_critical_path: bool | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AgentNodeMeta":
        missing = [field for field in REQUIRED_AGENT_META_FIELDS if field not in data]
        if missing:
            raise ValueError(f"AgentNodeMeta missing required fields: {missing}")
        prompt_block_types = data["prompt_block_types"]
        if isinstance(prompt_block_types, str):
            prompt_block_types = [prompt_block_types]
        if not isinstance(prompt_block_types, list):
            raise TypeError("prompt_block_types must be a list[str] or string")
        return cls(
            request_id=str(data["request_id"]),
            agent_id=str(data["agent_id"]),
            role=str(data["role"]),
            phase=str(data["phase"]),
            tool_type=None
            if data.get("tool_type") in (None, "", "null")
            else str(data["tool_type"]),
            graph_node_type=str(data["graph_node_type"]),
            prompt_block_types=[str(item) for item in prompt_block_types],
            ready_time=float(data.get("ready_time", 0.0)),
            deadline=None
            if data.get("deadline") in (None, "", "null")
            else float(data["deadline"]),
            run_probability=float(data.get("run_probability", 1.0)),
            criticality=float(data.get("criticality", 1.0)),
            trajectory_phase=None
            if data.get("trajectory_phase") in (None, "", "null")
            else str(data["trajectory_phase"]),
            event_outcome=None
            if data.get("event_outcome") in (None, "", "null")
            else str(data["event_outcome"]),
            dag_depth=None
            if data.get("dag_depth") in (None, "", "null")
            else int(data["dag_depth"]),
            group_local_step_index=None
            if data.get("group_local_step_index") in (None, "", "null")
            else int(data["group_local_step_index"]),
            on_critical_path=None
            if data.get("on_critical_path") in (None, "", "null")
            else bool(data["on_critical_path"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def block_type_key(self) -> str:
        if not self.prompt_block_types:
            return "none"
        return "+".join(self.prompt_block_types)

    def fallback_keys(self) -> list[tuple[str, ...]]:
        """Return RouteSig lookup keys from specific to global."""

        block = self.block_type_key
        return [
            (self.agent_id, self.role, self.phase, block),
            (self.role, self.phase, block),
            (self.role, self.phase),
            (self.role,),
            ("global",),
        ]

    def segment_fallback_keys(
        self, block_type: str, segment_position: int | str | None
    ) -> list[tuple[str, ...]]:
        """Return segment-aware RouteSig lookup keys from specific to global.

        Order is fixed by the tokenmoe-route-signature spec. Enriched levels
        (trajectory_phase / tool_type) are skipped when the workload does not
        provide those fields.
        """

        block = str(block_type or self.block_type_key or "unknown")
        position = str(segment_position if segment_position is not None else "unknown")
        keys: list[tuple[str, ...]] = []
        if self.trajectory_phase and self.tool_type:
            keys.append(
                (
                    self.agent_id,
                    self.role,
                    self.trajectory_phase,
                    self.tool_type,
                    block,
                    position,
                )
            )
            keys.append(
                (self.role, self.trajectory_phase, self.tool_type, block, position)
            )
        if self.trajectory_phase:
            keys.append((self.role, self.trajectory_phase, block, position))
        keys.extend(
            [
                (self.role, self.phase, block, position),
                (self.role, self.phase, block),
                (self.role, block),
                (block,),
                (self.role, self.phase),
                (self.role,),
                ("global",),
            ]
        )
        return keys

    def group_key(self, level: str) -> str:
        if level == "agent":
            return self.agent_id
        if level == "role":
            return self.role
        if level == "phase":
            return self.phase
        if level == "role_phase":
            return f"{self.role}/{self.phase}"
        if level == "block":
            return self.block_type_key
        if level == "graph_node_type":
            return self.graph_node_type
        raise ValueError(f"unknown group level: {level}")


@dataclass(frozen=True)
class PromptSegment:
    """A semantically typed prompt block with character and optional token span."""

    segment_id: str
    block_type: str
    segment_position: int
    char_start: int
    char_end: int
    token_start: int | None = None
    token_end: int | None = None
    alignment_status: str = "char_span_only"
    alignment_error: str | None = None
    text_sha1: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PromptSegment":
        return cls(
            segment_id=str(data["segment_id"]),
            block_type=str(data["block_type"]),
            segment_position=int(data.get("segment_position", 0)),
            char_start=int(data["char_start"]),
            char_end=int(data["char_end"]),
            token_start=None
            if data.get("token_start") in (None, "", "null")
            else int(data["token_start"]),
            token_end=None
            if data.get("token_end") in (None, "", "null")
            else int(data["token_end"]),
            alignment_status=str(data.get("alignment_status", "char_span_only")),
            alignment_error=None
            if data.get("alignment_error") in (None, "", "null")
            else str(data["alignment_error"]),
            text_sha1=None
            if data.get("text_sha1") in (None, "", "null")
            else str(data["text_sha1"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def token_count(self) -> int:
        if self.token_start is None or self.token_end is None:
            return 0
        return max(0, self.token_end - self.token_start)

    def validate(self, prompt: str) -> None:
        if self.char_start < 0 or self.char_end < self.char_start:
            raise ValueError(f"invalid char span for segment {self.segment_id}")
        if self.char_end > len(prompt):
            raise ValueError(f"segment {self.segment_id} extends past prompt length")
        if (self.token_start is None) != (self.token_end is None):
            raise ValueError(f"segment {self.segment_id} has a partial token span")
        if self.token_start is not None and self.token_end is not None:
            if self.token_start < 0 or self.token_end < self.token_start:
                raise ValueError(f"invalid token span for segment {self.segment_id}")


def default_prompt_segment(prompt: str, block_type: str = "prompt") -> PromptSegment:
    return PromptSegment(
        segment_id="prompt-000",
        block_type=block_type,
        segment_position=0,
        char_start=0,
        char_end=len(prompt),
        alignment_status="legacy_unsegmented",
    )


@dataclass(frozen=True)
class WorkloadRecord:
    """One normalized agent request used by trace collection and replay."""

    request_id: str
    workflow: str
    prompt: str
    meta: AgentNodeMeta
    dependencies: list[str] = field(default_factory=list)
    source: str = "synthetic"
    expected_output_tokens: int = 16
    schema_version: str = WORKLOAD_SCHEMA_V2
    prompt_segments: list[PromptSegment] = field(default_factory=list)
    source_dataset: str | None = None
    source_index: int | str | None = None
    source_group_id: str | None = None
    timestamp: str | float | int | None = None
    claim_scope: str = CLAIM_SCOPE_SYNTHETIC
    unavailable_fields: list[str] = field(default_factory=list)
    dag_available: bool = False
    dependency_edges: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "WorkloadRecord":
        meta_data = data.get("meta", data.get("agent_node_meta", data))
        meta = AgentNodeMeta.from_mapping(meta_data)
        request_id = str(data.get("request_id", meta.request_id))
        if request_id != meta.request_id:
            meta = AgentNodeMeta.from_mapping({**meta.to_dict(), "request_id": request_id})
        dependencies = data.get("dependencies", [])
        if dependencies is None:
            dependencies = []
        prompt = str(data["prompt"])
        raw_segments = data.get("prompt_segments", data.get("segments", []))
        prompt_segments = [
            PromptSegment.from_mapping(item) for item in raw_segments
        ]
        if not prompt_segments:
            prompt_segments = [default_prompt_segment(prompt, meta.block_type_key)]
        for segment in prompt_segments:
            segment.validate(prompt)
        dependency_edges = []
        for edge in data.get("dependency_edges", []) or []:
            if isinstance(edge, dict):
                dependency_edges.append((str(edge["from"]), str(edge["to"])))
            else:
                src, dst = edge
                dependency_edges.append((str(src), str(dst)))
        return cls(
            request_id=request_id,
            workflow=str(data.get("workflow", "unknown")),
            prompt=prompt,
            meta=meta,
            dependencies=[str(dep) for dep in dependencies],
            source=str(data.get("source", "synthetic")),
            expected_output_tokens=int(data.get("expected_output_tokens", 16)),
            schema_version=str(data.get("schema_version", WORKLOAD_SCHEMA_V1)),
            prompt_segments=prompt_segments,
            source_dataset=None
            if data.get("source_dataset") in (None, "", "null")
            else str(data["source_dataset"]),
            source_index=data.get("source_index"),
            source_group_id=None
            if data.get("source_group_id") in (None, "", "null")
            else str(data["source_group_id"]),
            timestamp=data.get("timestamp"),
            claim_scope=str(data.get("claim_scope", CLAIM_SCOPE_SYNTHETIC)),
            unavailable_fields=[str(item) for item in data.get("unavailable_fields", [])],
            dag_available=bool(data.get("dag_available", bool(dependencies))),
            dependency_edges=dependency_edges,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "workflow": self.workflow,
            "prompt": self.prompt,
            "meta": self.meta.to_dict(),
            "dependencies": list(self.dependencies),
            "source": self.source,
            "expected_output_tokens": self.expected_output_tokens,
            "prompt_segments": [segment.to_dict() for segment in self.prompt_segments],
            "source_dataset": self.source_dataset,
            "source_index": self.source_index,
            "source_group_id": self.source_group_id,
            "timestamp": self.timestamp,
            "claim_scope": self.claim_scope,
            "unavailable_fields": list(self.unavailable_fields),
            "dag_available": self.dag_available,
            "dependency_edges": [
                {"from": src, "to": dst} for src, dst in self.dependency_edges
            ],
        }


def read_workload_jsonl(path: str | Path) -> list[WorkloadRecord]:
    records: list[WorkloadRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(WorkloadRecord.from_mapping(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid workload record at {path}:{line_no}: {exc}") from exc
    return records


def write_workload_jsonl(records: Iterable[WorkloadRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
