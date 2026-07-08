from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


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
class WorkloadRecord:
    """One normalized agent request used by trace collection and replay."""

    request_id: str
    workflow: str
    prompt: str
    meta: AgentNodeMeta
    dependencies: list[str] = field(default_factory=list)
    source: str = "synthetic"
    expected_output_tokens: int = 16

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
        return cls(
            request_id=request_id,
            workflow=str(data.get("workflow", "unknown")),
            prompt=str(data["prompt"]),
            meta=meta,
            dependencies=[str(dep) for dep in dependencies],
            source=str(data.get("source", "synthetic")),
            expected_output_tokens=int(data.get("expected_output_tokens", 16)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "workflow": self.workflow,
            "prompt": self.prompt,
            "meta": self.meta.to_dict(),
            "dependencies": list(self.dependencies),
            "source": self.source,
            "expected_output_tokens": self.expected_output_tokens,
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
