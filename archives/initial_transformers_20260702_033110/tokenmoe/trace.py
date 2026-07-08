from __future__ import annotations

import base64
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from tokenmoe.schema import AgentNodeMeta


@dataclass(frozen=True)
class LayerTrace:
    layer_id: int
    selected_experts: list[list[int]]
    router_scores: list[list[float]] | None = None
    active_expert_histogram: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_arrays(
        cls,
        layer_id: int,
        selected_experts: np.ndarray,
        router_scores: np.ndarray | None = None,
    ) -> "LayerTrace":
        selected = np.asarray(selected_experts).astype(int)
        if selected.ndim != 2:
            raise ValueError("selected_experts must have shape [tokens, top_k]")
        hist = active_expert_histogram(selected)
        scores_list = None
        if router_scores is not None:
            scores = np.asarray(router_scores, dtype=float)
            if scores.shape != selected.shape:
                raise ValueError(
                    "router_scores must have the same [tokens, top_k] shape "
                    "as selected_experts"
                )
            scores_list = scores.tolist()
        return cls(
            layer_id=int(layer_id),
            selected_experts=selected.tolist(),
            router_scores=scores_list,
            active_expert_histogram={str(k): int(v) for k, v in hist.items()},
        )

    def selected_array(self) -> np.ndarray:
        return np.asarray(self.selected_experts, dtype=np.int64)

    def scores_array(self) -> np.ndarray | None:
        if self.router_scores is None:
            return None
        return np.asarray(self.router_scores, dtype=np.float32)


@dataclass(frozen=True)
class TraceRecord:
    request_id: str
    metadata: AgentNodeMeta
    model_id: str
    backend: str
    prompt: str
    prompt_token_count: int
    output_token_count: int
    layers: list[LayerTrace]
    token_ids: list[int] | None = None
    generated_token_ids: list[int] | None = None
    schema_version: str = "tokenmoe.trace.v1"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TraceRecord":
        meta_data = data.get("metadata", data.get("meta", {}))
        layers = [LayerTrace(**layer) for layer in data["layers"]]
        return cls(
            request_id=str(data["request_id"]),
            metadata=AgentNodeMeta.from_mapping(meta_data),
            model_id=str(data.get("model_id", "unknown")),
            backend=str(data.get("backend", "unknown")),
            prompt=str(data.get("prompt", "")),
            prompt_token_count=int(data.get("prompt_token_count", 0)),
            output_token_count=int(data.get("output_token_count", 0)),
            layers=layers,
            token_ids=data.get("token_ids"),
            generated_token_ids=data.get("generated_token_ids"),
            schema_version=str(data.get("schema_version", "tokenmoe.trace.v1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "metadata": self.metadata.to_dict(),
            "model_id": self.model_id,
            "backend": self.backend,
            "prompt": self.prompt,
            "prompt_token_count": self.prompt_token_count,
            "output_token_count": self.output_token_count,
            "token_ids": self.token_ids,
            "generated_token_ids": self.generated_token_ids,
            "layers": [asdict(layer) for layer in self.layers],
        }

    def layer(self, layer_id: int) -> LayerTrace:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(layer_id)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def top_k(self) -> int:
        if not self.layers or not self.layers[0].selected_experts:
            return 0
        return len(self.layers[0].selected_experts[0])


def active_expert_histogram(selected_experts: np.ndarray) -> dict[int, int]:
    selected = np.asarray(selected_experts)
    if selected.size == 0:
        return {}
    counts = Counter(int(item) for item in selected.reshape(-1) if int(item) >= 0)
    return dict(sorted(counts.items()))


def trace_from_selected_experts(
    *,
    request_id: str,
    metadata: AgentNodeMeta,
    model_id: str,
    backend: str,
    prompt: str,
    selected_experts: np.ndarray,
    router_scores: np.ndarray | None = None,
    token_ids: list[int] | None = None,
    output_token_count: int = 0,
) -> TraceRecord:
    """Create a TraceRecord from an array shaped [tokens, layers, top_k]."""

    selected = np.asarray(selected_experts)
    if selected.ndim != 3:
        raise ValueError("selected_experts must have shape [tokens, layers, top_k]")
    scores = None if router_scores is None else np.asarray(router_scores)
    if scores is not None and scores.shape != selected.shape:
        raise ValueError("router_scores must match selected_experts shape")
    layers = []
    for layer_id in range(selected.shape[1]):
        layers.append(
            LayerTrace.from_arrays(
                layer_id=layer_id,
                selected_experts=selected[:, layer_id, :],
                router_scores=None if scores is None else scores[:, layer_id, :],
            )
        )
    return TraceRecord(
        request_id=request_id,
        metadata=metadata,
        model_id=model_id,
        backend=backend,
        prompt=prompt,
        prompt_token_count=int(selected.shape[0]),
        output_token_count=int(output_token_count),
        layers=layers,
        token_ids=token_ids,
    )


def decode_vllm_routed_experts_b64(payload: str) -> np.ndarray:
    """Decode vLLM OpenAI API routed_experts base64 .npy payload."""

    return np.load(io.BytesIO(base64.b64decode(payload)))


def read_trace_jsonl(path: str | Path) -> list[TraceRecord]:
    records: list[TraceRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TraceRecord.from_mapping(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid trace record at {path}:{line_no}: {exc}") from exc
    return records


def iter_trace_jsonl(path: str | Path) -> Iterator[TraceRecord]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield TraceRecord.from_mapping(json.loads(line))
            except Exception as exc:
                raise ValueError(f"invalid trace record at {path}:{line_no}: {exc}") from exc


def write_trace_jsonl(records: Iterable[TraceRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def parquet_rows(records: Iterable[TraceRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metadata_json = json.dumps(record.metadata.to_dict(), ensure_ascii=False)
        for layer in record.layers:
            selected = layer.selected_array()
            scores = layer.scores_array()
            rows.append(
                {
                    "request_id": record.request_id,
                    "model_id": record.model_id,
                    "backend": record.backend,
                    "workflow_role": record.metadata.role,
                    "workflow_phase": record.metadata.phase,
                    "agent_id": record.metadata.agent_id,
                    "graph_node_type": record.metadata.graph_node_type,
                    "prompt_block_types": ",".join(record.metadata.prompt_block_types),
                    "metadata_json": metadata_json,
                    "layer_id": layer.layer_id,
                    "token_count": int(selected.shape[0]),
                    "top_k": int(selected.shape[1]) if selected.ndim == 2 else 0,
                    "selected_experts_json": json.dumps(layer.selected_experts),
                    "router_scores_json": None
                    if scores is None
                    else json.dumps(layer.router_scores),
                    "active_expert_histogram_json": json.dumps(
                        layer.active_expert_histogram
                    ),
                }
            )
    return rows


def write_trace_parquet(records: Iterable[TraceRecord], path: str | Path) -> None:
    """Write parquet-compatible one-row-per-request-layer trace records."""

    rows = parquet_rows(records)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write parquet output") from exc
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output)
