from __future__ import annotations

import base64
import io
import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from tokenmoe.schema import AgentNodeMeta, PromptSegment, default_prompt_segment


TRACE_SCHEMA_V1 = "tokenmoe.trace.v1"
TRACE_SCHEMA_V2 = "tokenmoe.trace.v2"
CURRENT_TRACE_SCHEMA = TRACE_SCHEMA_V2
CURRENT_TRACE_BACKEND = "vllm-routed-experts"


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
    schema_version: str = TRACE_SCHEMA_V2
    prompt_segments: list[PromptSegment] = field(default_factory=list)
    moe_layer_ids: list[int] = field(default_factory=list)
    routing_scope: str = "prompt_only"
    prompt_routing_start: int = 0
    decode_routing_excluded: bool = False
    backend_fallback_used: bool = False
    backend_fallback_reason: str | None = None
    segment_unavailable_reason: str | None = None
    unavailable_layer_ids: list[int] = field(default_factory=list)
    router_top_k: int | None = None
    num_experts: int | None = None
    source_dataset: str | None = None
    source_group_id: str | None = None
    source_index: int | str | None = None
    timestamp: str | float | int | None = None
    claim_scope: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TraceRecord":
        meta_data = data.get("metadata", data.get("meta", {}))
        layers = [LayerTrace(**layer) for layer in data["layers"]]
        prompt = str(data.get("prompt", ""))
        prompt_segments = [
            PromptSegment.from_mapping(item)
            for item in data.get("prompt_segments", data.get("segments", []))
        ]
        if not prompt_segments:
            prompt_segments = [default_prompt_segment(prompt, AgentNodeMeta.from_mapping(meta_data).block_type_key)]
        return cls(
            request_id=str(data["request_id"]),
            metadata=AgentNodeMeta.from_mapping(meta_data),
            model_id=str(data.get("model_id", "unknown")),
            backend=str(data.get("backend", "unknown")),
            prompt=prompt,
            prompt_token_count=int(data.get("prompt_token_count", 0)),
            output_token_count=int(data.get("output_token_count", 0)),
            layers=layers,
            token_ids=data.get("token_ids"),
            generated_token_ids=data.get("generated_token_ids"),
            schema_version=str(data.get("schema_version", TRACE_SCHEMA_V1)),
            prompt_segments=prompt_segments,
            moe_layer_ids=[int(item) for item in data.get("moe_layer_ids", [])],
            routing_scope=str(data.get("routing_scope", "unknown")),
            prompt_routing_start=int(data.get("prompt_routing_start", 0)),
            decode_routing_excluded=bool(data.get("decode_routing_excluded", False)),
            backend_fallback_used=bool(data.get("backend_fallback_used", False)),
            backend_fallback_reason=data.get("backend_fallback_reason"),
            segment_unavailable_reason=data.get("segment_unavailable_reason"),
            unavailable_layer_ids=[
                int(item) for item in data.get("unavailable_layer_ids", [])
            ],
            router_top_k=None
            if data.get("router_top_k") in (None, "", "null")
            else int(data["router_top_k"]),
            num_experts=None
            if data.get("num_experts") in (None, "", "null")
            else int(data["num_experts"]),
            source_dataset=data.get("source_dataset"),
            source_group_id=data.get("source_group_id"),
            source_index=data.get("source_index"),
            timestamp=data.get("timestamp"),
            claim_scope=data.get("claim_scope"),
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
            "prompt_segments": [segment.to_dict() for segment in self.prompt_segments],
            "moe_layer_ids": list(self.moe_layer_ids),
            "routing_scope": self.routing_scope,
            "prompt_routing_start": self.prompt_routing_start,
            "decode_routing_excluded": self.decode_routing_excluded,
            "backend_fallback_used": self.backend_fallback_used,
            "backend_fallback_reason": self.backend_fallback_reason,
            "segment_unavailable_reason": self.segment_unavailable_reason,
            "unavailable_layer_ids": list(self.unavailable_layer_ids),
            "router_top_k": self.router_top_k,
            "num_experts": self.num_experts,
            "source_dataset": self.source_dataset,
            "source_group_id": self.source_group_id,
            "source_index": self.source_index,
            "timestamp": self.timestamp,
            "claim_scope": self.claim_scope,
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
        if self.router_top_k is not None:
            return self.router_top_k
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
    generated_token_ids: list[int] | None = None,
    prompt_segments: list[PromptSegment] | None = None,
    layer_ids: list[int] | None = None,
    moe_layer_ids: list[int] | None = None,
    schema_version: str = TRACE_SCHEMA_V2,
    routing_scope: str = "prompt_only",
    prompt_routing_start: int = 0,
    decode_routing_excluded: bool = False,
    backend_fallback_used: bool = False,
    backend_fallback_reason: str | None = None,
    segment_unavailable_reason: str | None = None,
    unavailable_layer_ids: list[int] | None = None,
    router_top_k: int | None = None,
    num_experts: int | None = None,
    source_dataset: str | None = None,
    source_group_id: str | None = None,
    source_index: int | str | None = None,
    timestamp: str | float | int | None = None,
    claim_scope: str | None = None,
) -> TraceRecord:
    """Create a TraceRecord from an array shaped [tokens, layers, top_k]."""

    selected = np.asarray(selected_experts)
    if selected.ndim != 3:
        raise ValueError("selected_experts must have shape [tokens, layers, top_k]")
    scores = None if router_scores is None else np.asarray(router_scores)
    if scores is not None and scores.shape != selected.shape:
        raise ValueError("router_scores must match selected_experts shape")
    effective_moe_layer_ids = [int(item) for item in (moe_layer_ids or [])]
    if layer_ids is not None:
        effective_layer_ids = [int(item) for item in layer_ids]
        selected_by_output_layer = selected
        scores_by_output_layer = scores
    elif effective_moe_layer_ids and selected.shape[1] != len(effective_moe_layer_ids):
        max_layer_id = max(effective_moe_layer_ids)
        if max_layer_id >= selected.shape[1]:
            raise ValueError(
                "moe_layer_ids refer past selected_experts layer dimension"
            )
        effective_layer_ids = effective_moe_layer_ids
        selected_by_output_layer = selected[:, effective_layer_ids, :]
        scores_by_output_layer = None if scores is None else scores[:, effective_layer_ids, :]
    else:
        effective_layer_ids = (
            effective_moe_layer_ids
            if effective_moe_layer_ids
            else list(range(selected.shape[1]))
        )
        selected_by_output_layer = selected
        scores_by_output_layer = scores
    layers = []
    for output_idx, layer_id in enumerate(effective_layer_ids):
        layers.append(
            LayerTrace.from_arrays(
                layer_id=layer_id,
                selected_experts=selected_by_output_layer[:, output_idx, :],
                router_scores=None
                if scores_by_output_layer is None
                else scores_by_output_layer[:, output_idx, :],
            )
        )
    segments = prompt_segments or [default_prompt_segment(prompt, metadata.block_type_key)]
    return TraceRecord(
        request_id=request_id,
        metadata=metadata,
        model_id=model_id,
        backend=backend,
        prompt=prompt,
        prompt_token_count=int(selected_by_output_layer.shape[0]),
        output_token_count=int(output_token_count),
        layers=layers,
        token_ids=token_ids,
        generated_token_ids=generated_token_ids,
        schema_version=schema_version,
        prompt_segments=segments,
        moe_layer_ids=effective_moe_layer_ids or effective_layer_ids,
        routing_scope=routing_scope,
        prompt_routing_start=prompt_routing_start,
        decode_routing_excluded=decode_routing_excluded,
        backend_fallback_used=backend_fallback_used,
        backend_fallback_reason=backend_fallback_reason,
        segment_unavailable_reason=segment_unavailable_reason,
        unavailable_layer_ids=list(unavailable_layer_ids or []),
        router_top_k=router_top_k or int(selected_by_output_layer.shape[2]),
        num_experts=num_experts,
        source_dataset=source_dataset,
        source_group_id=source_group_id,
        source_index=source_index,
        timestamp=timestamp,
        claim_scope=claim_scope,
    )


def decode_vllm_routed_experts_b64(payload: str) -> np.ndarray:
    """Decode vLLM OpenAI API routed_experts base64 .npy payload."""

    return np.load(io.BytesIO(base64.b64decode(payload)))


def _tokenizer_encode_with_offsets(
    tokenizer: Any, prompt: str, *, add_special_tokens: bool
) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=add_special_tokens,
        )
    except (TypeError, NotImplementedError, ValueError):
        return None
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    offsets = (
        encoded["offset_mapping"]
        if isinstance(encoded, dict)
        else encoded.offset_mapping
    )
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    return [int(item) for item in input_ids], [
        (int(start), int(end)) for start, end in offsets
    ]


def align_prompt_segments_with_tokenizer(
    *,
    prompt: str,
    prompt_segments: list[PromptSegment],
    tokenizer: Any,
    prompt_token_ids: list[int],
) -> tuple[list[PromptSegment], str | None]:
    """Map character spans to token spans and verify IDs against vLLM tokens."""

    chosen: tuple[list[int], list[tuple[int, int]]] | None = None
    for add_special_tokens in (True, False):
        encoded = _tokenizer_encode_with_offsets(
            tokenizer, prompt, add_special_tokens=add_special_tokens
        )
        if encoded is None:
            continue
        input_ids, offsets = encoded
        if input_ids == [int(item) for item in prompt_token_ids]:
            chosen = (input_ids, offsets)
            break
    if chosen is None:
        reason = "tokenizer_prompt_token_ids_mismatch_or_offsets_unavailable"
        return [
            replace(
                segment,
                token_start=None,
                token_end=None,
                alignment_status="unavailable",
                alignment_error=reason,
            )
            for segment in prompt_segments
        ], reason

    _, offsets = chosen
    aligned: list[PromptSegment] = []
    reasons: list[str] = []
    for segment in prompt_segments:
        overlapping = [
            idx
            for idx, (start, end) in enumerate(offsets)
            if end > start and start < segment.char_end and end > segment.char_start
        ]
        if not overlapping:
            reason = f"no_tokens_overlap_segment:{segment.segment_id}"
            reasons.append(reason)
            aligned.append(
                replace(
                    segment,
                    token_start=None,
                    token_end=None,
                    alignment_status="unavailable",
                    alignment_error=reason,
                )
            )
            continue
        aligned.append(
            replace(
                segment,
                token_start=min(overlapping),
                token_end=max(overlapping) + 1,
                alignment_status="aligned",
                alignment_error=None,
            )
        )
    return aligned, ";".join(reasons) if reasons else None


class TraceValidationError(ValueError):
    pass


def validate_current_stage_traces(records: Iterable[TraceRecord]) -> None:
    """Reject trace inputs that cannot support current-stage claims."""

    materialized = list(records)
    versions = {record.schema_version for record in materialized}
    if len(versions) > 1:
        raise TraceValidationError(f"mixed trace schema versions: {sorted(versions)}")
    if versions and versions != {TRACE_SCHEMA_V2}:
        raise TraceValidationError(
            f"current-stage analysis requires {TRACE_SCHEMA_V2}, got {sorted(versions)}"
        )
    for record in materialized:
        if record.backend != CURRENT_TRACE_BACKEND:
            raise TraceValidationError(
                f"{record.request_id}: backend {record.backend!r} is not {CURRENT_TRACE_BACKEND!r}"
            )
        if record.backend_fallback_used:
            raise TraceValidationError(f"{record.request_id}: fallback trace is not allowed")
        if record.routing_scope != "prompt_only":
            raise TraceValidationError(
                f"{record.request_id}: routing_scope {record.routing_scope!r} is not prompt_only"
            )
        if record.prompt_routing_start != 0:
            raise TraceValidationError(
                f"{record.request_id}: prompt_routing_start must be 0"
            )
        if not record.moe_layer_ids:
            raise TraceValidationError(f"{record.request_id}: missing moe_layer_ids")
        moe_layers = {int(layer_id) for layer_id in record.moe_layer_ids}
        for layer in record.layers:
            if layer.layer_id not in moe_layers:
                raise TraceValidationError(
                    f"{record.request_id}: layer {layer.layer_id} not in moe_layer_ids"
                )
            selected = layer.selected_array()
            if selected.shape[0] != record.prompt_token_count:
                raise TraceValidationError(
                    f"{record.request_id}: layer {layer.layer_id} token dimension "
                    f"{selected.shape[0]} != prompt_token_count {record.prompt_token_count}"
                )
        for segment in record.prompt_segments:
            if segment.alignment_status == "aligned":
                if segment.token_start is None or segment.token_end is None:
                    raise TraceValidationError(
                        f"{record.request_id}: aligned segment {segment.segment_id} missing token span"
                    )
                if segment.token_end > record.prompt_token_count:
                    raise TraceValidationError(
                        f"{record.request_id}: segment {segment.segment_id} extends past prompt tokens"
                    )


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
