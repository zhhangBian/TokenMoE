"""Strict trace format for routed-expert data captured from vLLM."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from tokenmoe.schema import AgentNodeMeta, PromptSegment


TRACE_SCHEMA = "tokenmoe.trace.v3"
TRACE_BACKEND = "vllm-routed-experts"


@dataclass(frozen=True, slots=True)
class LayerTrace:
    layer_id: int
    selected_experts: tuple[tuple[int, ...], ...]
    router_scores: tuple[tuple[float, ...], ...] | None = None

    @classmethod
    def from_arrays(
        cls,
        layer_id: int,
        selected_experts: np.ndarray,
        router_scores: np.ndarray | None,
    ) -> "LayerTrace":
        selected = np.asarray(selected_experts, dtype=np.int64)
        if selected.ndim != 2 or selected.shape[0] == 0 or selected.shape[1] == 0:
            raise ValueError("selected_experts must have shape [tokens, top_k]")
        scores = None
        if router_scores is not None:
            score_array = np.asarray(router_scores, dtype=np.float32)
            if score_array.shape != selected.shape:
                raise ValueError("router_scores must match selected_experts")
            scores = tuple(tuple(float(value) for value in row) for row in score_array)
        return cls(
            layer_id=int(layer_id),
            selected_experts=tuple(
                tuple(int(value) for value in row) for row in selected
            ),
            router_scores=scores,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "LayerTrace":
        return cls.from_arrays(
            layer_id=int(data["layer_id"]),
            selected_experts=np.asarray(data["selected_experts"]),
            router_scores=(
                None
                if data.get("router_scores") is None
                else np.asarray(data["router_scores"])
            ),
        )

    def selected_array(self) -> np.ndarray:
        return np.asarray(self.selected_experts, dtype=np.int64)

    def scores_array(self) -> np.ndarray | None:
        if self.router_scores is None:
            return None
        return np.asarray(self.router_scores, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "selected_experts": [list(row) for row in self.selected_experts],
            "router_scores": (
                None
                if self.router_scores is None
                else [list(row) for row in self.router_scores]
            ),
        }


@dataclass(frozen=True, slots=True)
class TraceRecord:
    request_id: str
    metadata: AgentNodeMeta
    model_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    prompt_segments: tuple[PromptSegment, ...]
    layers: tuple[LayerTrace, ...]
    moe_layer_ids: tuple[int, ...]
    router_top_k: int
    num_experts: int
    router_score_semantics: str | None
    router_scores_unavailable_reason: str | None
    source_dataset: str
    source_group_id: str
    source_index: int | str
    claim_scope: str
    timestamp: str | float | int | None = None
    schema_version: str = TRACE_SCHEMA
    backend: str = TRACE_BACKEND

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TraceRecord":
        record = cls(
            request_id=str(data["request_id"]),
            metadata=AgentNodeMeta.from_mapping(data["metadata"]),
            model_id=str(data["model_id"]),
            prompt=str(data["prompt"]),
            prompt_token_ids=tuple(int(item) for item in data["prompt_token_ids"]),
            generated_token_ids=tuple(
                int(item) for item in data["generated_token_ids"]
            ),
            prompt_segments=tuple(
                PromptSegment.from_mapping(item) for item in data["prompt_segments"]
            ),
            layers=tuple(LayerTrace.from_mapping(item) for item in data["layers"]),
            moe_layer_ids=tuple(int(item) for item in data["moe_layer_ids"]),
            router_top_k=int(data["router_top_k"]),
            num_experts=int(data["num_experts"]),
            router_score_semantics=data.get("router_score_semantics"),
            router_scores_unavailable_reason=data.get(
                "router_scores_unavailable_reason"
            ),
            source_dataset=str(data["source_dataset"]),
            source_group_id=str(data["source_group_id"]),
            source_index=data["source_index"],
            claim_scope=str(data["claim_scope"]),
            timestamp=data.get("timestamp"),
            schema_version=str(data["schema_version"]),
            backend=str(data["backend"]),
        )
        validate_trace(record)
        return record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "request_id": self.request_id,
            "metadata": self.metadata.to_dict(),
            "model_id": self.model_id,
            "prompt": self.prompt,
            "prompt_token_ids": list(self.prompt_token_ids),
            "generated_token_ids": list(self.generated_token_ids),
            "prompt_segments": [segment.to_dict() for segment in self.prompt_segments],
            "layers": [layer.to_dict() for layer in self.layers],
            "moe_layer_ids": list(self.moe_layer_ids),
            "router_top_k": self.router_top_k,
            "num_experts": self.num_experts,
            "router_score_semantics": self.router_score_semantics,
            "router_scores_unavailable_reason": self.router_scores_unavailable_reason,
            "source_dataset": self.source_dataset,
            "source_group_id": self.source_group_id,
            "source_index": self.source_index,
            "claim_scope": self.claim_scope,
            "timestamp": self.timestamp,
        }

    @property
    def prompt_token_count(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_token_count(self) -> int:
        return len(self.generated_token_ids)

    @property
    def has_router_scores(self) -> bool:
        return all(layer.router_scores is not None for layer in self.layers)

    def layer(self, layer_id: int) -> LayerTrace:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(layer_id)


class TraceValidationError(ValueError):
    pass


def validate_trace(record: TraceRecord) -> None:
    prefix = record.request_id or "<missing-request-id>"

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise TraceValidationError(f"{prefix}: {message}")

    require(record.schema_version == TRACE_SCHEMA, "unsupported trace schema")
    require(record.backend == TRACE_BACKEND, "trace was not captured by vLLM")
    require(record.request_id == record.metadata.request_id, "metadata ID mismatch")
    require(bool(record.model_id), "model_id is required")
    require(bool(record.prompt), "prompt is required")
    require(bool(record.prompt_token_ids), "prompt_token_ids are required")
    require(record.router_top_k > 0, "router_top_k must be positive")
    require(record.num_experts > record.router_top_k, "num_experts is invalid")
    require(bool(record.moe_layer_ids), "moe_layer_ids are required")
    require(
        tuple(layer.layer_id for layer in record.layers) == record.moe_layer_ids,
        "layers must exactly match moe_layer_ids",
    )
    score_presence: list[bool] = []
    for layer in record.layers:
        selected = layer.selected_array()
        require(
            selected.shape == (record.prompt_token_count, record.router_top_k),
            f"layer {layer.layer_id} selected expert shape mismatch",
        )
        require(
            bool(np.all((selected >= 0) & (selected < record.num_experts))),
            f"layer {layer.layer_id} has out-of-range expert IDs",
        )
        scores = layer.scores_array()
        score_presence.append(scores is not None)
        if scores is not None:
            require(scores.shape == selected.shape, "router score shape mismatch")
            require(bool(np.isfinite(scores).all()), "router scores must be finite")
            require(bool((scores >= 0).all()), "router scores must be non-negative")
    require(all(score_presence) or not any(score_presence), "partial router scores")
    if all(score_presence):
        require(bool(record.router_score_semantics), "score semantics are required")
        require(
            record.router_scores_unavailable_reason is None,
            "score unavailability reason conflicts with captured scores",
        )
    else:
        require(
            bool(record.router_scores_unavailable_reason),
            "missing router scores require an explicit reason",
        )
        require(record.router_score_semantics is None, "semantics require scores")
    require(bool(record.prompt_segments), "prompt_segments are required")
    for segment in record.prompt_segments:
        segment.validate(record.prompt, record.prompt_token_count)
        if segment.token_start is None:
            require(bool(segment.alignment_error), "unaligned segment needs a reason")


def validate_traces(records: Iterable[TraceRecord]) -> None:
    model_id: str | None = None
    trace_profile: tuple[object, ...] | None = None
    for record in records:
        validate_trace(record)
        if model_id is None:
            model_id = record.model_id
        elif record.model_id != model_id:
            raise TraceValidationError("one analysis run must contain one model")
        current_profile = (
            record.moe_layer_ids,
            record.router_top_k,
            record.num_experts,
            record.has_router_scores,
            record.router_score_semantics,
        )
        if trace_profile is None:
            trace_profile = current_profile
        elif current_profile != trace_profile:
            raise TraceValidationError(
                "one analysis run must use one homogeneous Router capture profile"
            )


def trace_from_vllm(
    *,
    request_id: str,
    metadata: AgentNodeMeta,
    model_id: str,
    prompt: str,
    prompt_token_ids: Iterable[int],
    generated_token_ids: Iterable[int],
    prompt_segments: Iterable[PromptSegment],
    selected_experts: np.ndarray,
    router_scores: np.ndarray | None,
    moe_layer_ids: Iterable[int],
    router_top_k: int,
    num_experts: int,
    router_score_semantics: str | None,
    router_scores_unavailable_reason: str | None,
    source_dataset: str,
    source_group_id: str,
    source_index: int | str,
    claim_scope: str,
    timestamp: str | float | int | None = None,
) -> TraceRecord:
    """Build a strict prompt-only trace from vLLM arrays.

    vLLM may expose either all decoder-layer slots or compact MoE-layer slots;
    both are native capture layouts, not legacy trace formats.
    """

    selected = np.asarray(selected_experts, dtype=np.int64)
    if selected.ndim != 3:
        raise ValueError("selected_experts must have shape [tokens, layers, top_k]")
    scores = (
        None if router_scores is None else np.asarray(router_scores, dtype=np.float32)
    )
    if scores is not None and scores.shape != selected.shape:
        raise ValueError("router_scores must match selected_experts")
    layer_ids = tuple(int(item) for item in moe_layer_ids)
    if selected.shape[1] == len(layer_ids):
        layer_slots = range(len(layer_ids))
    elif layer_ids and max(layer_ids) < selected.shape[1]:
        layer_slots = layer_ids
    else:
        raise ValueError("vLLM layer dimension does not match moe_layer_ids")
    layers = tuple(
        LayerTrace.from_arrays(
            layer_id=layer_id,
            selected_experts=selected[:, slot, :],
            router_scores=None if scores is None else scores[:, slot, :],
        )
        for layer_id, slot in zip(layer_ids, layer_slots)
    )
    record = TraceRecord(
        request_id=request_id,
        metadata=metadata,
        model_id=model_id,
        prompt=prompt,
        prompt_token_ids=tuple(int(item) for item in prompt_token_ids),
        generated_token_ids=tuple(int(item) for item in generated_token_ids),
        prompt_segments=tuple(prompt_segments),
        layers=layers,
        moe_layer_ids=layer_ids,
        router_top_k=int(router_top_k),
        num_experts=int(num_experts),
        router_score_semantics=router_score_semantics,
        router_scores_unavailable_reason=router_scores_unavailable_reason,
        source_dataset=source_dataset,
        source_group_id=source_group_id,
        source_index=source_index,
        claim_scope=claim_scope,
        timestamp=timestamp,
    )
    validate_trace(record)
    return record


def decode_vllm_routed_experts_b64(payload: str) -> np.ndarray:
    return np.load(io.BytesIO(base64.b64decode(payload)))


def _encode_with_offsets(
    tokenizer: Any, prompt: str, add_special_tokens: bool
) -> tuple[list[int], list[tuple[int, int]]] | None:
    try:
        encoded = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=add_special_tokens,
        )
    except (TypeError, NotImplementedError, ValueError):
        return None
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
        offsets = offsets[0]
    return [int(item) for item in input_ids], [
        (int(start), int(end)) for start, end in offsets
    ]


def align_prompt_segments(
    *,
    prompt: str,
    segments: Iterable[PromptSegment],
    tokenizer: Any,
    prompt_token_ids: Iterable[int],
) -> tuple[PromptSegment, ...]:
    expected_ids = [int(item) for item in prompt_token_ids]
    encoded = next(
        (
            item
            for add_special_tokens in (True, False)
            if (item := _encode_with_offsets(tokenizer, prompt, add_special_tokens))
            is not None
            and item[0] == expected_ids
        ),
        None,
    )
    if encoded is None:
        reason = "tokenizer IDs differ from vLLM prompt_token_ids"
        return tuple(
            replace(
                segment,
                token_start=None,
                token_end=None,
                alignment_error=reason,
            )
            for segment in segments
        )
    offsets = encoded[1]
    aligned: list[PromptSegment] = []
    for segment in segments:
        overlap = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > start and start < segment.char_end and end > segment.char_start
        ]
        if overlap:
            aligned.append(
                replace(
                    segment,
                    token_start=overlap[0],
                    token_end=overlap[-1] + 1,
                    alignment_error=None,
                )
            )
        else:
            aligned.append(
                replace(
                    segment,
                    token_start=None,
                    token_end=None,
                    alignment_error=f"no token overlaps {segment.segment_id}",
                )
            )
    return tuple(aligned)


def iter_trace_jsonl(path: str | Path) -> Iterator[TraceRecord]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield TraceRecord.from_mapping(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise TraceValidationError(
                    f"invalid trace record at line {line_number}: {exc}"
                ) from exc


def read_trace_jsonl(path: str | Path) -> list[TraceRecord]:
    records = list(iter_trace_jsonl(path))
    validate_traces(records)
    return records


def write_trace_jsonl(records: Iterable[TraceRecord], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            validate_trace(record)
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
