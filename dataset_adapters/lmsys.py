#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from dataset_adapters.common import (
    CLAIM_SCOPE_CHAT,
    PromptBlock,
    first_text,
    iter_raw_records,
    make_record,
    run_converter,
)


SOURCE_DATASET = "lmsys/lmsys-chat-1m"
OUTPUT_NAME = "lmsys_prompt_workloads.jsonl"


def _conversation_blocks(record: dict[str, Any]) -> list[PromptBlock]:
    blocks: list[PromptBlock] = []
    messages = (
        record.get("conversation")
        or record.get("conversations")
        or record.get("messages")
        or record.get("turns")
        or []
    )
    if isinstance(messages, str):
        blocks.append(PromptBlock("conversation_message", messages))
        return blocks
    if isinstance(messages, list):
        last = messages[-1] if messages else None
        last_role = (
            str(last.get("role", last.get("from", ""))).lower()
            if isinstance(last, dict)
            else ""
        )
        target = (
            len(messages) - 1 if last_role in {"assistant", "gpt"} else len(messages)
        )
        for turn in messages[:target]:
            if isinstance(turn, dict):
                role = str(turn.get("role", turn.get("from", "message"))).lower()
                text = turn.get("content", turn.get("value", turn.get("text", "")))
            else:
                role = "message"
                text = str(turn)
            if text is None or not str(text).strip():
                continue
            block_type = (
                "user_message"
                if role in {"user", "human"}
                else (
                    "assistant_message"
                    if role in {"assistant", "gpt"}
                    else "conversation_message"
                )
            )
            blocks.append(PromptBlock(block_type, str(text)))
    return blocks


def convert(source_path, limit: int, repo_id: str):
    records = []
    unavailable_fields: list[str] = ["dependency_edges", "ready_times"]
    for source_index, _, raw in iter_raw_records(source_path, limit=None):
        group_text, group_field = first_text(
            raw,
            (
                "conversation_id",
                "conv_id",
                "id",
                "conversation_hash",
                "turn_identifier",
            ),
        )
        timestamp, timestamp_field = first_text(
            raw, ("tstamp", "timestamp", "created_at")
        )
        if group_field is None:
            unavailable_fields.append("source_group_id")
        if timestamp_field is None:
            unavailable_fields.append("timestamp")
        blocks = _conversation_blocks(raw)
        if not blocks:
            unavailable_fields.append("conversation")
            continue
        records.append(
            make_record(
                request_id=f"lmsys-{source_index:06d}",
                workflow="lmsys-chat",
                blocks=blocks,
                source_dataset=repo_id,
                source_index=source_index,
                source_group_id=group_text or str(source_index),
                timestamp=timestamp,
                claim_scope=CLAIM_SCOPE_CHAT,
                role="assistant",
                phase="chat",
                graph_node_type="conversation",
            )
        )
        if len(records) >= limit:
            break
    return records, unavailable_fields


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LMSYS-Chat-1M to TokenMoE prompt workloads."
    )
    run_converter(
        parser=parser,
        adapter_name="lmsys",
        source_dataset=SOURCE_DATASET,
        output_name=OUTPUT_NAME,
        claim_scope=CLAIM_SCOPE_CHAT,
        convert_fn=convert,
    )


if __name__ == "__main__":
    main()
