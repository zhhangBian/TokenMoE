#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from dataset_adapters.common import (
    CLAIM_SCOPE_CHAT,
    PromptBlock,
    iter_raw_records,
    make_record,
    run_converter,
)


SOURCE_DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
OUTPUT_NAME = "sharegpt_prompt_workloads.jsonl"


def _conversation_blocks(record: dict[str, Any]) -> list[PromptBlock]:
    blocks: list[PromptBlock] = []
    conversations = record.get("conversations") or record.get("messages") or []
    if isinstance(conversations, list):
        last = conversations[-1] if conversations else None
        last_role = (
            str(last.get("from", last.get("role", ""))).lower()
            if isinstance(last, dict)
            else ""
        )
        target = (
            len(conversations) - 1
            if last_role in {"gpt", "assistant"}
            else len(conversations)
        )
        for turn in conversations[:target]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", "message"))).lower()
            text = turn.get("value", turn.get("content", turn.get("text", "")))
            if text is None or not str(text).strip():
                continue
            block_type = {
                "human": "user_message",
                "user": "user_message",
                "gpt": "assistant_message",
                "assistant": "assistant_message",
                "system": "system",
            }.get(role, "conversation_message")
            blocks.append(PromptBlock(block_type, str(text)))
    return blocks


def convert(source_path, limit: int, repo_id: str):
    records = []
    unavailable_fields: list[str] = ["timestamp", "dependency_edges", "ready_times"]
    for source_index, _, raw in iter_raw_records(source_path, limit=None):
        group_id = str(raw.get("id", raw.get("conversation_id", source_index)))
        blocks = _conversation_blocks(raw)
        if not blocks:
            unavailable_fields.append("conversations")
            continue
        records.append(
            make_record(
                request_id=f"sharegpt-{source_index:06d}",
                workflow="sharegpt-chat",
                blocks=blocks,
                source_dataset=repo_id,
                source_index=source_index,
                source_group_id=group_id,
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
        description="Convert ShareGPT to TokenMoE prompt workloads."
    )
    run_converter(
        parser=parser,
        adapter_name="sharegpt",
        source_dataset=SOURCE_DATASET,
        output_name=OUTPUT_NAME,
        claim_scope=CLAIM_SCOPE_CHAT,
        convert_fn=convert,
    )


if __name__ == "__main__":
    main()
