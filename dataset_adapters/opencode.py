#!/usr/bin/env python3
from __future__ import annotations

import argparse

from dataset_adapters.common import (
    CLAIM_SCOPE_DOMAIN,
    PromptBlock,
    first_text,
    iter_raw_records,
    make_record,
    run_converter,
)


SOURCE_DATASET = "nvidia/OpenCodeInstruct"
OUTPUT_NAME = "opencode_prompt_workloads.jsonl"


def convert(source_path, limit: int, repo_id: str):
    records = []
    unavailable_fields: list[str] = ["dependency_edges", "ready_times"]
    for source_index, _, raw in iter_raw_records(source_path, limit=None):
        instruction, instruction_field = first_text(
            raw,
            ("instruction", "prompt", "question", "problem", "input", "messages"),
        )
        language, _ = first_text(raw, ("language", "lang", "programming_language"))
        group_id, group_field = first_text(
            raw, ("id", "problem_id", "source", "task_id")
        )
        missing = []
        if instruction_field is None:
            missing.append("instruction")
        if group_field is None:
            missing.append("source_group_id")
        blocks = [PromptBlock("instruction", instruction or "")]
        if language:
            blocks.append(PromptBlock("code_context", f"Language: {language}"))
        if not instruction:
            unavailable_fields.extend(missing)
            continue
        records.append(
            make_record(
                request_id=f"opencode-{source_index:06d}",
                workflow="opencode-instruction",
                blocks=blocks,
                source_dataset=repo_id,
                source_index=source_index,
                source_group_id=group_id or str(source_index),
                claim_scope=CLAIM_SCOPE_DOMAIN,
                role="coder",
                phase="act",
                graph_node_type="domain_instruction",
            )
        )
        if len(records) >= limit:
            break
    return records, unavailable_fields


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenCodeInstruct to TokenMoE prompt workloads."
    )
    run_converter(
        parser=parser,
        adapter_name="opencode",
        source_dataset=SOURCE_DATASET,
        output_name=OUTPUT_NAME,
        claim_scope=CLAIM_SCOPE_DOMAIN,
        convert_fn=convert,
    )


if __name__ == "__main__":
    main()
