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


SOURCE_DATASET = "nvidia/OpenMathInstruct-2"
OUTPUT_NAME = "openmath_prompt_workloads.jsonl"


def convert(source_path, limit: int, repo_id: str):
    records = []
    unavailable_fields: list[str] = ["dependency_edges", "ready_times"]
    for source_index, _, raw in iter_raw_records(source_path, limit=None):
        problem, problem_field = first_text(
            raw,
            ("problem", "question", "instruction", "prompt", "messages"),
        )
        solution, _ = first_text(raw, ("solution", "answer", "output", "response"))
        group_id, group_field = first_text(raw, ("id", "problem_id", "source", "task_id"))
        missing = []
        if problem_field is None:
            missing.append("problem")
        if group_field is None:
            missing.append("source_group_id")
        if not problem:
            unavailable_fields.extend(missing)
            continue
        blocks = [
            PromptBlock("system", "Math instruction workload."),
            PromptBlock("instruction", problem),
        ]
        if solution:
            blocks.append(PromptBlock("reference_answer", solution))
        records.append(
            make_record(
                request_id=f"openmath-{source_index:06d}",
                workflow="openmath-instruction",
                blocks=blocks,
                source_dataset=repo_id,
                source_index=source_index,
                source_group_id=group_id or str(source_index),
                claim_scope=CLAIM_SCOPE_DOMAIN,
                role="solver",
                phase="solve",
                graph_node_type="domain_instruction",
                unavailable_fields=missing + ["dependency_edges", "ready_times"],
            )
        )
        if len(records) >= limit:
            break
    return records, unavailable_fields, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenMathInstruct-2 to TokenMoE prompt workloads.")
    run_converter(
        parser=parser,
        adapter_name="openmath",
        source_dataset=SOURCE_DATASET,
        output_name=OUTPUT_NAME,
        claim_scope=CLAIM_SCOPE_DOMAIN,
        convert_fn=convert,
    )


if __name__ == "__main__":
    main()
