#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONVERTERS = (
    "sharegpt",
    "lmsys",
    "swe_agent",
    "opencode",
    "openmath",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all TokenMoE dataset converters.")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--dataset-root", default="/home/youwei/bzh/dataset")
    parser.add_argument(
        "--output-dir",
        default="/home/youwei/bzh/dataset/tokenmoe_artifacts/workloads",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_names = {
        "sharegpt": "sharegpt_prompt_workloads.jsonl",
        "lmsys": "lmsys_prompt_workloads.jsonl",
        "swe_agent": "swe_agent_prompt_workloads.jsonl",
        "opencode": "opencode_prompt_workloads.jsonl",
        "openmath": "openmath_prompt_workloads.jsonl",
    }
    for name in CONVERTERS:
        cmd = [
            sys.executable,
            "-m",
            f"dataset_adapters.{name}",
            "--limit",
            str(args.limit),
            "--dataset-root",
            args.dataset_root,
            "--output",
            str(output_dir / output_names[name]),
        ]
        print("running", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
