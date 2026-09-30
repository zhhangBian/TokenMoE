#!/usr/bin/env python3
"""Download the selected checkpoints and benchmark into explicit local paths."""

import argparse
import json
import os
from pathlib import Path

import yaml

MODEL_NAMES = ("gpt-oss-120b", "deepseek-v4-flash", "dots3-note")
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs/models"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--model-root", type=Path, help="Root containing <provider>/<model> directories"
    )
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--dataset-id", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--model-revision", action="append", default=[], metavar="MODEL=REVISION")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument(
        "--endpoint", help="Hugging Face endpoint; defaults to HF_ENDPOINT or the official endpoint"
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the download plan without network or writes"
    )
    args = parser.parse_args(argv)
    if args.skip_models and args.skip_dataset:
        parser.error("At least one of models or dataset must be selected")
    if not args.skip_models and args.model_root is None:
        parser.error("--model-root is required unless --skip-models is set")
    if not args.skip_dataset and args.dataset_dir is None:
        parser.error("--dataset-dir is required unless --skip-dataset is set")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    revisions = {}
    for setting in args.model_revision:
        model, separator, revision = setting.partition("=")
        if not separator or model not in args.models or not revision or model in revisions:
            parser.error(f"Invalid or repeated --model-revision: {setting}")
        revisions[model] = revision
    jobs = []
    if not args.skip_models:
        for model in dict.fromkeys(args.models):
            config = yaml.safe_load((CONFIG_ROOT / f"{model}.yaml").read_text())
            jobs.append(
                {
                    "repo_id": config["model_id"],
                    "repo_type": "model",
                    "revision": revisions.get(model, config.get("revision") or "main"),
                    "local_dir": str(
                        (args.model_root.expanduser().resolve() / config["model_id"]).resolve()
                    ),
                }
            )
    if not args.skip_dataset:
        jobs.append(
            {
                "repo_id": args.dataset_id,
                "repo_type": "dataset",
                "revision": args.dataset_revision,
                "local_dir": str(args.dataset_dir.expanduser().resolve()),
                "allow_patterns": ["*.parquet"],
            }
        )
    print(
        json.dumps({"downloads": jobs, "dry_run": args.dry_run}, ensure_ascii=False, indent=2),
        flush=True,
    )
    if args.dry_run:
        return 0
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    from huggingface_hub import snapshot_download

    for job in jobs:
        snapshot_download(
            **job,
            max_workers=args.max_workers,
            **({"endpoint": args.endpoint} if args.endpoint else {}),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
