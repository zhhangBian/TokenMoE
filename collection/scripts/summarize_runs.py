#!/usr/bin/env python3
"""Summarize automated acceptance checks without treating them as model scores."""

import argparse
import json
from collections import Counter
from pathlib import Path


def rate(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Rate must be between 0 and 1")
    return number


def summarize(root, min_success_rate=0.9, min_location_rate=0.99):
    root = Path(root).expanduser().resolve()
    if (root / "runtime").is_dir():
        directories = [root]
    else:
        # The managed runner writes this only after every configured concurrency.
        manifest = json.loads((root / "pilot_summary.json").read_text())
        directories = [root / Path(item["run_dir"]).name for item in manifest]
    reports = []
    for directory in directories:
        sessions = json.loads((directory / "run_summary.json").read_text())["sessions"]
        validation = json.loads((directory / "validation.json").read_text())
        alignment = json.loads((directory / "alignment.json").read_text())
        counts = Counter(row["status"] for row in sessions)
        success_rate = counts["succeeded"] / len(sessions) if sessions else 0.0
        passed = (
            bool(sessions)
            and validation["passed"]
            and success_rate >= min_success_rate
            and alignment["location_rate"] >= min_location_rate
        )
        reports.append(
            {
                "run_dir": str(directory),
                "sessions": len(sessions),
                "statuses": dict(counts),
                "success_rate": success_rate,
                "location_rate": alignment["location_rate"],
                "validation_passed": validation["passed"],
                "automated_checks_passed": passed,
            }
        )
    return {
        "input_dir": str(root),
        "runs": reports,
        "automated_checks_passed": bool(reports)
        and all(r["automated_checks_passed"] for r in reports),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="A managed pilot/full directory or one finalized N directory; repeat for multiple models",
    )
    parser.add_argument("--min-success-rate", type=rate, default=0.9)
    parser.add_argument("--min-location-rate", type=rate, default=0.99)
    parser.add_argument("--output", type=Path, help="Optional new JSON report file")
    args = parser.parse_args(argv)
    reports = []
    for root in args.run_dir:
        try:
            reports.append(summarize(root, args.min_success_rate, args.min_location_rate))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            reports.append(
                {"input_dir": str(root), "automated_checks_passed": False, "error": str(exc)}
            )
    result = {
        "reports": reports,
        "automated_checks_passed": all(r["automated_checks_passed"] for r in reports),
        "thresholds": {
            "success_rate": args.min_success_rate,
            "location_rate": args.min_location_rate,
        },
        "manual_checks_required": [
            "Tool-output/observation consistency",
            "Recorder overhead below 5%",
        ],
        "note": "succeeded means the agent submitted, not that SWE-bench tests passed",
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            destination.write(text + "\n")
    return 0 if result["automated_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
