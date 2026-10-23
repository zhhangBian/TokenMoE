"""Synthetic session around a real pytest tool for local controller qualification."""

import argparse
import json
from pathlib import Path

from tokenmoe_collect.events import emit
from tokenmoe_collect.executor import StreamingExecutor
from tokenmoe_collect.ids import new_id
from tokenmoe_collect.jsonl import write_json
from tokenmoe_collect.records import RecordStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads(args.predictor.read_text())["identity"]
    command = "/home/youwei/bzh/venvs/tokenmoe-harness/bin/python -m pytest /home/youwei/bzh/project/TokenMoE/optimization/tests -q"
    with RecordStore(args.output, control_events=True) as store:
        sid, tool = new_id("ses"), new_id("tc")
        emit(
            store,
            "session_started",
            session_id=sid,
            identity=identity,
            benchmark_item_id="synthetic-controller-mechanism",
        )
        emit(
            store,
            "tools_issued",
            session_id=sid,
            dependencies=[tool],
            dependency_mode="sequential",
            tools=[{"tool_id": tool, "command": command, "command_truncated": False}],
        )
        executor = StreamingExecutor(store, runtime="local", cwd=str(Path.cwd()), session_id=sid)
        result, timing = executor.execute(command, tool, timeout=30)
        emit(
            store,
            "session_ended",
            session_id=sid,
            status="succeeded" if result["returncode"] == 0 else "failed",
        )
    write_json(
        args.output / "fixture.json",
        {
            "schema_version": 1,
            "synthetic_session": True,
            "real_tool": "pytest",
            "agent_task_outcome": None,
            "result": result,
            "timing": timing,
        },
    )


if __name__ == "__main__":
    main()
