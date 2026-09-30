"""Join raw producers and materialize the collection storage layout."""

import errno
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from .align import align_prompt
from .jsonl import JsonlWriter, read_jsonl, write_json


def link_file(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        if not os.path.samefile(source, target):
            raise
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.copy2(source, target)


def write_table(path, rows):
    # Empty tables still produce a readable parquet file.
    pq.write_table(pa.Table.from_pylist(rows), path)


def finalize(run_dir):
    root = Path(run_dir).resolve()
    source = json.loads((root / "engine_source.json").read_text())
    engine_src = Path(source["path"])
    engine_dir = root / "raw/engine" / source["engine_instance_id"]
    engine_dir.mkdir(parents=True, exist_ok=True)
    for name in ["engine_meta.json", "layer_map.json", "requests.jsonl", "steps.jsonl"]:
        target = engine_dir / name
        # Snapshot append-only JSONL only after the sessions finish.
        shutil.copy2(engine_src / name, target)
    harness = {}
    for table in [
        "application_runs",
        "sessions",
        "llm_requests",
        "tool_calls",
        "tool_output_chunks",
        "host_load",
    ]:
        harness[table] = [
            record
            for path in sorted((root / "raw/harness").glob(f"*/{table}.jsonl"))
            for record in read_jsonl(path)
        ]
    metadata = json.loads((engine_dir / "engine_meta.json").read_text())
    profile = next(read_jsonl(root / "static/model_profiles.jsonl"))
    tokenizer = Tokenizer.from_file(profile["tokenizer_path"])
    engine_requests = {
        row["llm_request_id"]: row
        for row in read_jsonl(engine_dir / "requests.jsonl")
        if row.get("llm_request_id")
    }
    runtime = root / "runtime"
    if runtime.exists():
        raise FileExistsError("Runtime already finalized; use a new output directory")
    runtime.mkdir()
    (runtime / "prompts").mkdir()
    requests, segments = [], []
    reconstructed = {}
    alignment = {"message_fields": 0, "located": 0, "failed": 0}
    for raw in sorted(
        harness["llm_requests"], key=lambda r: (r["request_created_at"], r["llm_request_id"])
    ):
        rid = raw["llm_request_id"]
        if raw["messages_mode"] == "delta":
            base_messages, base_provenance = reconstructed[raw["messages_base_request_id"]]
            if len(base_messages) != raw["messages_base_count"]:
                raise ValueError(f"Invalid message delta base for {rid}")
        else:
            base_messages, base_provenance = [], []
        messages = base_messages + raw["messages_delta"]
        provenance = base_provenance + raw["message_provenance"]
        reconstructed[rid] = (messages, provenance)
        row = {k: v for k, v in raw.items() if k not in {"messages_delta", "message_provenance"}}
        row.update(
            model_profile_id=profile["model_profile_id"],
            engine_instance_id=metadata["engine_instance_id"],
            prompt_ref=None,
            output_ref=None,
            routing_ref=None,
        )
        engine = engine_requests.get(rid)
        if engine is None:
            row["request_status"] = "engine_record_missing"
        else:
            row.update(engine)
            if engine.get("routing_file"):
                saved = engine_dir / engine["routing_file"]
                link_file(engine_src / engine["routing_file"], saved)
                route = runtime / "routing" / profile["model_profile_id"] / f"{rid}.npz"
                link_file(saved, route)
                row["routing_ref"] = str(route.relative_to(root))
                with np.load(route, allow_pickle=False) as data:
                    tokens = data["token_ids"].tolist()
                    prompt_tokens = tokens[: row["num_prompt_tokens"]]
                    prompt, request_segments, metrics = align_prompt(
                        tokenizer, prompt_tokens, messages, provenance, rid
                    )
                    segments.extend(request_segments)
                    for key in alignment:
                        alignment[key] += metrics[key]
                    output = tokenizer.decode(
                        tokens[row["num_prompt_tokens"] :], skip_special_tokens=False
                    )
                for kind, text in [("prompt", prompt), ("output", output)]:
                    path = runtime / "prompts" / f"{rid}.{kind}.txt"
                    path.write_text(text, encoding="utf-8")
                    row[f"{kind}_ref"] = str(path.relative_to(root))
        requests.append(row)
    request_index = {r["llm_request_id"]: r for r in requests}
    consuming = {}
    for request in requests:
        for tcid in request["prev_tool_call_ids"]:
            consuming.setdefault(tcid, []).append(request["llm_request_id"])
    tools = []
    for raw in harness["tool_calls"]:
        row = dict(raw)
        issuing = request_index[row["llm_request_id_issuing"]]
        row["issued_at"] = issuing.get("inference_finished_at")
        row["llm_request_ids_consuming"] = consuming.get(row["tool_call_id"], [])
        target = runtime / "tool_outputs" / f"{row['tool_call_id']}.out"
        link_file(root / raw["output_ref"], target)
        row["output_ref"] = str(target.relative_to(root))
        target = runtime / "tool_args" / f"{row['tool_call_id']}.txt"
        link_file(root / raw["tool_args_ref"], target)
        row["tool_args_ref"] = str(target.relative_to(root))
        tools.append(row)
    steps = []
    for step in read_jsonl(engine_dir / "steps.jsonl"):
        steps.append(
            step
            | {
                "engine_instance_id": metadata["engine_instance_id"],
                "step_id": f"{metadata['engine_instance_id']}:{step['step_index']}",
            }
        )
    for name, rows in [
        ("application_runs", harness["application_runs"]),
        ("sessions", harness["sessions"]),
        ("llm_requests", requests),
        ("tool_calls", tools),
    ]:
        with JsonlWriter(runtime / f"{name}.jsonl") as writer:
            for row in rows:
                writer.write(row)
    for name, rows in [
        ("prompt_segments", segments),
        ("tool_output_chunks", harness["tool_output_chunks"]),
        ("engine_steps", steps),
        ("host_load", harness["host_load"]),
    ]:
        if name == "engine_steps":
            # Parquet structs preserve heterogeneous entry fields.
            rows = [
                row
                | {
                    "entries": [
                        dict(
                            zip(
                                [
                                    "llm_request_id",
                                    "vllm_request_id",
                                    "phase",
                                    "token_start",
                                    "token_end",
                                ],
                                entry,
                            )
                        )
                        for entry in row["entries"]
                    ]
                }
                for row in rows
            ]
        write_table(runtime / f"{name}.parquet", rows)
    write_json(
        root / "alignment.json",
        alignment
        | {
            "location_rate": alignment["located"] / alignment["message_fields"]
            if alignment["message_fields"]
            else 1.0
        },
    )
    from .validate import validate

    return validate(root)
