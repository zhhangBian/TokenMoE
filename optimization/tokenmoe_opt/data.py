"""Read-only validated first-computed routes with explicit operational limits."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tokenmoe_collect.jsonl import read_jsonl
from tokenmoe_collect.report import parquet_rows
from tokenmoe_collect.validate import validate

from .artifacts import digest
from .features import LABEL_VERSION, request_features

PHASES = ("new_prefill", "decode", "first32_decode")


@dataclass
class Sample:
    task: str
    session: str
    request: str
    decision: int
    observed: int
    hist: np.ndarray
    recent: np.ndarray
    features: dict
    root: Path
    metadata: dict


def histogram(experts, num_experts):
    return np.stack(
        [
            np.bincount(experts[:, layer].ravel(), minlength=num_experts)
            for layer in range(experts.shape[1])
        ]
    )


def phase_histograms(positions, experts, prompt, num_experts, *, recent_window=32):
    masks = [positions < prompt, positions >= prompt]
    decode_indices = np.flatnonzero(masks[1])[:32]
    early = np.zeros(len(positions), dtype=bool)
    early[decode_indices] = True
    masks.append(early)
    targets, recent = [], []
    for mask in masks:
        routes = experts[mask]
        targets.append(histogram(routes, num_experts))
        recent.append(histogram(routes[-recent_window:], num_experts))
    return np.stack(targets), np.stack(recent)


def model_identity(profile, template):
    return {
        "model_id": profile["model_id"],
        "weight_revision": profile["revision_hash"],
        "tokenizer_hash": profile["tokenizer_hash"],
        "template_id": template["agent_template_id"],
        "template_version": template["version_hash"],
        "layers": profile["moe_layer_ids"],
        "experts": profile["num_routed_experts"],
        "top_k": profile["top_k"],
    }


def load_runs(roots, *, label_version=LABEL_VERSION):
    samples, identities, summaries, ids = [], set(), [], set()
    entity_ids = set()
    for root in map(lambda p: Path(p).resolve(), roots):
        report = validate(root, write_report=False)
        if not report["passed"]:
            failed = [k for k, value in report["rules"].items() if value["status"] == "failed"]
            raise ValueError(f"Invalid collection {root}: rules {failed}")
        profiles = {
            r["model_profile_id"]: r for r in read_jsonl(root / "static/model_profiles.jsonl")
        }
        sessions = {r["session_id"]: r for r in read_jsonl(root / "runtime/sessions.jsonl")}
        runs = {
            r["application_run_id"]: r for r in read_jsonl(root / "runtime/application_runs.jsonl")
        }
        tools = {r["tool_call_id"]: r for r in read_jsonl(root / "runtime/tool_calls.jsonl")}
        for identifier in [*sessions, *runs, *tools]:
            if identifier in entity_ids:
                raise ValueError(f"Duplicate runtime entity ID across inputs: {identifier}")
            entity_ids.add(identifier)
        meta = json.loads((root / "static/engine_meta.json").read_text())
        clocks = {
            p["clock_domain_id"]
            for p in (
                json.loads(path.read_text())
                for path in (root / "raw/harness").glob("*/producer.json")
            )
        }
        clocks.add(meta["clock_domain_id"])
        if len(clocks) != 1:
            raise ValueError(f"Incomparable clock domains: {root}")
        counts, seen, keys = Counter(), set(), {}
        for request in sorted(
            read_jsonl(root / "runtime/llm_requests.jsonl"), key=lambda r: r["request_created_at"]
        ):
            sid, rid = request["session_id"], request["llm_request_id"]
            if rid in ids:
                raise ValueError(f"Duplicate request ID across inputs: {rid}")
            ids.add(rid)
            session = sessions[sid]
            profile = profiles[request["model_profile_id"]]
            identity = model_identity(profile, session)
            identities.add(json.dumps(identity, sort_keys=True))
            features = request_features(
                request,
                tools,
                root,
                first_request=sid not in seen,
                previous_key=keys.get(sid),
                label_version=label_version,
            )
            features["clock_domain_id"] = clocks.copy().pop()
            seen.add(sid)
            if features["tool_key"]:
                keys[sid] = features["tool_key"]
            counts["attempted_requests"] += 1
            if request.get("engine_admission_status") == "rejected_before_engine":
                counts["pre_engine_rejections"] += 1
                continue
            with np.load(root / request["routing_ref"], allow_pickle=False) as route:
                positions, experts = route["token_positions"], route["experts"]
                if not np.array_equal(route["layer_ids"], profile["moe_layer_ids"]):
                    raise ValueError(f"Layer identity mismatch: {rid}")
                hist, recent = phase_histograms(
                    positions, experts, request["num_prompt_tokens"], profile["num_routed_experts"]
                )
                counts["first_computed_positions"] += len(positions)
                counts["partial_requests"] += not bool(route["routing_complete"])
            boundaries = [
                request.get(key) for key in ("response_received_at", "inference_finished_at")
            ]
            observed = max((t for t in boundaries if t is not None), default=None)
            if observed is None:
                # A finalization timestamp is an upper bound for unfinished requests.
                observed = session["finished_at"]
            if observed is None or observed < request["request_created_at"]:
                raise ValueError(f"No causal observation boundary: {rid}")
            samples.append(
                Sample(
                    task=runs[session["application_run_id"]]["benchmark_item_id"],
                    session=sid,
                    request=rid,
                    decision=request["request_created_at"],
                    observed=observed,
                    hist=hist,
                    recent=recent,
                    features=features,
                    root=root,
                    metadata=request,
                )
            )
        for step in parquet_rows(root / "runtime/engine_steps.parquet"):
            for entry in step["entries"]:
                if entry["phase"] == "recompute":
                    counts["recompute_positions"] += entry["token_end"] - entry["token_start"]
        summaries.append(
            {
                "root": str(root),
                **counts,
                "validation": report,
                "operational_replay": "missing-recompute-routes"
                if counts["recompute_positions"]
                else "first-computed-routes-only",
            }
        )
    if len(identities) != 1:
        raise ValueError("Inputs must have one exact model/weight/tokenizer/template identity")
    identity = json.loads(identities.pop())
    identity["id"] = digest(identity)
    return samples, identity, summaries
