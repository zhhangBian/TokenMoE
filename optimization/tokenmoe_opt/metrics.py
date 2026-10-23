"""Metrics at observed engine/client boundaries, with explicit missing coverage."""

import json
from collections import defaultdict
from pathlib import Path

from tokenmoe_collect.jsonl import read_jsonl
from tokenmoe_collect.report import distribution

from .validation import validate_cache_events


def token_metrics(events):
    requests = defaultdict(lambda: {"tokens": {}, "batches": {}})
    clocks = set()
    for event in events:
        clocks.add(event["clock_domain_id"])
        rid = event.get("request_id")
        if rid is None:
            continue
        row = requests[rid]
        if event["kind"] == "request_received":
            row["received_at"] = event["source_at"]
            row["num_prompt_tokens"] = event.get("num_prompt_tokens")
            row["llm_request_id"] = event.get("llm_request_id")
        elif event["kind"] == "request_finished":
            row.update(
                finished_at=event["source_at"],
                status=event["status"],
                expected_tokens=event["num_output_tokens"],
            )
        elif event["kind"] == "token_readiness":
            batch = event["processing_batch"]
            old = row["batches"].setdefault(batch, event["source_at"])
            if old != event["source_at"]:
                raise ValueError("One processing batch has inconsistent observation times")
            for index in event["token_indices"]:
                if index in row["tokens"]:
                    raise ValueError(f"Duplicate token-readiness index: {rid}/{index}")
                row["tokens"][index] = (event["source_at"], batch)
    if len(clocks) > 1:
        raise ValueError("Token metrics require comparable clocks")
    result = []
    for rid, row in requests.items():
        tokens = row.pop("tokens")
        indices = sorted(tokens)
        first = tokens.get(0, (None, None))[0]
        last = tokens[indices[-1]][0] if indices else None
        received = row.get("received_at")
        complete = indices == list(range(row.get("expected_tokens", -1)))
        singleton_batches = len(set(batch for _, batch in tokens.values())) == len(tokens)
        ordered = [tokens[i][0] for i in indices]
        if any(b < a for a, b in zip(ordered, ordered[1:])):
            raise ValueError("Token indices regress in observed time")
        result.append(
            {
                "request_id": rid,
                **row,
                "observed_tokens": len(tokens),
                "first_token_at": first,
                "last_token_at": last,
                "token_observation_times_ns": ordered,
                "complete_token_coverage": complete,
                "engine_ttft_seconds": (first - received) / 1e9
                if first is not None and received is not None
                else None,
                "ttft_reason": None
                if first is not None and received is not None
                else "Missing engine receipt or token index zero",
                "endpoint_average_tpot_seconds": (last - first) / 1e9 / (len(tokens) - 1)
                if first is not None and last is not None and len(tokens) > 1 and complete
                else None,
                "observed_itl_seconds": [(b - a) / 1e9 for a, b in zip(ordered, ordered[1:])]
                if complete and singleton_batches
                else None,
                "itl_reason": None
                if complete and singleton_batches
                else "Incomplete indices or batched token delivery; no interpolation",
                "observation_point": "scheduler-output-processing-host",
                "network_delivery": None,
            }
        )
    return result


class Transfers:
    def __init__(self, window=None):
        self.started = self.completed = self.device_ns = self.evicted_unused = 0
        self.recovery_ns = self.sync_ns = 0
        self.recovery_samples = self.sync_samples = 0
        self.prefetched = {}
        self.window = window
        self.admitted = set()

    def observe(self, event):
        if self.window is not None and not self.window[0] <= event["source_at"] <= self.window[1]:
            return
        kind = event["kind"]
        if kind == "demand_ready":
            if self.window is not None and event["demand_at"] < self.window[0]:
                return
            self.recovery_ns += event["duration_ns"]
            self.recovery_samples += 1
        elif kind == "demand_sync":
            self.sync_ns += event["duration_ns"]
            self.sync_samples += 1
        if "slot" not in event:
            return
        key = (event.get("layer"), event["slot"], event["generation"])
        kind = event["kind"]
        if kind == "load_start":
            self.started += event["bytes"]
            self.admitted.add(key)
            if event["trigger"] == "prefetch":
                self.prefetched[key] = event["bytes"]
        elif kind == "load_complete":
            if self.window is not None and key not in self.admitted:
                return
            self.admitted.discard(key)
            self.completed += 1
            self.device_ns += event["device_duration_ns"]
        elif kind == "use":
            self.prefetched.pop(key, None)
        elif kind == "eviction":
            self.evicted_unused += self.prefetched.pop(key, 0)


def cache_metrics(observations, *, window=None):
    cache = []
    for path in sorted(Path(observations).glob("*/cache-rank-*.jsonl")):
        accumulator = Transfers(window)

        def observe():
            for event in read_jsonl(path):
                accumulator.observe(event)
                yield event

        status_path = path.with_suffix(".status.json")
        status = json.loads(status_path.read_text()) if status_path.exists() else None
        validation = validate_cache_events(observe(), status=status, expected_layers=range(48))
        cache.append(
            {
                "source": str(path),
                "counter_window_ns": window,
                "validation": validation,
                "bytes_started": accumulator.started
                if validation["exact_replay_available"]
                else None,
                "device_copy_seconds": accumulator.device_ns / 1e9
                if validation["exact_replay_available"]
                else None,
                "evicted_unused_prefetch_bytes": accumulator.evicted_unused
                if validation["exact_replay_available"]
                else None,
                "resident_unused_prefetch_bytes_at_end": sum(accumulator.prefetched.values())
                if validation["exact_replay_available"]
                else None,
                "counter_coverage": "complete"
                if validation["exact_replay_available"]
                else "incomplete-or-compact",
                "demand_recovery_seconds": accumulator.recovery_ns / 1e9
                if validation["exact_replay_available"] and accumulator.recovery_samples
                else None,
                "router_host_sync_seconds": accumulator.sync_ns / 1e9
                if validation["exact_replay_available"] and accumulator.sync_samples
                else None,
                "timing_scope": "per-rank host recovery and synchronization; overlaps copy duration",
            }
        )
    return cache


def resource_metrics(directory, *, window=None):
    root = Path(directory)
    if not (root / "peaks.json").exists():
        return None
    result = json.loads((root / "peaks.json").read_text())
    aggregate, phase_aggregate = 0, defaultdict(int)
    devices = [str(d["index"]) for d in result["metadata"]["devices"]]
    selected = {
        "samples": 0,
        "device_bytes": dict.fromkeys(devices, 0),
        "aggregate_device_bytes": 0,
        "process_tree_rss": 0,
    }
    for row in read_jsonl(root / "samples.jsonl"):
        values = row["gpu_mem_used"]
        if values is not None:
            aggregate = max(aggregate, sum(values))
            phase_aggregate[row["phase"]] = max(phase_aggregate[row["phase"]], sum(values))
        if window is None or not window[0] <= row["time"] <= window[1]:
            continue
        selected["samples"] += 1
        selected["process_tree_rss"] = max(
            selected["process_tree_rss"], sum(row["process_rss"].values())
        )
        if values is not None:
            selected["aggregate_device_bytes"] = max(
                selected["aggregate_device_bytes"], sum(values)
            )
            for device, value in zip(devices, values):
                selected["device_bytes"][device] = max(selected["device_bytes"][device], value)
    result.update(
        sampled_aggregate_device_peak_bytes=aggregate,
        phase_aggregate_device_peak_bytes=dict(phase_aggregate),
        measured_window_ns=window,
        measured=selected if window is not None and selected["samples"] else None,
    )
    return result


def extract_run(directory):
    root = Path(directory)
    run = json.loads((root / "result.json").read_text())
    tokens = []
    for path in sorted((root / "observations").glob("*/token-readiness.jsonl")):
        tokens.extend(token_metrics(read_jsonl(path)))
    cache = cache_metrics(root / "observations")
    resources = run.get("resources")
    return {
        "schema_version": 1,
        "evidence": "live-diagnostic-not-controlled",
        "requests": tokens,
        "engine_ttft_seconds": distribution([r["engine_ttft_seconds"] for r in tokens]),
        "endpoint_average_tpot_seconds": distribution(
            [r["endpoint_average_tpot_seconds"] for r in tokens]
        ),
        "cache": cache,
        "memory": resources,
        "torch_startup": run.get("startup_memory"),
        "torch_steady": run.get("final_memory"),
        "task_outcomes": None,
        "task_outcome_reason": "Qualification prompts are not agent task executions",
        "eligible_for_policy_comparison": False,
        "warmup_boundaries": None,
        "warmup_reason": "Smoke trials do not freeze three measured task repetitions",
        "limitations": [
            "Client network token delivery is unobserved",
            "No interpolation of batched observations",
            "Resource peaks are sampled at the recorded interval",
            "Qualification probes add overhead when enabled",
        ],
    }
