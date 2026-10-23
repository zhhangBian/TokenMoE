"""Causal dependency timing diagnostics; observations never supply future lead."""

import heapq
from collections import defaultdict
from math import isfinite

from .replay_cache import CacheModel


def simulate(nodes, *, capacity, costs, policy="lru", native_service=None, initial_snapshot=None):
    nodes = list(nodes)
    index = {node["id"]: node for node in nodes}
    order = {node["id"]: i for i, node in enumerate(nodes)}
    if len(index) != len(nodes):
        raise ValueError("Duplicate timing node identity")
    successors, waiting = defaultdict(list), {}
    queue, finished, timings = [], {}, {}
    for node in nodes:
        parents = node.get("dependencies", [])
        if len(set(parents)) != len(parents) or any(p not in index for p in parents):
            raise ValueError("Duplicate or unknown timing dependency")
        if parents and "release_at_seconds" in node:
            raise ValueError("Dependent nodes cannot reuse absolute observation timestamps")
        if "time_seconds" in node:
            raise ValueError("Convert recorded times into causal dependencies and lags")
        lag = node.get("lag_seconds", 0)
        if not isfinite(lag) or lag < 0:
            raise ValueError("Invalid causal lag")
        waiting[node["id"]] = len(parents)
        for parent in parents:
            successors[parent].append(node["id"])
        if not parents:
            ready = node.get("release_at_seconds", 0) + lag
            if not isfinite(ready) or ready < 0:
                raise ValueError("Invalid exogenous release time")
            heapq.heappush(
                queue, (ready, node["kind"] == "prefetch", order[node["id"]], node["id"])
            )
    cache = CacheModel(capacity, costs, policy=policy)
    if initial_snapshot is not None:
        cache.initialize(initial_snapshot["slots"], initial_snapshot.get("previously_loaded", ()))
    while queue:
        ready, _, _, identifier = heapq.heappop(queue)
        node = index[identifier]
        kind = node["kind"]
        duration = node.get("duration_seconds", 0)
        if kind == "model" and "service_profile_index" in node:
            if native_service is None or "duration_seconds" in node:
                raise ValueError("Model service references require a matched native profile")
            position = node["service_profile_index"]
            if type(position) is not int or not 0 <= position < len(native_service["requests"]):
                raise ValueError("Unknown measured model-service request")
            service = native_service["requests"][position]
            totals = defaultdict(int)
            for layer in node["layers"]:
                if "tokens" in layer:
                    count = len(layer["tokens"])
                elif layer.get("token_count_source") == "validated_compact_feedback":
                    count = layer["observed_token_count"]
                    if type(count) is not int or count < 0:
                        raise ValueError("Invalid observed model-work count")
                else:
                    raise ValueError("Service references require authoritative model-work counts")
                totals[layer["layer"]] += count
            if (
                set(totals) != set(native_service["layers"])
                or set(totals.values()) != {service["computed_tokens"]}
                or any(node.get(key) != service[key] for key in ("input_tokens", "output_tokens"))
            ):
                raise ValueError(
                    "Model work differs from measured native-service token/layer shape"
                )
            duration = service["model_service_seconds"]
        if not isfinite(duration) or duration < 0:
            raise ValueError("Invalid declared service duration")
        if kind == "model":
            if not node["layers"]:
                raise ValueError("Model nodes require authoritative expert demands")
            if (
                "duration_seconds" not in node
                and "service_profile_index" not in node
                and not all("compute_seconds" in layer for layer in node["layers"])
            ):
                raise ValueError("Model service durations must be explicitly declared")
            start = max(ready, cache.model_clock)
            if duration and any(layer.get("compute_seconds", 0) for layer in node["layers"]):
                raise ValueError("Model service and per-layer compute cannot be counted twice")
            cache.counts["queue_seconds"] += start - ready
            end = start
            for layer in node["layers"]:
                if "time_seconds" in layer:
                    raise ValueError("Layer timestamps cannot replace causal service dependencies")
                end = cache.demand(layer, max(end, start))
            if duration:
                cache._advance(end + duration, busy=True)
                cache.model_clock = cache.time
                end = cache.time
        elif kind == "prefetch":
            start, end = ready, ready + duration
            lease = node["lease_seconds"]
            if not isfinite(lease) or lease <= 0:
                raise ValueError("Invalid speculative plan lease")
            cache.prefetch({**node, "expires_at_seconds": end + lease}, end)
        elif kind in {"tool", "join", "return"}:
            if kind == "tool" and "duration_seconds" not in node:
                raise ValueError("Tool service durations must be explicitly declared")
            start, end = ready, ready + duration
        else:
            raise ValueError(f"Unsupported timing node: {kind}")
        if "deadline_seconds" in node and end > node["deadline_seconds"]:
            raise ValueError(
                "Timing crosses a workload deadline; explicit cancellation/branch modeling is required"
            )
        finished[identifier] = end
        timings[identifier] = {"kind": kind, "ready": ready, "started": start, "finished": end}
        for child in successors[identifier]:
            waiting[child] -= 1
            if waiting[child] == 0:
                child_node = index[child]
                at = max(finished[p] for p in child_node["dependencies"]) + child_node.get(
                    "lag_seconds", 0
                )
                heapq.heappush(queue, (at, child_node["kind"] == "prefetch", order[child], child))
    if len(finished) != len(nodes):
        raise ValueError("Timing dependencies contain a cycle")
    return {
        "schema_version": 1,
        "evidence": "uncalibrated-causal-timing-diagnostic",
        "calibrated_timing": False,
        "nodes": timings,
        "cache": cache.result(max(finished.values(), default=0)),
        "assumptions": [
            "fixed routing and token trajectories",
            "independent declared tool durations",
            "serialized shared copy link; no model/copy overlap calibration",
            "dependent arrivals follow simulated completion, never recorded future timestamps",
            "deadline-induced trajectory changes require explicit modeling",
        ],
    }
