"""Runtime identities and reproducible hashes, shared by all harness producers."""

import hashlib
import json
import time
import uuid


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_id(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:16]


def new_id(prefix):
    if prefix not in {"run", "ses", "req", "tc", "eng", "grp", "seg"}:
        raise ValueError(f"Unknown runtime ID prefix: {prefix}")
    return f"{prefix}_{time.time_ns():016x}{uuid.uuid4().hex[:16]}"


def sampling_seed(base_seed, benchmark_item_id, step_index, attempt_id):
    value = canonical_json([base_seed, benchmark_item_id, step_index, attempt_id])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big")
