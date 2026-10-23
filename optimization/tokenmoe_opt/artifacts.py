"""Versioned, immutable experiment inputs and exclusive output directories."""

import hashlib
import json
import subprocess
from pathlib import Path

from tokenmoe_collect.assets import sha256_file
from tokenmoe_collect.jsonl import write_json

SCHEMA_VERSION = 1
KINDS = {"input", "split", "predictor", "cost-profile", "experiment", "result"}
MILESTONES = ("analysis", "backend", "local_evaluation", "workload_expansion", "migration")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def source_version(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    files = sorted(
        p
        for directory in ("optimization", "collection/tokenmoe_collect", "vllm/vllm")
        for p in (root / directory).rglob("*")
        if p.is_file() and (p.suffix in {".py", ".toml", ".json"})
    )
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "files": {str(p.relative_to(root)): sha256_file(p) for p in files},
        "fork_commit": git("-C", "vllm", "rev-parse", "HEAD"),
    }


def manifest(kind, identity, **fields):
    if kind not in KINDS:
        raise ValueError(f"Unknown manifest kind: {kind}")
    if not identity:
        raise ValueError("A model/template identity is required")
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "identity": identity, **fields}


def read_manifest(path, kind):
    value = json.loads(Path(path).read_text())
    if value.get("schema_version") != SCHEMA_VERSION or value.get("kind") != kind:
        raise ValueError(f"Unsupported {kind} manifest: {path}")
    if not value.get("identity"):
        raise ValueError("Missing model/template identity")
    return value


class Output:
    def __init__(self, path, *, kind, identity, command):
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self.record = manifest(
            kind,
            identity,
            command=command,
            source=source_version(),
            status="running",
            milestones=dict.fromkeys(MILESTONES, "outstanding"),
        )
        self.write("manifest.json", self.record)

    def write(self, name, value):
        write_json(self.path / name, value)

    def finish(self, **fields):
        self.record.update(status="complete", **fields)
        self.write("manifest.json", self.record)

    def fail(self, error):
        self.record.update(status="failed", error=f"{type(error).__name__}: {error}")
        self.write("manifest.json", self.record)


def input_files(roots):
    """Hash every consumed finalized record, routing, label and validation input."""
    result = {}
    for root in roots:
        paths = [root / "validation.json"]
        for directory in ("static", "runtime", "derived/labels_v1"):
            paths.extend(p for p in (root / directory).rglob("*") if p.is_file())
        result[str(root)] = {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}
    return result
