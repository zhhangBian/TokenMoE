"""Verify local model/dataset assets before requesting any downloads."""

import hashlib
import json
import struct
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_model(directory):
    root = Path(directory)
    missing, damaged, shards = [], [], []
    metadata = {}
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]:
        path = root / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            metadata[name] = json.loads(path.read_text())
        except (ValueError, OSError):
            damaged.append(name)
    index = metadata.get("model.safetensors.index.json", {}).get("weight_map", {})
    if "model.safetensors.index.json" in metadata and not index:
        damaged.append("model.safetensors.index.json")
    for name in sorted(set(index.values())):
        path = root / name
        if Path(name).name != name:
            raise ValueError("Weight shard paths must be local file names")
        if not path.is_file():
            missing.append(name)
            continue
        size = path.stat().st_size
        try:
            with path.open("rb") as source:
                length = struct.unpack("<Q", source.read(8))[0]
                if not 0 < length <= min(size - 8, 64 * 1024 * 1024):
                    raise ValueError("Invalid safetensors header length")
                header = json.loads(source.read(length))
            for tensor, shard in index.items():
                if shard == name:
                    start, end = header[tensor]["data_offsets"]
                    if not 0 <= start < end <= size - 8 - length:
                        raise ValueError("Invalid tensor data offset")
            shards.append(
                {
                    "file": name,
                    "bytes": size,
                    "header_sha256": hashlib.sha256(
                        json.dumps(header, sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        except (ValueError, KeyError, TypeError, OSError, struct.error):
            damaged.append(name)
    return {
        "path": str(root.resolve()),
        "valid": not missing and not damaged,
        "missing": missing,
        "damaged": damaged,
        "shards": shards,
        "metadata_sha256": {name: sha256_file(root / name) for name in metadata},
    }


def inspect_dataset(directory, instances=None):
    import pyarrow.parquet as pq

    root = Path(directory)
    paths = [root] if root.is_file() else sorted(root.rglob("*.parquet"))
    files, identifiers, damaged = [], set(), []
    for path in paths:
        try:
            table = pq.read_table(path, columns=["instance_id", "problem_statement", "repo"])
            identifiers.update(table["instance_id"].to_pylist())
            files.append(
                {"path": str(path.resolve()), "rows": len(table), "sha256": sha256_file(path)}
            )
        except (ValueError, OSError, KeyError):
            damaged.append(str(path))
    missing_ids = sorted(set(instances or []) - identifiers)
    return {
        "path": str(root.resolve()),
        "valid": bool(files) and not damaged and not missing_ids,
        "files": files,
        "damaged": damaged,
        "missing_instances": missing_ids,
        "unique_tasks": len(identifiers),
        "revision": "unknown",
    }


def restore_asset(job, *, max_workers=8, endpoint=None):
    from huggingface_hub import snapshot_download

    options = {**job, "max_workers": max_workers}
    if endpoint:
        options["endpoint"] = endpoint
    root = Path(job["local_dir"])
    if job["repo_type"] == "dataset":
        report = inspect_dataset(root)
        if report["valid"]:
            return report
        if report["damaged"]:
            options["force_download"] = True
        snapshot_download(**options)
        report = inspect_dataset(root)
    else:
        report = inspect_model(root)
        if report["valid"]:
            return report
        names = report["missing"] + report["damaged"]
        # Fetch the index/tokenizer first when the local shard list is unknown.
        metadata = [name for name in names if not name.endswith(".safetensors")]
        if metadata:
            snapshot_download(
                **options, allow_patterns=metadata, force_download=bool(report["damaged"])
            )
            report = inspect_model(root)
        names = report["missing"] + report["damaged"]
        if names:
            snapshot_download(
                **options, allow_patterns=names, force_download=bool(report["damaged"])
            )
        report = inspect_model(root)
    if not report["valid"]:
        raise ValueError(f"Asset verification failed after download: {report}")
    return report
