"""CPU process-tree and selected-device sampling using the collection sampler."""

import json
import signal
import threading
from pathlib import Path

import psutil
from tokenmoe_collect.clock import clock_metadata, now
from tokenmoe_collect.hostload import HostSampler, HostSource
from tokenmoe_collect.jsonl import JsonlWriter, write_json


def monitor(pid, devices, output, *, interval=0.1):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parent = psutil.Process(pid)
    created = parent.create_time()
    source = HostSource()
    if source.nvml is None:
        raise RuntimeError("NVML is required for GPU peak sampling")
    source.handles = [source.nvml.nvmlDeviceGetHandleByIndex(index) for index in devices]
    metadata = {
        "schema_version": 1,
        "pid": pid,
        "create_time": created,
        "clock": clock_metadata(),
        "interval_seconds": interval,
        "devices": [
            {
                "index": index,
                "uuid": source.nvml.nvmlDeviceGetUUID(handle),
                "name": source.nvml.nvmlDeviceGetName(handle),
            }
            for index, handle in zip(devices, source.handles)
        ],
        "scope": "sampled total selected-device memory; process-tree RSS can double-count shared pages",
    }
    write_json(output / "metadata.json", metadata)
    peaks = {
        "device_bytes": dict.fromkeys(map(str, devices), 0),
        "process_tree_rss": 0,
        "aggregate_device_bytes": 0,
        "samples": 0,
        "phases": {},
    }
    clock = clock_metadata()
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    with JsonlWriter(output / "samples.jsonl") as writer:

        class Store:
            def __init__(self):
                self.clock = clock

            def write(self, table, record):
                try:
                    processes = [parent, *parent.children(recursive=True)]
                except psutil.NoSuchProcess:
                    processes = []
                memory = {}
                for process in processes:
                    try:
                        memory[str(process.pid)] = process.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                phase_path = output.parent / "resource_phase.json"
                phase = (
                    json.loads(phase_path.read_text())["phase"]
                    if phase_path.exists()
                    else "initialization"
                )
                record.update(process_rss=memory, phase=phase)
                peaks["samples"] += 1
                peaks["process_tree_rss"] = max(peaks["process_tree_rss"], sum(memory.values()))
                phase_peaks = peaks["phases"].setdefault(phase, dict.fromkeys(map(str, devices), 0))
                if record["gpu_mem_used"] is not None:
                    peaks["aggregate_device_bytes"] = max(
                        peaks["aggregate_device_bytes"], sum(record["gpu_mem_used"])
                    )
                    for index, value in zip(devices, record["gpu_mem_used"]):
                        key = str(index)
                        peaks["device_bytes"][key] = max(peaks["device_bytes"][key], value)
                        phase_peaks[key] = max(phase_peaks[key], value)
                writer.write(record)

        sampler = HostSampler(Store(), lambda: 0, source=source, interval=interval).start()
        write_json(output / "ready.json", {"ready_at": now()})
        try:
            while not stop.wait(interval):
                try:
                    if (
                        not parent.is_running()
                        or parent.status() == psutil.STATUS_ZOMBIE
                        or parent.create_time() != created
                    ):
                        break
                except psutil.NoSuchProcess:
                    break
        finally:
            sampler.close()
            write_json(output / "peaks.json", {**peaks, "status": "complete", "metadata": metadata})
