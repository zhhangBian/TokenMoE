"""Harness raw format version 1 and its producer-local append-only tables."""

from pathlib import Path

from .clock import clock_metadata
from .jsonl import JsonlWriter, write_json

RAW_FORMAT_VERSION = 1
TABLES = (
    "application_runs",
    "sessions",
    "llm_requests",
    "tool_calls",
    "tool_output_chunks",
    "host_load",
)


class RecordStore:
    def __init__(self, run_dir, *, control_events=False):
        self.run_dir = Path(run_dir)
        self.clock = clock_metadata()
        self.directory = (
            self.run_dir / "raw/harness" / f"{self.clock['host_id']}-{self.clock['pid']}"
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        write_json(
            self.directory / "producer.json",
            {"raw_format_version": RAW_FORMAT_VERSION, **self.clock},
        )
        self.writers = {table: JsonlWriter(self.directory / f"{table}.jsonl") for table in TABLES}
        self.outputs = self.directory / "tool_outputs"
        self.outputs.mkdir()
        self.event_sink = None
        if control_events:
            from .events import BoundedEventSink

            self.event_sink = BoundedEventSink(self.directory / "control_events.jsonl")

    def write(self, table, record):
        self.writers[table].write(record)

    def close(self):
        if self.event_sink is not None:
            status = self.event_sink.close()
            write_json(self.directory / "control_export_status.json", status)
        for writer in self.writers.values():
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
