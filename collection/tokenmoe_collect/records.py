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
    def __init__(self, run_dir):
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

    def write(self, table, record):
        self.writers[table].write(record)

    def close(self):
        for writer in self.writers.values():
            writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
