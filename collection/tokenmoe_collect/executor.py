"""Merged-pipe streaming execution; local runtime is for CPU tests only."""

import os
import selectors
import signal
import subprocess
import threading
import time

from .clock import now


class StreamingExecutor:
    def __init__(
        self,
        store,
        runtime="podman",
        container_id=None,
        cwd="/testbed",
        environment=None,
        interpreter=None,
    ):
        self.store = store
        self.runtime = runtime
        self.container_id = container_id
        self.cwd = cwd
        self.environment = environment or {}
        self.interpreter = interpreter or ["bash", "-lc"]
        self._lock = threading.Lock()
        self._active = 0

    @property
    def active(self):
        with self._lock:
            return self._active

    def command(self, command, cwd=None):
        if self.runtime == "local":
            return [*self.interpreter, command]
        if self.runtime not in {"docker", "podman"}:
            raise ValueError(f"Unsupported container runtime: {self.runtime}")
        if not self.container_id:
            raise ValueError("No session container")
        argv = [self.runtime, "exec", "-w", cwd or self.cwd]
        for key, value in self.environment.items():
            argv.extend(["-e", f"{key}={value}"])
        return [*argv, self.container_id, *self.interpreter, command]

    def execute(self, command, tool_call_id, timeout, *, cwd=None):
        argv = self.command(command, cwd)
        destination = self.store.outputs / f"{tool_call_id}.out"
        output = bytearray()
        chunks = 0
        started = now()
        deadline = time.monotonic() + timeout
        process = None
        timed_out = False
        error = None
        returncode = -1
        with self._lock:
            self._active += 1
        try:
            with destination.open("xb") as raw:
                process = subprocess.Popen(
                    argv,
                    cwd=(cwd or self.cwd) if self.runtime == "local" else None,
                    env=os.environ | self.environment if self.runtime == "local" else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    start_new_session=True,
                )
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while selector.get_map():
                        remaining = deadline - time.monotonic()
                        if remaining <= 0 and not timed_out:
                            timed_out = True
                            os.killpg(process.pid, signal.SIGKILL)
                        events = selector.select(max(0, remaining) if not timed_out else 0.1)
                        for key, _ in events:
                            data = os.read(key.fd, 65536)
                            if not data:
                                selector.unregister(key.fileobj)
                                continue
                            timestamp = now()
                            raw.write(data)
                            raw.flush()
                            self.store.write(
                                "tool_output_chunks",
                                {
                                    "tool_call_id": tool_call_id,
                                    "chunk_index": chunks,
                                    "chunk_time": timestamp,
                                    "byte_offset": len(output),
                                    "byte_length": len(data),
                                },
                            )
                            output.extend(data)
                            chunks += 1
                    try:
                        returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGKILL)
                        returncode = process.wait()
                if timed_out:
                    error = subprocess.TimeoutExpired(argv, timeout, output=bytes(output))
        except Exception as exc:
            error = exc
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            finished = now()
            with self._lock:
                self._active -= 1
        # subprocess.run(text=True) uses universal newline translation.
        text = (
            bytes(output)
            .decode("utf-8", errors="replace")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )
        result = {"output": text, "returncode": returncode, "exception_info": ""}
        if error is not None:
            result.update(
                returncode=-1,
                exception_info=f"An error occurred while executing the command: {error}",
                extra={"exception_type": type(error).__name__, "exception": str(error)},
            )
        timing = {
            "started_at": started,
            "finished_at": finished,
            "exit_status": returncode,
            "timed_out": timed_out,
            "output_bytes_raw": len(output),
            "output_ref": str(destination.relative_to(self.store.run_dir)),
        }
        return result, timing
