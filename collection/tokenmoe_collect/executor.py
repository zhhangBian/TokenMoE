"""Merged-pipe streaming execution; local runtime is for CPU tests only."""

import json
import os
import selectors
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .clock import now
from .events import emit
from .runtime import runtime_command


class StreamingExecutor:
    def __init__(
        self,
        store,
        runtime="podman",
        container_id=None,
        cwd="/testbed",
        environment=None,
        interpreter=None,
        runtime_socket=None,
        runtime_executable=None,
        session_id=None,
    ):
        self.store = store
        self.session_id = session_id
        self.runtime = runtime
        self.runtime_argv = runtime_command(runtime, runtime_socket, runtime_executable)
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
        argv = [*self.runtime_argv, "exec", "-w", cwd or self.cwd]
        for key, value in self.environment.items():
            argv.extend(["-e", f"{key}={value}"])
        return [*argv, self.container_id, *self.interpreter, command]

    def execute(self, command, tool_call_id, timeout, *, cwd=None):
        argv = self.command(command, cwd)
        launched = argv
        if self.runtime != "local":
            prefix = argv[: -len(self.interpreter) - 1]
            helper = Path(__file__).with_name("container_tool.py").read_text()
            status = f"/tmp/tokenmoe-tool-{uuid.uuid4().hex}.json"

            def container_command(mode, *args):
                select_python = (
                    'if command -v python3 >/dev/null 2>&1; then exec python3 -u -c "$@"; '
                    'else exec python -u -c "$@"; fi'
                )
                return [
                    *prefix,
                    "/bin/sh",
                    "-c",
                    select_python,
                    "tokenmoe-tool",
                    helper,
                    mode,
                    status,
                    *args,
                ]

            launched = container_command("launch", json.dumps([*self.interpreter, command]))
        destination = self.store.outputs / f"{tool_call_id}.out"
        output = bytearray()
        chunks = 0
        started = now()
        emit(
            self.store,
            "tool_started",
            source_at=started,
            session_id=self.session_id,
            tool_id=tool_call_id,
        )
        deadline = time.monotonic() + timeout
        process = None
        timed_out = False
        error = None
        returncode = -1
        stopped, fatal = self.runtime == "local", False

        def stop():
            nonlocal stopped, fatal
            if self.runtime != "local":
                detail = ""
                try:
                    result = subprocess.run(
                        container_command("kill"),
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    acknowledgement = json.loads(result.stdout)
                    stopped = acknowledgement["terminated"] is True
                    detail = json.dumps(acknowledgement) if not stopped else ""
                except Exception as exc:
                    stopped = False
                    detail = str(getattr(exc, "stderr", None) or exc)[-2048:]
                if not stopped:
                    fatal = True
                    cleanup = subprocess.run(
                        [*self.runtime_argv, "kill", self.container_id],
                        capture_output=True,
                        timeout=5,
                    )
                    stopped = cleanup.returncode == 0
                    raise RuntimeError(
                        f"Could not confirm tool termination ({detail}); aborting session container"
                    )
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

        with self._lock:
            self._active += 1
        try:
            with destination.open("xb") as raw:
                process = subprocess.Popen(
                    launched,
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
                            stop()
                        events = selector.select(max(0, remaining) if not timed_out else 0.1)
                        for key, _ in events:
                            data = os.read(key.fd, 65536)
                            if not data:
                                selector.unregister(key.fileobj)
                                continue
                            timestamp = now()
                            raw.write(data)
                            raw.flush()
                            emit(
                                self.store,
                                "tool_output",
                                source_at=timestamp,
                                session_id=self.session_id,
                                tool_id=tool_call_id,
                                output_ref=str(destination.relative_to(self.store.run_dir)),
                                byte_offset=len(output),
                                byte_length=len(data),
                                chunk_index=chunks,
                            )
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
                        stop()
                        returncode = process.wait()
                stopped = True
                if timed_out:
                    error = subprocess.TimeoutExpired(argv, timeout, output=bytes(output))
        except BaseException as exc:
            error = exc
            fatal = fatal or not isinstance(exc, Exception)
            if process is None:
                stopped = True
            if process is not None and process.poll() is None:
                try:
                    stop()
                except Exception as cleanup_error:
                    error, fatal = cleanup_error, True
                finally:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.wait()
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
            finished = now()
            emit(
                self.store,
                "tool_completed" if stopped and not fatal else "tool_termination_failed",
                source_at=finished,
                session_id=self.session_id,
                tool_id=tool_call_id,
                outcome="timeout" if timed_out else "success" if returncode == 0 else "failure",
            )
            with self._lock:
                self._active -= 1
        if fatal:
            raise error
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
