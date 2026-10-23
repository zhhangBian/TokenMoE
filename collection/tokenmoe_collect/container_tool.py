"""Isolate container tools and confirm group termination before reporting timeout."""

import json
import os
import signal
import sys
import time
from pathlib import Path


def process(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None


def terminate(path):
    if not path.exists():
        return False
    identity = json.loads(path.read_text())
    pid = identity["pid"]
    if type(pid) is not int or pid <= 1:
        return False
    current = process(pid)
    if current is not None and current[19] != identity["start"]:
        return False
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 1
    while True:
        members = [process(p.name) for p in Path("/proc").iterdir() if p.name.isdigit()]
        if not any(p is not None and p[2] == str(pid) and p[0] != "Z" for p in members):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)


def main():
    mode, status, *args = sys.argv[1:]
    path = Path(status)
    if mode == "kill":
        terminated = terminate(path)
        result = {"terminated": terminated}
        if not terminated:
            if not path.exists():
                result["reason"] = "process-group record missing"
            else:
                identity = json.loads(path.read_text())
                current = process(identity["pid"])
                result["reason"] = (
                    "process identity changed"
                    if current is not None and current[19] != identity["start"]
                    else "process group did not become terminal"
                )
                result["identity"] = identity
                result["current_state"] = current
        print(json.dumps(result))
        return
    if mode != "launch":
        raise ValueError("Unknown container tool operation")
    if os.getpgrp() != os.getpid():
        os.setsid()
    identity = {"pid": os.getpid(), "start": process(os.getpid())[19]}
    with path.open("x") as output:
        json.dump(identity, output)
    command = json.loads(args[0])
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
