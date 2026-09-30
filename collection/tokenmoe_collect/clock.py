"""Host clock in nanoseconds; never use a container's clock."""

import os
import socket
import time
from pathlib import Path

from .ids import content_id

now = time.monotonic_ns


def clock_metadata():
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return {
        "clock_domain_id": boot_id,
        "monotonic_anchor_ns": now(),
        "wall_anchor_ns": time.time_ns(),
        "host_id": content_id({"hostname": socket.gethostname(), "boot_id": boot_id}),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
