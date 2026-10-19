"""Read the running task's cgroup limits using only the Python standard library.

The runtime sends this source to the task image; it must not import the harness.
"""

import json
import re
from pathlib import Path


def _unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def snapshot(proc="/proc"):
    proc = Path(proc)
    memberships = {}
    for line in (proc / "self/cgroup").read_text().splitlines():
        _, controllers, group = line.split(":", 2)
        for controller in controllers.split(","):
            memberships[controller] = group

    def controller_files(controller, names):
        for line in (proc / "self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            fields, filesystem = before.split(), after.split()
            kind = filesystem[0]
            if kind == "cgroup2" and "" in memberships:
                group = memberships[""]
            elif kind == "cgroup" and controller in filesystem[2].split(","):
                if controller not in memberships:
                    continue
                group = memberships[controller]
            else:
                continue
            root, mount = Path(_unescape(fields[3])), Path(_unescape(fields[4]))
            group = Path(group)
            try:
                relative = group.relative_to(root)
            except ValueError:
                # A bind of the outer cgroup mount can have a different root
                # from the task's private cgroup namespace membership.
                relative = group.relative_to("/")
            if ".." in relative.parts:
                continue
            directory = mount / relative
            selected = names[2 if kind == "cgroup2" else 1]
            try:
                return (
                    kind,
                    str(directory),
                    [(directory / name).read_text().strip() for name in selected],
                )
            except FileNotFoundError:
                continue
        return None, None, []

    cpu_kind, cpu_path, cpu = controller_files(
        "cpu", {1: ["cpu.cfs_quota_us", "cpu.cfs_period_us"], 2: ["cpu.max"]}
    )
    memory_kind, memory_path, memory = controller_files(
        "memory", {1: ["memory.limit_in_bytes"], 2: ["memory.max"]}
    )
    if cpu_kind == "cgroup2":
        cpu = cpu[0].split()
    return {
        "boot_id": (proc / "sys/kernel/random/boot_id").read_text().strip(),
        "cpu_quota_us": int(cpu[0]) if cpu and cpu[0] != "max" else None,
        "cpu_period_us": int(cpu[1]) if cpu else None,
        "memory_limit_bytes": int(memory[0]) if memory and memory[0] != "max" else None,
        "cpu_cgroup": cpu_path,
        "memory_cgroup": memory_path,
        "cpu_cgroup_type": cpu_kind,
        "memory_cgroup_type": memory_kind,
        "uid_map": (proc / "self/uid_map").read_text().strip(),
        "gid_map": (proc / "self/gid_map").read_text().strip(),
    }


if __name__ == "__main__":
    print(json.dumps(snapshot()))
