"""Container transport and a real task-image startup check before GPU work."""

import subprocess
from pathlib import Path

from .ids import new_id
from .static import load_benchmark, load_yaml


def runtime_command(runtime, runtime_socket=None):
    if runtime not in {"docker", "podman", "local"}:
        raise ValueError(f"Unsupported container runtime: {runtime}")
    if runtime_socket is None:
        return [runtime]
    if runtime != "docker":
        raise ValueError("runtime_socket requires runtime: docker")
    if not isinstance(runtime_socket, str) or not Path(runtime_socket).is_absolute():
        raise ValueError("runtime_socket must be an absolute Unix socket path")
    return ["docker", "--host", f"unix://{runtime_socket}"]


def container_run_args(config):
    args = ["--rm", "--pull=never"]
    for key, flag in [
        ("container_network", "--network"),
        ("cpu_quota", "--cpus"),
        ("memory_limit", "--memory"),
    ]:
        if config.get(key):
            args += [flag, str(config[key])]
    return args


def check_runtime(run_config):
    """Start the first cached benchmark image with the configured resource limits.

    A host Docker socket creates sibling task containers. No harness paths or
    daemon sockets are mounted into those containers; tool I/O uses exec pipes.
    """
    from .launcher import image_name

    path = Path(run_config).expanduser().resolve()
    config = load_yaml(path)
    runtime = config["runtime"]
    prefix = runtime_command(runtime, config.get("runtime_socket"))
    if runtime == "local":
        return {"passed": True, "runtime": runtime, "scope": "CPU tests only"}
    hint = (
        "When the collector runs in Docker, mount the same Linux host's Docker "
        "socket into the collector and set runtime: docker and "
        "runtime_socket: /var/run/docker.sock. The socket must be accessible to "
        "the collector user. Pull the benchmark images before this check."
    )
    try:
        subprocess.run(prefix + ["info"], check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"Container runtime is unavailable: {detail}\n{hint}") from exc
    dataset = (path.parent / Path(config["dataset_path"]).expanduser()).resolve()
    items = load_benchmark(dataset, config["instances"])
    if not items:
        raise ValueError("Runtime check requires at least one benchmark instance")
    image = image_name(items[0])
    name = "tokenmoe-preflight-" + new_id("ses")
    command = [
        *prefix,
        "run",
        "--name",
        name,
        *container_run_args(config),
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        "cat /proc/sys/kernel/random/boot_id",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=config.get("container_start_timeout", 300),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Only this check's uniquely named container is eligible for cleanup.
        try:
            subprocess.run(prefix + ["rm", "-f", name], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            pass
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"Task container startup check failed: {detail}\n{hint}") from exc
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if result.stdout.strip() != boot_id:
        raise ValueError(
            "Task containers and collector must share the same Linux host boot clock; "
            "the selected daemon appears to run on another host or VM"
        )
    return {
        "passed": True,
        "runtime": runtime,
        "runtime_socket": config.get("runtime_socket"),
        "image": image,
        "boot_id": boot_id,
        "cpu_quota": config.get("cpu_quota"),
        "memory_limit": config.get("memory_limit"),
        "scope": "task image startup with configured limits and same-host boot clock",
    }
