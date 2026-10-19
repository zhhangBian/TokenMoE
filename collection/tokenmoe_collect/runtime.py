"""Container transport and a real task-image startup check before GPU work."""

import json
import math
import re
import subprocess
from pathlib import Path

from .ids import new_id
from .static import load_benchmark, load_yaml


def runtime_command(runtime, runtime_socket=None, runtime_executable=None):
    if runtime not in {"docker", "podman", "local"}:
        raise ValueError(f"Unsupported container runtime: {runtime}")
    if runtime_executable is not None and (
        runtime == "local"
        or not isinstance(runtime_executable, str)
        or not Path(runtime_executable).is_absolute()
    ):
        raise ValueError("runtime_executable must be an absolute Docker or Podman executable path")
    executable = runtime_executable or runtime
    if runtime_socket is None:
        # A platform-local Podman must not inherit a remote connection from
        # CONTAINER_HOST or the user's saved service settings.
        return [executable, "--remote=false"] if runtime == "podman" else [executable]
    if runtime != "docker":
        raise ValueError("runtime_socket requires runtime: docker")
    if not isinstance(runtime_socket, str) or not Path(runtime_socket).is_absolute():
        raise ValueError("runtime_socket must be an absolute Unix socket path")
    return [executable, "--host", f"unix://{runtime_socket}"]


def validate_limits(config, probe):
    quota, period = probe.get("cpu_quota_us"), probe.get("cpu_period_us")
    if config.get("cpu_quota") and (
        quota is None
        or quota <= 0
        or not period
        or not math.isclose(quota / period, float(config["cpu_quota"]), rel_tol=1e-5)
    ):
        raise ValueError(
            "Task cgroup CPU quota does not match cpu_quota; resource limits are required"
        )
    if config.get("memory_limit"):
        match = re.fullmatch(
            r"(\d+(?:\.\d+)?)\s*([bkmgtep]?)(?:i?b)?", str(config["memory_limit"]).lower()
        )
        if not match:
            raise ValueError("Unsupported memory_limit; use bytes or a size such as 4g")
        expected = int(float(match[1]) * 1024 ** "bkmgtep".index(match[2] or "b"))
        if probe.get("memory_limit_bytes") != expected:
            raise ValueError(
                "Task cgroup memory limit does not match memory_limit; resource limits are required"
            )


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

    Native Podman runs inside the current platform container. Docker sockets
    remain an optional transport when a caller explicitly provides one.
    """
    from .launcher import image_name

    path = Path(run_config).expanduser().resolve()
    config = load_yaml(path)
    runtime = config["runtime"]
    prefix = runtime_command(
        runtime, config.get("runtime_socket"), config.get("runtime_executable")
    )
    if runtime == "local":
        return {"passed": True, "runtime": runtime, "scope": "CPU tests only"}
    hint = (
        "For a platform container without host Docker access, use local Podman "
        "(runtime: podman). A tested private Podman entry point can be selected "
        "with runtime_executable. Check the platform's permitted user/mount/cgroup "
        "namespaces, UID/GID mappings and delegated cgroup permissions; keep CPU "
        "and memory limits enabled. Pull the benchmark images before this check."
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
        'if command -v python3 >/dev/null 2>&1; then exec python3 -c "$1"; else exec python -c "$1"; fi',
        "tokenmoe-probe",
        Path(__file__).with_name("container_probe.py").read_text(),
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
    probe = json.loads(result.stdout)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if probe.get("boot_id") != boot_id:
        raise ValueError(
            "Task containers and collector must share the same Linux host boot clock; "
            "the selected daemon appears to run on another host or VM"
        )
    validate_limits(config, probe)
    return {
        "passed": True,
        "runtime": runtime,
        "runtime_socket": config.get("runtime_socket"),
        "runtime_executable": config.get("runtime_executable"),
        "image": image,
        "boot_id": boot_id,
        "cpu_quota": config.get("cpu_quota"),
        "memory_limit": config.get("memory_limit"),
        "kernel_limits": probe,
        "scope": "task image startup, kernel cgroup limits and same-host boot clock",
    }
