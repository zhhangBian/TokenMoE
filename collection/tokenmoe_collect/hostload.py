"""One-hertz host load sampling with optional NVML metrics."""

import logging
import threading
import time

import psutil

from .clock import now

logger = logging.getLogger(__name__)


class HostSource:
    def __init__(self):
        self.nvml = None
        self.handles = []
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception as exc:
            logger.warning("GPU metrics unavailable: %s", exc)
            self.close()

    def sample(self):
        network = psutil.net_io_counters()
        util, memory = None, None
        if self.nvml is not None:
            try:
                util = [self.nvml.nvmlDeviceGetUtilizationRates(h).gpu for h in self.handles]
                memory = [self.nvml.nvmlDeviceGetMemoryInfo(h).used for h in self.handles]
            except Exception as exc:
                logger.warning("GPU metrics unavailable: %s", exc)
                self.close()
                util = memory = None
        return {
            "load_avg_1m": psutil.getloadavg()[0],
            "cpu_util": psutil.cpu_percent(),
            "mem_util": psutil.virtual_memory().percent,
            "gpu_util": util,
            "gpu_mem_used": memory,
            "net_rx_bytes": network.bytes_recv,
            "net_tx_bytes": network.bytes_sent,
        }

    def close(self):
        if self.nvml is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
            self.nvml = None


class HostSampler:
    def __init__(self, store, active_processes, source=None, interval=1.0):
        self.store = store
        self.active_processes = active_processes
        self.source = source or HostSource()
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="tokenmoe-hostload", daemon=True)
        self.error = None

    def _loop(self):
        try:
            deadline = time.monotonic()
            while not self.stop_event.is_set():
                self.store.write(
                    "host_load",
                    {
                        "time": now(),
                        "host_id": self.store.clock["host_id"],
                        "concurrent_tool_processes": self.active_processes(),
                        **self.source.sample(),
                    },
                )
                deadline += self.interval
                self.stop_event.wait(max(0, deadline - time.monotonic()))
        except Exception as exc:
            self.error = exc

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.source.close()
        if self.error:
            raise RuntimeError("Host load sampler failed") from self.error
