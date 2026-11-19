"""GPU telemetry in the DCGM exporter's Prometheus format, from NVML, for nodes
without a DCGM exporter. AIPerf's DCGM collector reads it through
--gpu-telemetry, and Kolupalka names each GPU by its `pod` label.

    python gpu_exporter.py --port 9401 --pods 0-3=prefill-0,4-7=decode-0

Beside the DCGM fields it keeps counters from NVML's GPM samples (Hopper and
newer), taken every --gpm-interval seconds: seconds the GPU was busy, and
seconds weighted by the share of SMs, tensor pipes and DRAM bandwidth in use.
Differences of two scrapes give exact shares over any window, so AIPerf scrapes
them through --server-metrics. GET /clock returns this node's wall clock in ns,
for the runner to measure the offsets between nodes.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import pynvml

# DCGM field name, NVML reading, Prometheus type and help, as the DCGM exporter names them.
FIELDS = (
    ("DCGM_FI_DEV_GPU_UTIL", "gauge", "GPU utilization (in %)."),
    ("DCGM_FI_DEV_MEM_COPY_UTIL", "gauge", "Memory utilization (in %)."),
    ("DCGM_FI_DEV_FB_USED", "gauge", "Framebuffer memory used (in MiB)."),
    ("DCGM_FI_DEV_FB_FREE", "gauge", "Framebuffer memory free (in MiB)."),
    ("DCGM_FI_DEV_POWER_USAGE", "gauge", "Power draw (in W)."),
    (
        "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION",
        "counter",
        "Total energy consumption since boot (in mJ).",
    ),
    ("DCGM_FI_DEV_GPU_TEMP", "gauge", "GPU temperature (in C)."),
    ("DCGM_FI_DEV_SM_CLOCK", "gauge", "SM clock frequency (in MHz)."),
)

# Counter name, GPM metric whose share (0-100) weights each sampled second, and help.
GPM_COUNTERS = (
    (
        "gpu_gpm_busy_seconds_total",
        "NVML_GPM_METRIC_GRAPHICS_UTIL",
        "Seconds any kernel ran on the GPU.",
    ),
    (
        "gpu_gpm_sm_active_seconds_total",
        "NVML_GPM_METRIC_SM_UTIL",
        "Seconds weighted by the share of busy SMs.",
    ),
    (
        "gpu_gpm_sm_occupancy_seconds_total",
        "NVML_GPM_METRIC_SM_OCCUPANCY",
        "Seconds weighted by warp occupancy.",
    ),
    (
        "gpu_gpm_tensor_active_seconds_total",
        "NVML_GPM_METRIC_ANY_TENSOR_UTIL",
        "Seconds weighted by tensor pipe activity.",
    ),
    (
        "gpu_gpm_dram_active_seconds_total",
        "NVML_GPM_METRIC_DRAM_BW_UTIL",
        "Seconds weighted by the share of DRAM bandwidth used.",
    ),
)
GPM_SAMPLED_SECONDS = "gpu_gpm_sampled_seconds_total"


def parse_pods(spec: str) -> dict[int, str]:
    """`0-3=prefill-0,4-7=decode-0` -> {0: "prefill-0", ..., 7: "decode-0"}."""
    pods = {}
    for part in filter(None, spec.split(",")):
        span, pod = part.split("=", 1)
        first, _, last = span.partition("-")
        for index in range(int(first), int(last or first) + 1):
            pods[index] = pod
    return pods


class GpmCounters:
    """Seconds weighted by GPM shares, per GPU, accumulated by a sampling thread."""

    def __init__(self, handles: list, interval_seconds: float) -> None:
        self.handles = handles
        self.interval_seconds = interval_seconds
        self.metric_ids = [getattr(pynvml, metric) for _, metric, _ in GPM_COUNTERS]
        self.totals = [[0.0] * len(GPM_COUNTERS) for _ in handles]
        self.sampled_seconds = [0.0] * len(handles)
        self.lock = threading.Lock()

    @staticmethod
    def start(handles: list, interval_seconds: float) -> Optional[GpmCounters]:
        """
        Returns:
            None when a GPU has no GPM support.
        """
        if not all(
            pynvml.nvmlGpmQueryDeviceSupport(handle).isSupportedDevice
            for handle in handles
        ):
            return None
        counters = GpmCounters(handles, interval_seconds)
        threading.Thread(target=counters.run, daemon=True).start()
        return counters

    def run(self) -> None:
        previous = [self._sample(handle) for handle in self.handles]
        previous_time = time.monotonic()
        while True:
            time.sleep(self.interval_seconds)
            current = [self._sample(handle) for handle in self.handles]
            now = time.monotonic()
            shares = [
                self._shares(before, after) for before, after in zip(previous, current)
            ]
            with self.lock:
                for index, gpu_shares in enumerate(shares):
                    self.sampled_seconds[index] += now - previous_time
                    for counter, share in enumerate(gpu_shares):
                        self.totals[index][counter] += (
                            share / 100.0 * (now - previous_time)
                        )
            for sample in previous:
                pynvml.nvmlGpmSampleFree(sample)
            previous, previous_time = current, now

    def snapshot(self, index: int) -> tuple[list[float], float]:
        with self.lock:
            return list(self.totals[index]), self.sampled_seconds[index]

    @staticmethod
    def _sample(handle):
        sample = pynvml.nvmlGpmSampleAlloc()
        pynvml.nvmlGpmSampleGet(handle, sample)
        return sample

    def _shares(self, before, after) -> list[float]:
        request = pynvml.c_nvmlGpmMetricsGet_t()
        request.version = pynvml.NVML_GPM_METRICS_GET_VERSION
        request.numMetrics = len(self.metric_ids)
        request.sample1 = before
        request.sample2 = after
        for index, metric_id in enumerate(self.metric_ids):
            request.metrics[index].metricId = metric_id
        pynvml.nvmlGpmMetricsGet(request)
        return [
            min(100.0, max(0.0, request.metrics[index].value))
            for index in range(len(self.metric_ids))
        ]


class NvmlReader:
    def __init__(
        self, pods: dict[int, str], hostname: str, gpm_interval_seconds: float
    ) -> None:
        pynvml.nvmlInit()
        self.pods = pods
        self.hostname = hostname
        self.handles = [
            pynvml.nvmlDeviceGetHandleByIndex(i)
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        self.gpm = (
            GpmCounters.start(self.handles, gpm_interval_seconds)
            if gpm_interval_seconds > 0
            else None
        )
        self.lock = threading.Lock()

    def render(self) -> str:
        with self.lock:
            rows = [
                self._read(index, handle) for index, handle in enumerate(self.handles)
            ]
        lines = []
        for name, kind, help_text in FIELDS:
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
            for labels, values in rows:
                if values.get(name) is not None:
                    lines.append(f"{name}{{{labels}}} {values[name]}")
        if self.gpm is not None:
            lines += self._gpm_lines([labels for labels, _ in rows])
        return "\n".join(lines) + "\n"

    def _gpm_lines(self, labels: list[str]) -> list[str]:
        snapshots = [self.gpm.snapshot(index) for index in range(len(self.handles))]
        lines = [
            f"# HELP {GPM_SAMPLED_SECONDS} Seconds covered by GPM samples.",
            f"# TYPE {GPM_SAMPLED_SECONDS} counter",
        ]
        lines += [
            f"{GPM_SAMPLED_SECONDS}{{{labels[index]}}} {seconds:.6f}"
            for index, (_, seconds) in enumerate(snapshots)
        ]
        for counter, (name, _, help_text) in enumerate(GPM_COUNTERS):
            lines += [f"# HELP {name} {help_text}", f"# TYPE {name} counter"]
            lines += [
                f"{name}{{{labels[index]}}} {totals[counter]:.6f}"
                for index, (totals, _) in enumerate(snapshots)
            ]
        return lines

    def _read(self, index: int, handle) -> tuple[str, dict[str, Optional[float]]]:
        def safe(read):
            try:
                return read()
            except pynvml.NVMLError:
                return None

        utilization = safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle))
        memory = safe(lambda: pynvml.nvmlDeviceGetMemoryInfo(handle))
        values = {
            "DCGM_FI_DEV_GPU_UTIL": utilization.gpu if utilization else None,
            "DCGM_FI_DEV_MEM_COPY_UTIL": utilization.memory if utilization else None,
            "DCGM_FI_DEV_FB_USED": memory.used // (1 << 20) if memory else None,
            "DCGM_FI_DEV_FB_FREE": memory.free // (1 << 20) if memory else None,
            "DCGM_FI_DEV_POWER_USAGE": safe(
                lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            ),
            "DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION": safe(
                lambda: pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
            ),
            "DCGM_FI_DEV_GPU_TEMP": safe(
                lambda: pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            ),
            "DCGM_FI_DEV_SM_CLOCK": safe(
                lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            ),
        }
        uuid = safe(lambda: pynvml.nvmlDeviceGetUUID(handle)) or ""
        model = safe(lambda: pynvml.nvmlDeviceGetName(handle)) or ""
        bus = safe(lambda: pynvml.nvmlDeviceGetPciInfo(handle).busId) or ""
        uuid, model, bus = (
            v.decode() if isinstance(v, bytes) else v for v in (uuid, model, bus)
        )
        labels = {
            "gpu": str(index),
            "UUID": uuid,
            "pci_bus_id": bus,
            "device": f"nvidia{index}",
            "modelName": model,
            "Hostname": self.hostname,
        }
        if index in self.pods:
            labels.update(namespace="franky", pod=self.pods[index], container="engine")
        return ",".join(f'{key}="{value}"' for key, value in labels.items()), values


def serve(
    port: int, pods: dict[int, str], hostname: str, gpm_interval_seconds: float
) -> ThreadingHTTPServer:
    reader = NvmlReader(pods, hostname, gpm_interval_seconds)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.startswith("/clock"):
                body = f"{time.time_ns()}\n".encode()
            elif self.path.startswith("/metrics"):
                body = reader.render().encode()
            else:
                body = b""
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9401)
    parser.add_argument(
        "--pods",
        default="",
        help="GPU index ranges to pod names, e.g. 0-3=prefill-0,4-7=decode-0",
    )
    parser.add_argument("--hostname", default=socket.gethostname())
    parser.add_argument(
        "--gpm-interval",
        type=float,
        default=0.1,
        help="seconds between GPM samples; 0 turns the GPM counters off",
    )
    args = parser.parse_args()
    serve(args.port, parse_pods(args.pods), args.hostname, args.gpm_interval)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
