"""VRAM + latency profiling on the RTX 5070. Wrap any callable to capture peak
GPU memory and wall-clock time — feeds the hardware-constraint chapter."""
from __future__ import annotations

import time
from contextlib import contextmanager


def _vram_used_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024 ** 2)
        pynvml.nvmlShutdown()
        return used
    except Exception:
        return -1.0


@contextmanager
def profile(label: str = "run"):
    start_vram = _vram_used_mb()
    t0 = time.perf_counter()
    stats = {}
    try:
        yield stats
    finally:
        stats["label"] = label
        stats["latency_s"] = round(time.perf_counter() - t0, 3)
        stats["vram_start_mb"] = round(start_vram, 1)
        stats["vram_end_mb"] = round(_vram_used_mb(), 1)
        print(f"[{label}] latency={stats['latency_s']}s "
              f"vram={stats['vram_end_mb']}MB")
