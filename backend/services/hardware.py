"""Hardware monitoring — CPU, RAM, GPU stats."""

import psutil
import time
from typing import Optional
from backend.models.schemas import GPUInfo, SystemStats

_start_time = time.time()


def get_system_stats() -> SystemStats:
    """Get current system resource usage."""
    mem = psutil.virtual_memory()
    gpu = _get_gpu_info()

    return SystemStats(
        cpu_percent=psutil.cpu_percent(interval=0.1),
        cpu_count=psutil.cpu_count(),
        ram_total_gb=round(mem.total / (1024**3), 1),
        ram_used_gb=round(mem.used / (1024**3), 1),
        ram_percent=mem.percent,
        gpu=gpu,
        uptime_seconds=int(time.time() - _start_time),
    )


def _get_gpu_info() -> Optional[GPUInfo]:
    """Try to get NVIDIA GPU info. Returns None if unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        name = pynvml.nvmlDeviceGetName(handle)

        return GPUInfo(
            name=name.decode() if isinstance(name, bytes) else name,
            memory_total_mb=mem_info.total // (1024**2),
            memory_used_mb=mem_info.used // (1024**2),
            memory_free_mb=mem_info.free // (1024**2),
            temperature_c=temp,
            utilization_pct=util.gpu,
        )
    except Exception:
        return None
