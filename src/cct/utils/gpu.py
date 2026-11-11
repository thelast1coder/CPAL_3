"""Shared-GPU helpers. Always cap memory per process before allocating (CLAUDE.md rules)."""
from __future__ import annotations

import subprocess

import torch


def free_memory_gb(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


def total_memory_gb(device: int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(device).total_memory / 1024 ** 3


def cap_memory(gb: float, device: int = 0) -> float:
    """Cap this process at `gb` GiB on `device`. Returns the fraction set (0 if no GPU)."""
    if not torch.cuda.is_available():
        return 0.0
    frac = min(1.0, gb / total_memory_gb(device))
    torch.cuda.set_per_process_memory_fraction(frac, device)
    return frac


def nvidia_smi() -> str:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # no GPU on the laptop
        return f"nvidia-smi unavailable: {e}"


def require_free(gb: float, device: int = 0) -> None:
    """Refuse to start if the shared GPU does not have `gb` GiB free (queue instead of competing)."""
    if torch.cuda.is_available() and free_memory_gb(device) < gb:
        raise RuntimeError(f"only {free_memory_gb(device):.1f} GiB free on GPU {device}; need {gb}.")
