"""CUDA-synchronised timers."""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(store: dict, key: str):
    sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        sync()
        store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def median_time(fn, repeats: int = 3) -> tuple[float, list[float]]:
    ts = []
    for _ in range(repeats):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    s = sorted(ts)
    return s[len(s) // 2], ts
