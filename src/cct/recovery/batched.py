"""Chunked recovery driver: masks per chunk, precision policy, timing."""
from __future__ import annotations

import time

import torch
from torch import Tensor

from ..operators.base import CausalMask
from .common import SparseColumns
from .cosamp import cosamp
from .fista import fista
from .iht import iht
from .omp import omp

ALGOS = {"omp": omp, "cosamp": cosamp, "iht": iht, "fista": fista}


def recover_columns(Phi: Tensor, Y: Tensor, cols: Tensor, mask: CausalMask | None, algo: str = "omp",
                    k_max: int = 32, tol: float = 1e-6, chunk: int = 1024,
                    device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32,
                    **kw) -> tuple[SparseColumns, float]:
    """Recover columns `cols` of A (over the sensed targets) from Y[:, cols].
    mask: causal mask restricted to the sensed targets (rows) and all sources (columns), or None.
    Returns (recovered columns, wall seconds). All accumulations in recovery are fp32 or fp64."""
    dev = torch.device(device)
    Phi_d = Phi.to(dev, dtype)
    parts = []
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for i in range(0, cols.numel(), chunk):
        c = cols[i : i + chunk]
        Yc = Y[:, c].to(dev, dtype)
        Mc = None if mask is None else mask.columns(c, device=dev)
        if algo == "omp":
            parts.append(omp(Phi_d, Yc, Mc, k_max=k_max, tol=tol, cols=c))
        elif algo == "cosamp":
            parts.append(cosamp(Phi_d, Yc, Mc, k=k_max, tol=tol, cols=c, **kw))
        elif algo == "iht":
            parts.append(iht(Phi_d, Yc, Mc, k=k_max, tol=tol, cols=c, **kw))
        elif algo == "fista":
            parts.append(fista(Phi_d, Yc, Mc, k_max=k_max, tol=tol, cols=c, **kw))
        else:
            raise ValueError(algo)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return SparseColumns.concat(parts), time.perf_counter() - t0
