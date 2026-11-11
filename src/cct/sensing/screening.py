"""Energy screening (PROJECT.md 5.4). With normalised Rademacher Phi, E ||y_s||^2 = ||a_s||^2, so the
measurements estimate every column's out-edge energy for free.

keep = the smallest set of columns whose estimated energy reaches a fraction q_eff of the total,
with q_eff = 1 - (1 - q) / safety. safety > 1 adds a margin for estimation noise so the kept set
holds at least q of the TRUE energy with high probability (test T8). safety = 1 is the plain rule.
"""
from __future__ import annotations

import torch
from torch import Tensor


def column_energy(Y: Tensor) -> Tensor:
    return Y.double().pow(2).sum(0)


def screen_columns(Y: Tensor, q: float | None, safety: float = 2.0) -> Tensor:
    """Boolean [N] mask of columns to recover. q None or >= 1 keeps every nonzero column."""
    e = column_energy(Y)
    if q is None or q >= 1.0:
        return e > 0
    q_eff = 1.0 - (1.0 - q) / safety
    total = e.sum()
    if total <= 0:
        return torch.zeros_like(e, dtype=torch.bool)
    vals, order = torch.sort(e, descending=True)
    cum = torch.cumsum(vals, 0) / total
    k = int(torch.searchsorted(cum, torch.tensor(q_eff, dtype=cum.dtype))) + 1
    k = min(k, e.numel())
    keep = torch.zeros_like(e, dtype=torch.bool)
    keep[order[:k]] = True
    return keep & (e > 0)
