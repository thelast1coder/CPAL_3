"""B4, structural colouring (Curtis, Powell and Reid 1974 style) using only the causal zero pattern.

Rows (targets) of one colour share no possibly-nonzero column, so one backward pass seeded with all
rows of a colour returns each of their entries unmixed. Number of colours = passes. With a causal
mask almost every pair of targets shares a reachable source (for example the first token), so this
gives little saving: structure alone is not enough.
"""
from __future__ import annotations

import torch
from torch import Tensor


def greedy_colouring(pattern: Tensor) -> Tensor:
    """Colour the COLUMNS of pattern [R, C] so no two columns of a colour share a True row.
    Returns colour id per column (greedy, largest-degree first)."""
    R, C = pattern.shape
    order = torch.argsort(pattern.sum(0), descending=True, stable=True)
    colours = torch.full((C,), -1, dtype=torch.long)
    occ = torch.zeros(0, R, dtype=torch.bool)
    for c in order.tolist():
        rows = pattern[:, c]
        if occ.shape[0]:
            clash = (occ & rows[None]).any(1)
            free = torch.nonzero(~clash).flatten()
        else:
            free = torch.zeros(0, dtype=torch.long)
        if free.numel():
            k = int(free[0])
            occ[k] |= rows
        else:
            k = occ.shape[0]
            occ = torch.cat([occ, rows[None]], 0)
        colours[c] = k
    return colours


def colouring_rows(op, pattern_TN: Tensor) -> tuple[Tensor, int]:
    """Recover A by backward passes, one per colour of the rows (targets)."""
    col = greedy_colouring(pattern_TN.T)  # colour targets
    n_col = int(col.max()) + 1 if col.numel() else 0
    seeds = torch.zeros(n_col, op.T, dtype=op.dtype)
    seeds[col, torch.arange(op.T)] = 1.0
    Y = op.rows(seeds).double().cpu()  # [n_col, N]
    A = Y[col] * pattern_TN.double()  # each target reads its colour's row on its own pattern
    return A, n_col
