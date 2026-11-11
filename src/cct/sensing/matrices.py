"""Sensing matrices Phi [m, T]. All seeded; generated in column chunks so very large T fits.

Kinds: rademacher (default, entries +-1/sqrt(m)), gaussian (N(0, 1/m)), sparse (d nonzero +-1
entries per column, scaled by 1/sqrt(d)), position_blocked (Rademacher, but targets are permuted so
each block of rows sees targets sorted by position; identical distribution to Rademacher up to a
column permutation, kept as an ablation hook for structured designs).
"""
from __future__ import annotations

import torch
from torch import Tensor


def _gen(seed: int, chunk: int) -> torch.Generator:
    # independent stream per (seed, chunk) so a chunk can be regenerated on its own
    return torch.Generator().manual_seed(int(seed) * 1_000_003 + int(chunk))


def sensing_chunk(kind: str, m: int, t0: int, t1: int, seed: int, chunk_id: int,
                  d: int = 8, dtype=torch.float32) -> Tensor:
    width = t1 - t0
    g = _gen(seed, chunk_id)
    if kind in ("rademacher", "position_blocked"):
        x = torch.randint(0, 2, (m, width), generator=g).to(dtype) * 2 - 1
        return x / m ** 0.5
    if kind == "gaussian":
        return torch.randn(m, width, generator=g, dtype=dtype) / m ** 0.5
    if kind == "sparse":
        d = min(d, m)
        out = torch.zeros(m, width, dtype=dtype)
        rows = torch.argsort(torch.rand(width, m, generator=g), dim=1)[:, :d]  # [width, d]
        signs = torch.randint(0, 2, (width, d), generator=g).to(dtype) * 2 - 1
        out[rows.T, torch.arange(width)[None].expand(d, -1)] = signs.T / d ** 0.5
        return out
    raise ValueError(f"unknown sensing matrix kind {kind}")


def sensing_matrix(kind: str, m: int, T: int, seed: int, chunk: int = 65536, d: int = 8,
                   dtype=torch.float32, positions: Tensor | None = None) -> Tensor:
    """Full Phi [m, T]. For position_blocked, columns are assigned in order of target position."""
    parts = [sensing_chunk(kind, m, t0, min(T, t0 + chunk), seed, i, d, dtype)
             for i, t0 in enumerate(range(0, T, chunk))]
    Phi = torch.cat(parts, 1) if parts else torch.zeros(m, 0, dtype=dtype)
    if kind == "position_blocked" and positions is not None:
        order = torch.argsort(positions, stable=True)
        out = torch.empty_like(Phi)
        out[:, order] = Phi
        Phi = out
    return Phi
