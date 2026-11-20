"""Count-sketch decoder D2 (PROJECT_V2.md T4; Charikar, Chen and Farach-Colton 2002).

R repetitions (default 5) x B buckets (m = R B mixed passes). Target t has bucket h_r(t) and sign g_r(t) in
repetition r. Mixed pass (r, b) is seeded with sum over {t : h_r(t) = b} of g_r(t) r_t, so by
linearity (PR1) its row is Y[(r, b)] = sum_t g_r(t) h-indicator A[t]. Estimate of A[t, s]:
median over r of g_r(t) Y[(r, h_r(t)), s]. Decoding keeps, per column, the `keep` largest estimates
by magnitude inside the candidate set (default B // 4: count-sketch with B = O(k) buckets recovers the
k heaviest entries); everything else is zero.
"""
from __future__ import annotations

import time

import torch
from torch import Tensor


class CountSketch:
    def __init__(self, T: int, m: int, R: int = 5, seed: int = 0, device="cpu"):
        if m < R:
            raise ValueError("m must be at least R")
        self.T, self.R = T, R
        self.B = m // R
        self.m = self.B * R
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(seed)
        self.h = torch.randint(0, self.B, (R, T), generator=g).to(self.device)
        self.sgn = (torch.randint(0, 2, (R, T), generator=g) * 2 - 1).to(self.device)
        self.row = self.h + torch.arange(R, device=self.device)[:, None] * self.B  # [R, T] row of Y

    def seeds(self, dtype=torch.float32) -> Tensor:
        """Dense seed matrix Phi [m, T] (small T only; real passes build seeds per bucket)."""
        Phi = torch.zeros(self.m, self.T, dtype=dtype, device=self.device)
        for r in range(self.R):
            Phi[self.row[r], torch.arange(self.T, device=self.device)] = self.sgn[r].to(dtype)
        return Phi

    def apply(self, A: Tensor) -> Tensor:
        """Y = Phi A for A [T, S] (what the mixed passes return): [m, S]."""
        A = A.to(self.device)
        Y = torch.zeros(self.m, A.shape[1], dtype=A.dtype, device=self.device)
        for r in range(self.R):
            Y.index_add_(0, self.row[r], A * self.sgn[r].to(A.dtype)[:, None])
        return Y

    def estimate(self, Y: Tensor) -> Tensor:
        """Raw median estimates for every (t, s): [T, S]."""
        Y = Y.to(self.device)
        est = torch.stack([Y[self.row[r]] * self.sgn[r].to(Y.dtype)[:, None] for r in range(self.R)])
        return est.median(dim=0).values  # odd R: the true median

    def decode(self, Y: Tensor, cand: Tensor | None = None, keep: int | None = None,
               chunk: int = 512) -> tuple[Tensor, float]:
        """Decoded columns [T, S] (dense, mostly zero) and decode seconds. cand: bool [T, S] candidate
        mask (None = all targets); keep: entries kept per column (default B // 4)."""
        keep = int(keep or max(1, self.B // 4))
        S = Y.shape[1]
        out = torch.zeros(self.T, S, dtype=Y.dtype, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(0, S, chunk):
            est = self.estimate(Y[:, i : i + chunk])
            if cand is not None:
                est = est * cand[:, i : i + chunk].to(self.device, est.dtype)
            k = min(keep, self.T)
            idx = est.abs().topk(k, dim=0).indices
            out[:, i : i + chunk].scatter_(0, idx, est.gather(0, idx))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t0


def column_rel_err(A_hat: Tensor, A: Tensor) -> Tensor:
    """Per-column relative error ||a_hat - a|| / ||a|| (nan for zero columns)."""
    num = (A_hat.double() - A.double().to(A_hat.device)).norm(dim=0)
    den = A.double().to(A_hat.device).norm(dim=0)
    return torch.where(den > 0, num / den.clamp_min(1e-300), torch.full_like(den, float("nan")))
