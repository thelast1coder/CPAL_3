"""Masked normalised iterative hard thresholding (Blumensath and Davies 2010), batched, with a final
least-squares debias on the support."""
from __future__ import annotations

import torch
from torch import Tensor

from .common import SparseColumns, batched_lstsq, dense_to_sparse


def _hard(X: Tensor, k: int) -> Tensor:
    idx = X.abs().topk(k, dim=0).indices
    out = torch.zeros_like(X)
    out.scatter_(0, idx, torch.gather(X, 0, idx))
    return out


@torch.no_grad()
def iht(Phi: Tensor, Y: Tensor, mask: Tensor | None, k: int, n_iter: int = 200,
        tol: float = 1e-6, cols: Tensor | None = None) -> SparseColumns:
    m, T = Phi.shape
    C = Y.shape[1]
    dev = Phi.device
    k = int(min(k, m, T))
    cols = torch.arange(C) if cols is None else cols
    Yd = Y.to(dev, Phi.dtype)
    ynorm = Yd.norm(dim=0).clamp_min(1e-300)
    msk = None if mask is None else mask.to(dev).to(Phi.dtype)
    X = torch.zeros(T, C, dtype=Phi.dtype, device=dev)
    for _ in range(n_iter):
        R = Yd - Phi @ X
        G = Phi.T @ R
        if msk is not None:
            G = G * msk
        S = (X != 0) if bool((X != 0).any()) else (_hard(G, k) != 0)
        gS = G * S
        num = gS.pow(2).sum(0)
        den = (Phi @ gS).pow(2).sum(0).clamp_min(1e-300)
        mu = num / den
        X = _hard(X + mu[None] * G, k)
        if msk is not None:
            X = X * msk
        res = (Yd - Phi @ X).norm(dim=0) / ynorm
        if bool((res <= tol).all()):
            break
    idx, _ = dense_to_sparse(X, k)
    valid = idx >= 0
    PhiK = Phi[:, idx.clamp_min(0).T].permute(2, 0, 1) * valid[:, None, :].to(Phi.dtype)
    val = batched_lstsq(PhiK, Yd) * valid.to(Phi.dtype)
    Xd = torch.zeros_like(X)
    Xd.scatter_add_(0, idx.clamp_min(0).T, val.T)
    res = (Yd - Phi @ Xd).norm(dim=0) / ynorm
    res[Yd.norm(dim=0) == 0] = 0
    return SparseColumns(cols.cpu(), idx.cpu(), val.cpu().double(), res.cpu().double(), valid.sum(1).cpu())
