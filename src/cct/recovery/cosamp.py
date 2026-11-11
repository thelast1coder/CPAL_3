"""Masked CoSaMP (Needell and Tropp 2009), batched over columns."""
from __future__ import annotations

import torch
from torch import Tensor

from .common import SparseColumns, batched_lstsq, dense_to_sparse


@torch.no_grad()
def cosamp(Phi: Tensor, Y: Tensor, mask: Tensor | None, k: int, n_iter: int = 30,
           tol: float = 1e-6, cols: Tensor | None = None) -> SparseColumns:
    m, T = Phi.shape
    C = Y.shape[1]
    dev = Phi.device
    k = int(min(k, max(1, m // 3), T))
    cols = torch.arange(C) if cols is None else cols
    Yd = Y.to(dev, Phi.dtype)
    ynorm = Yd.norm(dim=0).clamp_min(1e-300)
    msk = None if mask is None else mask.to(dev)
    X = torch.zeros(T, C, dtype=Phi.dtype, device=dev)
    R = Yd.clone()
    prev = torch.full((C,), float("inf"), device=dev, dtype=Phi.dtype)
    for _ in range(n_iter):
        proxy = (Phi.T @ R).abs()
        score = proxy.clone()
        score[X != 0] = float("inf")  # keep current support
        if msk is not None:
            score = score.masked_fill(~msk, -1.0)
        n_c = min(3 * k, T)
        cand = score.topk(n_c, dim=0).indices  # [3k, C]
        PhiS = Phi[:, cand].permute(2, 0, 1)  # [C, m, 3k]
        b = batched_lstsq(PhiS, Yd)  # [C, 3k]
        if msk is not None:
            b = b * torch.gather(msk, 0, cand).T.to(b.dtype)
        Xn = torch.zeros_like(X)
        Xn.scatter_(0, cand, b.T)
        idx, val = dense_to_sparse(Xn, k)  # prune to k
        X = torch.zeros_like(X)
        valid = idx >= 0
        X[idx.clamp_min(0).T, torch.arange(C, device=dev)[None].expand(idx.shape[1], -1)] = \
            torch.where(valid, val, torch.zeros_like(val)).T
        # re-fit on the pruned support (debias)
        PhiK = Phi[:, idx.clamp_min(0).T].permute(2, 0, 1) * valid[:, None, :].to(Phi.dtype)
        xk = batched_lstsq(PhiK, Yd, ridge=0.0) * valid.to(Phi.dtype)
        X = torch.zeros_like(X)
        X.scatter_add_(0, idx.clamp_min(0).T, xk.T)
        R = Yd - Phi @ X
        res = R.norm(dim=0) / ynorm
        if bool(((res <= tol) | (res >= prev * (1 - 1e-9))).all()):
            break
        prev = res
    idx, val = dense_to_sparse(X, k)
    res = (Yd - Phi @ X).norm(dim=0) / ynorm
    res[Yd.norm(dim=0) == 0] = 0
    return SparseColumns(cols.cpu(), idx.cpu(), val.cpu().double(), res.cpu().double(),
                         (idx >= 0).sum(1).cpu())
