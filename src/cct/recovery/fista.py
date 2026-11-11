"""Masked LASSO by FISTA (Beck and Teboulle 2009), batched over columns.

lambda per column by a residual-matching (discrepancy) rule with continuation: start at
lambda = lam0 * ||Phi^T y||_inf, halve it stage by stage until the relative residual is at most
target_resid (or a stage limit), warm-starting each stage. Then keep the k_max largest entries and
debias them by least squares.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .common import SparseColumns, batched_lstsq, dense_to_sparse


def _spectral_norm2(Phi: Tensor, n_iter: int = 50) -> float:
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(Phi.shape[1], 1, generator=g, dtype=Phi.dtype).to(Phi.device)
    for _ in range(n_iter):
        v = Phi.T @ (Phi @ v)
        v = v / v.norm()
    return float((Phi @ v).norm() ** 2)


@torch.no_grad()
def fista(Phi: Tensor, Y: Tensor, mask: Tensor | None, k_max: int, target_resid: float = 1e-3,
          lam0: float = 0.5, n_stages: int = 12, n_inner: int = 40, tol: float = 1e-6,
          cols: Tensor | None = None) -> SparseColumns:
    m, T = Phi.shape
    C = Y.shape[1]
    dev = Phi.device
    cols = torch.arange(C) if cols is None else cols
    Yd = Y.to(dev, Phi.dtype)
    ynorm = Yd.norm(dim=0).clamp_min(1e-300)
    msk = None if mask is None else mask.to(dev).to(Phi.dtype)
    Lip = _spectral_norm2(Phi) * 1.01
    G0 = Phi.T @ Yd
    if msk is not None:
        G0 = G0 * msk
    lam = lam0 * G0.abs().max(0).values  # [C]
    X = torch.zeros(T, C, dtype=Phi.dtype, device=dev)
    done = torch.zeros(C, dtype=torch.bool, device=dev)
    for _ in range(n_stages):
        Z = X.clone()
        t = 1.0
        for _ in range(n_inner):
            G = Phi.T @ (Phi @ Z - Yd)
            V = Z - G / Lip
            Xn = torch.sign(V) * (V.abs() - lam[None] / Lip).clamp_min(0)
            if msk is not None:
                Xn = Xn * msk
            tn = (1 + (1 + 4 * t * t) ** 0.5) / 2
            Z = Xn + ((t - 1) / tn) * (Xn - X)
            X, t = Xn, tn
        res = (Yd - Phi @ X).norm(dim=0) / ynorm
        done = done | (res <= target_resid)
        if bool(done.all()):
            break
        lam = torch.where(done, lam, lam / 2)
    # debias on the k_max largest entries
    idx, _ = dense_to_sparse(X, k_max)
    valid = idx >= 0
    PhiK = Phi[:, idx.clamp_min(0).T].permute(2, 0, 1) * valid[:, None, :].to(Phi.dtype)
    val = batched_lstsq(PhiK, Yd) * valid.to(Phi.dtype)
    Xd = torch.zeros_like(X)
    Xd.scatter_add_(0, idx.clamp_min(0).T, val.T)
    res = (Yd - Phi @ Xd).norm(dim=0) / ynorm
    res[Yd.norm(dim=0) == 0] = 0
    return SparseColumns(cols.cpu(), idx.cpu(), val.cpu().double(), res.cpu().double(), valid.sum(1).cpu())
