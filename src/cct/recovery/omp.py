"""Masked orthogonal matching pursuit, batched over columns, with incremental Cholesky updates.

For each column y_s (one column of Y = Phi A), greedily add the atom (target) with the largest
normalised correlation with the residual, restricted to the column's causal mask, then re-fit all
chosen atoms by least squares. A step costs one Phi^T R product plus O(C m k) work; the Gram factor
is extended by one row per step (no refactorisation).
Stopping: relative residual <= tol, k_max atoms, or no admissible atom left.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .common import SparseColumns


@torch.no_grad()
def omp(Phi: Tensor, Y: Tensor, mask: Tensor | None, k_max: int, tol: float = 1e-6,
        cols: Tensor | None = None) -> SparseColumns:
    """Phi [m, T], Y [m, C], mask [T, C] bool (None = no mask). Works on Phi's device and dtype
    (use float32 or float64; the Cholesky factor and coefficients are kept in float64)."""
    m, T = Phi.shape
    C = Y.shape[1]
    dev = Phi.device
    k_max = int(min(k_max, m, T))
    cols = torch.arange(C) if cols is None else cols
    Yd = Y.to(dev, Phi.dtype)
    ynorm = Yd.double().norm(dim=0)
    colnorm2 = Phi.double().pow(2).sum(0)  # [T]
    inv_cn = colnorm2.clamp_min(1e-300).rsqrt().to(Phi.dtype)

    R = Yd.clone()
    PhiS = torch.zeros(C, m, k_max, dtype=torch.float64, device=dev)
    Lc = torch.zeros(C, k_max, k_max, dtype=torch.float64, device=dev)
    bvec = torch.zeros(C, k_max, dtype=torch.float64, device=dev)
    sel = torch.full((C, k_max), -1, dtype=torch.long, device=dev)
    out_idx = torch.full((C, k_max), -1, dtype=torch.long, device=dev)
    out_val = torch.zeros(C, k_max, dtype=torch.float64, device=dev)
    out_res = torch.ones(C, dtype=torch.float64, device=dev)
    out_res[ynorm == 0] = 0.0
    n_atoms = torch.zeros(C, dtype=torch.long, device=dev)
    active = ynorm > 0
    ar = torch.arange(C, device=dev)
    Y64 = Yd.double()

    for it in range(k_max):
        if not bool(active.any()):
            break
        corr = (Phi.T @ R).abs() * inv_cn[:, None]  # [T, C]
        if mask is not None:
            corr = corr.masked_fill(~mask.to(dev), -1.0)
        if it > 0:
            corr[sel[:, :it].T, ar[None, :].expand(it, C)] = -1.0
        bestval, best = corr.max(0)
        active = active & (bestval > 0)
        if not bool(active.any()):
            break
        phi_new = Phi[:, best].double()  # [m, C]
        cn = colnorm2[best]
        if it == 0:
            Lc[:, 0, 0] = cn.sqrt()
        else:
            g = torch.einsum("cmi,mc->ci", PhiS[:, :, :it], phi_new)
            w = torch.linalg.solve_triangular(Lc[:, :it, :it], g[..., None], upper=False)[..., 0]
            dd = cn - w.pow(2).sum(1)
            bad = dd <= 1e-12 * cn
            active = active & ~bad
            dd = torch.where(bad, torch.ones_like(dd), dd)
            Lc[:, it, :it] = w
            Lc[:, it, it] = dd.sqrt()
        PhiS[:, :, it] = phi_new.T
        sel[:, it] = best
        bvec[:, it] = (phi_new * Y64).sum(0)
        # coefficients: L L^T x = b
        z = torch.linalg.solve_triangular(Lc[:, : it + 1, : it + 1], bvec[:, : it + 1, None], upper=False)
        x = torch.linalg.solve_triangular(Lc[:, : it + 1, : it + 1].transpose(1, 2), z, upper=True)[..., 0]
        Rn = Y64 - torch.einsum("cmi,ci->mc", PhiS[:, :, : it + 1], x)
        res = Rn.norm(dim=0) / ynorm.clamp_min(1e-300)
        upd = active
        out_idx[upd, : it + 1] = sel[upd, : it + 1]
        out_val[upd, : it + 1] = x[upd]
        out_res[upd] = res[upd]
        n_atoms[upd] = it + 1
        R = torch.where(upd[None, :], Rn.to(R.dtype), R)
        active = active & (res > tol)

    return SparseColumns(cols.cpu(), out_idx.cpu(), out_val.cpu(), out_res.cpu(), n_atoms.cpu())
