"""Shared types and helpers for sparse recovery."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SparseColumns:
    """Recovered columns in sparse form. idx/val are [C, k] with idx = -1 for padding."""

    cols: Tensor  # [C] column ids (into the operator's sources)
    idx: Tensor  # [C, k] target indices (into the sensed target set)
    val: Tensor  # [C, k]
    resid_rel: Tensor  # [C] ||y - Phi a_hat|| / ||y||
    n_atoms: Tensor  # [C]

    def to_dense(self, T: int, N: int | None = None, dtype=torch.float64) -> Tensor:
        """Dense [T, N] (or [T, C] if N is None) matrix with the recovered columns placed."""
        C = self.cols.numel()
        width = C if N is None else N
        out = torch.zeros(T, width, dtype=dtype)
        if C == 0:
            return out
        col_pos = torch.arange(C) if N is None else self.cols.cpu()
        valid = self.idx.cpu() >= 0
        r = self.idx.cpu()[valid]
        c = col_pos[:, None].expand_as(self.idx.cpu())[valid]
        out.index_put_((r, c), self.val.cpu()[valid].to(dtype), accumulate=True)
        return out

    @staticmethod
    def concat(parts: list["SparseColumns"]) -> "SparseColumns":
        if not parts:
            z = torch.zeros(0, dtype=torch.long)
            return SparseColumns(z, torch.zeros(0, 0, dtype=torch.long), torch.zeros(0, 0), torch.zeros(0), z)
        k = max(p.idx.shape[1] for p in parts)

        def pad(x, fill):
            if x.shape[1] == k:
                return x
            return torch.cat([x, torch.full((x.shape[0], k - x.shape[1]), fill, dtype=x.dtype,
                                            device=x.device)], 1)

        return SparseColumns(
            torch.cat([p.cols.cpu() for p in parts]),
            torch.cat([pad(p.idx, -1).cpu() for p in parts]),
            torch.cat([pad(p.val, 0).cpu() for p in parts]),
            torch.cat([p.resid_rel.cpu() for p in parts]),
            torch.cat([p.n_atoms.cpu() for p in parts]),
        )


def dense_to_sparse(X: Tensor, k: int | None = None) -> tuple[Tensor, Tensor]:
    """[T, C] dense -> (idx [C, k], val [C, k]) keeping the k largest |entries| (nonzero only)."""
    T, C = X.shape
    k = T if k is None else min(k, T)
    vals, idx = X.abs().topk(k, dim=0)  # [k, C]
    v = torch.gather(X, 0, idx)
    idx = torch.where(vals > 0, idx, torch.full_like(idx, -1))
    v = torch.where(vals > 0, v, torch.zeros_like(v))
    return idx.T.contiguous(), v.T.contiguous()


def batched_lstsq(PhiS: Tensor, Y: Tensor, ridge: float = 0.0) -> Tensor:
    """Least squares per column: PhiS [C, m, k], Y [m, C] -> x [C, k] (normal equations in fp64)."""
    A = PhiS.double()
    b = Y.T.double()[..., None]
    G = A.transpose(1, 2) @ A
    if ridge:
        G = G + ridge * torch.eye(G.shape[-1], dtype=G.dtype, device=G.device)
    rhs = A.transpose(1, 2) @ b
    # robust solve: lstsq handles rank deficiency (padded duplicate atoms)
    x = torch.linalg.lstsq(G, rhs, driver="gelsd" if G.device.type == "cpu" else None).solution
    return x[..., 0].to(Y.dtype)


def residual(Phi: Tensor, Y: Tensor, X: Tensor) -> Tensor:
    return Y - Phi @ X
