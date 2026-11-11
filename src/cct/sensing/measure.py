"""Measurement: Y = Phi A by mixed backward passes through an operator."""
from __future__ import annotations

import torch
from torch import Tensor

from ..operators.base import KIND_LOGIT


def sense(op, Phi: Tensor, target_idx: Tensor | None = None, batch: int | None = None) -> Tensor:
    """Y [m, N] = Phi @ A[target_idx]. Phi is [m, len(target_idx)]; the operator is called with
    seeds placed on target_idx (other targets get weight 0). Costs m mixed rows."""
    T = op.T
    if target_idx is None:
        target_idx = torch.arange(T)
    m = Phi.shape[0]
    old_bs = op.batch_size
    if batch is not None:
        op.batch_size = batch
    try:
        seeds = torch.zeros(m, T, dtype=Phi.dtype)
        seeds[:, target_idx] = Phi
        Y = op.rows(seeds)
    finally:
        op.batch_size = old_bs
    return Y


def feature_target_index(op) -> tuple[Tensor, Tensor]:
    """(logit target indices, feature target indices)."""
    is_logit = op.target_meta.kind == KIND_LOGIT
    return torch.nonzero(is_logit).flatten(), torch.nonzero(~is_logit).flatten()
