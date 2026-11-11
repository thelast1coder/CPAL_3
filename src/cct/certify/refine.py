"""Exact refinement (CCT-certified): recompute exact rows for kept nodes and for nodes whose
influence interval straddles the node-pruning threshold."""
from __future__ import annotations

import torch
from torch import Tensor


def rows_to_refine(kept_feature_nodes: Tensor, infl_lo: Tensor, infl_hi: Tensor, threshold: float,
                   n_features: int) -> Tensor:
    """Feature node indices (square-graph order, < n_features) to recompute exactly."""
    border = (infl_lo[:n_features] < threshold) & (infl_hi[:n_features] >= threshold)
    sel = kept_feature_nodes[:n_features].clone() | border
    return torch.nonzero(sel).flatten()


def refine_rows(op, A_sensed: Tensor, feature_rows: Tensor, node_to_target: Tensor) -> tuple[Tensor, int]:
    """Replace rows of A_sensed ([T, N] over all targets) for the given feature nodes with exact rows.
    node_to_target maps a feature node index to its target row (-1 if not a target)."""
    t = node_to_target[feature_rows]
    t = t[t >= 0]
    if t.numel() == 0:
        return A_sensed, 0
    exact = op.exact_rows(t).double().cpu()
    A = A_sensed.clone()
    A[t] = exact
    return A, int(t.numel())
