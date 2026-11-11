"""Hub columns (PROJECT.md 5.5): dense columns that sparse recovery cannot fit are computed exactly
with forward-mode passes. A hub only affects its own column."""
from __future__ import annotations

import torch
from torch import Tensor

from .operators.base import KIND_ERROR, KIND_FEATURE, KIND_TOKEN
from .recovery.common import SparseColumns


def flag_hubs(rec: SparseColumns, energy: Tensor, k_max: int, rho_hub: float, budget: int,
              source_meta=None, class_rules: dict | None = None) -> Tensor:
    """Return source ids to compute exactly, largest estimated energy first, capped at `budget`.
    Flag rule: relative residual > rho_hub, or recovered support reached k_max.
    class_rules: optional {"token": bool, "error": bool, "first_position": bool} forcing classes in."""
    flagged = (rec.resid_rel > rho_hub) | (rec.n_atoms >= k_max)
    ids = rec.cols[flagged]
    if class_rules and source_meta is not None:
        extra = torch.zeros(len(source_meta), dtype=torch.bool)
        if class_rules.get("token"):
            extra |= source_meta.kind == KIND_TOKEN
        if class_rules.get("error"):
            extra |= source_meta.kind == KIND_ERROR
        if class_rules.get("first_position"):
            extra |= (source_meta.kind == KIND_FEATURE) & (source_meta.pos == source_meta.pos[source_meta.kind == KIND_FEATURE].min()) if bool((source_meta.kind == KIND_FEATURE).any()) else extra
        extra &= energy > 0
        ids = torch.unique(torch.cat([ids, torch.nonzero(extra).flatten()]))
    if ids.numel() == 0 or budget <= 0:
        return ids[:0]
    order = torch.argsort(energy[ids], descending=True)
    return ids[order][:budget]


def compute_hub_columns(op, hub_ids: Tensor, target_idx: Tensor) -> Tensor:
    """Exact columns for hub sources over the sensed targets: [T_sensed, n_hubs]. Charged to the
    operator's forward-mode counter."""
    if hub_ids.numel() == 0:
        return torch.zeros(target_idx.numel(), 0, dtype=torch.float64)
    cols = op.exact_cols(hub_ids)  # [n_hubs, T]
    return cols[:, target_idx].T.double().cpu()
