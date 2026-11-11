"""B2, random rows at equal budget: logit rows plus random feature rows; other rows are empty."""
from __future__ import annotations

import torch
from torch import Tensor

from ..graph.export import GraphLayout, to_square


def random_rows_adjacency(A: Tensor, c: int, layout: GraphLayout, seed: int) -> Tensor:
    T = A.shape[0]
    is_logit = torch.zeros(T, dtype=torch.bool)
    is_logit[layout.logit_targets] = True
    feat = torch.nonzero(~is_logit).flatten()
    g = torch.Generator().manual_seed(seed)
    n = max(0, min(c - int(is_logit.sum()), feat.numel()))
    pick = feat[torch.randperm(feat.numel(), generator=g)[:n]]
    keep = is_logit.clone()
    keep[pick] = True
    At = A.clone()
    At[~keep] = 0
    return to_square(At, layout)
