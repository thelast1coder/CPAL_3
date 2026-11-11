"""B1, truncation: the graph from only the first c rows of the exact run's visiting order.

circuit-tracer with max_feature_nodes = c visits features in influence order and, at the end, drops
unvisited features as nodes entirely (attribute.py, Phase 5: col_read keeps selected features only).
On the exact node layout this is the same as zeroing both the rows and the columns of unvisited
features (zero rows and columns change no influence, threshold or edge score; K0.6 checks it against
a real capped run). c counts all rows, logits included (logit rows are always visited first).
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..graph.export import GraphLayout, to_square


def truncation_adjacency(A: Tensor, visit_order: Tensor, c: int, layout: GraphLayout) -> Tensor:
    """A [T, N] exact; visit_order: target rows in visiting order (logits first)."""
    visited = torch.zeros(A.shape[0], dtype=torch.bool)
    visited[visit_order[:c]] = True
    visited[layout.logit_targets] = True
    At = A.clone()
    At[~visited] = 0
    S = to_square(At, layout)
    f2t = layout.feature_node_to_target
    drop = (f2t < 0) | ~visited[f2t.clamp_min(0)]
    drop_nodes = torch.nonzero(drop).flatten()
    S[drop_nodes] = 0
    S[:, drop_nodes] = 0
    return S
