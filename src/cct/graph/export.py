"""Square adjacency in circuit-tracer's node order and export in its graph format.

Operator layout: A is T x N with targets [logits, feature targets] and sources
[source features, errors (layer-major), tokens]. circuit-tracer's adjacency is square over
[selected features, errors, tokens, logits] (rows = targets, columns = sources); rows of error and
token nodes are zero, columns of logit nodes are zero.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..operators.base import KIND_FEATURE, KIND_LOGIT, NodeMeta


@dataclass
class GraphLayout:
    n_features: int
    n_errors: int
    n_tokens: int
    n_logits: int
    feature_node_to_target: Tensor  # [n_features] target row (into A) of each feature node, -1 if none
    logit_targets: Tensor  # [n_logits] target rows of the logits
    logit_probs: Tensor

    @property
    def n_nodes(self) -> int:
        return self.n_features + self.n_errors + self.n_tokens + self.n_logits


def make_layout(target_meta: NodeMeta, source_meta: NodeMeta) -> GraphLayout:
    s_feat = torch.nonzero(source_meta.kind == KIND_FEATURE).flatten()
    n_f = s_feat.numel()
    n_tok = int((source_meta.layer == -1).sum())
    n_err = len(source_meta) - n_f - n_tok
    t_feat = torch.nonzero(target_meta.kind == KIND_FEATURE).flatten()
    key_t = {(int(target_meta.layer[t]), int(target_meta.pos[t]), int(target_meta.feature[t])): int(t)
             for t in t_feat}
    f2t = torch.tensor([key_t.get((int(source_meta.layer[s]), int(source_meta.pos[s]),
                                   int(source_meta.feature[s])), -1) for s in s_feat], dtype=torch.long)
    logit_t = torch.nonzero(target_meta.kind == KIND_LOGIT).flatten()
    probs = target_meta.extra[logit_t].double()
    return GraphLayout(n_f, n_err, n_tok, logit_t.numel(), f2t, logit_t, probs)


def to_square(A: Tensor, layout: GraphLayout) -> Tensor:
    """[T, N] -> square [n_nodes, n_nodes]."""
    n = layout.n_nodes
    N = A.shape[1]
    assert N == layout.n_features + layout.n_errors + layout.n_tokens
    S = torch.zeros(n, n, dtype=A.dtype)
    has = layout.feature_node_to_target >= 0
    S[torch.nonzero(has).flatten(), :N] = A[layout.feature_node_to_target[has]]
    S[n - layout.n_logits:, :N] = A[layout.logit_targets]
    return S


def square_mask(mask_TN: Tensor, layout: GraphLayout) -> Tensor:
    """Causal mask [T, N] -> square boolean mask in node order."""
    return to_square(mask_TN.to(torch.float64), layout) > 0


def to_circuit_tracer_dict(adjacency: Tensor, base: dict) -> dict:
    """Return a dict with the fields of circuit_tracer.graph.Graph.to_pt, copying every field from the
    exact graph's dict `base` and replacing the adjacency. On the VM: Graph(**d) builds the object so
    circuit-tracer's viewers and create_graph_files work unchanged."""
    d = dict(base)
    d["adjacency_matrix"] = adjacency.float().cpu()
    return d


def save_graph(path: str, adjacency: Tensor, base: dict) -> None:
    torch.save(to_circuit_tracer_dict(adjacency, base), path)
