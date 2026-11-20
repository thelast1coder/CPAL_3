"""Node and edge pruning, reproducing circuit-tracer v0.5.0 graph.prune_graph exactly (K0.4 checks
equality on real graphs). Works on the square adjacency in circuit-tracer's node order
[features, errors, tokens, logits]."""
from __future__ import annotations

import os
from typing import NamedTuple

import torch
from torch import Tensor

from .influence import compute_influence, normalize_matrix


class PruneResult(NamedTuple):
    node_mask: Tensor
    edge_mask: Tensor
    cumulative_scores: Tensor
    node_influence: Tensor


def find_threshold(scores: Tensor, threshold: float) -> Tensor:
    sorted_scores = torch.sort(scores, descending=True).values
    cumulative = torch.cumsum(sorted_scores, dim=0) / torch.sum(sorted_scores)
    i = int(torch.searchsorted(cumulative, torch.tensor(threshold, dtype=cumulative.dtype,
                                                        device=cumulative.device)).item())
    i = min(i, len(cumulative) - 1)
    return sorted_scores[i]


def logit_weight_vector(n_nodes: int, logit_probs: Tensor, device=None, dtype=None) -> Tensor:
    w = torch.zeros(n_nodes, device=device, dtype=dtype)
    w[-logit_probs.numel():] = logit_probs.to(w)
    return w


def prune_adjacency(adjacency: Tensor, n_features: int, n_tokens: int, logit_probs: Tensor,
                    node_threshold: float = 0.8, edge_threshold: float = 0.98) -> PruneResult:
    """If the environment variable CCT_PRUNE_DEVICE names a device (e.g. cuda) and the adjacency is on
    the CPU, pruning runs there (same dtype, same operations) and the result comes back to the CPU; on
    out-of-memory it falls back to the CPU."""
    dev = os.environ.get("CCT_PRUNE_DEVICE")
    if dev and adjacency.device.type == "cpu" and torch.device(dev).type != "cpu":
        try:
            r = _prune(adjacency.to(dev), n_features, n_tokens, logit_probs, node_threshold, edge_threshold)
            return PruneResult(*(x.cpu() for x in r))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    return _prune(adjacency, n_features, n_tokens, logit_probs, node_threshold, edge_threshold)


def _prune(adjacency: Tensor, n_features: int, n_tokens: int, logit_probs: Tensor,
           node_threshold: float, edge_threshold: float) -> PruneResult:
    n_logits = logit_probs.numel()
    w = logit_weight_vector(adjacency.shape[0], logit_probs, adjacency.device, adjacency.dtype)
    node_infl = compute_influence(normalize_matrix(adjacency), w)[0]
    node_mask = node_infl >= find_threshold(node_infl, node_threshold)
    node_mask[-n_logits - n_tokens:] = True
    pruned = adjacency.clone()
    pruned[~node_mask] = 0
    pruned[:, ~node_mask] = 0
    norm_pruned = normalize_matrix(pruned)
    pinf = compute_influence(norm_pruned, w)[0] + w
    edge_scores = norm_pruned * pinf[:, None]
    edge_mask = edge_scores >= find_threshold(edge_scores.flatten(), edge_threshold)
    old = node_mask.clone()
    node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
    node_mask[:n_features] &= edge_mask[:n_features].any(1)
    while not torch.all(node_mask == old):
        old[:] = node_mask
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[: -n_logits - n_tokens] &= edge_mask[:, : -n_logits - n_tokens].any(0)
        node_mask[:n_features] &= edge_mask[:n_features].any(1)
    s, idx = torch.sort(node_infl, descending=True)
    cum = torch.cumsum(s, 0) / torch.sum(s)
    final = torch.zeros_like(node_infl)
    final[idx] = cum
    return PruneResult(node_mask, edge_mask, final, node_infl)


def prune_reference(adjacency: Tensor, n_features: int, n_tokens: int, logit_probs: Tensor,
                    node_threshold: float = 0.8, edge_threshold: float = 0.98) -> tuple[Tensor, Tensor]:
    """Slow, loop-based reference used only by test T13."""
    n = adjacency.shape[0]
    n_logits = logit_probs.numel()
    A = adjacency.double()

    def norm(M):
        out = torch.zeros_like(M)
        for i in range(n):
            s = sum(abs(float(x)) for x in M[i])
            s = max(s, 1e-10)
            for j in range(n):
                out[i, j] = abs(float(M[i, j])) / s
        return out

    def infl(M, w):
        tot = torch.zeros(n, dtype=torch.float64)
        cur = w.clone()
        for _ in range(n + 1):
            nxt = torch.zeros(n, dtype=torch.float64)
            for i in range(n):
                if cur[i] != 0:
                    for j in range(n):
                        nxt[j] += cur[i] * M[i, j]
            cur = nxt
            tot += cur
        return tot

    def thresh(scores, t):
        v = sorted([float(x) for x in scores], reverse=True)
        total = sum(v)
        c = 0.0
        for x in v:
            c += x
            if c / total >= t - 1e-12:
                return x
        return v[-1]

    w = torch.zeros(n, dtype=torch.float64)
    w[-n_logits:] = logit_probs.double()
    ni = infl(norm(A), w)
    th = thresh(ni, node_threshold)
    keep = [float(ni[i]) >= th - 1e-15 for i in range(n)]
    for i in range(n - n_logits - n_tokens, n):
        keep[i] = True
    P = A.clone()
    for i in range(n):
        if not keep[i]:
            P[i, :] = 0
            P[:, i] = 0
    NP = norm(P)
    pi = infl(NP, w) + w
    es = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(n):
            es[i, j] = NP[i, j] * pi[i]
    eth = thresh(es.flatten(), edge_threshold)
    E = es >= eth - 1e-15
    keep_t = torch.tensor(keep)
    while True:
        old = keep_t.clone()
        for j in range(n - n_logits - n_tokens):
            keep_t[j] = keep_t[j] and bool(E[:, j].any())
        for i in range(n_features):
            keep_t[i] = keep_t[i] and bool(E[i].any())
        E[~keep_t] = False
        E[:, ~keep_t] = False
        if torch.equal(old, keep_t):
            break
    return keep_t, E
