"""Fidelity metrics against the exact graph (PROJECT.md 9). Both graphs share the node layout.
Node-set metrics use feature nodes only (the prereg's "pruned feature-node sets")."""
from __future__ import annotations

import torch
from torch import Tensor

from .influence import compute_influence, normalize_matrix
from .prune import PruneResult


def iw_node_recall(exact: PruneResult, approx_mask: Tensor, n_features: int) -> float:
    ke = exact.node_mask[:n_features]
    infl = exact.node_influence[:n_features].double()
    den = infl[ke].sum()
    if den <= 0:
        return 1.0
    return float(infl[ke & approx_mask[:n_features]].sum() / den)


def node_jaccard(exact_mask: Tensor, approx_mask: Tensor, n_features: int) -> float:
    a, b = exact_mask[:n_features], approx_mask[:n_features]
    u = int((a | b).sum())
    return 1.0 if u == 0 else int((a & b).sum()) / u


def topk_overlap(infl_exact: Tensor, infl_approx: Tensor, n_features: int, k: int) -> float:
    k = min(k, n_features)
    if k == 0:
        return 1.0
    a = set(torch.topk(infl_exact[:n_features], k).indices.tolist())
    b = set(torch.topk(infl_approx[:n_features], k).indices.tolist())
    return len(a & b) / k


def _rank(x: Tensor) -> Tensor:
    order = torch.argsort(x, stable=True)
    r = torch.empty_like(x, dtype=torch.float64)
    r[order] = torch.arange(x.numel(), dtype=torch.float64)
    # average ties
    xs = x[order]
    i = 0
    n = x.numel()
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a: Tensor, b: Tensor) -> float:
    if a.numel() < 2:
        return 1.0
    ra, rb = _rank(a.double()), _rank(b.double())
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = ra.norm() * rb.norm()
    return 1.0 if den == 0 else float((ra * rb).sum() / den)


def rel_frobenius_on_edges(A_exact: Tensor, A_approx: Tensor, edge_mask: Tensor) -> float:
    E = A_exact.double()[edge_mask]
    F = A_approx.double()[edge_mask]
    den = E.norm()
    return 0.0 if den == 0 else float((F - E).norm() / den)


def graph_scores(adjacency: Tensor, n_features: int, n_errors: int, n_tokens: int,
                 logit_probs: Tensor) -> tuple[float, float]:
    """Replacement and completeness scores, as circuit-tracer v0.5.0 compute_graph_scores. Runs on
    CCT_PRUNE_DEVICE when set (see cct.graph.prune.prune_adjacency), falling back to the CPU."""
    import os

    dev = os.environ.get("CCT_PRUNE_DEVICE")
    if dev and adjacency.device.type == "cpu" and torch.device(dev).type != "cpu":
        try:
            return graph_scores(adjacency.to(dev), n_features, n_errors, n_tokens, logit_probs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    n = adjacency.shape[0]
    n_logits = logit_probs.numel()
    error_start, error_end = n_features, n_features + n_errors
    token_end = error_end + n_tokens
    w = torch.zeros(n, dtype=adjacency.dtype, device=adjacency.device)
    w[-n_logits:] = logit_probs.to(adjacency.device, adjacency.dtype)
    Abar = normalize_matrix(adjacency)
    infl = compute_influence(Abar, w)[0]
    tok = infl[error_end:token_end].sum()
    err = infl[error_start:error_end].sum()
    repl = tok / (tok + err) if float(tok + err) > 0 else torch.tensor(float("nan"))
    non_err = 1 - Abar[:, error_start:error_end].sum(-1)
    out_infl = infl + w
    comp = (non_err * out_infl).sum() / out_infl.sum()
    return float(repl), float(comp)


def fidelity(A_exact: Tensor, exact: PruneResult, A_approx: Tensor, approx: PruneResult,
             n_features: int, n_errors: int, n_tokens: int, logit_probs: Tensor) -> dict:
    re, ce = graph_scores(A_exact, n_features, n_errors, n_tokens, logit_probs)
    ra, ca = graph_scores(A_approx, n_features, n_errors, n_tokens, logit_probs)
    return {
        "iw_node_recall": iw_node_recall(exact, approx.node_mask, n_features),
        "node_jaccard": node_jaccard(exact.node_mask, approx.node_mask, n_features),
        "top20_overlap": topk_overlap(exact.node_influence, approx.node_influence, n_features, 20),
        "top50_overlap": topk_overlap(exact.node_influence, approx.node_influence, n_features, 50),
        "spearman_influence": spearman(exact.node_influence[:n_features], approx.node_influence[:n_features]),
        "rel_frobenius_pruned_edges": rel_frobenius_on_edges(A_exact, A_approx, exact.edge_mask),
        "replacement_score_diff": ra - re,
        "completeness_score_diff": ca - ce,
        "n_kept_exact": int(exact.node_mask[:n_features].sum()),
        "n_kept_approx": int(approx.node_mask[:n_features].sum()),
    }


def meets_fidelity(f: dict, recall_min: float, jaccard_min: float) -> bool:
    return f["iw_node_recall"] >= recall_min and f["node_jaccard"] >= jaccard_min
