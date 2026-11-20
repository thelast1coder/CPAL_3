"""Influence on the logits (PROJECT.md 5.6), matching circuit-tracer v0.5.0 graph.py:
normalize_matrix (row-normalised |A|, row sums clamped at 1e-10) and compute_influence
(w A + w A^2 + ... until the product is exactly zero, which happens after at most D steps because the
graph is a DAG)."""
from __future__ import annotations

import torch
from torch import Tensor


def normalize_matrix(A: Tensor) -> Tensor:
    Ab = A.abs()
    return Ab / Ab.sum(dim=1, keepdim=True).clamp(min=1e-10)


def compute_influence(Abar: Tensor, w: Tensor, max_iter: int = 1000) -> tuple[Tensor, int]:
    """Returns (w^T (Abar + Abar^2 + ...), number of nonzero terms)."""
    cur = w @ Abar
    infl = cur.clone()
    it = 1
    while bool(cur.any()):
        if it > max_iter:
            raise RuntimeError("influence did not terminate: graph is not a DAG")
        cur = cur @ Abar
        infl = infl + cur
        it += 1
    return infl, it - 1


def partial_influences(edge_rows: Tensor, logit_p: Tensor, row_to_node: Tensor, max_iter: int = 128) -> Tensor:
    """Copy of circuit-tracer 0.5.0 graph.compute_partial_influences (used to order the visits).
    edge_rows [r, n_nodes] (rows computed so far), row_to_node [r] node index of each row."""
    M = edge_rows.abs()
    M = M / M.sum(dim=1, keepdim=True).clamp(min=1e-8)
    infl = torch.zeros(edge_rows.shape[1], dtype=M.dtype, device=M.device)
    prod = torch.zeros(edge_rows.shape[1], dtype=M.dtype, device=M.device)
    prod[-logit_p.numel():] = logit_p.to(M.dtype)
    for _ in range(max_iter):
        prod = prod[row_to_node.long().to(M.device)] @ M
        if not bool(prod.any()):
            break
        infl += prod
    else:
        raise RuntimeError("Failed to converge")
    return infl


def node_influence(adjacency: Tensor, w: Tensor) -> Tensor:
    return compute_influence(normalize_matrix(adjacency), w)[0]


def neumann_B(Abar: Tensor, D: int | None = None) -> Tensor:
    """B = Abar + Abar^2 + ... (finite); small matrices only (tests)."""
    n = Abar.shape[0]
    B = torch.zeros_like(Abar)
    P = torch.eye(n, dtype=Abar.dtype)
    for _ in range(D if D is not None else n + 1):
        P = P @ Abar
        if not bool(P.any()):
            break
        B = B + P
    return B


def nilpotency_index(Abar: Tensor) -> int:
    """Smallest k with Abar^k = 0 (exact zero pattern), computed on the boolean pattern."""
    P = (Abar != 0).to(torch.float64)
    M = P.clone()
    k = 1
    while bool(M.any()):
        M = ((M @ P) > 0).to(torch.float64)
        k += 1
        if k > Abar.shape[0] + 1:
            raise RuntimeError("not nilpotent")
    return k
