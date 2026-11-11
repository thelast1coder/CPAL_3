"""Influence bounds from the finite Neumann series (PROJECT.md 5.6).

(a) Global bound. If the row-normalised matrix is off by E with eps = ||E||_inf (largest absolute
    row sum) and ||w||_1 <= 1, then ||w^T (B' - B)||_1 <= sum_{k=1..D} ((1 + eps)^k - 1), because
    ||Abar||_inf <= 1 and ||X^k - Y^k||_inf <= (||Y|| + eps)^k - ||Y||^k.
(b) Data-dependent per-node interval. From per-column error bounds c_s >= ||a_s - a_hat_s||_2
    (so |A[t,s] - A_hat[t,s]| <= c_s for every row t; this is the conservative column-to-entry
    conversion) we get entrywise bounds on the normalised matrix:
        r_t in [r_hat_t - eps_t, r_hat_t + eps_t],  eps_t = sum_{s reachable from t} c_s,
        Lo[t,s] = max(|A_hat[t,s]| - c_s, 0) / (r_hat_t + eps_t),
        Hi[t,s] = min(1, (|A_hat[t,s]| + c_s) / (r_hat_t - eps_t))  (= 1 if r_hat_t <= eps_t),
    for (t, s) inside the causal mask; rows known exactly (logits, refined rows) get c = 0.
    Total influence g = w + w Abar + w Abar^2 + ... is monotone in the nonnegative entries of Abar,
    so g_lo = w sum_k Lo^k <= g <= w sum_k Hi^k = g_hi; both series are finite because Lo and Hi
    share the DAG pattern. Node influence interval = [g_lo - w, g_hi - w].
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..graph.influence import compute_influence


def global_bound(eps: float, D: int) -> float:
    return float(sum((1.0 + eps) ** k - 1.0 for k in range(1, D + 1)))


def entry_intervals(A_hat: Tensor, col_bound: Tensor, mask: Tensor, exact_rows: Tensor | None = None
                    ) -> tuple[Tensor, Tensor]:
    """A_hat [n, n] square adjacency estimate; col_bound [n] per source node (0 for exact columns);
    mask [n, n] bool causal pattern (True where an edge can exist); exact_rows [n] bool."""
    Ah = A_hat.abs().double()
    C = col_bound.double()[None, :].expand_as(Ah) * mask.double()
    if exact_rows is not None:
        C = C * (~exact_rows).double()[:, None]
    r_hat = Ah.sum(1)
    eps = C.sum(1)
    lo_den = (r_hat + eps).clamp(min=1e-10)
    Lo = (Ah - C).clamp(min=0) / lo_den[:, None]
    hi_den = r_hat - eps
    Hi = torch.where(hi_den[:, None] > 1e-10, (Ah + C) / hi_den.clamp(min=1e-10)[:, None],
                     torch.ones_like(Ah))
    Hi = Hi.clamp(max=1.0) * mask.double()
    Hi = torch.where(Ah + C > 0, Hi, torch.zeros_like(Hi))
    return Lo, Hi


def influence_interval(Lo: Tensor, Hi: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    w = w.double()
    lo = compute_influence(Lo, w)[0]
    hi = compute_influence(Hi, w)[0]
    return lo, hi


def row_l1_error_bound(A_hat: Tensor, col_bound: Tensor, mask: Tensor,
                       exact_rows: Tensor | None = None) -> Tensor:
    """Per-row bound on || normalised row error ||_1 = 2 eps_t / (r_t), capped at 2."""
    Ah = A_hat.abs().double()
    C = col_bound.double()[None, :] * mask.double()
    if exact_rows is not None:
        C = C * (~exact_rows).double()[:, None]
    eps = C.sum(1)
    r = Ah.sum(1)
    den = (r - eps)
    out = torch.where(den > 1e-10, 2 * eps / den.clamp(min=1e-10), torch.full_like(eps, 2.0))
    out = torch.where(eps == 0, torch.zeros_like(out), out)
    return out.clamp(max=2.0)
