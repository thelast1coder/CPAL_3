"""B3, low-rank sketch at the same budget (randomised range finder, single-pass form of
Tropp, Yurtsever, Udell and Cevher 2017 (verify citation)).

Budget m is split: m_f = m // 2 forward-mode probes W = A Omega give the column space Q = orth(W);
m_b = m - m_f mixed backward passes give Y = Phi A; then A ~ Q (Phi Q)^+ Y. Cost = m_b + kappa * m_f
row-equivalents. Exact for matrices of rank <= m_f (test T14). Logit rows are computed exactly, as in
CCT, and the sketch is applied to the feature-target rows.
"""
from __future__ import annotations

import torch
from torch import Tensor

from ..sensing.measure import feature_target_index


def lowrank_sketch(op, m: int, seed: int) -> tuple[Tensor, dict]:
    logit_idx, feat_idx = feature_target_index(op)
    c0 = op.counters.snapshot()
    A_hat = torch.zeros(op.T, op.N, dtype=torch.float64)
    A_hat[logit_idx] = op.exact_rows(logit_idx).double().cpu()
    m_f = max(1, m // 2)
    m_b = max(m_f, m - m_f)
    g = torch.Generator().manual_seed(seed)
    Omega = torch.randn(m_f, op.N, generator=g, dtype=torch.float64)
    W = op.cols(Omega.to(op.dtype)).double().cpu()[:, feat_idx].T  # [T_f, m_f]
    Q, _ = torch.linalg.qr(W)
    Phi = torch.randn(m_b, feat_idx.numel(), generator=g, dtype=torch.float64) / m_b ** 0.5
    seeds = torch.zeros(m_b, op.T, dtype=torch.float64)
    seeds[:, feat_idx] = Phi
    Y = op.rows(seeds.to(op.dtype)).double().cpu()
    A_hat[feat_idx] = Q @ torch.linalg.pinv(Phi @ Q) @ Y
    c1 = op.counters.snapshot()
    cost = {"mixed_rows": c1["mixed_rows"] - c0["mixed_rows"], "exact_rows": c1["exact_rows"] - c0["exact_rows"],
            "forward_passes": c1["forward_passes"] - c0["forward_passes"]}
    return A_hat, cost
