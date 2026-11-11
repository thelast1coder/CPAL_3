"""Group testing for interventions (PROJECT.md 5.9, P4).

Find the k features whose ablation changes a metric most, among N candidates, with about C k log N
group ablations instead of N single ones. Designs: Bernoulli inclusion with probability p.
Decoders:
- nnls: sparse regression with non-negativity on effect sizes |f(ablate group) - f(base)|
  (projected gradient), then top k;
- comp_dd: standard non-adaptive group testing on binarised outcomes (positive if the change exceeds
  tau): COMP clears every item in a negative test; DD marks as definite any remaining candidate that
  is alone among candidates in a positive test; ranking: definites first, then by the nnls score.
"""
from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor


class Ablator(Protocol):
    N: int

    def base(self) -> float: ...

    def ablate(self, mask: Tensor) -> float:
        """Metric with features in mask (bool [N]) set to zero."""


class SyntheticAblator:
    """Additive synthetic model for tests: f(mask) = f0 - sum_{i in mask} beta_i + noise."""

    def __init__(self, N: int, k: int, seed: int = 0, noise: float = 0.0, big: float = 1.0,
                 small: float = 0.0):
        g = torch.Generator().manual_seed(seed)
        self.N = N
        self.beta = small * torch.rand(N, generator=g, dtype=torch.float64)
        self.true = torch.randperm(N, generator=g)[:k]
        self.beta[self.true] = big * (1 + torch.rand(k, generator=g, dtype=torch.float64))
        self.noise = noise
        self.g = g
        self.calls = 0

    def base(self) -> float:
        return 0.0

    def ablate(self, mask: Tensor) -> float:
        self.calls += 1
        e = float(self.noise * torch.randn(1, generator=self.g, dtype=torch.float64))
        return -float(self.beta[mask].sum()) + e


class TinyAblator:
    """Ablations in the real (nonlinear) tiny replacement model."""

    def __init__(self, op):
        self.op = op
        self.N = op.active.shape[0]
        self._base = float(op.replacement_forward())
        self.calls = 0

    def base(self) -> float:
        return self._base

    def ablate(self, mask: Tensor) -> float:
        self.calls += 1
        scale = torch.ones(self.N, dtype=self.op.model.dtype)
        scale[mask] = 0
        return float(self.op.replacement_forward(scale))


def bernoulli_design(n_tests: int, N: int, p: float, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n_tests, N, generator=g) < p


def default_p(k: int) -> float:
    return min(0.5, 1.0 / max(k, 1))


def n_tests_for(k: int, N: int, C: float = 2.0) -> int:
    return int(math.ceil(C * k * math.log(max(N, 2))))


def run_design(ablator: Ablator, design: Tensor) -> Tensor:
    b = ablator.base()
    return torch.tensor([abs(ablator.ablate(row) - b) for row in design], dtype=torch.float64)


def nnls_decoder(design: Tensor, y: Tensor, n_iter: int = 2000) -> Tensor:
    X = design.double()
    L = float(torch.linalg.matrix_norm(X, ord=2) ** 2) + 1e-12
    beta = torch.zeros(X.shape[1], dtype=torch.float64)
    for _ in range(n_iter):
        beta = (beta - X.T @ (X @ beta - y) / L).clamp_min(0)
    return beta


def comp_dd_decoder(design: Tensor, y: Tensor, tau: float) -> tuple[Tensor, Tensor]:
    pos = y > tau
    neg_items = design[~pos].any(0) if bool((~pos).any()) else torch.zeros(design.shape[1], dtype=torch.bool)
    cand = ~neg_items
    definite = torch.zeros_like(cand)
    for t in torch.nonzero(pos).flatten():
        in_t = design[t] & cand
        if int(in_t.sum()) == 1:
            definite |= in_t
    return cand, definite


def decode_topk(design: Tensor, y: Tensor, k: int, method: str = "nnls", tau: float | None = None) -> Tensor:
    score = nnls_decoder(design, y)
    if method == "comp_dd":
        tau = float(y.median()) / 2 if tau is None else tau
        cand, definite = comp_dd_decoder(design, y, tau)
        score = score + 1e6 * definite.double() + 1e3 * cand.double()
    return torch.topk(score, k).indices


def exhaustive_topk(ablator: Ablator, k: int) -> tuple[Tensor, Tensor]:
    b = ablator.base()
    eff = torch.zeros(ablator.N, dtype=torch.float64)
    for i in range(ablator.N):
        m = torch.zeros(ablator.N, dtype=torch.bool)
        m[i] = True
        eff[i] = abs(ablator.ablate(m) - b)
    return torch.topk(eff, k).indices, eff


def recall(found: Tensor, truth: Tensor) -> float:
    return len(set(found.tolist()) & set(truth.tolist())) / max(1, truth.numel())
