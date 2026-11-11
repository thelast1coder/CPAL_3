"""B5, one-pass total-effect node ranking: gradient times activation of the prob-weighted logit
through the linearised model with features unfrozen (node ranking only, no edges).

Implemented on the tiny backend (TinyOperator.total_effect). On the circuit-tracer backend this is a
P2 baseline (EXPERIMENTS.md X1) and is not needed for the kill test; it raises until implemented.
"""
from __future__ import annotations

import torch
from torch import Tensor


def total_effect_scores(op) -> Tensor:
    if not hasattr(op, "total_effect"):
        raise NotImplementedError("B5 total effect is implemented for the tiny backend only (P2 item)")
    return op.total_effect().abs().double().cpu()


def total_effect_keep(scores: Tensor, n_keep: int) -> Tensor:
    keep = torch.zeros_like(scores, dtype=torch.bool)
    keep[torch.topk(scores, min(n_keep, scores.numel())).indices] = True
    return keep
