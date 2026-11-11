"""DenseOperator: an attribution operator backed by a stored matrix A (tests and offline simulation).

Because mixed passes are exactly linear (K0.2), a mixed backward pass on the real model equals a
matrix product with the exact matrix, so simulation through this operator is legitimate. Its
counters count products as if they were real passes.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .base import CostCounters, NodeMeta, OperatorMixin


class DenseOperator(OperatorMixin):
    def __init__(self, A: Tensor, target_meta: NodeMeta, source_meta: NodeMeta,
                 batch_size: int = 256, device: str | torch.device = "cpu",
                 dtype: torch.dtype = torch.float32):
        assert A.shape == (len(target_meta), len(source_meta)), "A must be T x N"
        self.device = torch.device(device)
        self.dtype = dtype
        self.A = A.to(self.device, dtype)
        self.T, self.N = A.shape
        self.target_meta = target_meta
        self.source_meta = source_meta
        self.batch_size = batch_size
        self.counters = CostCounters()

    def _rows_impl(self, seeds: Tensor) -> Tensor:
        return seeds @ self.A

    def _cols_impl(self, tangents: Tensor) -> Tensor:
        return tangents @ self.A.T

    @staticmethod
    def from_arrays(A: Tensor, t_layer, t_pos, s_layer, s_pos, t_kind=None, s_kind=None,
                    logit_probs: Tensor | None = None, **kw) -> "DenseOperator":
        """Convenience constructor for synthetic tests."""
        from .base import KIND_FEATURE, KIND_LOGIT

        T, N = A.shape
        t_kind = torch.full((T,), KIND_FEATURE) if t_kind is None else t_kind
        s_kind = torch.full((N,), KIND_FEATURE) if s_kind is None else s_kind
        t_extra = torch.zeros(T)
        if logit_probs is not None:
            t_extra[t_kind == KIND_LOGIT] = logit_probs.float()
        tm = NodeMeta(t_kind.long(), torch.as_tensor(t_layer).long(), torch.as_tensor(t_pos).long(),
                      torch.full((T,), -1), t_extra)
        sm = NodeMeta(s_kind.long(), torch.as_tensor(s_layer).long(), torch.as_tensor(s_pos).long(),
                      torch.full((N,), -1), torch.ones(N))
        return DenseOperator(A, tm, sm, **kw)
