"""The operator interface that all method code talks to.

A is the attribution matrix, T x N: A[t, s] is the direct edge from source s to target t.
Targets are ordered logits first, then features. Sources are ordered features, then error nodes
(layer-major: layer 0 positions 0..n-1, layer 1, ...), then token embeddings. This matches
circuit-tracer's node order (graph.py docstring), so a T x N matrix maps to its square adjacency
without reordering (see cct.graph.export).

Layer convention used by the causal mask:
- token embedding sources have layer -1;
- feature and error sources have the layer they are read/written at (for a cross-layer transcoder
  feature, its encoder layer, which is also its lowest write layer);
- feature targets have their encoder layer; logit targets have layer n_layers.
A source (l_s, p_s) can reach a target (l_t, p_t) only if l_t > l_s and p_t >= p_s.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

KIND_LOGIT, KIND_FEATURE, KIND_ERROR, KIND_TOKEN = 0, 1, 2, 3
KIND_NAMES = {KIND_LOGIT: "logit", KIND_FEATURE: "feature", KIND_ERROR: "error", KIND_TOKEN: "token"}


@dataclass
class NodeMeta:
    """Per-node metadata (targets or sources). All tensors have length T or N."""

    kind: Tensor  # long, one of KIND_*
    layer: Tensor  # long, see module docstring
    pos: Tensor  # long
    feature: Tensor  # long, transcoder feature id (-1 if not a feature)
    extra: Tensor | None = None  # float: activation for feature sources, prob for logit targets

    def __len__(self) -> int:
        return int(self.kind.shape[0])

    def subset(self, idx: Tensor) -> "NodeMeta":
        return NodeMeta(
            self.kind[idx], self.layer[idx], self.pos[idx], self.feature[idx],
            None if self.extra is None else self.extra[idx],
        )

    def to_dict(self) -> dict:
        return {k: (None if v is None else v.cpu()) for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "NodeMeta":
        return NodeMeta(**d)


TargetMeta = NodeMeta
SourceMeta = NodeMeta


@dataclass
class CausalMask:
    """Compact reachability: target (l_t, p_t) reachable from source (l_s, p_s) iff
    l_t > l_s and p_t >= p_s. Stored as layer and position vectors only."""

    t_layer: Tensor
    t_pos: Tensor
    s_layer: Tensor
    s_pos: Tensor

    def block(self, t_idx: Tensor | slice | None = None, s_idx: Tensor | slice | None = None,
              device: torch.device | str | None = None) -> Tensor:
        """Boolean [len(t_idx), len(s_idx)] block of the mask."""
        tl = self.t_layer if t_idx is None else self.t_layer[t_idx]
        tp = self.t_pos if t_idx is None else self.t_pos[t_idx]
        sl = self.s_layer if s_idx is None else self.s_layer[s_idx]
        sp = self.s_pos if s_idx is None else self.s_pos[s_idx]
        if device is not None:
            tl, tp, sl, sp = tl.to(device), tp.to(device), sl.to(device), sp.to(device)
        return (tl[:, None] > sl[None, :]) & (tp[:, None] >= sp[None, :])

    def dense(self) -> Tensor:
        return self.block()

    def columns(self, s_idx: Tensor, device=None) -> Tensor:
        """Mask for a set of columns, shape [T, len(s_idx)]."""
        return self.block(None, s_idx, device=device)

    def subset_targets(self, t_idx: Tensor) -> "CausalMask":
        return CausalMask(self.t_layer[t_idx], self.t_pos[t_idx], self.s_layer, self.s_pos)

    def subset_sources(self, s_idx: Tensor) -> "CausalMask":
        return CausalMask(self.t_layer, self.t_pos, self.s_layer[s_idx], self.s_pos[s_idx])

    def depth_bound(self) -> int:
        """Upper bound on the longest path (in edges) of any graph respecting the mask.
        Layers strictly increase along a path, so the number of distinct layers bounds it."""
        layers = torch.cat([self.t_layer, self.s_layer]).unique()
        return int(layers.numel())


@dataclass
class CostCounters:
    """Counts every pass an operator performs, with wall time per kind."""

    mixed_rows: int = 0  # rows obtained by backward passes with arbitrary (mixed) seeds
    exact_rows: int = 0  # rows obtained by one-hot seeds
    forward_passes: int = 0  # columns obtained by forward-mode passes
    backward_calls: int = 0  # number of batched backward calls
    forward_calls: int = 0
    seconds: dict = field(default_factory=lambda: {"rows": 0.0, "cols": 0.0})

    def reset(self) -> None:
        self.mixed_rows = self.exact_rows = self.forward_passes = 0
        self.backward_calls = self.forward_calls = 0
        self.seconds = {"rows": 0.0, "cols": 0.0}

    def snapshot(self) -> dict:
        return {
            "mixed_rows": self.mixed_rows, "exact_rows": self.exact_rows,
            "forward_passes": self.forward_passes, "backward_calls": self.backward_calls,
            "forward_calls": self.forward_calls, "seconds": dict(self.seconds),
        }


@runtime_checkable
class AttributionOperator(Protocol):
    T: int
    N: int
    target_meta: NodeMeta
    source_meta: NodeMeta
    counters: CostCounters

    def rows(self, seeds: Tensor) -> Tensor:
        """seeds [b, T] -> seeds @ A, [b, N] (one batched mixed backward pass)."""

    def cols(self, tangents: Tensor) -> Tensor:
        """tangents [b, N] -> tangents @ A.T, [b, T] (one batched forward-mode pass)."""

    def exact_rows(self, idx: Tensor) -> Tensor:
        """Rows idx of A by one-hot seeds; must equal rows(onehot)."""

    def causal_mask(self) -> CausalMask: ...

    def logit_weights(self) -> Tensor:
        """w [T]: logit probabilities on logit targets, zero elsewhere."""


class OperatorMixin:
    """Shared helpers: batching, counting and timing around _rows_impl / _cols_impl."""

    T: int
    N: int
    target_meta: NodeMeta
    source_meta: NodeMeta
    counters: CostCounters
    batch_size: int = 256
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32

    def _rows_impl(self, seeds: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _cols_impl(self, tangents: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def rows(self, seeds: Tensor, _exact: bool = False) -> Tensor:
        seeds = seeds.to(self.device, self.dtype)
        out = []
        for i in range(0, seeds.shape[0], self.batch_size):
            chunk = seeds[i : i + self.batch_size]
            self._sync()
            t0 = time.perf_counter()
            out.append(self._rows_impl(chunk))
            self._sync()
            self.counters.seconds["rows"] += time.perf_counter() - t0
            self.counters.backward_calls += 1
            if _exact:
                self.counters.exact_rows += chunk.shape[0]
            else:
                self.counters.mixed_rows += chunk.shape[0]
        return torch.cat(out, 0) if out else seeds.new_zeros(0, self.N)

    def cols(self, tangents: Tensor) -> Tensor:
        tangents = tangents.to(self.device, self.dtype)
        out = []
        for i in range(0, tangents.shape[0], self.batch_size):
            chunk = tangents[i : i + self.batch_size]
            self._sync()
            t0 = time.perf_counter()
            out.append(self._cols_impl(chunk))
            self._sync()
            self.counters.seconds["cols"] += time.perf_counter() - t0
            self.counters.forward_calls += 1
            self.counters.forward_passes += chunk.shape[0]
        return torch.cat(out, 0) if out else tangents.new_zeros(0, self.T)

    def exact_rows(self, idx: Tensor) -> Tensor:
        idx = torch.as_tensor(idx, dtype=torch.long)
        seeds = torch.zeros(idx.numel(), self.T, dtype=self.dtype)
        seeds[torch.arange(idx.numel()), idx] = 1.0
        return self.rows(seeds, _exact=True)

    def exact_cols(self, idx: Tensor) -> Tensor:
        """Columns idx of A, returned as [len(idx), T] (one forward-mode pass each)."""
        idx = torch.as_tensor(idx, dtype=torch.long)
        tangents = torch.zeros(idx.numel(), self.N, dtype=self.dtype)
        tangents[torch.arange(idx.numel()), idx] = 1.0
        return self.cols(tangents)

    def causal_mask(self) -> CausalMask:
        return CausalMask(self.target_meta.layer, self.target_meta.pos,
                          self.source_meta.layer, self.source_meta.pos)

    def logit_weights(self) -> Tensor:
        w = torch.zeros(self.T, dtype=torch.float64)
        is_logit = self.target_meta.kind == KIND_LOGIT
        w[is_logit] = self.target_meta.extra[is_logit].double()
        return w
