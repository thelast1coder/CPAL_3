"""Closed-form local part A_0 of the attribution matrix (PROJECT_V2.md Section 4, T2).

In the local replacement model the only live paths from a source's write point to a target's read
point are the residual stream and attention heads with frozen patterns (MLPs are stop-gradiented;
a transcoder skip connection, if present, is a live linear map at the same position). A_0 is the
part of A carried by paths with no attention head, so it only links nodes at the SAME position and
is block-diagonal over positions. With frozen normalisation scales the residual-to-read map at
(layer l, position p) is diagonal: read = D[l, p] * residual. Hence, without skip connections,

    A_0[t, s] = 1[pos(s) = pos(t)] * sum over writes (l_w, w) of s with l_w < l_t of (D[l_t, p] * r_t) . w

and with skip connections the residual gradient is propagated down through the skip maps
(g_{j-1} = g_j + D[j] * S_j^T g_j). Layer conventions: tokens write at layer -1, features and
errors at their (output) layer, feature targets read at their layer, logits read at n_layers.

Checked numerically against the attention-detached backward pass (CPU test, K0.7 on the VM).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

SkipVJP = Callable[[int, Tensor], Tensor]  # (layer, V [k, d]) -> S_layer^T V  [k, d]


@dataclass
class Reads:
    vecs: Tensor   # [T, d] read vectors in the read-hook space (encoder rows, logit vectors)
    layer: Tensor  # [T] read layer (n_layers for logits)
    pos: Tensor    # [T]


@dataclass
class Writes:
    vecs: Tensor   # [W, d] write vectors (activation x decoder, error vectors, token embeddings)
    layer: Tensor  # [W] write layer (-1 for tokens)
    pos: Tensor    # [W]
    src: Tensor    # [W] source column


class LocalPart:
    """A_0 stored as dense per-position blocks: for position p, targets tgt[p], sources src[p] and
    block[p] = A_0[tgt[p]][:, src[p]]."""

    def __init__(self, reads: Reads, writes: Writes, D: Tensor, T: int, N: int,
                 skip_vjp: SkipVJP | None = None, device=None, dtype=torch.float32):
        self.T, self.N = T, N
        self.device = torch.device(device) if device is not None else D.device
        self.dtype = dtype
        self.blocks: dict[int, tuple[Tensor, Tensor, Tensor]] = {}
        n_layers = D.shape[0] - 1
        D = D.to(self.device, dtype)
        rv, rl, rp = reads.vecs.to(self.device, dtype), reads.layer.cpu(), reads.pos.cpu()
        wv, wl, wp, ws = writes.vecs.to(self.device, dtype), writes.layer.cpu(), writes.pos.cpu(), writes.src.cpu()
        for p in torch.unique(rp).tolist():
            t_idx = torch.nonzero(rp == p).flatten()
            w_idx = torch.nonzero(wp == p).flatten()
            if w_idx.numel() == 0:
                continue
            srcs, inv = torch.unique(ws[w_idx], return_inverse=True)
            t_lay = rl[t_idx]
            w_lay = wl[w_idx]
            Wp = wv[w_idx.to(self.device)]  # [Wp, d]
            block = torch.zeros(t_idx.numel(), srcs.numel(), device=self.device, dtype=dtype)
            if skip_vjp is None:
                g = rv[t_idx.to(self.device)] * D[t_lay.to(self.device), p]  # [Tp, d]
                S = g @ Wp.T  # [Tp, Wp]
                S = S * (w_lay[None, :] < t_lay[:, None]).to(self.device, dtype)
                block.index_add_(1, inv.to(self.device), S)
            else:
                # g_l: gradient at the residual just after layer l's write, for every target at p
                g = torch.zeros(t_idx.numel(), rv.shape[1], device=self.device, dtype=dtype)
                for lw in range(n_layers - 1, -2, -1):
                    # targets reading at layer lw + 1 start here
                    start = (t_lay == lw + 1).to(self.device)
                    if bool(start.any()):
                        g[start] = rv[t_idx.to(self.device)][start] * D[lw + 1, p]
                    sel = torch.nonzero(w_lay == lw).flatten()
                    if sel.numel():
                        block.index_add_(1, inv[sel].to(self.device), g @ Wp[sel.to(self.device)].T)
                    if lw >= 0:  # move below layer lw: add the skip path of layer lw
                        g = g + D[lw, p] * skip_vjp(lw, g)
                # A target's g stays zero until the sweep reaches its read layer, so writes at or
                # above the read layer contribute nothing (the causal mask holds by construction).
            self.blocks[int(p)] = (t_idx, srcs, block)

    # -- views ---------------------------------------------------------------------------------
    def dense(self, device=None) -> Tensor:
        dev = torch.device(device) if device is not None else self.device
        A = torch.zeros(self.T, self.N, device=dev, dtype=self.dtype)
        for t_idx, srcs, b in self.blocks.values():
            A[t_idx.to(dev)[:, None], srcs.to(dev)[None, :]] = b.to(dev)
        return A

    def rows(self, idx: Tensor) -> Tensor:
        idx = torch.as_tensor(idx).long().cpu()
        out = torch.zeros(idx.numel(), self.N, device=self.device, dtype=self.dtype)
        where = torch.full((self.T,), -1, dtype=torch.long)
        where[idx] = torch.arange(idx.numel())
        for t_idx, srcs, b in self.blocks.values():
            loc = where[t_idx]
            keep = loc >= 0
            if bool(keep.any()):
                out[loc[keep].to(self.device)[:, None], srcs.to(self.device)[None, :]] = b[keep.to(self.device)]
        return out

    def cols(self, idx: Tensor) -> Tensor:
        """A_0[:, idx] as [T, len(idx)]."""
        idx = torch.as_tensor(idx).long().cpu()
        out = torch.zeros(self.T, idx.numel(), device=self.device, dtype=self.dtype)
        where = torch.full((self.N,), -1, dtype=torch.long)
        where[idx] = torch.arange(idx.numel())
        for t_idx, srcs, b in self.blocks.values():
            loc = where[srcs]
            keep = loc >= 0
            if bool(keep.any()):
                out[t_idx.to(self.device)[:, None], loc[keep].to(self.device)[None, :]] = b[:, keep.to(self.device)]
        return out

    def sketch(self, Phi: Tensor, target_idx: Tensor | None = None) -> Tensor:
        """Phi @ A_0[target_idx] for Phi [m, len(target_idx)] (all targets if None): [m, N]."""
        m = Phi.shape[0]
        Phi = Phi.to(self.device, self.dtype)
        if target_idx is not None:
            full = torch.zeros(m, self.T, device=self.device, dtype=self.dtype)
            full[:, torch.as_tensor(target_idx).long().to(self.device)] = Phi
            Phi = full
        out = torch.zeros(m, self.N, device=self.device, dtype=self.dtype)
        for t_idx, srcs, b in self.blocks.values():
            out[:, srcs.to(self.device)] += Phi[:, t_idx.to(self.device)] @ b
        return out

    def nnz(self) -> int:
        return int(sum(b.numel() for _, _, b in self.blocks.values()))


# -------------------------------------------------------------------------------------------------
# builders
# -------------------------------------------------------------------------------------------------
def from_tiny(op) -> LocalPart:
    """LocalPart of a TinyOperator (PLT reads ln2-normalised input; CLT reads resid_mid)."""
    m, cfg, c = op.model, op.cfg, op.cache
    L, n, d = cfg.n_layers, op.n_pos, cfg.d_model
    D = torch.ones(L + 1, n, d, dtype=m.dtype)
    for l in range(L):
        if cfg.transcoder == "plt":
            D[l] = m.g2[l][None, :] / c["scale2"][l][0]
    D[L] = m.gf[None, :] / c["scalef"][0]
    reads = Reads(torch.cat([op.logit_vecs, op.enc_vecs]), op.target_meta.layer, op.target_meta.pos)
    n_sf = op.source_features.numel()
    writes = Writes(
        vecs=torch.cat([op.dec_vecs, op.errors.reshape(L * n, d), op.tok_vecs]),
        layer=torch.cat([op.dec_lay, torch.arange(L).repeat_interleave(n), torch.full((n,), -1)]),
        pos=torch.cat([op.dec_pos, torch.arange(n).repeat(L), torch.arange(n)]),
        src=torch.cat([op.dec_src, n_sf + torch.arange(L * n), n_sf + L * n + torch.arange(n)]),
    )
    skip = None
    if getattr(m, "W_skip", None) is not None:
        def skip(l: int, V: Tensor) -> Tensor:  # skip(x) = x @ W_skip[l]  ->  S^T v = v @ W_skip[l]^T
            return V @ m.W_skip[l].T
    return LocalPart(reads, writes, D, op.T, op.N, skip_vjp=skip, dtype=m.dtype)


def from_circuit_tracer(op, dtype=torch.float32) -> LocalPart:
    """LocalPart of a CircuitTracerOperator (VM only). Read diagonals are detected from the cached
    forward pass (op.read_diagonals); the skip map, if the transcoders have one, through autograd."""
    ctx = op.ctx
    L, n = op.n_layers, op.n_pos
    D = op.read_diagonals().to(dtype)
    reads = Reads(torch.cat([op.logit_vecs.to(dtype), op.enc_vecs.to(dtype)]), op.target_meta.layer,
                  op.target_meta.pos)
    dec_layers, dec_pos = (x.cpu() for x in ctx.decoder_locations)
    dec_src = op.active_to_source[ctx.encoder_to_decoder_map.cpu()]
    keep = dec_src >= 0
    nsf = op.source_features.numel()
    d = ctx.error_vectors.shape[-1]
    writes = Writes(
        vecs=torch.cat([ctx.decoder_vecs[keep.to(ctx.decoder_vecs.device)].to(op.device, dtype),
                        ctx.error_vectors.reshape(L * n, d).to(op.device, dtype),
                        ctx.token_vectors.to(op.device, dtype)]),
        layer=torch.cat([dec_layers[keep], torch.arange(L).repeat_interleave(n), torch.full((n,), -1)]),
        pos=torch.cat([dec_pos[keep], torch.arange(n).repeat(L), torch.arange(n)]),
        src=torch.cat([dec_src[keep], nsf + torch.arange(L * n), nsf + L * n + torch.arange(n)]),
    )
    skip = op.skip_vjp() if getattr(op.model, "skip_transcoder", False) else None
    return LocalPart(reads, writes, D, op.T, op.N, skip_vjp=skip, device=op.device, dtype=dtype)


def energy_share(A: Tensor, A0: Tensor) -> dict:
    """Share of the squared Frobenius norm of A carried by A_0, and by the remainder."""
    A = A.double()
    A0 = A0.double().to(A.device)
    tot = float(A.pow(2).sum())
    rem = float((A - A0).pow(2).sum())
    return {"a0_energy_share": 1.0 - rem / tot if tot > 0 else float("nan"),
            "remainder_energy_share": rem / tot if tot > 0 else float("nan"),
            "a0_nnz_fraction": float((A0 != 0).double().mean())}
