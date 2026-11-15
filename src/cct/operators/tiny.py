"""A tiny transformer with random JumpReLU transcoders, in pure PyTorch (CPU tests, tiny pipeline).

It freezes the same things circuit-tracer freezes (replacement_model_transformerlens.py,
_configure_gradient_flow and _configure_skip_connection, v0.5.0): attention patterns, the
normalisation scales and the MLP outputs (stop-gradient; no skip connection). With these frozen the
map from "what each node writes" to "each target's pre-activation" is linear, so:
- rows(seeds) is one batched VJP seeded with sum_t seed[t] * e_t at every target's (layer, pos)
  (additive, multi-location injection);
- cols(tangents) is one batched forward-mode pass (torch.autograd.forward_ad);
- brute_force_A() builds A from explicit Jacobian matrices of the frozen linear map, independent of
  autograd (test T2).

Transcoder conventions (as circuit-tracer): PLT reads the normalised MLP input (ln2 output) and writes
to its own layer's MLP output; CLT reads the residual stream at resid_mid and writes to its own and
every later layer's MLP output. Features and error nodes at position 0 are zeroed
(circuit-tracer's zero_positions = slice(0, 1)). Edges go to feature pre-activations (encoder row,
not scaled by activation); source write vectors are activation times decoder row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from .base import (KIND_ERROR, KIND_FEATURE, KIND_LOGIT, KIND_TOKEN, CostCounters, NodeMeta,
                   OperatorMixin)


@dataclass
class TinyConfig:
    n_layers: int = 3
    d_model: int = 32
    n_heads: int = 2
    d_head: int = 16
    vocab: int = 64
    d_mlp: int = 64
    n_features: int = 32  # per layer
    transcoder: str = "plt"  # "plt" or "clt"
    active_fraction: float = 0.2
    eps: float = 1e-6
    seed: int = 0
    skip: bool = False  # linear skip connection from the transcoder input to the MLP output


def _rms(x: Tensor, eps: float) -> Tensor:
    return (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


class TinyModel:
    """Random weights; the transcoder thresholds are calibrated so about active_fraction fire."""

    def __init__(self, cfg: TinyConfig, dtype=torch.float64):
        self.cfg = cfg
        self.dtype = dtype
        g = torch.Generator().manual_seed(cfg.seed)
        L, d, H, dh, V, dm, F = (cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_head, cfg.vocab,
                                 cfg.d_mlp, cfg.n_features)

        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g, dtype=dtype) * scale)

        self.W_E = rnd(V, d)
        self.g1 = 1.0 + 0.1 * rnd(L, d)
        self.g2 = 1.0 + 0.1 * rnd(L, d)
        self.gf = 1.0 + 0.1 * rnd(d)
        self.W_Q = rnd(L, H, d, dh, scale=d ** -0.5)
        self.W_K = rnd(L, H, d, dh, scale=d ** -0.5)
        self.W_V = rnd(L, H, d, dh, scale=d ** -0.5)
        self.W_O = rnd(L, H, dh, d, scale=dh ** -0.5)
        self.W_in = rnd(L, d, dm, scale=d ** -0.5)
        self.W_out = rnd(L, dm, d, scale=dm ** -0.5)
        self.W_U = rnd(d, V, scale=d ** -0.5)
        # transcoders
        self.W_enc = rnd(L, F, d, scale=d ** -0.5)
        self.b_enc = torch.zeros(L, F, dtype=dtype)
        self.b_dec = 0.1 * rnd(L, d)
        if cfg.transcoder == "plt":
            self.W_dec = rnd(L, F, d, scale=F ** -0.5)  # writes to its own layer
        elif cfg.transcoder == "clt":
            # W_dec[l][f, j] writes to layer l + j, for j = 0..L-1-l
            self.W_dec_clt = [rnd(F, L - l, d, scale=(F * (L - l)) ** -0.5) for l in range(L)]
        else:
            raise ValueError(cfg.transcoder)
        self.threshold = torch.zeros(L, F, dtype=dtype)
        # skip transcoder: skip(x) = x @ W_skip[l], live in the frozen model (as circuit-tracer)
        self.W_skip = rnd(L, d, d, scale=0.5 * d ** -0.5) if cfg.skip else None
        self._calibrate(g)

    # -- forward pieces ------------------------------------------------------------------------
    def _attn(self, l: int, h: Tensor, pattern: Tensor | None = None) -> tuple[Tensor, Tensor]:
        n = h.shape[-2]
        if pattern is None:
            q = torch.einsum("bnd,hde->bhne", h, self.W_Q[l])
            k = torch.einsum("bnd,hde->bhne", h, self.W_K[l])
            s = torch.einsum("bhne,bhme->bhnm", q, k) / math.sqrt(self.cfg.d_head)
            causal = torch.ones(n, n, dtype=torch.bool).tril()
            s = s.masked_fill(~causal, float("-inf"))
            pattern = torch.softmax(s, -1)
        v = torch.einsum("bnd,hde->bhne", h, self.W_V[l])
        z = torch.einsum("bhnm,bhme->bhne", pattern, v)
        return torch.einsum("bhne,hed->bnd", z, self.W_O[l]), pattern

    def _mlp(self, l: int, u: Tensor) -> Tensor:
        return torch.nn.functional.gelu(u @ self.W_in[l]) @ self.W_out[l]

    def feature_input(self, l: int, r_mid: Tensor, scale2: Tensor) -> Tensor:
        """What the transcoder at layer l reads (PLT: normalised; CLT: resid_mid)."""
        if self.cfg.transcoder == "plt":
            return r_mid / scale2 * self.g2[l]
        return r_mid

    def pre_acts(self, l: int, x: Tensor) -> Tensor:
        return x @ self.W_enc[l].T + self.b_enc[l]

    def jumprelu(self, l: int, pre: Tensor) -> Tensor:
        return pre * (pre > self.threshold[l])

    def decoder_out(self, acts: list[Tensor], l_out: int) -> Tensor:
        """Reconstruction written to layer l_out from per-layer activations acts[l] [b, n, F]."""
        if self.cfg.transcoder == "plt":
            return acts[l_out] @ self.W_dec[l_out] + self.b_dec[l_out]
        out = self.b_dec[l_out].clone()
        for l in range(l_out + 1):
            out = out + acts[l] @ self.W_dec_clt[l][:, l_out - l]
        return out

    def _calibrate(self, g: torch.Generator) -> None:
        tokens = torch.randint(0, self.cfg.vocab, (16, 12), generator=g)
        cache = self.clean_forward(tokens)
        for l in range(self.cfg.n_layers):
            pre = self.pre_acts(l, cache["feat_in"][l]).reshape(-1, self.cfg.n_features)
            self.threshold[l] = torch.quantile(pre, 1 - self.cfg.active_fraction, dim=0)

    @torch.no_grad()
    def clean_forward(self, tokens: Tensor) -> dict:
        """The real (nonlinear) forward pass; caches everything the frozen model needs."""
        cfg = self.cfg
        if tokens.ndim == 1:
            tokens = tokens[None]
        r = self.W_E[tokens]
        cache = {"scale1": [], "pattern": [], "scale2": [], "mlp_out": [], "feat_in": [],
                 "r_mid": []}
        for l in range(cfg.n_layers):
            s1 = _rms(r, cfg.eps)
            a, pat = self._attn(l, r / s1 * self.g1[l])
            r = r + a
            s2 = _rms(r, cfg.eps)
            u = r / s2 * self.g2[l]
            m = self._mlp(l, u)
            cache["scale1"].append(s1)
            cache["pattern"].append(pat)
            cache["scale2"].append(s2)
            cache["r_mid"].append(r)
            cache["feat_in"].append(self.feature_input(l, r, s2))
            cache["mlp_out"].append(m)
            r = r + m
        sf = _rms(r, cfg.eps)
        f = r / sf * self.gf
        cache["scalef"] = sf
        cache["final"] = f
        cache["logits"] = f @ self.W_U
        return cache

    def skip_out(self, l: int, x: Tensor) -> Tensor:
        return x @ self.W_skip[l] if self.W_skip is not None else torch.zeros_like(x)

    def frozen_forward(self, tokens: Tensor, delta_emb: Tensor, delta: Tensor,
                       detach_attn: bool = False) -> tuple[Tensor, Tensor]:
        """The local replacement model as a hooked forward: stop-gradients (detach) on attention
        patterns, normalisation scales and MLP outputs. delta_emb [b, n, d] is added to the token
        embeddings and delta [b, L, n, d] to each layer's MLP output (the write locations).
        detach_attn stop-gradients every attention output (the attention-free part A_0, K0.7).
        Returns (feature inputs [b, L, n, d], final normalised residual [b, n, d])."""
        cfg = self.cfg
        b = delta.shape[0]
        r = self.W_E[tokens].expand(b, -1, -1) + delta_emb
        feats = []
        for l in range(cfg.n_layers):
            s1 = _rms(r, cfg.eps).detach()
            h = r / s1 * self.g1[l]
            q = torch.einsum("bnd,hde->bhne", h, self.W_Q[l])
            k = torch.einsum("bnd,hde->bhne", h, self.W_K[l])
            s = torch.einsum("bhne,bhme->bhnm", q, k) / math.sqrt(cfg.d_head)
            n = s.shape[-1]
            s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
            pattern = torch.softmax(s, -1).detach()
            a, _ = self._attn(l, h, pattern)
            r = r + (a.detach() if detach_attn else a)
            s2 = _rms(r, cfg.eps).detach()
            u = r / s2 * self.g2[l]
            fi = self.feature_input(l, r, s2)
            feats.append(fi)
            if self.W_skip is None:
                m = self._mlp(l, u).detach()  # MLP output frozen (no skip connection)
            else:
                sk = self.skip_out(l, fi)
                m = sk + (self._mlp(l, u) - sk).detach()
            r = r + m + delta[:, l]
        sf = _rms(r, cfg.eps).detach()
        return torch.stack(feats, 1), r / sf * self.gf


class TinyOperator(OperatorMixin):
    """Attribution operator for one prompt on a TinyModel."""

    def __init__(self, model: TinyModel, tokens: Tensor, max_n_logits: int = 10,
                 desired_logit_prob: float = 0.95, target_features: Tensor | None = None,
                 source_features: Tensor | None = None, batch_size: int = 256,
                 dtype: torch.dtype = torch.float64, detach_attn: bool = False):
        self.model = model
        self.detach_attn = detach_attn
        self.cfg = cfg = model.cfg
        self.tokens = tokens.long().flatten()
        self.n_pos = n = self.tokens.numel()
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.counters = CostCounters()
        L = cfg.n_layers

        cache = model.clean_forward(self.tokens)
        self.cache = cache
        # -- active features (zero position 0, as circuit-tracer) ------------------------------
        acts = []
        for l in range(L):
            a = model.jumprelu(l, model.pre_acts(l, cache["feat_in"][l][0]))
            a[0] = 0
            acts.append(a)
        self.acts = acts  # list of [n, F]
        A = torch.stack(acts)  # [L, n, F]
        lay, pos, fid = A.nonzero(as_tuple=True)
        self.active = torch.stack([lay, pos, fid], 1)  # same order as circuit-tracer (coalesced)
        self.active_vals = A[lay, pos, fid]
        n_active = lay.numel()
        # -- reconstruction and error vectors --------------------------------------------------
        acts_b = [a[None] for a in acts]
        errors = torch.stack([cache["mlp_out"][l][0] - model.decoder_out(acts_b, l)[0]
                              - model.skip_out(l, cache["feat_in"][l][0]) for l in range(L)])  # [L, n, d]
        errors[:, 0] = 0
        self.errors = errors
        # -- logit targets (circuit-tracer AttributionTargets._from_salient) --------------------
        logits = cache["logits"][0, -1]
        probs = torch.softmax(logits, -1)
        top_p, top_idx = torch.topk(probs, min(max_n_logits, cfg.vocab))
        cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
        self.logit_idx = top_idx[:cutoff]
        self.logit_probs = probs[self.logit_idx]
        W_U = model.W_U
        self.logit_vecs = (W_U[:, self.logit_idx] - W_U.mean(dim=-1, keepdim=True)).T  # [k, d]
        n_logits = self.logit_idx.numel()

        # -- targets: logits then features -----------------------------------------------------
        tf = torch.arange(n_active) if target_features is None else torch.as_tensor(target_features).long()
        self.target_features = tf
        self.T = n_logits + tf.numel()
        self.target_meta = NodeMeta(
            kind=torch.cat([torch.full((n_logits,), KIND_LOGIT), torch.full((tf.numel(),), KIND_FEATURE)]),
            layer=torch.cat([torch.full((n_logits,), L), lay[tf]]),
            pos=torch.cat([torch.full((n_logits,), n - 1), pos[tf]]),
            feature=torch.cat([torch.full((n_logits,), -1), fid[tf]]),
            extra=torch.cat([self.logit_probs.float(), torch.zeros(tf.numel())]),
        )
        self.enc_vecs = model.W_enc[lay[tf], fid[tf]]  # [T_f, d]

        # -- sources: features, errors (layer-major), tokens -----------------------------------
        sf = torch.arange(n_active) if source_features is None else torch.as_tensor(source_features).long()
        self.source_features = sf
        n_sf = sf.numel()
        el = torch.arange(L).repeat_interleave(n)
        ep = torch.arange(n).repeat(L)
        self.N = n_sf + L * n + n
        self.source_meta = NodeMeta(
            kind=torch.cat([torch.full((n_sf,), KIND_FEATURE), torch.full((L * n,), KIND_ERROR),
                            torch.full((n,), KIND_TOKEN)]),
            layer=torch.cat([lay[sf], el, torch.full((n,), -1)]),
            pos=torch.cat([pos[sf], ep, torch.arange(n)]),
            feature=torch.cat([fid[sf], torch.full((L * n + n,), -1)]),
            extra=torch.cat([self.active_vals[sf].float(), torch.ones(L * n + n)]),
        )
        # decoder entries: one per (source feature, write layer)
        d_src, d_lay, d_pos, d_vec = [], [], [], []
        for j, a_idx in enumerate(sf.tolist()):
            l, p, f = (int(x) for x in self.active[a_idx])
            val = self.active_vals[a_idx]
            if cfg.transcoder == "plt":
                writes = [(l, model.W_dec[l, f])]
            else:
                writes = [(l + k, model.W_dec_clt[l][f, k]) for k in range(L - l)]
            for lw, vec in writes:
                d_src.append(j)
                d_lay.append(lw)
                d_pos.append(p)
                d_vec.append(val * vec)
        self.dec_src = torch.tensor(d_src, dtype=torch.long)
        self.dec_lay = torch.tensor(d_lay, dtype=torch.long)
        self.dec_pos = torch.tensor(d_pos, dtype=torch.long)
        self.dec_vecs = torch.stack(d_vec) if d_vec else torch.zeros(0, cfg.d_model, dtype=model.dtype)
        self.tok_vecs = model.W_E[self.tokens]
        self.n_logits = n_logits

    # -- helpers -------------------------------------------------------------------------------
    def _seed_to_grad_outputs(self, seeds: Tensor) -> tuple[Tensor, Tensor]:
        """Additive multi-location injection: G_in[b, l, p] = sum_t seed[b, t] e_t over feature
        targets at (l, p); G_final[b, n-1] = sum over logit targets."""
        L, n, d = self.cfg.n_layers, self.n_pos, self.cfg.d_model
        b = seeds.shape[0]
        k = self.n_logits
        G_in = torch.zeros(b, L * n, d, dtype=self.model.dtype)
        loc = self.target_meta.layer[k:] * n + self.target_meta.pos[k:]
        contrib = seeds[:, k:, None] * self.enc_vecs[None]  # [b, T_f, d]
        G_in.index_add_(1, loc, contrib)
        G_fin = torch.zeros(b, n, d, dtype=self.model.dtype)
        G_fin[:, n - 1] = seeds[:, :k] @ self.logit_vecs
        return G_in.view(b, L, n, d), G_fin

    def _scores_from_grads(self, g_emb: Tensor, g_mlp: Tensor) -> Tensor:
        """Contract gradients at write locations with every source's write vector."""
        b = g_mlp.shape[0]
        L, n = self.cfg.n_layers, self.n_pos
        out = torch.zeros(b, self.N, dtype=self.model.dtype)
        n_sf = self.source_features.numel()
        if self.dec_vecs.shape[0]:
            vals = (g_mlp[:, self.dec_lay, self.dec_pos] * self.dec_vecs[None]).sum(-1)
            out[:, :n_sf].index_add_(1, self.dec_src, vals)
        out[:, n_sf : n_sf + L * n] = (g_mlp * self.errors[None]).sum(-1).reshape(b, L * n)
        out[:, n_sf + L * n :] = (g_emb * self.tok_vecs[None]).sum(-1)
        return out

    def _rows_impl(self, seeds: Tensor) -> Tensor:
        b = seeds.shape[0]
        L, n, d = self.cfg.n_layers, self.n_pos, self.cfg.d_model
        seeds = seeds.to(self.model.dtype)
        delta_emb = torch.zeros(b, n, d, dtype=self.model.dtype, requires_grad=True)
        delta = torch.zeros(b, L, n, d, dtype=self.model.dtype, requires_grad=True)
        with torch.enable_grad():
            feats, fin = self.model.frozen_forward(self.tokens, delta_emb, delta, self.detach_attn)
            G_in, G_fin = self._seed_to_grad_outputs(seeds)
            g_emb, g_mlp = torch.autograd.grad([feats, fin], [delta_emb, delta], [G_in, G_fin],
                                               allow_unused=True)
        g_emb = torch.zeros_like(delta_emb) if g_emb is None else g_emb
        g_mlp = torch.zeros_like(delta) if g_mlp is None else g_mlp
        return self._scores_from_grads(g_emb, g_mlp).to(self.dtype)

    def _write_tangents(self, tangents: Tensor) -> tuple[Tensor, Tensor]:
        b = tangents.shape[0]
        L, n, d = self.cfg.n_layers, self.n_pos, self.cfg.d_model
        tangents = tangents.to(self.model.dtype)
        n_sf = self.source_features.numel()
        t_mlp = torch.zeros(b, L * n, d, dtype=self.model.dtype)
        if self.dec_vecs.shape[0]:
            loc = self.dec_lay * n + self.dec_pos
            t_mlp.index_add_(1, loc, tangents[:, self.dec_src, None] * self.dec_vecs[None])
        t_mlp = t_mlp + tangents[:, n_sf : n_sf + L * n, None] * self.errors.reshape(1, L * n, d)
        t_emb = tangents[:, n_sf + L * n :, None] * self.tok_vecs[None]
        return t_emb, t_mlp.view(b, L, n, d)

    def _cols_impl(self, tangents: Tensor) -> Tensor:
        b = tangents.shape[0]
        L, n, d = self.cfg.n_layers, self.n_pos, self.cfg.d_model
        t_emb, t_mlp = self._write_tangents(tangents)
        k = self.n_logits
        with fwAD.dual_level():
            d_emb = fwAD.make_dual(torch.zeros(b, n, d, dtype=self.model.dtype), t_emb)
            d_mlp = fwAD.make_dual(torch.zeros(b, L, n, d, dtype=self.model.dtype), t_mlp)
            feats, fin = self.model.frozen_forward(self.tokens, d_emb, d_mlp, self.detach_attn)
            tf = fwAD.unpack_dual(feats).tangent
            tfin = fwAD.unpack_dual(fin).tangent
        out = torch.zeros(b, self.T, dtype=self.model.dtype)
        if tfin is not None:
            out[:, :k] = tfin[:, n - 1] @ self.logit_vecs.T
        if tf is not None:
            tl, tp = self.target_meta.layer[k:], self.target_meta.pos[k:]
            out[:, k:] = (tf[:, tl, tp] * self.enc_vecs[None]).sum(-1)
        return out.to(self.dtype)

    def exact_matrix(self) -> Tensor:
        return self.exact_rows(torch.arange(self.T))

    # -- brute force from explicit Jacobians (test T2) -----------------------------------------
    def brute_force_A(self) -> Tensor:
        """Build A from PROJECT.md 5.1 with explicit Jacobian matrices of the frozen linear map
        (no autograd). Tiny sizes only."""
        m, cfg, c = self.model, self.cfg, self.cache
        L, n, d = cfg.n_layers, self.n_pos, cfg.d_model
        nd = n * d
        I = torch.eye(nd, dtype=m.dtype)

        def kron_rowmajor(P, B):  # vec(P X B) = kron(P, B^T) vec(X) for row-major vec
            return torch.kron(P.contiguous(), B.T.contiguous())

        R = []  # attention residual update of layer l: r_mid = R_l r
        Uin = []  # feature-input read map of layer l
        for l in range(L):
            s1 = c["scale1"][l][0, :, 0]
            Rl = I.clone()
            for h in range(cfg.n_heads):
                P = c["pattern"][l][0, h] @ torch.diag(1.0 / s1)
                B = torch.diag(m.g1[l]) @ m.W_V[l, h] @ m.W_O[l, h]
                Rl = Rl + kron_rowmajor(P, B)
            R.append(Rl)
            if cfg.transcoder == "plt":
                s2 = c["scale2"][l][0, :, 0]
                Uin.append(kron_rowmajor(torch.diag(1.0 / s2), torch.diag(m.g2[l])))
            else:
                Uin.append(I)
        sf = c["scalef"][0, :, 0]
        Fin = kron_rowmajor(torch.diag(1.0 / sf), torch.diag(m.gf))

        def J(l_write: int, l_read: int) -> Tensor:
            """Jacobian from a write after layer l_write (-1 = embedding) to the read at l_read
            (n_layers = final)."""
            M = I
            for k in range(l_write + 1, min(l_read + 1, L)):
                M = R[k] @ M
            if l_read == L:
                return Fin @ M
            return Uin[l_read] @ M

        k = self.n_logits
        # read vectors per target placed at their position
        t_lay = self.target_meta.layer
        t_pos = self.target_meta.pos
        read = torch.cat([self.logit_vecs, self.enc_vecs])  # [T, d]
        # write list per source: (source col, layer, pos, vec)
        n_sf = self.source_features.numel()
        w_src = [self.dec_src, n_sf + torch.arange(L * n), n_sf + L * n + torch.arange(n)]
        w_lay = [self.dec_lay, torch.arange(L).repeat_interleave(n), torch.full((n,), -1)]
        w_pos = [self.dec_pos, torch.arange(n).repeat(L), torch.arange(n)]
        w_vec = [self.dec_vecs, self.errors.reshape(L * n, d), self.tok_vecs]
        w_src, w_lay, w_pos, w_vec = (torch.cat(x) for x in (w_src, w_lay, w_pos, w_vec))
        A = torch.zeros(self.T, self.N, dtype=m.dtype)
        Jcache: dict = {}
        for t in range(self.T):
            lt, pt = int(t_lay[t]), int(t_pos[t])
            for lw in range(-1, L):
                if lw >= lt:
                    continue
                sel = w_lay == lw
                if not sel.any():
                    continue
                if (lw, lt) not in Jcache:
                    Jcache[(lw, lt)] = J(lw, lt).reshape(n, d, n, d)
                Jb = Jcache[(lw, lt)]
                v = torch.einsum("i,ipj->pj", read[t], Jb[pt])  # [n, d]: gradient at writes
                vals = (v[w_pos[sel]] * w_vec[sel]).sum(-1)
                A[t].index_add_(0, w_src[sel], vals)
        return A.to(self.dtype)

    # -- real-model quantities for baselines and interventions ---------------------------------
    def replacement_forward(self, feature_scale: Tensor | None = None, error_scale: Tensor | None = None,
                            token_scale: Tensor | None = None, frozen: bool = False) -> Tensor:
        """Forward pass with each MLP replaced by its transcoder (live JumpReLU) plus the frozen error
        vector. feature_scale [n_active] multiplies each active feature's activation (0 = ablate).
        If frozen, attention patterns and norm scales are stop-gradiented (B5 linearisation).
        Returns the prob-weighted logit-difference metric sum_t p_t (f_{n-1} . logit_vec_t)."""
        m, cfg = self.model, self.cfg
        L, n = cfg.n_layers, self.n_pos
        r = self.W_tok_scaled(token_scale)
        acts_out: list[Tensor] = []
        for l in range(L):
            s1 = _rms(r, cfg.eps)
            s1 = s1.detach() if frozen else s1
            h = r / s1 * m.g1[l]
            q = torch.einsum("bnd,hde->bhne", h, m.W_Q[l])
            kk = torch.einsum("bnd,hde->bhne", h, m.W_K[l])
            s = torch.einsum("bhne,bhme->bhnm", q, kk) / math.sqrt(cfg.d_head)
            s = s.masked_fill(~torch.ones(n, n, dtype=torch.bool).tril(), float("-inf"))
            pat = torch.softmax(s, -1)
            pat = pat.detach() if frozen else pat
            a, _ = m._attn(l, h, pat)
            r = r + a
            s2 = _rms(r, cfg.eps)
            s2 = s2.detach() if frozen else s2
            x = m.feature_input(l, r, s2)
            act = m.jumprelu(l, m.pre_acts(l, x))
            act = act.clone()
            act[:, 0] = 0
            if feature_scale is not None:
                sel = self.active[:, 0] == l
                scale = torch.ones_like(act)
                scale[0, self.active[sel, 1], self.active[sel, 2]] = feature_scale[sel]
                act = act * scale
            acts_out.append(act)
            err = self.errors[l][None]
            if error_scale is not None:
                err = err * error_scale[l * n : (l + 1) * n, None]
            r = r + m.decoder_out(acts_out, l) + m.skip_out(l, x) + err
        sf = _rms(r, cfg.eps)
        sf = sf.detach() if frozen else sf
        f = r / sf * m.gf
        return (f[0, n - 1] @ self.logit_vecs.T * self.logit_probs).sum()

    def W_tok_scaled(self, token_scale: Tensor | None) -> Tensor:
        e = self.tok_vecs[None]
        if token_scale is not None:
            e = e * token_scale[None, :, None]
        return e

    def total_effect(self) -> Tensor:
        """B5: gradient times activation of the logit metric through the linearised model with
        features unfrozen, one backward pass. Returns a score per source [N]."""
        L, n = self.cfg.n_layers, self.n_pos
        fs = torch.ones(self.active.shape[0], dtype=self.model.dtype, requires_grad=True)
        es = torch.ones(L * n, dtype=self.model.dtype, requires_grad=True)
        ts = torch.ones(n, dtype=self.model.dtype, requires_grad=True)
        with torch.enable_grad():
            metric = self.replacement_forward(fs, es, ts, frozen=True)
            gf, ge, gt = torch.autograd.grad(metric, [fs, es, ts])
        return torch.cat([gf[self.source_features], ge, gt]).to(self.dtype)


def random_prompts(n_prompts: int, vocab: int, lengths: tuple[int, int], seed: int) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_prompts):
        ln = int(torch.randint(lengths[0], lengths[1] + 1, (1,), generator=g))
        out.append(torch.randint(0, vocab, (ln,), generator=g))
    return out
