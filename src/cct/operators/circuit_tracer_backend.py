"""Attribution operator on circuit-tracer's ReplacementModel (TransformerLens backend), pinned to
circuit-tracer 0.5.0 (source read from the PyPI wheel; docs/circuit_tracer_notes.md).

Written on the laptop against the source, NOT run there (no model downloads). First tested at K0 on
the VM. Everything below mirrors attribution/context_transformerlens.py and attribution/attribute.py:

- Forward state: model.setup_attribution(tokens) builds the AttributionContext (active features,
  encoder vectors, activation-scaled decoder vectors, error and token vectors); then
  model.run_forward_pass(tokens, batch_size, ctx) runs the hooked forward on the batch-expanded prompt,
  caching the residual activations at every layer's feature-input hook (and ln_final output as layer
  n_layers) with the graph retained, and with circuit-tracer's backward score hooks installed.
- rows(seeds): one backward pass per batch, exactly like AttributionContext.compute_batch, except that
  the injected gradient is ADDITIVE and multi-location: at every layer l a dense [B, n_pos, d] tensor
  sum_{t at layer l} seed[b, t] e_t (at p_t) is ADDED to the gradient flowing into
  _resid_activations[l]. circuit-tracer's _inject uses index_put_ (replace). Replace equals add for
  one-hot rows (nothing downstream in the same batch row), but for mixed rows it would overwrite
  gradient arriving from later-layer targets (always for CLTs, whose input hook is hook_resid_mid,
  and for skip transcoders). Hence the patch.
- cols(tangents): forward-mode AD (torch.autograd.forward_ad) through the same hooked model: tangents
  are added at each layer's feature-output hook (mlp hook_out_grad, where circuit-tracer reads feature
  and error scores) and at hook_embed (token scores); tangents are read at each feature-input hook and
  at ln_final's output. The stop-gradient hooks (.detach()) also stop tangents, so this is the same
  frozen linear map. Fallback: forward_mode="double_backward" computes J u as the gradient of
  <J^T v, u> with respect to v (VJP of a VJP); K0.3 records which one is used.

Node layout of the context's buffer rows: [all active features, errors (layer-major), tokens].
"""
from __future__ import annotations

import time
from typing import Literal

import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor

from .base import KIND_ERROR, KIND_FEATURE, KIND_LOGIT, KIND_TOKEN, CostCounters, NodeMeta, OperatorMixin


def load_replacement_model(model_name: str, transcoder_set: str, dtype: torch.dtype = torch.float32,
                           device: str = "cuda", **kw):
    from circuit_tracer import ReplacementModel  # VM only

    return ReplacementModel.from_pretrained(model_name, transcoder_set, backend="transformerlens",
                                            device=torch.device(device), dtype=dtype, **kw)


class CircuitTracerOperator(OperatorMixin):
    def __init__(self, model, prompt, *, max_n_logits: int = 10, desired_logit_prob: float = 0.95,
                 batch_size: int = 256, target_features: Tensor | None = None,
                 source_features: Tensor | None = None,
                 forward_mode: Literal["fwad", "double_backward"] = "fwad",
                 detach_attention: bool = False):
        """detach_attention=True stop-gradients every attention output (hook_attn_out) in the cached
        forward and in forward-mode passes: the forward values are unchanged and only the attention-free
        paths remain, so rows and columns are those of the local part A_0 (K0.7)."""
        from circuit_tracer.attribution.targets import AttributionTargets  # VM only

        self.model = model
        self.detach_attention = detach_attention
        self.batch_size = batch_size
        self.forward_mode = forward_mode
        self.counters = CostCounters()
        self.device = torch.device(model.cfg.device)
        self.dtype = model.cfg.dtype
        t0 = time.perf_counter()
        self.tokens = model.ensure_tokenized(prompt)
        ctx = model.setup_attribution(self.tokens)
        with model.hooks(fwd_hooks=self._extra_fwd_hooks()):
            model.run_forward_pass(self.tokens, batch_size, ctx)
        self.ctx = ctx
        self.n_layers, self.n_pos, _ = ctx.activation_matrix.shape
        L, n = self.n_layers, self.n_pos
        feat_layers, feat_pos, feat_ids = ctx.activation_matrix.indices()
        self.feat_layers, self.feat_pos, self.feat_ids = feat_layers.cpu(), feat_pos.cpu(), feat_ids.cpu()
        self.n_active = int(ctx.activation_matrix._nnz())
        self.activation_values = ctx.activation_matrix.values().detach().cpu()
        self.targets = AttributionTargets(attribution_targets=None, logits=ctx.logits[0, -1],
                                          unembed_proj=model.unembed_proj, tokenizer=model.tokenizer,
                                          max_n_logits=max_n_logits, desired_logit_prob=desired_logit_prob)
        self.logit_vecs = self.targets.logit_vectors.to(self.device, self.dtype)
        self.logit_probs = self.targets.logit_probabilities.detach().cpu()
        n_logits = len(self.targets)
        self.n_logits = n_logits
        self.setup_seconds = time.perf_counter() - t0
        self.set_targets(target_features)
        self.set_sources(source_features)

    def _extra_fwd_hooks(self) -> list:
        if not self.detach_attention:
            return []

        def detach(acts, hook):
            return acts.detach()

        return [(f"blocks.{l}.hook_attn_out", detach) for l in range(self.model.cfg.n_layers)]

    # -- node sets -----------------------------------------------------------------------------
    def set_targets(self, target_features: Tensor | None) -> None:
        tf = torch.arange(self.n_active) if target_features is None else torch.as_tensor(target_features).long().cpu()
        self.target_features = tf
        k, L, n = self.n_logits, self.n_layers, self.n_pos
        self.T = k + tf.numel()
        self.target_meta = NodeMeta(
            kind=torch.cat([torch.full((k,), KIND_LOGIT), torch.full((tf.numel(),), KIND_FEATURE)]),
            layer=torch.cat([torch.full((k,), L), self.feat_layers[tf]]),
            pos=torch.cat([torch.full((k,), n - 1), self.feat_pos[tf]]),
            feature=torch.cat([torch.full((k,), -1), self.feat_ids[tf]]),
            extra=torch.cat([self.logit_probs.float(), torch.zeros(tf.numel())]),
        )
        self.enc_vecs = self.ctx.encoder_vecs[tf.to(self.ctx.encoder_vecs.device)].to(self.device, self.dtype)

    def set_sources(self, source_features: Tensor | None) -> None:
        sf = torch.arange(self.n_active) if source_features is None else torch.as_tensor(source_features).long().cpu()
        self.source_features = sf
        L, n = self.n_layers, self.n_pos
        self.source_cols = torch.cat([sf, torch.arange(self.n_active, self.n_active + (L + 1) * n)])
        self.N = self.source_cols.numel()
        self.source_meta = NodeMeta(
            kind=torch.cat([torch.full((sf.numel(),), KIND_FEATURE), torch.full((L * n,), KIND_ERROR),
                            torch.full((n,), KIND_TOKEN)]),
            layer=torch.cat([self.feat_layers[sf], torch.arange(L).repeat_interleave(n), torch.full((n,), -1)]),
            pos=torch.cat([self.feat_pos[sf], torch.arange(n).repeat(L), torch.arange(n)]),
            feature=torch.cat([self.feat_ids[sf], torch.full((L * n + n,), -1)]),
            extra=torch.cat([self.activation_values[sf].float(), torch.ones(L * n + n)]),
        )
        # map active feature -> source column (-1 if not a source), for forward-mode tangents
        a2s = torch.full((self.n_active,), -1, dtype=torch.long)
        a2s[sf] = torch.arange(sf.numel())
        self.active_to_source = a2s

    # -- injections ----------------------------------------------------------------------------
    def _injections(self, seeds: Tensor) -> dict[int, Tensor]:
        """Per layer, a dense [B, n_pos, d] additive gradient (B = cached forward batch)."""
        B = self.ctx._resid_activations[0].shape[0]
        b = seeds.shape[0]
        d = self.enc_vecs.shape[1]
        k = self.n_logits
        out: dict[int, Tensor] = {}
        if k:
            inj = torch.zeros(B, self.n_pos, d, device=self.device, dtype=self.dtype)
            inj[:b, self.n_pos - 1] = seeds[:, :k] @ self.logit_vecs
            out[self.n_layers] = inj
        tl = self.target_meta.layer[k:]
        tp = self.target_meta.pos[k:]
        for l in tl.unique().tolist():
            sel_l = torch.nonzero(tl == l).flatten()
            inj = torch.zeros(B, self.n_pos, d, device=self.device, dtype=self.dtype)
            pos_l = tp[sel_l]
            for p in pos_l.unique().tolist():
                sel = sel_l[pos_l == p]
                selk = (sel + k).to(self.device)
                inj[:b, p] = seeds[:, selk] @ self.enc_vecs[sel.to(self.device)]
            out[int(l)] = inj
        return out

    def _rows_impl(self, seeds: Tensor) -> Tensor:
        ctx = self.ctx
        B = ctx._resid_activations[0].shape[0]
        b = seeds.shape[0]
        assert b <= B, "batch larger than the cached forward batch"
        ctx._batch_buffer = torch.zeros(ctx._row_size, B, dtype=self.dtype, device=self.device)
        injections = self._injections(seeds)
        handles = []

        def make_hook(inj):
            def _add(grads):
                return grads + inj.to(grads.dtype)
            return _add

        for l, inj in injections.items():
            handles.append(ctx._resid_activations[l].register_hook(make_hook(inj)))
        try:
            last = max(injections)
            ctx._resid_activations[last].backward(gradient=torch.zeros_like(ctx._resid_activations[last]),
                                                  retain_graph=True)
        finally:
            for h in handles:
                h.remove()
        buf, ctx._batch_buffer = ctx._batch_buffer, None
        return buf.T[:b][:, self.source_cols.to(self.device)]

    def ct_rows(self, idx: Tensor) -> Tensor:
        """Rows by circuit-tracer's own compute_batch (one location per batch row), for K0 checks."""
        idx = torch.as_tensor(idx).long()
        k = self.n_logits
        out = []
        for i in range(0, idx.numel(), self.batch_size):
            chunk = idx[i : i + self.batch_size]
            is_logit = chunk < k
            layers = torch.where(is_logit, torch.full_like(chunk, self.n_layers), self.target_meta.layer[chunk])
            positions = self.target_meta.pos[chunk]
            vals = torch.zeros(chunk.numel(), self.enc_vecs.shape[1], device=self.device, dtype=self.dtype)
            if bool(is_logit.any()):
                vals[is_logit.to(self.device)] = self.logit_vecs[chunk[is_logit].to(self.device)]
            if bool((~is_logit).any()):
                vals[(~is_logit).to(self.device)] = self.enc_vecs[(chunk[~is_logit] - k).to(self.device)]
            rows = self.ctx.compute_batch(layers=layers.to(self.device), positions=positions.to(self.device),
                                          inject_values=vals, retain_graph=True)
            out.append(rows[:, self.source_cols.to(self.device)])
        return torch.cat(out, 0)

    # -- forward mode --------------------------------------------------------------------------
    def _tangent_writes(self, tangents: Tensor) -> tuple[dict[int, Tensor], Tensor]:
        """Per-layer [b, n_pos, d] tangents at the feature-output hook, and [b, n_pos, d] at hook_embed."""
        ctx = self.ctx
        b = tangents.shape[0]
        L, n = self.n_layers, self.n_pos
        d = ctx.error_vectors.shape[-1]
        nsf = self.source_features.numel()
        dec_layers, dec_pos = ctx.decoder_locations
        dec_src = self.active_to_source.to(self.device)[ctx.encoder_to_decoder_map]  # source col per decoder row
        writes = {}
        for l in range(L):
            t = torch.zeros(b, n, d, device=self.device, dtype=self.dtype)
            sel = torch.nonzero((dec_layers == l) & (dec_src >= 0)).flatten()
            if sel.numel():
                coef = tangents[:, dec_src[sel]]  # [b, n_sel]
                vec = ctx.decoder_vecs[sel].to(self.dtype)  # [n_sel, d]
                pos = dec_pos[sel]
                for p in pos.unique().tolist():
                    s2 = pos == p
                    t[:, p] += coef[:, s2] @ vec[s2]
            t += tangents[:, nsf + l * n : nsf + (l + 1) * n, None] * ctx.error_vectors[l][None].to(self.dtype)
            writes[l] = t
        tok = tangents[:, nsf + L * n :, None] * ctx.token_vectors[None].to(self.dtype)
        return writes, tok

    def _read_targets(self, t_in: list[Tensor], t_fin: Tensor) -> Tensor:
        k = self.n_logits
        b = t_fin.shape[0]
        out = torch.zeros(b, self.T, device=self.device, dtype=self.dtype)
        out[:, :k] = t_fin[:, self.n_pos - 1] @ self.logit_vecs.T
        tl = self.target_meta.layer[k:].to(self.device)
        tp = self.target_meta.pos[k:].to(self.device)
        stacked = torch.stack(t_in, 1)  # [b, L, n, d]
        out[:, k:] = (stacked[:, tl, tp] * self.enc_vecs[None]).sum(-1)
        return out

    def _cols_impl(self, tangents: Tensor) -> Tensor:
        if self.forward_mode == "fwad":
            return self._cols_fwad(tangents)
        return self._cols_double_backward(tangents)

    def _hooked_forward(self, b: int, add_out: dict, add_embed, store: dict):
        """Run the hooked model on the batch-expanded prompt. add_out[l](acts) and add_embed(acts) modify
        the feature-output / embedding activations; feature inputs and the final normalised residual are
        stored in `store`."""
        m = self.model
        L = self.n_layers

        def out_hook(acts, hook, l):
            return add_out[l](acts)

        def emb_hook(acts, hook):
            return add_embed(acts)

        def in_hook(acts, hook, l):
            store[l] = acts
            return acts

        from functools import partial
        fwd = [(f"blocks.{l}.{m.feature_output_hook}", partial(out_hook, l=l)) for l in range(L)]
        fwd += [(f"blocks.{l}.{m.feature_input_hook}", partial(in_hook, l=l)) for l in range(L)]
        fwd.append(("hook_embed", emb_hook))
        with m.hooks(fwd_hooks=fwd + self._extra_fwd_hooks()):
            resid = m.forward(self.tokens.expand(b, -1), stop_at_layer=L)
        store[L] = m.ln_final(resid)

    def _cols_fwad(self, tangents: Tensor) -> Tensor:
        b = tangents.shape[0]
        writes, tok = self._tangent_writes(tangents)
        store: dict = {}
        with fwAD.dual_level():
            add_out = {l: (lambda acts, t=writes[l]: acts + fwAD.make_dual(torch.zeros_like(acts), t.to(acts.dtype)))
                       for l in writes}
            add_emb = lambda acts: acts + fwAD.make_dual(torch.zeros_like(acts), tok.to(acts.dtype))  # noqa: E731
            self._hooked_forward(b, add_out, add_emb, store)
            t_in = []
            for l in range(self.n_layers):
                tt = fwAD.unpack_dual(store[l]).tangent
                t_in.append(torch.zeros_like(fwAD.unpack_dual(store[l]).primal) if tt is None else tt)
            tf = fwAD.unpack_dual(store[self.n_layers]).tangent
            tf = torch.zeros_like(fwAD.unpack_dual(store[self.n_layers]).primal) if tf is None else tf
            return self._read_targets([x.to(self.dtype) for x in t_in], tf.to(self.dtype)).detach()

    def _cols_double_backward(self, tangents: Tensor) -> Tensor:
        """J u = d/dv <J^T v, u> (VJP of a VJP). J^T v is linear in v, so its gradient in v is exact."""
        b = tangents.shape[0]
        L, n = self.n_layers, self.n_pos
        writes, tok = self._tangent_writes(tangents)
        d = tok.shape[-1]
        deltas = {l: torch.zeros(b, n, d, device=self.device, dtype=self.dtype, requires_grad=True) for l in range(L)}
        d_emb = torch.zeros(b, n, d, device=self.device, dtype=self.dtype, requires_grad=True)
        store: dict = {}
        with torch.enable_grad():
            self._hooked_forward(b, {l: (lambda acts, dl=deltas[l]: acts + dl.to(acts.dtype)) for l in range(L)},
                                 lambda acts: acts + d_emb.to(acts.dtype), store)
            outs = [store[l] for l in range(L + 1)]
            vs = [torch.zeros_like(o, requires_grad=True) for o in outs]
            g = torch.autograd.grad(outs, [d_emb] + [deltas[l] for l in range(L)], vs, create_graph=True,
                                    allow_unused=True)
            inner = sum(((gi * ui.to(gi.dtype)).sum() for gi, ui in zip(g, [tok] + [writes[l] for l in range(L)])
                         if gi is not None), torch.zeros((), device=self.device))
            Ju = torch.autograd.grad(inner, vs, allow_unused=True)
        Ju = [torch.zeros_like(o) if j is None else j for j, o in zip(Ju, outs)]
        return self._read_targets([x.to(self.dtype) for x in Ju[:L]], Ju[L].to(self.dtype)).detach()

    # -- exact graph with circuit-tracer's visiting order ---------------------------------------
    def exact_attribution(self, max_feature_nodes: int | None, update_interval: int = 4,
                          influence_ordered: bool = True) -> dict:
        """Reproduce attribute.py's Phase 3-4 loop with circuit-tracer's own compute_batch and
        compute_partial_influences, returning the rows over ALL buffer columns and the visiting order.
        influence_ordered=True ranks features by influence even when every feature is visited (circuit-
        tracer visits in index order in that case; the matrix is identical, only the order differs,
        which regime F's truncation baseline needs)."""
        from circuit_tracer.graph import compute_partial_influences  # VM only

        ctx = self.ctx
        k, bs = self.n_logits, self.batch_size
        n_act = self.n_active
        logit_offset = ctx._row_size
        cap = min(max_feature_nodes or n_act, n_act)
        edge = torch.zeros(cap + k, logit_offset + k)
        row_to_node = torch.zeros(cap + k, dtype=torch.int32)
        t0 = time.perf_counter()
        for i in range(0, k, bs):
            vecs = self.targets.logit_vectors[i : i + bs]
            rows = ctx.compute_batch(layers=torch.full((vecs.shape[0],), self.n_layers),
                                     positions=torch.full((vecs.shape[0],), self.n_pos - 1), inject_values=vecs)
            edge[i : i + vecs.shape[0], :logit_offset] = rows.cpu()
            row_to_node[i : i + vecs.shape[0]] = torch.arange(i, i + vecs.shape[0]) + logit_offset
        st, visited, n_visited = k, torch.zeros(n_act, dtype=torch.bool), 0
        order: list[Tensor] = []
        fl, fp = ctx.activation_matrix.indices()[:2]
        while n_visited < cap:
            if cap == n_act and not influence_ordered:
                pending = torch.arange(n_act)
            else:
                infl = compute_partial_influences(edge[:st], self.targets.logit_probabilities, row_to_node[:st])
                rank = torch.argsort(infl[:n_act], descending=True).cpu()
                pending = rank[~visited[rank]][: min(update_interval * bs, cap - n_visited)]
            for j in range(0, len(pending), bs):
                idx = pending[j : j + bs]
                n_visited += len(idx)
                rows = ctx.compute_batch(layers=fl[idx], positions=fp[idx], inject_values=ctx.encoder_vecs[idx],
                                         retain_graph=True)
                end = st + rows.shape[0]
                edge[st:end, :logit_offset] = rows.cpu()
                row_to_node[st:end] = idx
                visited[idx] = True
                order.append(idx)
                st = end
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return {
            "rows": edge[:, :logit_offset],  # [k + cap, row_size]: logits first, then features in visiting order
            "visit_order_features": torch.cat(order) if order else torch.zeros(0, dtype=torch.long),
            "selected_features": torch.where(visited)[0],
            "seconds": time.perf_counter() - t0,
        }

    # -- local part A_0 (PROJECT_V2 T2) and attention rollout -------------------------------------
    @torch.no_grad()
    def read_diagonals(self, rtol: float | None = None) -> Tensor:
        """D [L + 1, n_pos, d]: the frozen residual-to-read map at each (layer, position), which is
        diagonal. Detected from a cached forward pass by checking which candidate reproduces the
        feature-input hook from hook_resid_mid: identity (CLT reading hook_resid_mid), 1 / scale
        (ln2.hook_normalized, or a folded RMSNormPre) or w / scale (the ln2 output, e.g. mlp.hook_in).
        Row L is ln_final (its output is the logit read point in circuit-tracer). Raises if no
        candidate matches."""
        m = self.model
        L = self.n_layers
        rtol = rtol or (1e-3 if self.dtype == torch.float32 else 3e-2)
        cache: dict = {}

        def keep(acts, hook):
            cache[hook.name] = acts.detach()[0].float()
            return acts

        names = [f"blocks.{l}.hook_resid_mid" for l in range(L)] +                 [f"blocks.{l}.ln2.hook_scale" for l in range(L)] +                 [f"blocks.{l}.{m.feature_input_hook}" for l in range(L)] + ["ln_final.hook_scale"]
        with m.hooks(fwd_hooks=[(nm, keep) for nm in names]):
            resid = m.forward(self.tokens[None], stop_at_layer=L)
        final = m.ln_final(resid)[0].float()
        resid = resid[0].float()
        n, d = resid.shape
        D = torch.empty(L + 1, n, d, device=self.device, dtype=torch.float32)
        self.read_map_kind = []
        for l in range(L):
            x = cache[f"blocks.{l}.hook_resid_mid"]
            y = cache[f"blocks.{l}.{m.feature_input_hook}"]
            sc = cache[f"blocks.{l}.ln2.hook_scale"]
            w = getattr(m.blocks[l].ln2, "w", None)
            cands = {"identity": torch.ones_like(x), "inv_scale": (1.0 / sc).expand_as(x)}
            if w is not None:
                cands["w_over_scale"] = w.float()[None, :] / sc
            D[l], kind = self._pick(x, y, cands, rtol, f"layer {l}")
            self.read_map_kind.append(kind)
        scf = cache["ln_final.hook_scale"]
        wf = getattr(m.ln_final, "w", None)
        cands = {"inv_scale": (1.0 / scf).expand_as(resid)}
        if wf is not None:
            cands["w_over_scale"] = wf.float()[None, :] / scf
        D[L], kind = self._pick(resid, final, cands, rtol, "ln_final")
        self.read_map_kind.append(kind)
        return D

    @staticmethod
    def _pick(x: Tensor, y: Tensor, cands: dict, rtol: float, where: str) -> tuple[Tensor, str]:
        errs = {}
        for k, Dk in cands.items():
            e = float((x * Dk - y).norm() / y.norm().clamp_min(1e-30))
            errs[k] = e
            if e <= rtol:
                return Dk, k
        raise RuntimeError(f"read map at {where} matches no candidate (relative errors {errs})")

    def skip_vjp(self):
        """(layer, V [k, d]) -> S_layer^T V for the transcoders' linear skip connection."""
        tr = self.model.transcoders

        def f(l: int, V: Tensor) -> Tensor:
            with torch.enable_grad():
                x = torch.zeros(V.shape, device=V.device, dtype=self.dtype, requires_grad=True)
                y = tr.compute_skip(l, x)
                (g,) = torch.autograd.grad(y, x, V.to(y.dtype))
            return g.to(V.dtype)

        return f

    @torch.no_grad()
    def attention_patterns(self) -> Tensor:
        """[L, H, n, n] attention patterns of the prompt (query x key)."""
        m = self.model
        pats: dict = {}

        def keep(acts, hook):
            pats[hook.name] = acts.detach()[0].float()
            return acts

        names = [f"blocks.{l}.attn.hook_pattern" for l in range(self.n_layers)]
        with m.hooks(fwd_hooks=[(nm, keep) for nm in names]):
            m.forward(self.tokens[None], stop_at_layer=self.n_layers)
        return torch.stack([pats[nm] for nm in names])

    def ov_weights(self) -> tuple[Tensor, Tensor]:
        """W_V [L, H, d, dh] and W_O [L, H, dh, d], key/value heads repeated for grouped-query
        attention so that heads match the patterns."""
        Wv, Wo = [], []
        for blk in self.model.blocks:
            v, o = blk.attn.W_V.detach().float(), blk.attn.W_O.detach().float()
            if v.shape[0] != o.shape[0]:
                v = v.repeat_interleave(o.shape[0] // v.shape[0], dim=0)
            Wv.append(v)
            Wo.append(o)
        return torch.stack(Wv), torch.stack(Wo)
