"""Gate 0, Stage F (GATE0_PREMISES.md Section 3; PREREG_GATE0.md): PR11, do feature-level step
dependencies predict real-model step interventions, and are the carrier features nameable?

Per committed trace (results/gate0/traces.json), with sentences from cct.premise_traces (Thought
Anchors' rule on the thinking text) and the 17 sampled outputs whose exact reference graphs Stage B
built (artifacts/<run>/<set>/premise_refs/<trace>/<q>.pt):

1. Feature-level dependency D[j, i]: for each sampled output of sentence j, the graph's node influence
   (circuit-tracer's influence on the output's logits, all feature, error and token nodes, unpruned)
   summed over nodes whose trace position (graph position + prefix_offset) lies in sentence i; summed
   over j's sampled outputs; normalised per j over i < j. Pairs: i < j with j holding sampled outputs.
2. Ground truth in the real model, teacher forced: change in the summed log-probability of sentence j's
   sampled tokens when (a) sentence i is replaced by the filler token repeated to equal token length,
   (b) sentence i is deleted (later tokens shift left; positions aligned by content). The actual effect
   used for the metrics is |change| (the signed change is stored).
   Real model: the TransformerLens model inside circuit-tracer's ReplacementModel. Its permanent hooks
   only stop gradients (forward values equal the plain model), so forward passes with those hooks in
   place are the real model; for gradients (S2) every permanent hook is removed first.
3. Baselines on the same pairs:
   S4  Thought Anchors' attention suppression (see cct.premise_df for the verified source): every
       layer and head, pre-softmax scores to sentence i's keys set to finfo.min for every query; effect
       = mean over sentence j's token positions of log(KL(P_base || Q_supp) + 1e-9), KL on top-p
       (0.9999, at most 100 tokens) logits at temperature 0.6, as their get_suppression_KL_matrix.
       S4_logprob (extra): |change| in j's sampled-token log-probability under the same suppression.
   S1  attention rollout (cct.reach.rollout_layers over all layers, product of the per-layer
       matrices): sum over j's sampled outputs (query = the position predicting the token) of the
       rollout mass on sentence i's positions.
   S2  gradient x input: |sum over sentence i's positions of grad . embedding| of j's summed sampled
       log-probability, real-model gradient.
   S5  FlashTrace (github.com/wbopan/flashtrace, PyPI flashtrace): run only if the package imports in
       the pinned environment; otherwise skipped with the reason.
4. Metrics: Spearman per trace between each predictor and each ground truth; AUROC for the top-quartile
   actual effects (labels per trace: |change| >= its 75th percentile), per trace and pooled over traces.
5. Carriers: for the top 5 dependencies per trace by D, the 5 feature nodes at sentence i with the most
   influence into j's sampled outputs (summed over j's graphs), with their top-activating tokens from
   the transcoder repository's feature-example files (features/index.json.gz + layer_<l>.bin, the format
   circuit-tracer's frontend reads), fetched by HTTP range requests and cached under $HF_HOME; fallback:
   top-activating positions over the 3 traces and the problem texts. Writes
   results/gate0/carriers_for_labelling.md; reads Dawar's labels from results/gate0/carrier_labels.json
   ({"<trace>/<dep>": true|false}).
PR11 (PREREG_GATE0 stage_f.pr11) uses the filler ground truth and pooled AUROC: feature-level D against
S4. Status pass / fail / weak; "pending" if the AUROC part passes and labels are missing.

--tiny: tiny traces and tiny premise model; "real model" = the tiny model's nonlinear forward
(TinyModel.clean_forward logits); suppression edits the tiny attention scores; carriers = top-activating
positions within the traces. Reference graphs from premise_B --tiny if present, else built here on the
prefixes (artifacts/tiny/tiny_premise/premise_refs_F/).
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import types
import zlib

import torch

from cct import premise_traces as pt
from cct import scriptutil as su
from cct.premise_df import (TA_P_NUCLEUS, align_after_deletion, auroc, delete_span, dependency_matrix,
                            filler_replace, graph_node_positions, output_sentence_influence, pairs_from,
                            pr11_status, rollout_matrix, spearman_nan, ta_sentence_effect, ta_sparse_kl,
                            tiny_forward, top_quartile, suppress_scores)
from cct.reach import head_weights, rollout_layers
from cct.utils.io import atomic_write_json, atomic_write_text, load_pt, read_json, save_pt
from cct.utils.logging import get_logger, run_record

CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")
PREREG = os.path.join(su.ROOT, "configs", "prereg_gate0.yaml")
TINY_FILLER_ID = 0
PREDICTORS = ["D", "D_raw", "S4", "S4_logprob", "S1", "S2", "S5"]
TRUTHS = ["filler", "deletion"]


# =================================================================================================
# language-model wrappers ("real model")
# =================================================================================================
class TinyLM:
    """The tiny model's nonlinear forward as the 'real model'."""

    def __init__(self, model):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.name = "tiny model nonlinear forward (TinyModel.clean_forward equivalent)"

    @torch.no_grad()
    def logits(self, tokens, suppress=None):
        return tiny_forward(self.model, tokens, suppress=suppress).double()

    @torch.no_grad()
    def suppression_check(self, tokens, span) -> float:
        _, pats = tiny_forward(self.model, tokens, suppress=[span], return_patterns=True)
        return float(pats[..., span[0]:span[1]].sum(-1).max())

    @torch.no_grad()
    def rollout(self, tokens):
        _, pats = tiny_forward(self.model, tokens, return_patterns=True)
        return rollout_matrix(rollout_layers(pats, head_weights(self.model.W_V, self.model.W_O)))

    def grad_x_input(self, tokens, positions) -> torch.Tensor:
        x = self.model.W_E[tokens.long()].clone().requires_grad_(True)
        with torch.enable_grad():
            lg = tiny_forward(self.model, tokens, embed=x)
            lp = torch.log_softmax(lg.double(), -1)
            tot = sum(lp[p - 1, int(tokens[p])] for p in positions)
            (g,) = torch.autograd.grad(tot, x)
        return (g * x.detach()).sum(-1).double()

    def decode(self, ids) -> str:
        return " ".join(f"<{int(t)}>" for t in ids)

    def feature_acts(self, tokens) -> torch.Tensor:
        """[L, n, F] transcoder activations (position 0 zeroed, as attribution)."""
        c = self.model.clean_forward(tokens.long())
        acts = []
        for l in range(self.n_layers):
            a = self.model.jumprelu(l, self.model.pre_acts(l, c["feat_in"][l][0]))
            a[0] = 0
            acts.append(a)
        return torch.stack(acts)


class RealLM:
    """TransformerLens model of circuit-tracer's ReplacementModel (VM only)."""

    def __init__(self, model):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.device = model.cfg.device
        self.hooks_removed = False
        self.name = ("TransformerLens model inside circuit-tracer's ReplacementModel; forward values equal the "
                     "plain model (its permanent hooks only stop gradients); gradients after removing every "
                     "permanent hook (reset_hooks(including_permanent=True))")

    def _prep(self, tokens) -> tuple[torch.Tensor, int]:
        """Model tokens as circuit-tracer builds them (ensure_tokenized may prepend a special token) and the
        offset between model and trace coordinates; every output below is in trace coordinates."""
        mt = self.model.ensure_tokenized(tokens.long().to(self.device))
        return mt, int(mt.numel() - tokens.numel())

    def _run(self, tokens, fwd_hooks=()):
        """Logits [n, V] in trace coordinates (rows of a prepended token dropped)."""
        mt, off = self._prep(tokens)
        with self.model.hooks(fwd_hooks=list(fwd_hooks)):
            return self.model(mt[None])[0][off:]

    def _supp_hooks(self, spans, off: int):
        sp = [(s + off, e + off) for s, e in spans]

        def hook(scores, hook):
            return suppress_scores(scores, sp)

        return [(f"blocks.{l}.attn.hook_attn_scores", hook) for l in range(self.n_layers)]

    @torch.no_grad()
    def logits(self, tokens, suppress=None):
        off = self._prep(tokens)[1]
        return self._run(tokens, self._supp_hooks(suppress, off) if suppress else ()).double()

    @torch.no_grad()
    def suppression_check(self, tokens, span) -> float:
        """Largest attention mass on the suppressed keys over all layers, heads and queries (should be 0;
        checks that hook_attn_scores sits before the softmax)."""
        off = self._prep(tokens)[1]
        mx = [0.0]

        def keep(p, hook):
            mx[0] = max(mx[0], float(p[..., span[0] + off:span[1] + off].sum(-1).max()))
            return p

        hooks = self._supp_hooks([span], off) + [(f"blocks.{l}.attn.hook_pattern", keep) for l in range(self.n_layers)]
        self._run(tokens, hooks)
        return mx[0]

    @torch.no_grad()
    def rollout(self, tokens):
        from cct.operators.circuit_tracer_backend import CircuitTracerOperator

        mt, off = self._prep(tokens)
        ns = types.SimpleNamespace(model=self.model, tokens=mt, n_layers=self.n_layers)
        pats = CircuitTracerOperator.attention_patterns(ns)
        Wv, Wo = CircuitTracerOperator.ov_weights(ns)
        return rollout_matrix(rollout_layers(pats, head_weights(Wv, Wo)))[off:, off:].cpu()

    def remove_permanent_hooks(self):
        if not self.hooks_removed:
            self.model.reset_hooks(including_permanent=True)
            self.hooks_removed = True

    def grad_x_input(self, tokens, positions) -> torch.Tensor:
        """Per trace position: grad . embedding of the summed log-probability of the tokens at positions."""
        self.remove_permanent_hooks()
        store = {}

        def emb(acts, hook):
            x = acts.detach().clone().requires_grad_(True)
            store["x"] = x
            return x

        t = tokens.long().to(self.device)
        off = self._prep(tokens)[1]
        with torch.enable_grad():
            lg = self._run(t, [("hook_embed", emb)])
            lp = torch.log_softmax(lg.float(), -1)
            idx = torch.tensor(positions, device=lp.device)
            tot = lp[idx - 1, t[idx]].sum()
            (g,) = torch.autograd.grad(tot, store["x"])
        return (g * store["x"].detach())[0].sum(-1)[off:].double().cpu()

    def decode(self, ids) -> str:
        return self.model.tokenizer.decode([int(x) for x in ids])

    @torch.no_grad()
    def feature_acts_sparse(self, tokens):
        mt, off = self._prep(tokens)
        _, acts = self.model.get_activations(mt, sparse=True)
        acts = acts.coalesce().cpu()
        if off:  # back to trace coordinates (drop the prepended position)
            idx, val = acts.indices(), acts.values()
            keep = idx[1] >= off
            idx = idx[:, keep].clone()
            idx[1] -= off
            acts = torch.sparse_coo_tensor(idx, val[keep], (acts.shape[0], acts.shape[1] - off, acts.shape[2])).coalesce()
        return acts


# =================================================================================================
# reference graphs
# =================================================================================================
def find_refs(run: str, set_name: str, tid: str) -> dict[int, str]:
    """{position: path} of Stage B reference graphs for trace tid: artifacts/<run>/<set>/premise_refs/<tid>/<q>.pt."""
    d = os.path.join(su.ROOT, "artifacts", run, set_name, "premise_refs", tid)
    out = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".pt"):
                try:
                    out[int(os.path.splitext(f)[0])] = os.path.join(d, f)
                except ValueError:
                    pass
    return out


def load_ref(path: str) -> dict:
    """Reference graph record, memory-mapped when possible (only the metadata and node influence are read)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001
        return load_pt(path)


def tiny_ref(model, tokens, q: int, path: str, cfg: dict) -> dict:
    """Exact graph of the tiny model on tokens[:q] (every feature visited), saved to path."""
    from cct.killtest import exact_graph_tiny

    if os.path.exists(path):
        return load_pt(path)
    rec = exact_graph_tiny(model, tokens[:q], cap=10 ** 9, batch_size=64, regime="F",
                           prune_cfg={"node_threshold": cfg["exact"]["node_threshold"],
                                      "edge_threshold": cfg["exact"]["edge_threshold"]})
    rec.update({"position": q, "prefix_offset": 0, "built_by": "premise_F tiny fallback"})
    save_pt(path, rec)
    return rec


# =================================================================================================
# carriers: feature examples
# =================================================================================================
class FeatureExamples:
    """Top-activating examples from the transcoder repository's features/ folder (circuit-tracer frontend
    format: features/index.json.gz maps a layer to {"filename", "offsets"}; the bytes of feature f are
    offsets[f]..offsets[f+1] of features/<filename>: a 4-byte little-endian length, then a zlib stream
    of the feature's JSON (frontend/feature_models.py Model). Fetched by HTTP range requests, cached
    under $HF_HOME/cct_feature_examples/<repo>/."""

    def __init__(self, repo: str, log):
        self.repo, self.log = repo, log
        self.base = f"https://huggingface.co/{repo}/resolve/main/features/"
        home = os.environ.get("HF_HOME")
        if not home:
            raise RuntimeError("HF_HOME is not set: feature examples are downloaded only under $HF_HOME")
        self.cache = os.path.join(home, "cct_feature_examples", repo.replace("/", "__"))
        os.makedirs(self.cache, exist_ok=True)
        self.index = None

    def _get(self, url: str, rng: tuple[int, int] | None = None) -> bytes:
        import requests

        headers = {}
        tok = os.environ.get("HF_TOKEN")
        if tok:
            headers["Authorization"] = f"Bearer {tok}"
        if rng:
            headers["Range"] = f"bytes={rng[0]}-{rng[1]}"
        r = requests.get(url, headers=headers, timeout=60)
        r.raise_for_status()
        return r.content

    def _index(self):
        if self.index is None:
            p = os.path.join(self.cache, "index.json.gz")
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(self._get(self.base + "index.json.gz"))
            with open(p, "rb") as f:
                self.index = json.loads(gzip.decompress(f.read()).decode("utf-8"))
        return self.index

    def feature(self, layer: int, feat: int) -> dict:
        p = os.path.join(self.cache, f"{layer}_{feat}.json")
        if os.path.exists(p):
            return read_json(p)
        idx = self._index()
        ent = idx[str(layer)] if isinstance(idx, dict) else idx[layer]
        offs, fn = ent["offsets"], ent["filename"]
        raw = self._get(self.base + fn, (int(offs[feat]), int(offs[feat + 1]) - 1))
        n = int.from_bytes(raw[:4], "little")
        body = raw[4:4 + n]
        try:
            txt = zlib.decompress(body)
        except zlib.error:
            txt = zlib.decompress(body, -15)
        d = json.loads(txt.decode("utf-8"))
        atomic_write_json(p, d)
        return d


def summarise_examples(feat: dict, k: int = 3, width: int = 6) -> list[str]:
    """Top-activating token in context for the first k examples of the top quantile."""
    out = []
    qs = feat.get("examples_quantiles") or []
    exs = qs[0]["examples"] if qs else []
    for ex in exs[:k]:
        toks, acts = ex["tokens"], ex["tokens_acts_list"]
        if not toks:
            continue
        m = max(range(len(acts)), key=lambda i: acts[i])
        ctx = "".join(toks[max(0, m - width):m]) + "[[" + toks[m] + "]]" + "".join(toks[m + 1:m + 3])
        out.append(ctx.replace("\n", "\\n").replace("|", "/"))
    return out


def local_top_contexts(acts_list, toks_list, decode, layer: int, feat: int, k: int = 3, width: int = 6) -> list[str]:
    """Top-activating positions of (layer, feat) over a list of texts. acts_list: per text either a dense
    [L, n, F] tensor or a sparse COO tensor."""
    cands = []
    for ti, (A, toks) in enumerate(zip(acts_list, toks_list)):
        if A.is_sparse:
            idx, val = A.indices(), A.values()
            sel = (idx[0] == layer) & (idx[2] == feat)
            for p, v in zip(idx[1][sel].tolist(), val[sel].tolist()):
                cands.append((float(v), ti, int(p)))
        else:
            col = A[layer, :, feat]
            for p in torch.nonzero(col > 0).flatten().tolist():
                cands.append((float(col[p]), ti, int(p)))
    cands.sort(reverse=True)
    out = []
    for v, ti, p in cands[:k]:
        toks = toks_list[ti]
        ctx = decode(toks[max(0, p - width):p]) + "[[" + decode(toks[p:p + 1]) + "]]" + decode(toks[p + 1:p + 3])
        ctx = ctx.replace("\n", "\\n").replace("|", "/")
        out.append(f"{ctx} ({v:.2f})")
    return out


# =================================================================================================
# S5 FlashTrace
# =================================================================================================
def flashtrace_available() -> tuple[bool, str]:
    try:
        import flashtrace  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, (f"flashtrace is not importable in the pinned environment ({type(e).__name__}); public code "
                       "exists (github.com/wbopan/flashtrace, PyPI flashtrace 0.1.1) but needs numpy>=2, spacy and "
                       "transformers>=4.53, and installing it could change the pinned environment")
    return True, ""


def run_flashtrace(model_name: str, tokens, spans, j_targets: dict, decode, dtype: str, log) -> torch.Tensor | None:
    """S5[j, i] = sum over sentence i's positions of FlashTrace scores of j's sampled tokens, with the
    prompt = the trace before sentence j and the target = sentence j (output_span = the sampled tokens,
    inclusive, relative to the start of j). Positions are aligned only if FlashTrace's prompt tokens
    re-tokenise to exactly tokens[:start_j]; otherwise that row is left nan. Unverified on real runs."""
    from flashtrace import FlashTrace, load_model_and_tokenizer

    m, tok = load_model_and_tokenizer(model_name, device_map="auto", dtype=dtype)
    tracer = FlashTrace(m, tok)
    n = len(spans)
    S5 = torch.full((n, n), float("nan"), dtype=torch.float64)
    for j, poss in j_targets.items():
        sj, ej = spans[j]
        prompt, target = decode(tokens[:sj]), decode(tokens[sj:ej])
        rel = [p - sj for p in poss]
        tr = tracer.trace(prompt=prompt, target=target, output_span=(min(rel), max(rel)), hops=1)
        if len(tr.prompt_tokens) != sj:
            log.info(f"S5: prompt re-tokenised to {len(tr.prompt_tokens)} tokens, expected {sj}; row {j} skipped")
            continue
        sc = torch.as_tensor(tr.scores, dtype=torch.float64).flatten()
        for i in range(j):
            s, e = spans[i]
            S5[j, i] = float(sc[s:e].sum())
    del m
    return S5


# =================================================================================================
# per trace
# =================================================================================================
def run_trace(args, cfg, tr: dict, lm, tiny_model, filler_id: int, log) -> dict:
    t0 = time.perf_counter()
    tid = tr["id"]
    toks = torch.as_tensor(tr["token_ids"]).long()
    # sentences: stored by gen_traces (Thought Anchors' rule via premise_traces.split_sentences); else split here
    if tr.get("sentences"):
        spans = [(int(a), int(b)) for a, b in tr["sentences"]]
        sent_src = "stored in traces.json"
        if not args.tiny:
            s0, e0 = pt.thinking_span(tr)
            again = pt.split_sentences(tr["token_ids"], lm.decode, s0, e0)
            if again != spans:
                log.info(f"{tid}: stored sentences differ from split_sentences now; using the stored ones")
                sent_src += " (differs from a fresh split_sentences)"
    elif args.tiny:
        spans = pt.trace_sentences(tr)
        sent_src = "premise_traces.tiny_sentences"
    else:
        s0, e0 = pt.thinking_span(tr)
        spans = pt.split_sentences(tr["token_ids"], lm.decode, s0, e0)
        sent_src = "premise_traces.split_sentences on the thinking span"
    outs = pt.trace_outputs(tr, cfg["traces"]["sentences_per_trace"], cfg["traces"]["tokens_per_sentence"])
    n = len(spans)
    # -- reference graphs ---------------------------------------------------------------------------------
    refs = find_refs("tiny" if args.tiny else "real", args.model_set, tid)
    built = 0
    recs = {}
    for o in outs:
        q = int(o["position"])
        if q in refs:
            rec = load_ref(refs[q])
        elif args.tiny:
            d = os.path.join(su.ROOT, "artifacts", "tiny", "tiny_premise", "premise_refs_F", tid)
            rec = tiny_ref(tiny_model, toks, q, os.path.join(d, f"{q}.pt"), cfg)
            built += 1
        else:
            raise FileNotFoundError(f"{tid}: no reference graph for position {q} (run premise_B first)")
        # keep only what Stage F needs: the packed matrices of real graphs are several GB each
        recs[q] = {"source_meta": rec["source_meta"], "prefix_offset": rec.get("prefix_offset", 0),
                   "exact_prune": {"node_influence": rec["exact_prune"]["node_influence"]}}
        del rec
    # -- feature-level dependency -----------------------------------------------------------------------
    infl_outputs, j_tokens, node_tables = [], {}, {}
    for o in outs:
        j = o.get("sentence_index")
        q = int(o["position"])
        rec = recs[q]
        npos, nkind = graph_node_positions(rec)
        last = int(npos.max())
        if last + 1 != q:
            log.info(f"{tid}: graph for q={q} ends at trace position {last} (expected {q - 1})")
        if j is None:
            continue
        ni = rec["exact_prune"]["node_influence"].double().cpu()
        infl_outputs.append({"sentence_index": int(j), "influence_by_sentence": output_sentence_influence(ni, npos, spans)})
        j_tokens.setdefault(int(j), []).append(q)
        node_tables.setdefault(int(j), []).append((rec, ni, npos, nkind))
    D, raw = dependency_matrix(infl_outputs, spans)
    pairs = pairs_from(D)
    js = sorted(j_tokens)
    # -- ground truth and S4 ----------------------------------------------------------------------------------
    base = logprobs(lm.logits(toks))
    lp0 = {j: sum_lp(base, toks, j_tokens[j]) for j in js}
    max_j = max(js) if js else 0
    eff = {k: torch.full((n, n), float("nan"), dtype=torch.float64)
           for k in ("filler", "deletion", "S4", "S4_logprob", "filler_signed", "deletion_signed")}
    span_pos = sorted({p for j in js for p in range(*spans[j])})
    supp_check = None
    for i in range(max_j):
        if not any(j > i for j in js):
            continue
        si = spans[i]
        lf = logprobs(lm.logits(filler_replace(toks, si, filler_id)))
        dt = delete_span(toks, si)
        ld = logprobs(lm.logits(dt))
        ls = logprobs(lm.logits(toks, suppress=[si]))
        if supp_check is None:
            supp_check = lm.suppression_check(toks, si)
        idx = torch.tensor(span_pos)
        kl = torch.full((toks.numel(),), float("nan"), dtype=torch.float64)
        kl[idx] = ta_sparse_kl(base[idx], ls[idx])
        for j in js:
            if j <= i:
                continue
            pj = j_tokens[j]
            df = sum_lp(lf, toks, pj) - lp0[j]
            pd = align_after_deletion(toks, dt, pj, si)
            dd = sum_lp(ld, dt, pd) - lp0[j]
            eff["filler_signed"][j, i], eff["filler"][j, i] = df, abs(df)
            eff["deletion_signed"][j, i], eff["deletion"][j, i] = dd, abs(dd)
            eff["S4"][j, i] = ta_sentence_effect(kl, spans[j])
            eff["S4_logprob"][j, i] = abs(sum_lp(ls, toks, pj) - lp0[j])
        del lf, ld, ls
    # -- S1 rollout -------------------------------------------------------------------------------------------
    Rho = lm.rollout(toks).double()
    S1 = torch.full((n, n), float("nan"), dtype=torch.float64)
    for j in js:
        for i in range(j):
            s, e = spans[i]
            S1[j, i] = float(sum(Rho[p - 1, s:e].sum() for p in j_tokens[j]))
    # -- carriers (before any hook removal) -------------------------------------------------------------------
    top = sorted(pairs, key=lambda ji: -float(D[ji]))[: cfg["stage_f"]["top_dependencies_per_trace"]]
    carriers = []
    for (j, i) in top:
        s, e = spans[i]
        agg, kind_mass = {}, {"feature": 0.0, "error": 0.0, "token": 0.0}
        for rec, ni, npos, nkind in node_tables[j]:
            sm = rec["source_meta"]
            sel = torch.nonzero((npos >= s) & (npos < e)).flatten()
            for k in sel.tolist():
                kn = {1: "feature", 2: "error", 3: "token"}.get(int(nkind[k]), "other")
                kind_mass[kn] = kind_mass.get(kn, 0.0) + float(ni[k])
                if int(nkind[k]) == 1:
                    key = (int(sm["layer"][k]), int(npos[k]), int(sm["feature"][k]))
                    agg[key] = agg.get(key, 0.0) + float(ni[k])
        tot = sum(kind_mass.values()) or 1.0
        best = sorted(agg.items(), key=lambda kv: -kv[1])[: cfg["stage_f"]["carriers_per_dependency"]]
        carriers.append({"key": f"{tid}/j{j}_i{i}", "j": j, "i": i, "D": float(D[j, i]),
                         "sentence_i": lm.decode(toks[s:e].tolist()), "sentence_j": lm.decode(toks[spans[j][0]:spans[j][1]].tolist()),
                         "kind_share": {k: v / tot for k, v in kind_mass.items()},
                         "features": [{"layer": l, "pos": p, "feature": f, "influence": v} for (l, p, f), v in best]})
    return {"trace_id": tid, "spans": spans, "sentence_source": sent_src, "outputs": outs, "j_tokens": j_tokens,
            "refs_built_here": built, "pairs": pairs, "D": D, "D_raw": raw, "eff": eff, "S1": S1,
            "suppression_max_mass": supp_check, "carriers": carriers, "tokens": toks,
            "seconds": time.perf_counter() - t0}


def sum_lp(lp, tokens, positions) -> float:
    """Sum over positions p of lp[p - 1, tokens[p]]; lp are log-probabilities [n, V]."""
    idx = torch.as_tensor(positions, dtype=torch.long)
    t = torch.as_tensor(tokens).long().cpu()
    return float(lp[(idx - 1).to(lp.device), t[idx].to(lp.device)].double().sum())


def logprobs(logits):
    """Log-probabilities; softmax(logprobs / T) equals softmax(logits / T), so the Thought Anchors KL is
    unchanged when computed on them."""
    return torch.log_softmax(logits.double(), -1)


# =================================================================================================
# metrics
# =================================================================================================
def pair_vector(M: torch.Tensor, pairs) -> torch.Tensor:
    return torch.tensor([float(M[j, i]) for j, i in pairs], dtype=torch.float64)


def metrics(per: dict) -> dict:
    out = {"per_trace": {}, "pooled": {}}
    pooled = {t: {"labels": [], **{p: [] for p in PREDICTORS}} for t in TRUTHS}
    for tid, r in per.items():
        pairs = r["pairs"]
        pv = {p: pair_vector(r["pred"][p], pairs) if r["pred"].get(p) is not None else None for p in PREDICTORS}
        out["per_trace"][tid] = {}
        for t in TRUTHS:
            act = pair_vector(r["eff"][t], pairs)
            lab = top_quartile(act) if act.numel() else torch.zeros(0, dtype=torch.bool)
            row = {"n_pairs": len(pairs), "n_top_quartile": int(lab.sum())}
            for p in PREDICTORS:
                if pv[p] is None:
                    row[p] = {"spearman": None, "auroc": None}
                    continue
                row[p] = {"spearman": spearman_nan(pv[p], act), "auroc": auroc(pv[p], lab)}
                pooled[t][p].append(pv[p])
            pooled[t]["labels"].append(lab)
            out["per_trace"][tid][t] = row
    for t in TRUTHS:
        lab = torch.cat(pooled[t]["labels"]) if pooled[t]["labels"] else torch.zeros(0, dtype=torch.bool)
        out["pooled"][t] = {}
        for p in PREDICTORS:
            if len(pooled[t][p]) != len(pooled[t]["labels"]) or not pooled[t][p]:
                out["pooled"][t][p] = {"auroc": None}
                continue
            out["pooled"][t][p] = {"auroc": auroc(torch.cat(pooled[t][p]), lab)}
    return out


def write_carriers_md(path: str, per: dict, source: str, labels: dict | None) -> None:
    lines = ["# Carrier features for labelling (Gate 0, PR11)", "",
             "Plain summary: these are the 5 strongest feature-level step dependencies per trace and, for each,",
             "the 5 features at the earlier sentence that carry most influence into the later sentence's sampled",
             "tokens. Please judge whether each set of carriers is nameable (one shared, describable role).", "",
             f"Top-activating examples: {source}. [[token]] marks the most active token.", "",
             "How to label: write results/gate0/carrier_labels.json as {\"<key>\": true or false}, one entry per row.",
             "", "| key | D | sentence i (earlier) | sentence j (later) | carriers: layer/feature @pos (influence): top contexts | nameable? |",
             "|---|---|---|---|---|---|"]
    for r in per.values():
        for c in r["carriers"]:
            cs = "<br>".join(f"L{f['layer']}/{f['feature']} @{f['pos']} ({f['influence']:.3g}): " +
                             " // ".join(f.get("examples") or ["(no examples)"]) for f in c["features"])
            esc = lambda s: s.replace("|", "/").replace("\n", " ")  # noqa: E731
            lab = "" if not labels or c["key"] not in labels else ("yes" if labels[c["key"]] else "no")
            lines.append(f"| {c['key']} | {c['D']:.3f} | {esc(c['sentence_i'])} | {esc(c['sentence_j'])} | {cs} | {lab} |")
    atomic_write_text(path, "\n".join(lines) + "\n")


# =================================================================================================
def main() -> None:
    p = su.parser("Gate 0 Stage F: PR11 (feature-level step dependencies against interventions)")
    p.set_defaults(config=CONFIG)
    p.add_argument("--model-set", default=None)
    p.add_argument("--out-dir", default=None, help="output directory (default: <results>/gate0; tests)")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    prereg = su.load_yaml(PREREG)
    args.model_set = args.model_set or cfg["sets"]["qwen"]
    tag = pt.model_tag(args.model_set, cfg["sets"]["qwen"])
    res_dir = pt.gate0_results_dir(su.ROOT, args.tiny, tag)
    out_dir = args.out_dir or res_dir
    os.makedirs(out_dir, exist_ok=True)
    log = get_logger("premise_F", os.path.join(su.ROOT, "logs"))
    traces = pt.load_traces(pt.traces_file(su.ROOT, args.tiny, tag, cfg))
    t_start = time.perf_counter()
    notes = []
    if args.tiny:
        tiny_model = pt.tiny_premise_model()
        lm, filler_id, filler_note = TinyLM(tiny_model), TINY_FILLER_ID, f"tiny token id {TINY_FILLER_ID}"
        model_name = "tiny"
    else:
        from cct.operators.circuit_tracer_backend import load_replacement_model
        from cct.utils.gpu import cap_memory, nvidia_smi

        log.info(nvidia_smi())
        mc = su.model_config(args.model_set)
        cap_memory(mc["memory_cap_gb"])
        model = load_replacement_model(mc["model_name"], mc["transcoder_set"], dtype=su.torch_dtype(mc["dtype"]))
        lm, tiny_model, model_name = RealLM(model), None, mc["model_name"]
        ft = cfg["stage_f"]["filler_token"]
        ids = model.tokenizer.encode(ft, add_special_tokens=False)
        if len(ids) != 1:
            raise SystemExit(f"filler {ft!r} tokenises to {len(ids)} tokens ({ids}); choose a one-token filler "
                             "(ask Dawar: the filler is part of the committed settings)")
        filler_id, filler_note = int(ids[0]), f"{ft!r} -> token id {ids[0]} (one token, checked)"
    per = {}
    for tr in traces:
        r = run_trace(args, cfg, tr, lm, tiny_model, filler_id, log)
        per[tr["id"]] = r
        log.info(f"{tr['id']}: {len(r['spans'])} sentences, {len(r['pairs'])} pairs, "
                 f"suppression max mass {r['suppression_max_mass']}")
    # -- carriers' top-activating tokens ------------------------------------------------------------------------
    labels_path = os.path.join(out_dir, "carrier_labels.json")
    labels = read_json(labels_path) if os.path.exists(labels_path) else None
    ex_source = None
    if not args.tiny:
        try:
            fe = FeatureExamples(su.model_config(args.model_set)["transcoder_set"], log)
            for r in per.values():
                for c in r["carriers"]:
                    for f in c["features"]:
                        f["examples"] = summarise_examples(fe.feature(f["layer"], f["feature"]))
            ex_source = (f"feature-example files of {fe.repo} (features/index.json.gz and layer_<l>.bin, top "
                         "quantile, first 3 examples)")
        except Exception as e:  # noqa: BLE001
            log.info(f"feature examples unavailable ({e!r}); computing top-activating contexts locally")
            notes.append(f"feature-example files unavailable ({e!r})")
    if ex_source is None:
        texts = [r["tokens"] for r in per.values()]
        if not args.tiny:
            probs = su.load_yaml(os.path.join(su.ROOT, cfg["traces"]["problems_file"])) \
                if cfg["traces"]["problems_file"].endswith(".yaml") else read_json(os.path.join(su.ROOT, cfg["traces"]["problems_file"]))
            texts += [lm.model.ensure_tokenized(pp["question"]).cpu() for pp in probs]
            acts = [lm.feature_acts_sparse(t) for t in texts]
            ex_source = "computed here: top-activating positions over the 3 traces and the problem texts (no repository examples)"
        else:
            acts = [lm.feature_acts(t) for t in texts]
            ex_source = "computed here: top-activating positions within the tiny traces"
        tl = [t.tolist() for t in texts]
        for r in per.values():
            for c in r["carriers"]:
                for f in c["features"]:
                    f["examples"] = local_top_contexts(acts, tl, lm.decode, f["layer"], f["feature"])
    # -- S2 (real: after removing the permanent hooks) -------------------------------------------------------
    for tid, r in per.items():
        n = len(r["spans"])
        S2 = torch.full((n, n), float("nan"), dtype=torch.float64)
        for j, poss in r["j_tokens"].items():
            gx = lm.grad_x_input(r["tokens"], poss)
            for i in range(j):
                s, e = r["spans"][i]
                S2[j, i] = abs(float(gx[s:e].sum()))
        r["pred"] = {"D": r["D"], "D_raw": r["D_raw"], "S4": r["eff"]["S4"], "S4_logprob": r["eff"]["S4_logprob"],
                     "S1": r["S1"], "S2": S2, "S5": None}
    # -- S5 ----------------------------------------------------------------------------------------------------
    ok, why = flashtrace_available()
    s5 = {"status": "skipped", "reason": why}
    if ok and not args.tiny:
        try:
            for tid, r in per.items():
                r["pred"]["S5"] = run_flashtrace(model_name, r["tokens"].tolist(), r["spans"], r["j_tokens"], lm.decode,
                                                 su.model_config(args.model_set)["dtype"] or "float32", log)
            s5 = {"status": "run", "reason": "flashtrace imported; method flashtrace, hops 1 (unverified alignment)"}
        except Exception as e:  # noqa: BLE001
            s5 = {"status": "skipped", "reason": f"flashtrace failed: {e!r}"}
            for r in per.values():
                r["pred"]["S5"] = None
    elif ok and args.tiny:
        s5 = {"status": "skipped", "reason": "tiny run (FlashTrace needs a Hugging Face model)"}
    write_carriers_md(os.path.join(out_dir, "carriers_for_labelling.md"), per, ex_source, labels)
    # -- metrics and PR11 -----------------------------------------------------------------------------------------
    met = metrics(per)
    pr = prereg["stage_f"]["pr11"]
    a_feat = met["pooled"]["filler"]["D"]["auroc"]
    a_s4 = met["pooled"]["filler"]["S4"]["auroc"]
    keys = [c["key"] for r in per.values() for c in r["carriers"]]
    n_nameable = None
    if labels is not None:
        n_nameable = sum(1 for k in keys if labels.get(k) is True)
        missing = [k for k in keys if k not in labels]
        if missing:
            notes.append(f"{len(missing)} carrier keys have no label: {missing}")
    st = pr11_status(a_feat if a_feat is not None else float("nan"), a_s4 if a_s4 is not None else float("nan"),
                     n_nameable, pr)
    pr11 = {"status": st["status"],
            "values": {"auroc_feature_level": a_feat, "auroc_s4": a_s4,
                       "auroc_minus_s4": None if a_feat is None or a_s4 is None else a_feat - a_s4,
                       "n_nameable": n_nameable, "n_top_dependencies": len(keys),
                       "auroc_condition": st["auroc_condition"], "nameable_condition": st["nameable_condition"],
                       "auroc_deletion_feature_level": met["pooled"]["deletion"]["D"]["auroc"],
                       "auroc_deletion_s4": met["pooled"]["deletion"]["S4"]["auroc"]},
            "thresholds": pr,
            "notes": ("pooled AUROC over the 3 traces against the filler ground truth (top-quartile labels per trace); "
                      "deletion variant in values and detail" +
                      ("; nameable condition pending_dawar (no results/gate0/carrier_labels.json)" if labels is None else ""))}
    gpu_s = time.perf_counter() - t_start if (torch.cuda.is_available() and not args.tiny) else 0.0
    detail = {
        "real_model": lm.name, "filler": filler_note, "metrics": met, "S5": s5,
        "S4_definition": {"source": "github.com/interp-reasoning/thought-anchors whitebox-analyses/attention_analysis/"
                                    "attn_supp_funcs.py get_suppression_KL_matrix; logits_funcs.py compress_logits_top_p",
                          "mask": "pre-softmax scores to sentence i's keys set to finfo.min, all layers, all heads, all queries",
                          "effect": "mean over sentence j's token positions of log(KL(base || suppressed) + 1e-9), "
                                    f"top-p {TA_P_NUCLEUS}, max 100 tokens, temperature 0.6"},
        "carrier_examples_source": ex_source, "notes": notes,
        "decisions": ["actual effect = |change in summed log-probability| (signed change stored)",
                      "D uses every node's influence in the unpruned reference graph; D_raw is the unnormalised sum",
                      "top-quartile labels are set per trace; pooled AUROC concatenates the traces' pairs",
                      "carriers are feature nodes only; the share of sentence i's influence held by error and token "
                      "nodes is reported per dependency",
                      "S1 query position = the position whose logits predict the sampled token (q - 1)"],
        "per_trace": {tid: {"n_sentences": len(r["spans"]), "spans": r["spans"], "sentence_source": r["sentence_source"],
                            "n_pairs": len(r["pairs"]), "pairs": r["pairs"], "refs_built_here": r["refs_built_here"],
                            "suppression_max_mass": r["suppression_max_mass"], "seconds": r["seconds"],
                            "D": r["D"].tolist(), "D_raw": r["D_raw"].tolist(),
                            "effects": {k: v.tolist() for k, v in r["eff"].items()},
                            "S1": r["S1"].tolist(), "S2": r["pred"]["S2"].tolist(),
                            "S5": None if r["pred"]["S5"] is None else r["pred"]["S5"].tolist(),
                            "carriers": r["carriers"]} for tid, r in per.items()},
    }
    result = {"run": run_record(cfg, cfg["seed"], time.perf_counter() - t_start, model_set=args.model_set, tiny=args.tiny),
              "stage": "F", "premises": {"PR11": pr11}, "stop": False, "stop_reason": None, "gpu_seconds": gpu_s,
              "detail": json.loads(json.dumps(detail, default=_nan_safe))}
    atomic_write_json(os.path.join(out_dir, "stage_F.json"), result)
    log.info(f"Stage F: PR11 {pr11['status']} (AUROC D {a_feat}, S4 {a_s4}, nameable {n_nameable})")


def _nan_safe(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    main()
