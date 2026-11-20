"""Pure logic for Gate 0 Stages D and F (GATE0_PREMISES.md Section 3; PREREG_GATE0.md Section 2).

Stage D (PR7, PR8, PR9): absolute s95 and k_eff, causal sets, mass recall inside Omega_s, the
count-sketch (D2) simulation on exact remainder columns, hub flags, m_trace, and the energy-screening
count used by Stage E.

Stage F (PR11): the feature-level dependency matrix D[j, i] from reference graphs, sentence
interventions (filler replacement and deletion), Thought Anchors' attention-suppression effect
measure, AUROC and top-quartile labels, and the PR11 status rule. A small tiny-model forward with
attention suppression and an embedding override lives here too (the --tiny "real model").

Everything here runs on the CPU in the tests; the scripts move tensors to the GPU where it matters.
"""
from __future__ import annotations

import math
import time

import torch
from torch import Tensor

KIND_NAMES = {1: "feature", 2: "error", 3: "token", 0: "logit"}


# =================================================================================================
# Stage D
# =================================================================================================
def abs_s95(v: Tensor, frac: float = 0.95) -> Tensor:
    """Per column of v [T, C]: number of largest-|.| entries holding `frac` of the column's absolute (L1)
    mass (0 for zero columns). PREREG_GATE0 'Operational definitions', s95 (absolute)."""
    a = v.double().abs()
    tot = a.sum(0)
    s = torch.sort(a, dim=0, descending=True).values
    cum = torch.cumsum(s, 0)
    cnt = (cum < (frac * tot)[None] * (1 - 1e-12)).sum(0) + 1
    return torch.where(tot > 0, cnt, torch.zeros_like(cnt))


def causal_set(t_kind: Tensor, t_layer: Tensor, t_pos: Tensor, src_min_write_layer: Tensor,
               src_pos: Tensor) -> Tensor:
    """Bool [T, S]: the causal candidate set of each source (PREREG_GATE0): feature targets (logits
    excluded) at a read layer above the source's lowest write layer, at the same or a later position."""
    feat = (t_kind == 1)[:, None]
    lay = t_layer[:, None] > src_min_write_layer[None, :]
    pos = t_pos[:, None] >= src_pos[None, :]
    return feat & lay & pos


def mass_recall(cols: Tensor, mask: Tensor) -> dict:
    """Absolute-mass recall of each column inside mask, and its energy-weighted mean over columns
    (weights = squared column norms; PREREG_GATE0 'Mass recall (PR8)'). cols, mask [T, S]."""
    a = cols.double().abs()
    m = mask.to(a.device, torch.bool)
    tot = a.sum(0)
    per = torch.where(tot > 0, (a * m).sum(0) / tot.clamp_min(1e-300), torch.full_like(tot, float("nan")))
    w = cols.double().pow(2).sum(0)
    ok = (tot > 0) & (w > 0)
    rec = float((per[ok] * w[ok]).sum() / w[ok].sum()) if bool(ok.any()) else float("nan")
    return {"recall": rec, "per_column": per.cpu()}


def omega_fraction(mask: Tensor, causal: Tensor) -> Tensor:
    """Per column |Omega_s inside the causal set| / |causal set| (nan if the causal set is empty)."""
    c = causal.to(torch.bool)
    num = (mask.to(c.device, torch.bool) & c).sum(0).double()
    den = c.sum(0).double()
    return torch.where(den > 0, num / den.clamp_min(1), torch.full_like(den, float("nan"))).cpu()


def nanmedian(x: Tensor) -> float:
    x = x.double().flatten()
    x = x[~torch.isnan(x)]
    return float(x.median()) if x.numel() else float("nan")


def pr8_select(per_tau: dict, recall_min: float, frac_max: float) -> dict:
    """per_tau: {tau_str: {"recall": float, "median_frac": float}}. PR8 passes if some tau has
    recall >= recall_min and median fraction <= frac_max. The tau used for PR9 decoding is the passing
    tau with the smallest median fraction (most selective); if none passes, the tau with the highest
    recall (ties: smaller fraction)."""
    ok = {t: v for t, v in per_tau.items()
          if v["recall"] >= recall_min and v["median_frac"] <= frac_max}
    if ok:
        best = min(ok, key=lambda t: (ok[t]["median_frac"], -ok[t]["recall"]))
        return {"passed": True, "tau": best, "passing_taus": sorted(ok, key=float, reverse=True),
                "reason": "passing tau with the smallest median |Omega_s| / |causal set|"}
    best = max(per_tau, key=lambda t: (per_tau[t]["recall"], -per_tau[t]["median_frac"]))
    return {"passed": False, "tau": best, "passing_taus": [],
            "reason": "no tau passes PR8: tau with the highest mass recall used for PR9"}


def simulate_d2(cols: Tensor, m: int, seed: int, cand: Tensor | None, R: int = 5, device="cpu",
                dtype=torch.float32) -> dict:
    """Count-sketch D2 on exact columns cols [T, S] over the full target set: Y = Phi cols, decode over
    the candidate mask (keep B // 4 per column, PREREG_GATE0). Returns per-column relative errors and
    decode seconds per column."""
    from .sketch import CountSketch, column_rel_err

    T, S = cols.shape
    cs = CountSketch(T, m, R=R, seed=seed, device=device)
    A = cols.to(device, dtype)
    Y = cs.apply(A)
    A_hat, secs = cs.decode(Y, cand=None if cand is None else cand.to(device), keep=max(1, cs.B // 4))
    err = column_rel_err(A_hat, A).cpu()
    energy_est = (Y.double().pow(2).sum(0) / R).cpu()  # each repetition is unbiased for ||a||^2
    return {"err": err, "decode_seconds_per_column": secs / max(S, 1), "energy_est": energy_est,
            "B": cs.B, "m": cs.m}


def hub_flags(err_by_seed: Tensor, eps: float) -> Tensor:
    """err_by_seed [n_seeds, S] at the largest m. A column is a hub if its error is above eps on a
    majority of seeds (median over 3 seeds above eps); zero columns (nan) are not hubs."""
    e = err_by_seed.double()
    bad = (e > eps) & ~torch.isnan(e)
    return bad.sum(0) * 2 > e.shape[0]


def non_hub_share_within(err: Tensor, w: Tensor, hub: Tensor, eps: float) -> float:
    """Share of non-hub column energy w whose error is <= eps."""
    keep = (~hub) & (w > 0) & ~torch.isnan(err)
    tot = float(w[keep].sum())
    if tot <= 0:
        return float("nan")
    return float(w[keep & (err <= eps)].sum()) / tot


def select_m_trace(errs: dict, w: Tensor, hub: Tensor, eps: float, share_min: float = 0.90,
                   seeds_required: int = 2) -> tuple[int | None, dict]:
    """errs: {m: Tensor [n_seeds, S]}. m_trace = smallest m at which >= share_min of non-hub column
    energy has error <= eps, on at least seeds_required seeds."""
    table = {}
    m_trace = None
    for m in sorted(errs, key=int):
        shares = [non_hub_share_within(errs[m][k], w, hub, eps) for k in range(errs[m].shape[0])]
        n_ok = sum(1 for s in shares if not math.isnan(s) and s >= share_min)
        table[str(m)] = {"share_by_seed": shares, "seeds_ok": n_ok, "meets": n_ok >= seeds_required}
        if m_trace is None and n_ok >= seeds_required:
            m_trace = int(m)
    return m_trace, table


def hub_rate_by_kind(hub: Tensor, kinds: Tensor) -> dict:
    out = {}
    for k in sorted(set(kinds.tolist())):
        sel = kinds == k
        out[KIND_NAMES.get(int(k), str(k))] = float(hub[sel].double().mean()) if bool(sel.any()) else float("nan")
    return out


def weighted_screen_count(energy: Tensor, weights: Tensor, q: float, safety: float = 2.0) -> float:
    """Estimated number of columns kept by cct.sensing.screening.screen_columns(q, safety) on the
    whole source set, from a stratified sample: each sampled column stands for `weights` columns
    (stratum size / sampled count). Same rule as screen_columns: the smallest set of largest-energy
    columns holding q_eff = 1 - (1 - q) / safety of the total energy."""
    e = energy.double()
    w = weights.double()
    keep = e > 0
    e, w = e[keep], w[keep]
    if e.numel() == 0:
        return 0.0
    if q is None or q >= 1.0:
        return float(w.sum())
    q_eff = 1.0 - (1.0 - q) / safety
    vals, order = torch.sort(e, descending=True)
    ww = w[order]
    cum = torch.cumsum(vals * ww, 0) / (vals * ww).sum()
    k = int(torch.searchsorted(cum, torch.tensor(q_eff, dtype=cum.dtype))) + 1
    k = min(k, e.numel())
    return float(ww[:k].sum())


def ratio_status(ratio: float, pass_at_most: float, kill_above: float) -> str:
    if ratio is None or math.isnan(ratio):
        return "not_run"
    if ratio > kill_above:
        return "kill"
    return "pass" if ratio <= pass_at_most else "fail"


# =================================================================================================
# Stage F
# =================================================================================================
def graph_node_positions(rec: dict) -> tuple[Tensor, Tensor]:
    """(positions in trace coordinates, kinds) for the non-logit nodes of a reference graph record, in
    circuit-tracer node order [features, errors, tokens] (= source_meta order). Record positions are model
    token coordinates = trace coordinates + prefix_offset (1 if circuit-tracer prepended a special token;
    cct.premise_be), so a prepended token maps to trace position -1."""
    sm = rec["source_meta"]
    off = int(rec.get("prefix_offset", 0) or 0)
    return sm["pos"].long() - off, sm["kind"].long()


def output_sentence_influence(node_influence: Tensor, node_pos: Tensor, spans: list) -> Tensor:
    """Sum of node influence over the nodes whose (trace) position lies in each sentence span
    [start, end). node_influence covers at least the non-logit nodes, in node order."""
    n = node_pos.numel()
    infl = node_influence[:n].double()
    out = torch.zeros(len(spans), dtype=torch.float64)
    for i, (s, e) in enumerate(spans):
        sel = (node_pos >= s) & (node_pos < e)
        out[i] = infl[sel].sum()
    return out


def dependency_matrix(outputs: list, spans: list) -> tuple[Tensor, Tensor]:
    """outputs: [{"sentence_index": j, "influence_by_sentence": Tensor [n_sent]}]. Returns
    (D, raw): raw[j, i] = summed influence over j's sampled outputs from sentence i; D = raw normalised
    per row over i < j (row sum 1). Rows without sampled outputs and entries i >= j are nan."""
    n = len(spans)
    raw = torch.full((n, n), float("nan"), dtype=torch.float64)
    for o in outputs:
        j = int(o["sentence_index"])
        if j < 0 or j >= n:
            continue
        row = torch.nan_to_num(raw[j, :j], nan=0.0) + o["influence_by_sentence"][:j].double()
        raw[j, :j] = row
    D = raw.clone()
    for j in range(n):
        r = raw[j, :j]
        if r.numel() and not bool(torch.isnan(r).all()):
            s = float(torch.nan_to_num(r, nan=0.0).sum())
            D[j, :j] = r / s if s > 0 else torch.zeros_like(r)
    return D, raw


def pairs_from(D: Tensor) -> list[tuple[int, int]]:
    """(j, i) pairs with i < j where row j has sampled outputs (non-nan)."""
    n = D.shape[0]
    return [(j, i) for j in range(n) for i in range(j) if not math.isnan(float(D[j, i]))]


# -- interventions ----------------------------------------------------------------------------------
def filler_replace(tokens: Tensor, span: tuple[int, int], filler_id: int) -> Tensor:
    """Replace tokens[start:end] with filler_id repeated (equal token length)."""
    s, e = span
    out = tokens.clone()
    out[s:e] = int(filler_id)
    return out


def delete_span(tokens: Tensor, span: tuple[int, int]) -> Tensor:
    s, e = span
    return torch.cat([tokens[:s], tokens[e:]])


def align_after_deletion(orig: Tensor, new: Tensor, positions: list[int], span: tuple[int, int]) -> list[int]:
    """Map positions of the original sequence to the sequence with span deleted, by content: a
    position after the span shifts left by the span length, and the token there must be the same.
    Positions inside the span raise; positions before it are unchanged."""
    s, e = span
    out = []
    for p in positions:
        if s <= p < e:
            raise ValueError(f"position {p} lies inside the deleted span {span}")
        q = p if p < s else p - (e - s)
        if int(new[q]) != int(orig[p]):
            raise ValueError(f"content mismatch after deletion at {p} -> {q}")
        out.append(q)
    return out


def sum_logprob(logits: Tensor, tokens: Tensor, positions: list[int]) -> float:
    """Sum over positions p of log softmax(logits[p - 1])[tokens[p]] (teacher forcing)."""
    lp = torch.log_softmax(logits.double(), -1)
    return float(sum(lp[p - 1, int(tokens[p])] for p in positions))


# -- Thought Anchors' attention-suppression effect measure ------------------------------------------
# Source: github.com/interp-reasoning/thought-anchors, whitebox-analyses/attention_analysis/
# attn_supp_funcs.py (get_suppression_KL_matrix, calculate_kl_divergence_sparse) and logits_funcs.py
# (compress_logits_top_p): logits of every position are compressed to the top-p nucleus (p = 0.9999 at
# their call site, at most 100 tokens) of softmax(logits / 0.6); KL(P_base || Q_suppressed) is taken
# over the union of both kept index sets (missing logits = -1e9), with softmax at temperature 0.6,
# clipped at 0; the sentence-to-sentence score is the mean over the target sentence's token positions
# of log(KL + 1e-9). Suppression: in every layer and head, the pre-softmax attention scores to the
# suppressed sentence's key positions are set to finfo(dtype).min for every query.
TA_P_NUCLEUS, TA_MAX_K, TA_TEMPERATURE, TA_LOG_EPS = 0.9999, 100, 0.6, 1e-9


def _nucleus_batch(logits: Tensor, p: float, max_k: int, temperature: float) -> tuple[Tensor, Tensor, Tensor]:
    """Per row: the top-p (cumulative softmax(logits / temperature) >= p) indices, capped at max_k, as
    compress_logits_top_p. Returns (idx [n, max_k], logits [n, max_k], k [n]); entries past k are unused."""
    lg = logits.float()
    probs = torch.softmax(lg / temperature, -1)
    kk = min(max_k, lg.shape[-1])
    tp, ti = probs.topk(kk, dim=-1)  # sorted descending; only the first max_k can ever be kept
    cum = torch.cumsum(tp, -1)
    k = (cum < p).sum(-1) + 1  # = searchsorted(cum, p, right=False) + 1
    k = k.clamp(1, kk)
    return ti.cpu(), lg.gather(-1, ti).cpu(), k.cpu()


def ta_sparse_kl(base_logits: Tensor, supp_logits: Tensor, p_nucleus: float = TA_P_NUCLEUS,
                 max_k: int = TA_MAX_K, temperature: float = TA_TEMPERATURE) -> Tensor:
    """Per position KL(P || Q) as calculate_kl_divergence_sparse. base_logits, supp_logits [n, V]."""
    bi, bl, bk = _nucleus_batch(base_logits, p_nucleus, max_k, temperature)
    si, sl, sk = _nucleus_batch(supp_logits, p_nucleus, max_k, temperature)
    n = bi.shape[0]
    out = torch.zeros(n, dtype=torch.float64)
    for t in range(n):
        b_idx, b_val = bi[t, : bk[t]], bl[t, : bk[t]]
        s_idx, s_val = si[t, : sk[t]], sl[t, : sk[t]]
        union = torch.unique(torch.cat([b_idx, s_idx]))
        bu = torch.full((union.numel(),), -1e9, dtype=torch.float32)
        su = torch.full((union.numel(),), -1e9, dtype=torch.float32)
        bu[torch.searchsorted(union, b_idx)] = b_val
        su[torch.searchsorted(union, s_idx)] = s_val
        lp = torch.log_softmax(bu / temperature, 0)
        lq = torch.log_softmax(su / temperature, 0)
        pr = lp.exp()
        terms = torch.where(pr == 0, torch.zeros_like(pr), pr * (lp - lq))
        out[t] = max(float(terms.sum()), 0.0)
    return out


def ta_sentence_effect(kl: Tensor, span: tuple[int, int]) -> float:
    """Mean over the span's positions of log(KL + 1e-9) (their take_log=True default)."""
    s, e = span
    e = min(e, kl.numel())
    if s >= e:
        return float("nan")
    return float(torch.log(kl[s:e] + TA_LOG_EPS).mean())


def suppress_scores(scores: Tensor, spans: list[tuple[int, int]]) -> Tensor:
    """Set attention scores [..., q, k] to finfo.min at key positions in spans (entries already -inf by
    the causal mask stay -inf)."""
    out = scores.clone()
    mv = torch.finfo(scores.dtype).min
    for s, e in spans:
        out[..., s:e] = torch.minimum(out[..., s:e], torch.full_like(out[..., s:e], mv))
    return out


# -- metrics --------------------------------------------------------------------------------------------
def auroc(scores: Tensor, labels: Tensor) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties); nan if one class is empty."""
    from .graph.metrics import _rank

    s = scores.double().flatten()
    y = labels.bool().flatten()
    keep = ~torch.isnan(s)
    s, y = s[keep], y[keep]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(s) + 1.0
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def top_quartile(actual: Tensor) -> Tensor:
    """Labels for 'top-quartile actual effect': actual >= its 75th percentile (linear interpolation)."""
    a = actual.double()
    q = torch.quantile(a[~torch.isnan(a)], 0.75)
    return a >= q


def spearman_nan(a: Tensor, b: Tensor) -> float:
    from .graph.metrics import spearman

    a, b = a.double().flatten(), b.double().flatten()
    keep = ~torch.isnan(a) & ~torch.isnan(b)
    if int(keep.sum()) < 3:
        return float("nan")
    return spearman(a[keep], b[keep])


def pr11_status(auroc_feat: float, auroc_s4: float, n_nameable: int | None, pr11: dict) -> dict:
    """PREREG_GATE0 stage_f.pr11. AUROC part: pass if auroc_feat >= auroc_s4 + auroc_minus_s4_at_least
    and auroc_feat >= auroc_at_least; weak if auroc_feat < auroc_s4 + auroc_minus_s4_below.
    Nameable part: pass if n_nameable >= nameable_at_least; weak if < nameable_below; pending if no
    labels. Overall: weak if either part is weak; pass if both pass; pending if the AUROC part passes
    and labels are missing; fail otherwise."""
    p, w = pr11["pass"], pr11["weak"]
    if any(x is None or math.isnan(x) for x in (auroc_feat, auroc_s4)):
        auc = "not_run"
    elif auroc_feat < auroc_s4 + w["auroc_minus_s4_below"]:
        auc = "weak"
    elif auroc_feat >= auroc_s4 + p["auroc_minus_s4_at_least"] and auroc_feat >= p["auroc_at_least"]:
        auc = "pass"
    else:
        auc = "fail"
    if n_nameable is None:
        nm = "pending_dawar"
    elif n_nameable < w["nameable_below"]:
        nm = "weak"
    elif n_nameable >= p["nameable_at_least"]:
        nm = "pass"
    else:
        nm = "fail"
    if auc == "not_run":
        st = "not_run"
    elif "weak" in (auc, nm):
        st = "weak"
    elif auc == "pass" and nm == "pass":
        st = "pass"
    elif auc == "pass" and nm == "pending_dawar":
        st = "pending"
    else:
        st = "fail"
    return {"status": st, "auroc_condition": auc, "nameable_condition": nm}


# -- tiny "real model" ----------------------------------------------------------------------------------
def tiny_forward(model, tokens: Tensor, suppress: list[tuple[int, int]] | None = None,
                 embed: Tensor | None = None, return_patterns: bool = False):
    """TinyModel's real (nonlinear) forward, as clean_forward, with optional attention suppression
    (scores to keys in `suppress` set to finfo.min in every layer and head) and an optional embedding
    override embed [n, d] (for gradient x input). Returns logits [n, V] (and patterns [L, H, n, n])."""
    from .operators.tiny import _rms

    cfg = model.cfg
    toks = tokens.long().flatten()
    r = (model.W_E[toks] if embed is None else embed)[None]
    n = toks.numel()
    causal = torch.ones(n, n, dtype=torch.bool).tril()
    pats = []
    for l in range(cfg.n_layers):
        s1 = _rms(r, cfg.eps)
        h = r / s1 * model.g1[l]
        q = torch.einsum("bnd,hde->bhne", h, model.W_Q[l])
        k = torch.einsum("bnd,hde->bhne", h, model.W_K[l])
        s = torch.einsum("bhne,bhme->bhnm", q, k) / math.sqrt(cfg.d_head)
        s = s.masked_fill(~causal, float("-inf"))
        if suppress:
            s = suppress_scores(s, suppress)
        pat = torch.softmax(s, -1)
        pats.append(pat[0])
        a, _ = model._attn(l, h, pat)
        r = r + a
        s2 = _rms(r, cfg.eps)
        r = r + model._mlp(l, r / s2 * model.g2[l])
    f = r / _rms(r, cfg.eps) * model.gf
    logits = (f @ model.W_U)[0]
    return (logits, torch.stack(pats)) if return_patterns else logits


def rollout_matrix(R_layers: Tensor) -> Tensor:
    """Product R_{L-1} ... R_0 of per-layer rollout matrices [L, n, n] (rows: query, cols: key)."""
    M = torch.eye(R_layers.shape[-1], dtype=R_layers.dtype, device=R_layers.device)
    for l in range(R_layers.shape[0]):
        M = R_layers[l] @ M
    return M


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, key: str, secs: float) -> None:
        self.t[key] = self.t.get(key, 0.0) + secs

    def time(self, key: str):
        timer = self

        class _C:
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.t0 = time.perf_counter()

            def __exit__(self, *a):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timer.add(key, time.perf_counter() - self.t0)

        return _C()
