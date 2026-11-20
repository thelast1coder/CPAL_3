"""Shared logic of Gate 0 Stages A and C (GATE0_PREMISES.md Section 3, PREREG_GATE0.md Section 2).

Stage A: PR1 (mixed passes are exactly linear and cost about one pass), PR2 (a prefix's rows are a
slice of the full-trace frozen pass), PR3 (closed-form A_0 equals the attention-detached rows).
Stage C: full exact matrices on short prompts, PR5 (compressibility relative to each source's causal
candidate set) and PR6 (D1 and D2 recovery reproduce the pruned graph; epsilon* calibration).

Everything here is backend-free (works on operators that follow cct.operators.base, or on stored
matrices) so it runs in the CPU tests with the tiny model. Thresholds come from
configs/prereg_gate0.yaml; nothing in this module hard-codes a prereg number.

Interpretations of points the prereg leaves open are listed in INTERPRETATIONS (copied into every
stage file so the report can show them).
"""
from __future__ import annotations

import math
import os
import statistics
import time

import torch
from torch import Tensor

from .graph.export import make_layout, to_square
from .graph.metrics import fidelity, meets_fidelity
from .graph.prune import prune_adjacency
from .killtest import exact_prune_result, finalise_record, s95, weighted_median
from .operators.base import KIND_ERROR, KIND_FEATURE, KIND_LOGIT, KIND_TOKEN, CausalMask, NodeMeta
from .operators.dense import DenseOperator
from .pipeline import merge, run_cct
from .sketch import CountSketch
from .utils.io import pack_matrix, unpack_matrix

# Hub (PR5 hub share): a remainder column whose absolute-value s95 exceeds this fraction of its causal
# candidate set (the PR5 kill level). Not a prereg threshold: a descriptive cut for the report.
HUB_FRAC_CAUSAL = 0.10

INTERPRETATIONS = {
    "PR1_kill": "kill if any relative error exceeds the tolerance of the dtype in use (one dtype is run; "
                "the 'fails in both dtypes' clause needs a bf16 rerun, which Dawar decides on); a time-ratio "
                "failure alone is 'fail'",
    "PR1_rel_err": "per mix: ||Y_mix - phi_mix A||_2 / ||phi_mix A||_2 over all source columns, phi A summed in "
                   "float64 from one-hot rows; the status uses the maximum",
    "PR1_timing": "same operator and batch shape: a batch of B dense random-sign seeds over all targets against "
                  "a batch of B one-hot seeds; median of timing_repeats after one warm-up; ratio = mixed / one-hot",
    "PR2": "feature targets only (logit rows depend on the last position); sources aligned by (kind, layer, "
           "position, feature id); a prefix target missing from the full trace, or any mass on sources that "
           "exist only in the full trace, counts as a mismatch",
    "PR3_tolerance": "1e-4 in fp32 and fp64; the PR1/PR2 bf16 tolerance (2e-2) if the model runs in bf16",
    "PR3_rows": "every row (logits and features) of the attention-detached pass against A_0 (Frobenius)",
    "PR3_s95": "reduction factor = energy-weighted median column s95 of A over that of A - A_0 (feature "
               "targets, squared version; the absolute version is reported too); the informative flag uses "
               "the squared version and the median over every (set, prompt)",
    "PR5_hub": f"hub = remainder column with absolute s95 > {HUB_FRAC_CAUSAL} of its causal set",
    "PR6_T": "T in 'm <= T/10' and 'm <= T/4' is the number of feature targets of the prompt (logit rows are "
             "computed exactly, not sensed)",
    "PR6_seeds": "fidelity at (prompt, m, decoder) needs >= 2 of 3 seeds (prereg fidelity.seeds_required)",
    "epsilon_run_error": "energy-weighted per-column error of a run = sum_s w_s e_s / sum_s w_s with w_s the "
                         "squared norm of the remainder column (feature targets), e_s its relative error; "
                         "energy-weighted median and 90% energy quantile are stored as alternatives",
    "k_eff_short": "median over the 3 Qwen short prompts of the per-prompt energy-weighted median remainder "
                   "absolute s95 (entries, feature targets)",
}


# -------------------------------------------------------------------------------------------------
# small numerics
# -------------------------------------------------------------------------------------------------
def rel_err(a: Tensor, b: Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def tol_for(dtype_name: str, fp32_tol: float, bf16_tol: float) -> float:
    """fp32 and fp64 (tiny) share the fp32 tolerance."""
    return bf16_tol if "bf" in str(dtype_name) or "16" in str(dtype_name) else fp32_tol


def all_rows(op, chunk: int | None = None, device: str = "cpu") -> Tensor:
    """Every row of op (one-hot seeds, in batches of the operator's batch size) as [T, N] float32
    (float64 for a float64 operator) on `device`."""
    chunk = chunk or op.batch_size
    out_dt = torch.float64 if op.dtype == torch.float64 else torch.float32
    out = torch.empty(op.T, op.N, dtype=out_dt, device=device)
    for i in range(0, op.T, chunk):
        idx = torch.arange(i, min(op.T, i + chunk))
        out[idx.to(device)] = op.exact_rows(idx).to(device, out_dt)
    return out


def rademacher(n: int, k: int, gen: torch.Generator) -> Tensor:
    return (torch.randint(0, 2, (n, k), generator=gen) * 2 - 1).double()


def chunked_left_mul(Phi: Tensor, A: Tensor, chunk: int = 4096) -> Tensor:
    """Phi [m, T] @ A [T, N] in float64 without materialising A in float64."""
    out = torch.zeros(Phi.shape[0], A.shape[1], dtype=torch.float64)
    for i in range(0, A.shape[0], chunk):
        out += Phi[:, i : i + chunk].double().cpu() @ A[i : i + chunk].double().cpu()
    return out


# -------------------------------------------------------------------------------------------------
# PR1: linearity of mixed passes and their cost
# -------------------------------------------------------------------------------------------------
def mix_check(op, n_mixes: int, gen: torch.Generator, targets: Tensor | None = None,
              A: Tensor | None = None) -> dict:
    """n_mixes random-sign mixes over `targets` (all targets if None) in ONE batched mixed pass, against
    Phi times the exact (one-hot) rows. A: the full exact matrix if already computed (all targets)."""
    T = op.T
    tgt = torch.arange(T) if targets is None else torch.as_tensor(targets).long()
    Phi_s = rademacher(n_mixes, tgt.numel(), gen)
    Phi = torch.zeros(n_mixes, T, dtype=torch.float64)
    Phi[:, tgt] = Phi_s
    Y = op.rows(Phi.to(op.dtype)).double().cpu()
    if A is not None:
        ref = chunked_left_mul(Phi_s, A if targets is None else A[tgt])
    else:
        ref = torch.zeros(n_mixes, op.N, dtype=torch.float64)
        for i in range(0, tgt.numel(), op.batch_size):
            idx = tgt[i : i + op.batch_size]
            ref += Phi_s[:, i : i + idx.numel()] @ op.exact_rows(idx).double().cpu()
    errs = ((Y - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-300)).tolist()
    return {"rel_err": errs, "rel_err_max": max(errs), "n_targets": int(tgt.numel()), "n_mixes": n_mixes}


def time_ratio(op, batch: int, repeats: int, gen: torch.Generator) -> dict:
    """One mixed pass (batch of dense random-sign seeds over all targets) against one single-target pass
    (batch of one-hot seeds), same batch shape. Median of `repeats` after one warm-up each."""
    from .utils.timing import median_time

    b = min(batch, op.batch_size)
    seeds_mixed = rademacher(b, op.T, gen).to(op.dtype)
    onehot_idx = torch.randperm(op.T, generator=gen)[:b]
    if onehot_idx.numel() < b:  # fewer targets than the batch: repeat targets (same batch shape)
        onehot_idx = onehot_idx.repeat(math.ceil(b / max(1, onehot_idx.numel())))[:b]
    seeds_one = torch.zeros(b, op.T, dtype=op.dtype)
    seeds_one[torch.arange(b), onehot_idx] = 1.0
    op.rows(seeds_mixed)
    op.rows(seeds_one, _exact=True)
    t_mix, all_mix = median_time(lambda: op.rows(seeds_mixed), repeats)
    t_one, all_one = median_time(lambda: op.rows(seeds_one, _exact=True), repeats)
    return {"batch": b, "mixed_seconds": t_mix, "onehot_seconds": t_one, "ratio": t_mix / max(t_one, 1e-12),
            "mixed_all": all_mix, "onehot_all": all_one, "T": int(op.T), "N": int(op.N)}


def pr1_status(gemma_errs: list[float], trace_errs: list[float], ratios: list[float], tol_gemma: float,
               tol_trace: float, ratio_max: float) -> tuple[str, str]:
    if not gemma_errs or not trace_errs or not ratios:
        return "not_run", "some PR1 measurements are missing"
    bad_err = [e for e in gemma_errs if not e <= tol_gemma] + [e for e in trace_errs if not e <= tol_trace]
    if bad_err:
        return "kill", f"{len(bad_err)} mixed-row relative errors above tolerance (max {max(bad_err):.3e})"
    bad_t = [r for r in ratios if not r <= ratio_max]
    if bad_t:
        return "fail", f"rows exact, but time ratio {max(bad_t):.3f} > {ratio_max}"
    return "pass", "mixed rows equal phi times exact rows within tolerance; time ratio within bound"


# -------------------------------------------------------------------------------------------------
# PR2: prefix rows are a slice of the full-trace frozen pass
# -------------------------------------------------------------------------------------------------
def node_keys(meta: NodeMeta) -> list[tuple]:
    """(kind, layer, position, feature id) per node; logits get their index as feature id."""
    k, l, p, f = (meta.kind.tolist(), meta.layer.tolist(), meta.pos.tolist(), meta.feature.tolist())
    out = []
    li = 0
    for i in range(len(k)):
        if k[i] == KIND_LOGIT:
            out.append((k[i], l[i], p[i], li))
            li += 1
        else:
            out.append((k[i], l[i], p[i], f[i]))
    return out


def align_sources(sm_prefix: NodeMeta, sm_full: NodeMeta) -> tuple[Tensor, Tensor]:
    """(full column of each prefix source, -1 if absent; bool mask of full columns absent from the
    prefix)."""
    full_index = {key: i for i, key in enumerate(node_keys(sm_full))}
    m = torch.tensor([full_index.get(key, -1) for key in node_keys(sm_prefix)], dtype=torch.long)
    only_full = torch.ones(len(sm_full), dtype=torch.bool)
    only_full[m[m >= 0]] = False
    return m, only_full


def pr2_prefix_part(op_prefix, n_targets: int, gen: torch.Generator) -> dict:
    """Exact rows of n_targets random FEATURE targets on the prefix operator, with node identities."""
    feat = torch.nonzero(op_prefix.target_meta.kind != KIND_LOGIT).flatten()
    pick = feat[torch.randperm(feat.numel(), generator=gen)[: min(n_targets, feat.numel())]]
    rows = op_prefix.exact_rows(pick).double().cpu()
    tk = node_keys(op_prefix.target_meta)
    return {"target_keys": [tk[i] for i in pick.tolist()], "rows": rows,
            "source_meta": op_prefix.source_meta.subset(torch.arange(op_prefix.N)), "n_pos": int(op_prefix.n_pos)}


def pr2_full_rows(op_full, parts: list[dict]) -> list[dict]:
    """For each prefix part, the full-trace rows of the same targets (None where a target is absent)."""
    t_index = {key: i for i, key in enumerate(node_keys(op_full.target_meta))}
    out = []
    for part in parts:
        idx = [t_index.get(k, -1) for k in part["target_keys"]]
        found = torch.tensor([i for i in idx if i >= 0], dtype=torch.long)
        rows = op_full.exact_rows(found).double().cpu() if found.numel() else torch.zeros(0, op_full.N)
        out.append({"idx": idx, "rows": rows})
    return out


def pr2_compare(part: dict, full: dict, sm_full: NodeMeta) -> dict:
    present = torch.tensor([i >= 0 for i in full["idx"]], dtype=torch.bool)
    m, only_full = align_sources(part["source_meta"], sm_full)
    R_p = part["rows"][present]
    R_f = full["rows"]
    missing_src = int((m < 0).sum())
    keep = m >= 0
    aligned = torch.zeros_like(R_p)
    aligned[:, keep] = R_f[:, m[keep]]
    out = {
        "n_targets": int(present.numel()), "n_targets_missing_in_full": int((~present).sum()),
        "n_prefix_sources_missing_in_full": missing_src,
        "n_sources_only_in_full": int(only_full.sum()),
        "rel_err": rel_err(aligned, R_p) if R_p.numel() else float("nan"),
        "only_full_rel_mass": float(R_f[:, only_full].norm() / R_f.norm().clamp_min(1e-300)) if R_f.numel() else 0.0,
        "max_abs_diff": float((aligned - R_p).abs().max()) if R_p.numel() else 0.0,
    }
    if missing_src:  # prefix mass on sources the full trace does not have
        out["prefix_mass_on_missing_sources"] = float(R_p[:, ~keep].norm() / R_p.norm().clamp_min(1e-300))
    return out


def pr2_status(comps: list[dict], tol: float) -> tuple[str, str]:
    if not comps:
        return "not_run", "PR2 not measured"
    bad = []
    for c in comps:
        if c["n_targets_missing_in_full"] or c["n_prefix_sources_missing_in_full"]:
            bad.append("node set differs between prefix and full trace")
        if not c["rel_err"] <= tol:
            bad.append(f"rel err {c['rel_err']:.3e} > {tol}")
        if not c["only_full_rel_mass"] <= tol:
            bad.append(f"mass {c['only_full_rel_mass']:.3e} on sources only in the full trace")
    if bad:
        return "kill", "; ".join(sorted(set(bad)))
    return "pass", "prefix rows equal the full-trace rows of the same nodes within tolerance"


# -------------------------------------------------------------------------------------------------
# column structure (PR3, PR5)
# -------------------------------------------------------------------------------------------------
def s95_abs(v: Tensor, frac: float = 0.95) -> Tensor:
    """Per column of v [T, C]: number of largest entries holding `frac` of the absolute (L1) mass."""
    e = v.double().abs()
    tot = e.sum(0)
    s = torch.sort(e, dim=0, descending=True).values
    cum = torch.cumsum(s, 0)
    cnt = (cum < (frac * tot)[None] * (1 - 1e-12)).sum(0) + 1
    return torch.where(tot > 0, cnt, torch.zeros_like(cnt))


def causal_set_sizes(tm: NodeMeta, sm: NodeMeta) -> Tensor:
    """Per source: number of FEATURE targets with read layer > the source's (lowest write) layer and
    position >= the source's position (PREREG_GATE0 Section 2)."""
    feat = tm.kind != KIND_LOGIT
    tl, tp = tm.layer[feat].long(), tm.pos[feat].long()
    sl, sp = sm.layer.long(), sm.pos.long()
    n_l = int(max(int(tl.max()) if tl.numel() else 0, int(sl.max()) + 1)) + 2
    n_p = int(max(int(tp.max()) if tp.numel() else 0, int(sp.max()))) + 2
    H = torch.zeros(n_l, n_p, dtype=torch.long)
    if tl.numel():
        H.index_put_((tl, tp), torch.ones_like(tl), accumulate=True)
    C = H.flip(0).cumsum(0).flip(0).flip(1).cumsum(1).flip(1)  # C[l, p] = #targets with l' >= l, p' >= p
    return C[(sl + 1).clamp(0, n_l - 1), sp.clamp(0, n_p - 1)]


def _feature_rows(tm: NodeMeta) -> Tensor:
    return tm.kind != KIND_LOGIT


def column_stats(M: Tensor, tm: NodeMeta, sm: NodeMeta, causal: Tensor | None = None) -> dict:
    """Energy-weighted median s95 (squared and absolute), in entries and as a fraction of the causal
    set, over feature targets. Weights = squared column norm of M."""
    Mf = M[_feature_rows(tm)].double()
    causal = causal_set_sizes(tm, sm) if causal is None else causal
    w = Mf.pow(2).sum(0)
    ok = (w > 0) & (causal > 0)
    sq, ab = s95(Mf), s95_abs(Mf)
    out = {"n_cols_nonzero": int(ok.sum()), "energy": float(w.sum())}
    for name, s in (("sq", sq), ("abs", ab)):
        frac = s.double() / causal.clamp_min(1).double()
        out[f"s95_{name}_ewm"] = weighted_median(s[ok].double(), w[ok])
        out[f"s95_{name}_ewm_frac_causal"] = weighted_median(frac[ok], w[ok])
    out["_per_col"] = {"s95_sq": sq, "s95_abs": ab, "w": w, "causal": causal}
    return out


def hub_share(per_col: dict, sm: NodeMeta, hub_frac: float = HUB_FRAC_CAUSAL) -> dict:
    """Hubs (remainder columns with absolute s95 > hub_frac of the causal set) by node kind. First-token
    nodes (position 0, any kind) are reported as their own group as well."""
    s, w, c = per_col["s95_abs"], per_col["w"], per_col["causal"]
    nz = (w > 0) & (c > 0)
    hub = nz & (s.double() / c.clamp_min(1).double() > hub_frac)
    tot = float(w[nz].sum()) or 1.0
    groups = {"feature": sm.kind == KIND_FEATURE, "error": sm.kind == KIND_ERROR,
              "token": sm.kind == KIND_TOKEN, "first_token": sm.pos == 0}
    out = {}
    for g, sel in groups.items():
        n = int((nz & sel).sum())
        out[g] = {"n_cols": n, "n_hubs": int((hub & sel).sum()),
                  "hub_count_share": float((hub & sel).sum()) / n if n else float("nan"),
                  "hub_energy_share_of_total": float(w[hub & sel].sum()) / tot,
                  "energy_share_of_total": float(w[nz & sel].sum()) / tot}
    out["all"] = {"n_cols": int(nz.sum()), "n_hubs": int(hub.sum()),
                  "hub_energy_share_of_total": float(w[hub].sum()) / tot}
    return out


def structure_AR(A: Tensor, A0: Tensor, tm: NodeMeta, sm: NodeMeta) -> dict:
    """PR3 and PR5 numbers for A and the remainder A - A_0."""
    A, A0 = A.double().cpu(), A0.double().cpu()
    R = A - A0
    causal = causal_set_sizes(tm, sm)
    sA = column_stats(A, tm, sm, causal)
    sR = column_stats(R, tm, sm, causal)
    tot = float(A.pow(2).sum())
    out = {"T": int(A.shape[0]), "T_features": int(_feature_rows(tm).sum()), "N": int(A.shape[1]),
           "a0_energy_share": 1.0 - float(R.pow(2).sum()) / tot if tot > 0 else float("nan"),
           "causal_set_median": float(causal[causal > 0].double().median()) if bool((causal > 0).any()) else 0.0,
           "A": {k: v for k, v in sA.items() if k != "_per_col"},
           "remainder": {k: v for k, v in sR.items() if k != "_per_col"},
           "hub_share_remainder": hub_share(sR["_per_col"], sm)}
    for name in ("sq", "abs"):
        a, r = sA[f"s95_{name}_ewm"], sR[f"s95_{name}_ewm"]
        out[f"s95_{name}_reduction"] = a / r if r and r > 0 else float("nan")
    return out


def pr3_status(errs: list[float], tol: float, shares: list[float], reductions: list[float],
               rule: dict) -> tuple[str, str, dict]:
    if not errs:
        return "not_run", "PR3 not measured", {}
    share = statistics.median(shares) if shares else float("nan")
    red = statistics.median(reductions) if reductions else float("nan")
    drop = bool(share < rule["a0_energy_share_below"] and red < rule["s95_reduction_factor_below"])
    info = {"a0_energy_share_median": share, "s95_reduction_median": red, "drop_local_part_from_thesis": drop}
    if all(e <= tol for e in errs):
        return "pass", "closed-form A_0 equals the attention-detached rows within tolerance", info
    return "fail", (f"closed-form A_0 differs from the detached rows (max rel err {max(errs):.3e} > {tol}); "
                    "not a kill: fix the formula or drop A_0 subtraction and record it"), info


def pr5_status(fracs: dict[str, dict[str, float]], qwen_set: str, pre5: dict) -> tuple[str, str]:
    """fracs: {set: {prompt: remainder absolute s95 energy-weighted-median fraction of the causal set}}."""
    if qwen_set not in fracs or not fracs[qwen_set]:
        return "not_run", "Qwen short prompts not measured"
    p, k = pre5["pass"], pre5["kill"]
    n_ok = sum(v <= p["remainder_abs_s95_frac_causal_at_most"] for v in fracs[qwen_set].values())
    every = [v for s in fracs.values() for v in s.values()]
    if len(fracs) >= 3 and every and all(v > k["remainder_abs_s95_frac_causal_above"] for v in every):
        return "kill", "remainder s95 above the kill level on every prompt of all three sets"
    if n_ok >= p["prompts_at_least"]:
        return "pass", f"{n_ok} of {len(fracs[qwen_set])} Qwen prompts within the pass level"
    return "fail", f"only {n_ok} of {len(fracs[qwen_set])} Qwen prompts within the pass level"


# -------------------------------------------------------------------------------------------------
# full exact record (Stage A PR3 and Stage C share it)
# -------------------------------------------------------------------------------------------------
def build_record(A: Tensor, A0: Tensor, tm: NodeMeta, sm: NodeMeta, prune_cfg: dict, extra: dict) -> dict:
    """Packed full exact matrix (fp16 with per-row scale and checksum), A_0, metadata and the exact
    prune result (from the full-precision A)."""
    rec = finalise_record(A.double(), tm, sm, "F", extra, prune_cfg)
    rec["A0"] = pack_matrix(A0)
    return rec


def record_arrays(rec: dict) -> tuple[Tensor, Tensor, NodeMeta, NodeMeta]:
    return (unpack_matrix(rec["A"]).double(), unpack_matrix(rec["A0"]).double(),
            NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"]))


# -------------------------------------------------------------------------------------------------
# PR6: recovery simulation (D1, D2) and fidelity
# -------------------------------------------------------------------------------------------------
def remainder_col_errors(A_hat: Tensor, A: Tensor, A0: Tensor, tm: NodeMeta) -> dict:
    """Per-column relative error on the remainder (feature targets): energy-weighted mean (the epsilon*
    quantity), median and 90% energy quantile."""
    f = _feature_rows(tm)
    R = (A[f] - A0[f]).double()
    E = (A_hat[f].double() - A[f].double())
    w = R.pow(2).sum(0)
    keep = w > 0
    e = E.norm(dim=0)[keep] / w[keep].sqrt()
    w = w[keep]
    if e.numel() == 0:
        return {"ew_mean": float("nan"), "ew_median": float("nan"), "ew_q90": float("nan"), "n_cols": 0}
    o = torch.argsort(e)
    c = torch.cumsum(w[o], 0) / w.sum()
    q90 = float(e[o][int(torch.searchsorted(c, torch.tensor(0.9, dtype=c.dtype)).clamp_max(e.numel() - 1))])
    return {"ew_mean": float((w * e).sum() / w.sum()), "ew_median": weighted_median(e, w), "ew_q90": q90,
            "n_cols": int(e.numel())}


def d2_estimate(A: Tensor, A0: Tensor, tm: NodeMeta, sm: NodeMeta, m: int, seed: int, R: int = 5,
                device: str = "cpu") -> tuple[Tensor, float]:
    """Count-sketch D2 on the remainder of the feature-target rows: Y = Phi (A - A_0) (by PR1 exactly what
    mixed passes minus the known Phi A_0 return), decoded over the causal set keeping B // 4 entries
    per column, A_0 added back; logit rows exact. Returns (A_hat [T, N] float64 CPU, decode seconds)."""
    f = _feature_rows(tm)
    fi = torch.nonzero(f).flatten()
    Rm = (A[fi] - A0[fi]).to(torch.float32)
    cs = CountSketch(fi.numel(), m, R=R, seed=seed, device=device)
    Y = cs.apply(Rm.to(device))
    cand = CausalMask(tm.layer[fi], tm.pos[fi], sm.layer, sm.pos).dense()
    R_hat, secs = cs.decode(Y, cand=cand)
    A_hat = A.double().clone()
    A_hat[fi] = R_hat.double().cpu() + A0[fi].double()
    return A_hat, secs


class FidelityScorer:
    """Prunes an estimate and scores it against the exact pruned graph of one record."""

    def __init__(self, rec: dict, A: Tensor, prune_cfg: dict, fid: dict):
        self.tm, self.sm = NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"])
        self.lay = make_layout(self.tm, self.sm)
        self.sq = to_square(A.double(), self.lay)
        self.ex = exact_prune_result(rec)
        self.pc, self.fid = prune_cfg, fid

    def __call__(self, A_hat: Tensor) -> dict:
        lay = self.lay
        adj = to_square(A_hat.double().cpu(), lay)
        pr = prune_adjacency(adj, lay.n_features, lay.n_tokens, lay.logit_probs, self.pc["node_threshold"],
                             self.pc["edge_threshold"])
        f = fidelity(self.sq, self.ex, adj, pr, lay.n_features, lay.n_errors, lay.n_tokens, lay.logit_probs)
        f["meets"] = meets_fidelity(f, self.fid["influence_weighted_node_recall_at_least"],
                                    self.fid["node_jaccard_at_least"])
        return f


def recovery_runs(rec: dict, ms: list[int], seeds: list[int], d1_cfg: dict, d2_R: int, prune_cfg: dict,
                  fid: dict, device: str = "cpu", decoders: tuple = ("D1", "D2")) -> list[dict]:
    """Every (decoder, m, seed) run on one record: fidelity, remainder per-column errors, recovery time."""
    A, A0, tm, sm = record_arrays(rec)
    score = FidelityScorer(rec, A, prune_cfg, fid)
    runs = []
    op = None
    if "D1" in decoders:
        op = DenseOperator(A, tm, sm, device=device, dtype=torch.float64 if device == "cpu" else torch.float32)
    for m in ms:
        for sd in seeds:
            for dec in decoders:
                t0 = time.perf_counter()
                if dec == "D1":
                    cfg = merge(d1_cfg, {"m": int(m), "recovery": {"device": device}})
                    res = run_cct(op, cfg, seed=sd, layout=score.lay, local_part=A0)
                    A_hat = res.A_hat
                    rsec = res.ledger.recovery_seconds
                    extra = {"n_hubs": res.diagnostics.get("n_hubs"),
                             "n_cols_recovered": res.diagnostics.get("n_cols_recovered"),
                             "mixed_rows": res.ledger.mixed_rows, "forward_passes": res.ledger.forward_passes}
                else:
                    A_hat, rsec = d2_estimate(A, A0, tm, sm, int(m), sd, d2_R, device)
                    extra = {"R": d2_R, "B": int(m) // d2_R, "keep": max(1, (int(m) // d2_R) // 4)}
                f = score(A_hat)
                err = remainder_col_errors(A_hat, A, A0, tm)
                runs.append({"decoder": dec, "m": int(m), "seed": int(sd), "meets": bool(f["meets"]),
                             "iw_node_recall": f["iw_node_recall"], "node_jaccard": f["node_jaccard"],
                             "n_kept_exact": f["n_kept_exact"], "n_kept_approx": f["n_kept_approx"],
                             "col_err": err, "recovery_seconds": float(rsec),
                             "run_seconds": time.perf_counter() - t0, **extra})
    return runs


def meets_table(runs: list[dict], seeds_required: int = 2) -> dict:
    """{decoder: {m: {"n_meet", "n_runs", "meets"}}} for one prompt (seed rule)."""
    out: dict = {}
    for r in runs:
        d = out.setdefault(r["decoder"], {}).setdefault(str(r["m"]), {"n_meet": 0, "n_runs": 0})
        d["n_meet"] += int(r["meets"])
        d["n_runs"] += 1
    for dec in out.values():
        for d in dec.values():
            d["meets"] = d["n_meet"] >= min(seeds_required, d["n_runs"])
    return out


def pr6_status(tables: dict[str, dict], T_by_prompt: dict[str, int], pre6: dict) -> tuple[str, str, dict]:
    """tables: {prompt: meets_table(...)} for the Qwen prompts; T_by_prompt: feature targets per prompt."""
    if not tables:
        return "not_run", "Qwen short prompts not measured", {}
    p, k = pre6["pass"], pre6["kill"]
    ms = sorted({int(m) for t in tables.values() for dec in t.values() for m in dec})
    per_m = {}
    for m in ms:
        ok = [pr for pr, t in tables.items()
              if m <= T_by_prompt[pr] / p["m_at_most_T_over"] and t.get(p["decoder"], {}).get(str(m), {}).get("meets")]
        per_m[str(m)] = ok
    passing = [m for m in ms if len(per_m[str(m)]) >= p["prompts_at_least"]]
    any_within_quarter = any(t[dec][str(m)]["meets"] for pr, t in tables.items() for dec in t
                             for m in ms if str(m) in t[dec] and m <= T_by_prompt[pr] / k["m_at_most_T_over"])
    info = {"d2_prompts_meeting_within_T_over_10_by_m": per_m, "passing_m": passing,
            "any_meet_within_T_over_4": any_within_quarter,
            "T_features": T_by_prompt}
    if passing:
        return "pass", f"D2 meets fidelity with m <= T/10 on >= {p['prompts_at_least']} Qwen prompts at m = {passing[0]}", info
    if not any_within_quarter:
        return "kill", "no m <= T/4 meets fidelity on any Qwen prompt with either decoder", info
    return "fail", "fidelity reached only above T/10, or with D1 only, or on fewer than 2 Qwen prompts", info


def epsilon_star(runs: list[dict], seeds_required: int = 2, prompts_required: int = 2,
                 err_key: str = "ew_mean") -> dict:
    """PREREG_GATE0 epsilon*: per decoder, the smallest m at which fidelity holds (seed rule) on at least
    `prompts_required` prompts; there, the largest run error among runs that meet fidelity. The minimum
    over the decoders that qualify; None if none does. runs carry "prompt", "decoder", "m", "seed",
    "meets", "col_err"."""
    by_dec: dict = {}
    for dec in sorted({r["decoder"] for r in runs}):
        rs = [r for r in runs if r["decoder"] == dec]
        ms = sorted({r["m"] for r in rs})
        m_min = None
        for m in ms:
            n_prompts = 0
            for pr in sorted({r["prompt"] for r in rs}):
                sel = [r for r in rs if r["m"] == m and r["prompt"] == pr]
                if sel and sum(r["meets"] for r in sel) >= min(seeds_required, len(sel)):
                    n_prompts += 1
            if n_prompts >= prompts_required:
                m_min = m
                break
        if m_min is None:
            by_dec[dec] = {"m_min": None, "epsilon": None}
            continue
        errs = [r["col_err"][err_key] for r in rs if r["m"] == m_min and r["meets"]
                and not math.isnan(r["col_err"][err_key])]
        by_dec[dec] = {"m_min": m_min, "epsilon": max(errs) if errs else None, "n_runs_used": len(errs)}
    q = {d: v for d, v in by_dec.items() if v["epsilon"] is not None}
    if not q:
        return {"epsilon_star": None, "decoder": None, "by_decoder": by_dec, "error_key": err_key}
    best = min(q, key=lambda d: q[d]["epsilon"])
    return {"epsilon_star": q[best]["epsilon"], "decoder": best, "by_decoder": by_dec, "error_key": err_key}


# -------------------------------------------------------------------------------------------------
# sets, prompts, traces and the model backend (tiny or circuit-tracer)
# -------------------------------------------------------------------------------------------------
ROLES = ("gemma_plt", "gemma_clt", "qwen")
TINY_SETS = {"gemma_plt": "tiny_plt", "gemma_clt": "tiny_clt", "qwen": "tiny_qwen_plt"}


def role_set(cfg: dict, role: str, tiny: bool) -> str:
    return TINY_SETS[role] if tiny else cfg["sets"][role]


def role_prompts(cfg: dict, role: str, tiny: bool, kt: dict | None = None) -> list[dict]:
    """[{"id", "prompt"}] (real) or [{"id", "tokens"}] (tiny: test prompts 0-2 for the Gemma roles,
    3-5 for the Qwen stand-in)."""
    from . import scriptutil as su

    if tiny:
        tp = su.tiny_prompts(kt)["test"]
        sel = tp[3:6] if role == "qwen" else tp[0:3]
        return [{"id": p["id"], "tokens": p["tokens"]} for p in sel]
    from .killtest import load_prompts

    spec = cfg["prompts"]["qwen" if role == "qwen" else "gemma"]
    allp = {p["id"]: p for p in load_prompts(os.path.join(su.ROOT, spec["file"]))}
    return [{"id": i, "prompt": allp[i]["prompt"]} for i in spec["ids"]]


def load_trace(cfg: dict, tiny: bool, index: int = 0) -> dict:
    """The first committed trace: {"id", "token_ids", "n_prompt_tokens", "source"}. Tiny: the tiny traces
    file if present, else 24 random tokens (8 of them 'prompt')."""
    from . import scriptutil as su
    from .utils.io import read_json

    path = (os.path.join(su.ROOT, "results", "tiny", "gate0", "traces.json") if tiny
            else os.path.join(su.ROOT, cfg["traces"]["traces_file"]))
    if os.path.exists(path):
        t = read_json(path)["traces"][index]
        ids = t.get("token_ids", t.get("tokens"))
        if ids is None:
            raise KeyError(f"trace {t.get('id')} has neither 'token_ids' nor 'tokens'")
        return {"id": t["id"], "token_ids": [int(x) for x in ids], "n_prompt_tokens": int(t["n_prompt_tokens"]),
                "source": os.path.relpath(path, su.ROOT)}
    if not tiny:
        raise FileNotFoundError(f"{path} is missing: generate and commit the traces first (scripts/gen_traces.py)")
    g = torch.Generator().manual_seed(1234)
    return {"id": "tiny_random_trace", "token_ids": torch.randint(0, 64, (24,), generator=g).tolist(),
            "n_prompt_tokens": 8, "source": "random tokens (no tiny traces file)"}


def is_oom(e: BaseException) -> bool:
    return isinstance(e, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in str(e).lower()


class Backend:
    """One model (tiny or circuit-tracer ReplacementModel) and operator construction for a set.
    Real models load with the requested dtype (fp32 by default); load_model raises on OOM so that the
    caller can restart the whole set in bf16 (one dtype per set, never mixed)."""

    def __init__(self, set_name: str, tiny: bool, exact_cfg: dict, kt: dict | None = None, log=None,
                 dtype_name: str | None = None):
        self.set_name, self.tiny, self.ex, self.kt, self.log = set_name, tiny, exact_cfg, kt, log
        self.model = None
        self.notes: list[str] = []
        self.load_seconds = 0.0
        if tiny:
            self.dtype_name = "float64"
            self.mc = None
        else:
            from . import scriptutil as su

            self.mc = su.model_config(set_name)
            self.dtype_name = dtype_name or self.mc.get("dtype") or "float32"

    def load(self):
        if self.model is not None:
            return self.model
        t0 = time.perf_counter()
        from . import scriptutil as su

        if self.tiny:
            self.model = su.tiny_model(self.kt, self.set_name)
        else:
            from .operators.circuit_tracer_backend import load_replacement_model
            from .utils.gpu import cap_memory, nvidia_smi

            if self.log:
                self.log.info(nvidia_smi())
            cap_memory(self.mc["memory_cap_gb"])
            dt = su.torch_dtype(self.dtype_name)
            try:
                self.model = load_replacement_model(self.mc["model_name"], self.mc["transcoder_set"], dtype=dt)
            except Exception as e:  # noqa: BLE001
                if is_oom(e) or not self.mc.get("transcoder_set_alt"):
                    raise
                msg = (f"loading {self.mc['transcoder_set']} failed ({e!r}); using "
                       f"{self.mc['transcoder_set_alt']} (configs/models/{self.set_name}.yaml fallback)")
                if self.log:
                    self.log.warning(msg)
                self.notes.append(msg)
                self.model = load_replacement_model(self.mc["model_name"], self.mc["transcoder_set_alt"], dtype=dt)
        self.load_seconds += time.perf_counter() - t0
        return self.model

    def free(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def op(self, inp, batch_size: int, detach: bool = False):
        """inp: prompt string, or a list / tensor of token ids."""
        model = self.load()
        if self.tiny:
            from .operators.tiny import TinyOperator

            return TinyOperator(model, torch.as_tensor(inp).long(), batch_size=batch_size, detach_attn=detach,
                                max_n_logits=self.ex["max_n_logits"], desired_logit_prob=self.ex["desired_logit_prob"])
        from .operators.circuit_tracer_backend import CircuitTracerOperator

        if not isinstance(inp, str):
            inp = torch.as_tensor(inp).long()
        return CircuitTracerOperator(model, inp, max_n_logits=self.ex["max_n_logits"],
                                     desired_logit_prob=self.ex["desired_logit_prob"], batch_size=batch_size,
                                     detach_attention=detach)

    def local_part(self, op):
        from .local_part import from_circuit_tracer, from_tiny

        return from_tiny(op) if self.tiny else from_circuit_tracer(op)


def prompt_input(p: dict):
    return p["tokens"] if "tokens" in p else p["prompt"]


def full_matrix_and_a0(be: Backend, inp, batch_size: int):
    """Every row of the exact matrix (all active features as targets and sources) and the closed-form
    A_0 in the same layout, both on the CPU. Returns (A, A0, target_meta, source_meta, info, op); the
    caller deletes op when done with it."""
    t0 = time.perf_counter()
    op = be.op(inp, batch_size)
    t_setup = time.perf_counter() - t0
    A = all_rows(op)
    t_rows = time.perf_counter() - t0 - t_setup
    lp = be.local_part(op)
    A0 = lp.dense(device="cpu").to(A.dtype)
    del lp
    tm = op.target_meta.subset(torch.arange(op.T))
    sm = op.source_meta.subset(torch.arange(op.N))
    info = {"n_pos": int(op.n_pos), "T": int(op.T), "N": int(op.N), "n_logits": int(op.n_logits),
            "setup_seconds": t_setup, "rows_seconds": t_rows, "a0_seconds": time.perf_counter() - t0 - t_setup - t_rows,
            "dtype": be.dtype_name}
    if hasattr(op, "read_map_kind"):
        info["read_map_kind"] = sorted(set(op.read_map_kind))
    return A, A0, tm, sm, info, op


# -------------------------------------------------------------------------------------------------
# stage file contract
# -------------------------------------------------------------------------------------------------
def premise(status: str, values: dict, thresholds: dict, notes: str) -> dict:
    return {"status": status, "values": values, "thresholds": thresholds, "notes": notes}


def stage_doc(run: dict, stage: str, premises: dict, stop: bool, stop_reason: str | None, gpu_seconds: float,
              dtype_by_set: dict, detail: dict, **extra) -> dict:
    return {"run": run, "stage": stage, "premises": premises, "stop": bool(stop), "stop_reason": stop_reason,
            "gpu_seconds": float(gpu_seconds), "dtype_by_set": dtype_by_set, "interpretations": INTERPRETATIONS,
            **extra, "detail": detail}


def jsonable(x):
    """Tensors and non-finite floats to JSON-friendly values (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, Tensor):
        return jsonable(x.tolist())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def item_path(root: str, *parts: str) -> str:
    p = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
