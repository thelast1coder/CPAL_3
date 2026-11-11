"""Kill-test logic shared by the scripts (PHASE1_KILL_TEST.md, PREREG_KILL_TEST.md).

Records: one exact graph per (transcoder set, prompt, regime), stored with cct.utils.io.save_pt:
  A (packed T x N), target_meta, source_meta, logit_probs, T_D, regime, exact prune result
  (node_mask, edge_mask, node_influence, computed in float64 from the unpacked matrix), timing.
Targets are ordered [logits, features in visiting order], so the visiting order in target rows is
simply 0..T-1 and truncation at c rows keeps rows 0..c-1.
"""
from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import dataclass

import torch
from torch import Tensor

from .baselines.lowrank import lowrank_sketch
from .baselines.random_rows import random_rows_adjacency
from .baselines.truncation import truncation_adjacency
from .graph.export import make_layout, to_square
from .graph.influence import partial_influences
from .graph.metrics import fidelity, meets_fidelity
from .graph.prune import prune_adjacency
from .operators.base import KIND_FEATURE, KIND_LOGIT, NodeMeta
from .operators.dense import DenseOperator
from .pipeline import merge, run_cct
from .utils.io import pack_matrix, unpack_matrix


# -------------------------------------------------------------------------------------------------
# prompts
# -------------------------------------------------------------------------------------------------
def load_prompts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def first_of_each_category(prompts: list[dict]) -> list[dict]:
    seen, out = set(), []
    for p in prompts:
        if p["category"] not in seen:
            seen.add(p["category"])
            out.append(p)
    return out


# -------------------------------------------------------------------------------------------------
# exact graphs
# -------------------------------------------------------------------------------------------------
def visit_order_from_full(A_full: Tensor, n_logits: int, n_active: int, logit_probs: Tensor, cap: int,
                          batch_size: int, update_interval: int = 4) -> Tensor:
    """Emulate circuit-tracer's influence-ordered visiting (attribute.py Phase 4) given every row.
    A_full: [n_logits + n_active, N] with targets [logits, active features in index order] and sources
    [active features, errors, tokens]. Returns active-feature indices in visiting order (length cap)."""
    N = A_full.shape[1]
    n_nodes = N + n_logits
    rows = torch.zeros(n_logits + cap, n_nodes, dtype=torch.float64)
    r2n = torch.zeros(n_logits + cap, dtype=torch.long)
    rows[:n_logits, :N] = A_full[:n_logits]
    r2n[:n_logits] = torch.arange(N, N + n_logits)
    st, visited, order = n_logits, torch.zeros(n_active, dtype=torch.bool), []
    while len(order) < cap:
        infl = partial_influences(rows[:st], logit_probs, r2n[:st])
        rank = torch.argsort(infl[:n_active], descending=True)
        pending = rank[~visited[rank]][: min(update_interval * batch_size, cap - len(order))]
        for j in range(0, pending.numel(), batch_size):
            idx = pending[j : j + batch_size]
            end = st + idx.numel()
            rows[st:end, :N] = A_full[n_logits + idx]
            r2n[st:end] = idx
            visited[idx] = True
            order.extend(idx.tolist())
            st = end
    return torch.tensor(order, dtype=torch.long)


def regime_record(A_rows: Tensor, n_logits: int, visit: Tensor, active_meta: NodeMeta, nonfeat_meta: NodeMeta,
                  n_active: int, logit_probs: Tensor, regime: str, n_layers: int, extra: dict | None = None,
                  prune_cfg: dict | None = None, fp32_storage: bool = False) -> dict:
    """Build a stored record from rows over [all active features, nonfeature sources].
    A_rows: [n_logits + len(visit), n_active + n_nonfeat], rows ordered [logits, visit order]."""
    selected = torch.sort(visit).values
    cols = torch.cat([selected, torch.arange(n_active, A_rows.shape[1])])
    A = A_rows[:, cols].double()
    k = n_logits
    tm = NodeMeta(
        kind=torch.cat([torch.full((k,), KIND_LOGIT), torch.full((visit.numel(),), KIND_FEATURE)]),
        layer=torch.cat([torch.full((k,), n_layers), active_meta.layer[visit]]),
        pos=torch.cat([torch.full((k,), int(nonfeat_meta.pos.max())), active_meta.pos[visit]]),
        feature=torch.cat([torch.full((k,), -1), active_meta.feature[visit]]),
        extra=torch.cat([logit_probs.float(), torch.zeros(visit.numel())]),
    )
    sm = NodeMeta(
        kind=torch.cat([active_meta.kind[selected], nonfeat_meta.kind]),
        layer=torch.cat([active_meta.layer[selected], nonfeat_meta.layer]),
        pos=torch.cat([active_meta.pos[selected], nonfeat_meta.pos]),
        feature=torch.cat([active_meta.feature[selected], nonfeat_meta.feature]),
        extra=torch.cat([active_meta.extra[selected], nonfeat_meta.extra]),
    )
    return finalise_record(A, tm, sm, regime, extra, prune_cfg, fp32_storage)


def finalise_record(A: Tensor, tm: NodeMeta, sm: NodeMeta, regime: str, extra: dict | None,
                    prune_cfg: dict | None, fp32_storage: bool = False) -> dict:
    prune_cfg = prune_cfg or {"node_threshold": 0.8, "edge_threshold": 0.98}
    lay = make_layout(tm, sm)
    sq = to_square(A.double(), lay)
    pr = prune_adjacency(sq, lay.n_features, lay.n_tokens, lay.logit_probs,
                         prune_cfg["node_threshold"], prune_cfg["edge_threshold"])
    rec = {
        "A": pack_matrix(A, fp32=fp32_storage), "target_meta": tm.to_dict(), "source_meta": sm.to_dict(),
        "T_D": int(A.shape[0]), "N": int(A.shape[1]), "regime": regime,
        "exact_prune": {"node_mask": pr.node_mask, "edge_mask": pr.edge_mask.to_sparse(),
                        "node_influence": pr.node_influence},
        "prune_cfg": prune_cfg,
    }
    rec.update(extra or {})
    return rec


def tiny_meta(op) -> tuple[NodeMeta, NodeMeta]:
    """Metadata for all active features and the nonfeature sources of a TinyOperator built with all
    features as sources."""
    n_act = op.active.shape[0]
    sm = op.source_meta
    return sm.subset(torch.arange(n_act)), sm.subset(torch.arange(n_act, len(sm)))


def exact_graph_tiny(model, tokens: Tensor, cap: int, batch_size: int, regime: str = "D",
                     prune_cfg: dict | None = None) -> dict:
    import time

    from .operators.tiny import TinyOperator

    t0 = time.perf_counter()
    op = TinyOperator(model, tokens, batch_size=batch_size)
    A_full = op.exact_matrix().double()
    n_act = op.active.shape[0]
    cap = n_act if regime == "F" else min(cap, n_act)
    visit = visit_order_from_full(A_full, op.n_logits, n_act, op.logit_probs, cap, batch_size)
    rows = torch.cat([A_full[: op.n_logits], A_full[op.n_logits + visit]])
    am, nm = tiny_meta(op)
    secs = time.perf_counter() - t0
    return regime_record(rows, op.n_logits, visit, am, nm, n_act, op.logit_probs, regime, model.cfg.n_layers,
                         {"tokens": tokens, "n_active": n_act, "visit_features": visit, "exact_seconds": secs,
                          "rows_computed": int(rows.shape[0])}, prune_cfg, fp32_storage=True)


def exact_graph_real(op, cap: int | None, regime: str, update_interval: int = 4,
                     prune_cfg: dict | None = None) -> dict:
    """op: CircuitTracerOperator with all active features as sources."""
    out = op.exact_attribution(cap if regime == "D" else None, update_interval=update_interval,
                               influence_ordered=True)
    n_act = op.n_active
    sm = op.source_meta
    am, nm = sm.subset(torch.arange(n_act)), sm.subset(torch.arange(n_act, len(sm)))
    rows = out["rows"]
    if rows.shape[1] != len(sm):
        rows = rows[:, op.source_cols]
    return regime_record(rows, op.n_logits, out["visit_order_features"], am, nm, n_act, op.logit_probs, regime,
                         op.n_layers,
                         {"n_active": n_act, "visit_features": out["visit_order_features"],
                          "exact_seconds": out["seconds"], "setup_seconds": op.setup_seconds,
                          "rows_computed": int(rows.shape[0])}, prune_cfg)


def record_matrix(rec: dict) -> Tensor:
    return unpack_matrix(rec["A"]).double()


def record_operator(rec: dict, device="cpu", batch_size: int = 256) -> DenseOperator:
    return DenseOperator(record_matrix(rec), NodeMeta.from_dict(rec["target_meta"]),
                         NodeMeta.from_dict(rec["source_meta"]), batch_size=batch_size, device=device,
                         dtype=torch.float64 if device == "cpu" else torch.float32)


def exact_prune_result(rec: dict):
    from .graph.prune import PruneResult

    e = rec["exact_prune"]
    em = e["edge_mask"].to_dense() if e["edge_mask"].is_sparse else e["edge_mask"]
    return PruneResult(e["node_mask"], em, torch.zeros(0), e["node_influence"])


# -------------------------------------------------------------------------------------------------
# K1: structure
# -------------------------------------------------------------------------------------------------
def s95(v: Tensor, frac: float = 0.95) -> Tensor:
    """Per column of v [T, C]: number of largest entries holding `frac` of the squared norm (0 for zero
    columns)."""
    e = v.double().pow(2)
    tot = e.sum(0)
    s = torch.sort(e, dim=0, descending=True).values
    cum = torch.cumsum(s, 0)
    need = frac * tot
    cnt = (cum < need[None] * (1 - 1e-12)).sum(0) + 1
    return torch.where(tot > 0, cnt, torch.zeros_like(cnt))


def weighted_median(x: Tensor, w: Tensor) -> float:
    keep = w > 0
    x, w = x[keep].double(), w[keep].double()
    if x.numel() == 0:
        return float("nan")
    o = torch.argsort(x)
    x, w = x[o], w[o]
    c = torch.cumsum(w, 0) / w.sum()
    return float(x[int(torch.searchsorted(c, torch.tensor(0.5, dtype=c.dtype)))])


def structure_stats(A: Tensor, tm: NodeMeta, sm: NodeMeta) -> dict:
    """Column statistics over feature targets (out-edges within the causal mask: entries outside the
    mask are exactly zero), row statistics over sources. Logit rows are excluded from columns."""
    feat_t = tm.kind != KIND_LOGIT
    Af = A[feat_t].double()
    T = int(feat_t.sum())
    cs = s95(Af)
    cw = Af.pow(2).sum(0)
    rs = s95(A.double().T)
    rw = A.double().pow(2).sum(1)
    hub = cs > T / 20
    out = {
        "T_features": T, "T_D": int(A.shape[0]), "N": int(A.shape[1]),
        "col_s95_ewm": weighted_median(cs.double(), cw),
        "row_s95_ewm": weighted_median(rs.double(), rw),
        "hub_energy_share": float(cw[hub].sum() / cw.sum().clamp_min(1e-300)),
        "hub_count": int(hub.sum()),
        "hub_count_by_kind": {str(int(k)): int((hub & (sm.kind == k)).sum()) for k in sm.kind.unique()},
        "col_s95_ewm_by_kind": {str(int(k)): weighted_median(cs[sm.kind == k].double(), cw[sm.kind == k])
                                for k in sm.kind.unique()},
    }
    out["col_s95_ewm_frac"] = out["col_s95_ewm"] / max(1, out["T_D"])
    out["row_s95_ewm_frac"] = out["row_s95_ewm"] / max(1, out["N"])
    return out


def k1_decision(fracs: list[float], prereg: dict) -> dict:
    k1 = prereg["k1"]
    n = len(fracs)
    share_go = sum(f <= k1["go"]["s95_fraction_of_T_at_most"] for f in fracs) / max(1, n)
    share_nogo = sum(f > k1["nogo"]["s95_fraction_of_T_above"] for f in fracs) / max(1, n)
    if share_go >= k1["go"]["prompt_share_at_least"]:
        v = "GO"
    elif share_nogo >= k1["nogo"]["prompt_share_at_least"]:
        v = "NO-GO"
    else:
        v = "AMBER"
    return {"verdict": v, "share_go": share_go, "share_nogo": share_nogo, "n_prompts": n}


# -------------------------------------------------------------------------------------------------
# K2: simulation
# -------------------------------------------------------------------------------------------------
def grid_settings(grid: dict) -> list[dict]:
    keys = list(grid)
    out = []
    for vals in itertools.product(*[grid[k] for k in keys]):
        cfg: dict = {}
        for k, v in zip(keys, vals):
            a, b = k.split(".")
            cfg.setdefault(a, {})[b] = v
        out.append(cfg)
    return out


def eval_cct(rec: dict, cfg: dict, m: int, seed: int, row_seconds: float, kappa: float, device="cpu") -> dict:
    op = record_operator(rec, device=device)
    A = op.A.double().cpu()
    lay = make_layout(op.target_meta, op.source_meta)
    res = run_cct(op, merge(cfg, {"m": m}), seed=seed, row_seconds=row_seconds, kappa=kappa, layout=lay)
    ex = exact_prune_result(rec)
    f = fidelity(to_square(A, lay), ex, res.adjacency, res.prune, lay.n_features, lay.n_errors, lay.n_tokens,
                 lay.logit_probs)
    return {"fidelity": f, "cost": res.ledger.row_equivalents(), "ledger": res.ledger.to_dict(),
            "diag": {k: v for k, v in res.diagnostics.items() if k != "config"}, "A_hat": res.A_hat}


def eval_truncation(rec: dict, c: int) -> dict:
    A = record_matrix(rec)
    tm, sm = NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"])
    lay = make_layout(tm, sm)
    c_eff = min(c, rec["T_D"])
    adj = truncation_adjacency(A, torch.arange(rec["T_D"]), c_eff, lay)
    pr = prune_adjacency(adj, lay.n_features, lay.n_tokens, lay.logit_probs, rec["prune_cfg"]["node_threshold"],
                         rec["prune_cfg"]["edge_threshold"])
    f = fidelity(to_square(A, lay), exact_prune_result(rec), adj, pr, lay.n_features, lay.n_errors, lay.n_tokens,
                 lay.logit_probs)
    return {"fidelity": f, "cost": float(c_eff)}


def eval_random_rows(rec: dict, c: int, seed: int) -> dict:
    A = record_matrix(rec)
    tm, sm = NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"])
    lay = make_layout(tm, sm)
    c_eff = min(c, rec["T_D"])
    adj = random_rows_adjacency(A, c_eff, lay, seed)
    pr = prune_adjacency(adj, lay.n_features, lay.n_tokens, lay.logit_probs, rec["prune_cfg"]["node_threshold"],
                         rec["prune_cfg"]["edge_threshold"])
    f = fidelity(to_square(A, lay), exact_prune_result(rec), adj, pr, lay.n_features, lay.n_errors, lay.n_tokens,
                 lay.logit_probs)
    return {"fidelity": f, "cost": float(c_eff)}


def eval_lowrank(rec: dict, m: int, seed: int, kappa: float) -> dict:
    op = record_operator(rec)
    A = op.A.double().cpu()
    lay = make_layout(op.target_meta, op.source_meta)
    A_hat, cost = lowrank_sketch(op, m, seed)
    A_hat = A_hat * op.causal_mask().dense().double()  # graphs must be DAGs (docs/decisions.md)
    adj = to_square(A_hat, lay)
    pr = prune_adjacency(adj, lay.n_features, lay.n_tokens, lay.logit_probs, rec["prune_cfg"]["node_threshold"],
                         rec["prune_cfg"]["edge_threshold"])
    f = fidelity(to_square(A, lay), exact_prune_result(rec), adj, pr, lay.n_features, lay.n_errors, lay.n_tokens,
                 lay.logit_probs)
    return {"fidelity": f, "cost": cost["mixed_rows"] + cost["exact_rows"] + kappa * cost["forward_passes"]}


def prompt_meets(fids: list[dict], prereg: dict) -> bool:
    fp = prereg["fidelity"]
    ok = sum(meets_fidelity(f, fp["influence_weighted_node_recall_at_least"], fp["node_jaccard_at_least"])
             for f in fids)
    need = fp["seeds_required"] if len(fids) > 1 else 1
    return ok >= need


def c75(curve: dict, prereg: dict) -> dict | None:
    """curve: {budget: {"meets": [bool per prompt], "costs": [cost per prompt]}} -> smallest budget with
    share >= c75_prompt_share, and the median cost there."""
    share = prereg["k2"]["c75_prompt_share"]
    for b in sorted(curve, key=lambda x: float("inf") if x == "T_D" else float(x)):
        meets = curve[b]["meets"]
        if meets and sum(meets) / len(meets) >= share:
            return {"budget": b, "cost": statistics.median(curve[b]["costs"]), "share": sum(meets) / len(meets)}
    return None


def k2_decision(cct: dict | None, b1: dict | None, T_D_median: float, prereg: dict) -> dict:
    k2 = prereg["k2"]
    if cct is None:
        return {"verdict": "NO-GO", "reason": "CCT never reaches the 75% fidelity rate on its grid",
                "s_exact": None, "s_b1": None}
    s_exact = T_D_median / cct["cost"]
    s_b1 = (b1["cost"] / cct["cost"]) if b1 is not None else float("inf")
    if s_exact >= k2["go"]["s_exact_at_least"] and s_b1 >= k2["go"]["s_b1_at_least"]:
        v = "GO"
    elif s_exact >= k2["amber"]["s_exact_at_least"] and s_b1 >= k2["amber"]["s_b1_at_least"]:
        v = "AMBER"
    else:
        v = "NO-GO"
    return {"verdict": v, "s_exact": s_exact, "s_b1": s_b1}


def go_f_upgrade(k2_d: dict, k2_f: dict) -> dict:
    """Prereg upgrade path: K2 AMBER on both sets in regime D, but regime F gives s_exact >= 20 and
    s_b1 >= 3 on at least one set -> that set is GO-F."""
    out = dict(k2_d)
    if all(v["verdict"] == "AMBER" for v in k2_d.values()):
        for s, f in k2_f.items():
            if f and f.get("s_exact") is not None and f["s_exact"] >= 20 and (f["s_b1"] or 0) >= 3:
                out[s] = dict(out[s], verdict="GO-F", regime_f=f)
    return out


# -------------------------------------------------------------------------------------------------
# K3 and the overall decision
# -------------------------------------------------------------------------------------------------
def k3_decision(n_meet: int, speedup: float, rel_frob: float, dtype: str, prereg: dict) -> dict:
    k3 = prereg["k3"]
    g, a = k3["go"], k3["amber"]
    tol = g["real_vs_simulated_rel_frobenius_at_most"]["bf16" if "bf" in dtype else "fp32"]
    if n_meet >= g["prompts_meeting_fidelity_at_least"] and speedup >= g["wallclock_speedup_at_least"] and rel_frob <= tol:
        v = "GO"
    elif n_meet >= a["prompts_meeting_fidelity_at_least"] and speedup >= a["wallclock_speedup_at_least"]:
        v = "AMBER"
    else:
        v = "NO-GO"
    return {"verdict": v, "n_meet": n_meet, "speedup": speedup, "rel_frob": rel_frob, "rel_frob_tol": tol}


def overall_decision(k1: dict, k2: dict, k3: dict) -> dict:
    """k1, k2, k3: {set: {"verdict": ...}} (k3 may lack sets it was not run on)."""
    sets = list(k2)
    v = lambda d, s: (d.get(s) or {}).get("verdict")  # noqa: E731
    if all(v(k1, s) == "NO-GO" for s in sets) or all(v(k2, s) == "NO-GO" for s in sets) or \
            any(v(k3, s) == "NO-GO" for s in sets):
        return {"verdict": "NO-GO", "set": None}
    for s in sets:
        others_ok = all(v(k2, o) != "NO-GO" for o in sets if o != s)
        if v(k2, s) in ("GO", "GO-F") and others_ok and v(k3, s) == "GO" and v(k1, s) != "NO-GO":
            return {"verdict": "GO", "set": s}
    return {"verdict": "AMBER", "set": None}


def median_or_nan(x: list[float]) -> float:
    return statistics.median(x) if x else math.nan
