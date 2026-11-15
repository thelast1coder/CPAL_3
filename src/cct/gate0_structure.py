"""G0-A analysis (PROJECT_V2.md Section 7): structure of A and of the remainder A - A_0, the truncation
curve, and simulated sensing of A - A_0 (A_0 added back). Backend-free: works on stored records
(cct.killtest) and a dense A_0 in the record's layout."""
from __future__ import annotations

import statistics

import torch
from torch import Tensor

from .graph.export import make_layout, to_square
from .graph.metrics import fidelity, meets_fidelity
from .killtest import (eval_truncation, exact_prune_result, record_matrix, record_operator, s95,
                       structure_stats, weighted_median)
from .operators.base import KIND_LOGIT, KIND_NAMES, NodeMeta
from .pipeline import merge, run_cct


def record_index_map(rec: dict, n_logits: int, n_active: int) -> tuple[Tensor, Tensor]:
    """Rows and columns of a regime record in the operator layout (targets [logits, all active features],
    sources [all active features, errors, tokens])."""
    visit = torch.as_tensor(rec["visit_features"]).long()
    rows = torch.cat([torch.arange(n_logits), n_logits + visit])
    selected = torch.sort(visit).values
    n_nonfeat = rec["N"] - selected.numel()
    cols = torch.cat([selected, n_active + torch.arange(n_nonfeat)])
    return rows, cols


def rel_err(a: Tensor, b: Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def structure(A: Tensor, A0: Tensor, tm: NodeMeta, sm: NodeMeta) -> dict:
    A, A0 = A.double().cpu(), A0.double().cpu()
    R = A - A0
    sA = structure_stats(A, tm, sm)
    sR = structure_stats(R, tm, sm)
    feat_t = tm.kind != KIND_LOGIT
    cs = s95(R[feat_t])
    cw = R[feat_t].pow(2).sum(0)
    hub = cs > int(feat_t.sum()) / 20
    tot = float(cw.sum()) or 1.0
    tot_A = float(A.pow(2).sum()) or 1.0
    return {
        "T": int(A.shape[0]), "N": int(A.shape[1]),
        "a0_energy_share": 1.0 - float(R.pow(2).sum()) / tot_A,
        "s95_A_ewm": sA["col_s95_ewm"], "s95_rem_ewm": sR["col_s95_ewm"],
        "s95_A_frac_T": sA["col_s95_ewm_frac"], "s95_rem_frac_T": sR["col_s95_ewm"] / max(1, int(A.shape[0])),
        "row_s95_rem_ewm": sR["row_s95_ewm"],
        "hub_share_by_kind": {KIND_NAMES.get(int(k), str(int(k))): float(cw[hub & (sm.kind == k)].sum()) / tot
                              for k in sm.kind.unique()},
        "hub_count_rem": int(hub.sum()),
        "s95_rem_ewm_by_kind": {KIND_NAMES.get(int(k), str(int(k))): weighted_median(cs[sm.kind == k].double(),
                                                                                    cw[sm.kind == k])
                                for k in sm.kind.unique()},
    }


def truncation_curve(rec: dict, budgets: list, fid: dict) -> dict:
    curve = {}
    c_min = None
    for b in budgets:
        c = rec["T_D"] if b == "T_D" else int(b)
        r = eval_truncation(rec, c)
        ok = meets_fidelity(r["fidelity"], fid["influence_weighted_node_recall_at_least"], fid["node_jaccard_at_least"])
        curve[str(b)] = {"cost": r["cost"], "meets": ok, "iw_recall": r["fidelity"].get("iw_node_recall"),
                         "jaccard": r["fidelity"].get("node_jaccard")}
        if ok and c_min is None:
            c_min = int(r["cost"])
    return {"c_trunc": c_min, "curve": curve}


def sensing_curve(rec: dict, A0: Tensor, ms: list[int], seeds: list[int], cct_cfg: dict, fid: dict,
                  row_seconds: float, kappa: float, device: str = "cpu") -> dict:
    """Simulated CCT on the stored exact matrix with A_0 subtracted before recovery and added back."""
    op = record_operator(rec, device=device)
    A = record_matrix(rec)
    lay = make_layout(op.target_meta, op.source_meta)
    ex = exact_prune_result(rec)
    sq = to_square(A, lay)
    out, m_min = {}, None
    need = fid.get("seeds_required", 2) if len(seeds) > 1 else 1
    for m in ms:
        runs = []
        for sd in seeds:
            cfg = merge(cct_cfg, {"m": int(m), "recovery": {"device": device}})
            res = run_cct(op, cfg, seed=sd, row_seconds=row_seconds, kappa=kappa, layout=lay, local_part=A0)
            f = fidelity(sq, ex, res.adjacency, res.prune, lay.n_features, lay.n_errors, lay.n_tokens, lay.logit_probs)
            runs.append({"seed": sd, "meets": meets_fidelity(f, fid["influence_weighted_node_recall_at_least"],
                                                             fid["node_jaccard_at_least"]),
                         "iw_recall": f.get("iw_node_recall"), "jaccard": f.get("node_jaccard"),
                         "cost": res.ledger.row_equivalents(), "recovery_seconds": res.ledger.recovery_seconds,
                         "n_hubs": res.diagnostics.get("n_hubs")})
        n_ok = sum(r["meets"] for r in runs)
        out[str(m)] = {"runs": runs, "n_meet": n_ok, "meets": n_ok >= need,
                       "median_cost": statistics.median(r["cost"] for r in runs),
                       "median_recovery_seconds": statistics.median(r["recovery_seconds"] for r in runs)}
        if n_ok >= need and m_min is None:
            m_min = int(m)
    return {"m_min": m_min, "by_m": out}
