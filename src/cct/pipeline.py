"""run_cct: sense, screen, recover, hubs, certify, optional refine, prune (PROJECT.md 7).

Every component is switched by config so ablations (A1 to A10) and B6 (plain Jacobian sensing:
no mask, no screening, no hubs, no certification) are config changes.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch
from torch import Tensor

from .certify.neumann import entry_intervals, influence_interval
from .certify.posterior import posterior_check
from .certify.refine import refine_rows, rows_to_refine
from .cost import Ledger
from .graph.export import GraphLayout, make_layout, square_mask, to_square
from .graph.prune import PruneResult, find_threshold, prune_adjacency
from .hubs import compute_hub_columns, flag_hubs
from .recovery.batched import recover_columns
from .sensing.matrices import sensing_matrix
from .sensing.measure import feature_target_index, sense
from .sensing.screening import column_energy, screen_columns

DEFAULT_CCT = {
    "m": 256,
    "sensing": "rademacher",
    "sparse_d": 8,
    "exact_logit_rows": True,
    "mask": True,
    "screening": {"enabled": True, "q": 0.995, "safety": 2.0},
    "recovery": {"algo": "omp", "k_max": 32, "tol": 1e-6, "chunk": 1024, "device": "cpu",
                 "dtype": "float32"},
    "hubs": {"enabled": True, "rho_hub": 0.3, "budget": 128, "class_rules": {}},
    "certify": {"enabled": True, "m_probe": 16, "delta": 0.01, "seed_offset": 10_000},
    "refine": {"enabled": False},
    "pruning": {"node_threshold": 0.8, "edge_threshold": 0.98},
}

B6_OVERRIDES = {"mask": False, "screening": {"enabled": False}, "hubs": {"enabled": False},
                "certify": {"enabled": False}, "refine": {"enabled": False}}


def merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class CCTResult:
    A_hat: Tensor  # [T, N]
    adjacency: Tensor  # square, circuit-tracer node order
    prune: PruneResult
    ledger: Ledger
    diagnostics: dict


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def run_cct(op, cfg: dict | None = None, seed: int = 0, row_seconds: float = 1.0, kappa: float = 1.0,
            layout: GraphLayout | None = None, local_part: Tensor | None = None) -> CCTResult:
    """local_part: the closed-form A_0 [T, N] (PROJECT_V2 T2), known exactly at zero passes. When given,
    sensing works on the remainder: Y_rem = Y - Phi A_0, recovery and hub columns are of A - A_0, and
    A_0 is added back before certification and pruning."""
    cfg = merge(DEFAULT_CCT, cfg)
    t_wall = time.perf_counter()
    led = Ledger(row_seconds=row_seconds, kappa=kappa)
    diag: dict = {"config": cfg, "seed": seed}
    c0 = op.counters.snapshot()
    layout = layout or make_layout(op.target_meta, op.source_meta)

    logit_idx, feat_idx = feature_target_index(op)
    A_hat = torch.zeros(op.T, op.N, dtype=torch.float64)
    if cfg["exact_logit_rows"]:
        A_hat[logit_idx] = op.exact_rows(logit_idx).double().cpu()
        sensed = feat_idx
    else:
        sensed = torch.arange(op.T)
    T_s = sensed.numel()
    c1 = op.counters.snapshot()
    led.exact_rows += c1["exact_rows"] - c0["exact_rows"]

    # -- sense ---------------------------------------------------------------------------------
    m = int(cfg["m"])
    rdt = _dtype(cfg["recovery"]["dtype"])
    Phi = sensing_matrix(cfg["sensing"], m, T_s, seed, d=cfg["sparse_d"], dtype=torch.float64,
                         positions=op.target_meta.pos[sensed])
    Y = sense(op, Phi.to(op.dtype), sensed).double().cpu()
    c2 = op.counters.snapshot()
    led.mixed_rows += c2["mixed_rows"] - c1["mixed_rows"]
    A0_s = None
    if local_part is not None:
        t0 = time.perf_counter()
        A0_s = local_part[sensed].double().cpu()
        Y = Y - Phi @ A0_s
        led.other_seconds += time.perf_counter() - t0
        diag["local_part_energy_share_est"] = float(
            1.0 - Y.pow(2).sum() / (Y + Phi @ A0_s).pow(2).sum().clamp_min(1e-300))

    # -- screen --------------------------------------------------------------------------------
    t0 = time.perf_counter()
    energy = column_energy(Y)
    if cfg["screening"]["enabled"]:
        keep = screen_columns(Y, cfg["screening"]["q"], cfg["screening"].get("safety", 2.0))
    else:
        keep = energy > 0
    cols = torch.nonzero(keep).flatten()
    mask = op.causal_mask().subset_targets(sensed) if cfg["mask"] else None
    led.other_seconds += time.perf_counter() - t0
    diag["n_cols_total"] = int(op.N)
    diag["n_cols_recovered"] = int(cols.numel())
    diag["screened_energy_fraction_est"] = float(energy[keep].sum() / energy.sum().clamp_min(1e-300))

    # -- recover -------------------------------------------------------------------------------
    rc = cfg["recovery"]
    # a support of size close to m interpolates any column; cap k_max at m // 2 (docs/decisions.md)
    k_eff = int(max(1, min(rc["k_max"], m // 2)))
    diag["k_max_effective"] = k_eff
    extra = {k: v for k, v in rc.items() if k not in ("algo", "k_max", "tol", "chunk", "device", "dtype")}
    rec, rsec = recover_columns(Phi.to(rdt), Y.to(rdt), cols, mask, algo=rc["algo"], k_max=k_eff,
                                tol=rc["tol"], chunk=rc["chunk"], device=rc["device"], dtype=rdt, **extra)
    led.recovery_seconds += rsec
    A_s = rec.to_dense(T_s, op.N)  # [T_s, N]
    diag["median_atoms"] = float(rec.n_atoms.double().median()) if rec.n_atoms.numel() else 0.0

    # -- hubs ----------------------------------------------------------------------------------
    hub_ids = torch.zeros(0, dtype=torch.long)
    if cfg["hubs"]["enabled"]:
        t0 = time.perf_counter()
        hub_ids = flag_hubs(rec, energy, k_eff, cfg["hubs"]["rho_hub"], cfg["hubs"]["budget"],
                            op.source_meta, cfg["hubs"].get("class_rules"))
        led.other_seconds += time.perf_counter() - t0
        c3 = op.counters.snapshot()
        H = compute_hub_columns(op, hub_ids, sensed)
        led.forward_passes += op.counters.forward_passes - c3["forward_passes"]
        if hub_ids.numel():
            A_s[:, hub_ids] = H.double().cpu() - (A0_s[:, hub_ids] if A0_s is not None else 0.0)
    diag["n_hubs"] = int(hub_ids.numel())
    diag["hub_ids"] = hub_ids
    if A0_s is not None:
        A_s = A_s + A0_s
    A_hat[sensed] = A_s
    if not cfg["mask"]:
        # Recovery ran without the mask (B6 / ablation A3). The graph must still be a DAG for influence
        # and pruning to be defined, so the estimate is projected onto the causal pattern afterwards.
        A_hat = A_hat * op.causal_mask().dense().double()
        A_s = A_hat[sensed]

    # -- certify -------------------------------------------------------------------------------
    col_bound = None
    if cfg["certify"]["enabled"]:
        c4 = op.counters.snapshot()
        post = posterior_check(op, A_s, sensed, m_probe=cfg["certify"]["m_probe"],
                               seed=seed + cfg["certify"]["seed_offset"], delta=cfg["certify"]["delta"])
        led.cert_rows += op.counters.mixed_rows - c4["mixed_rows"]
        col_bound = post["col_bound"].clone()
        if hub_ids.numel():
            col_bound[hub_ids] = 0.0  # exact columns
        diag["posterior"] = {"fro_est": post["fro_est"], "fro_bound": post["fro_bound"],
                             "constant": post["constant"], "delta": post["delta"]}

    # -- prune (and optional refinement) -------------------------------------------------------
    pr = cfg["pruning"]
    t0 = time.perf_counter()
    adj = to_square(A_hat, layout)
    res = prune_adjacency(adj, layout.n_features, layout.n_tokens, layout.logit_probs,
                          pr["node_threshold"], pr["edge_threshold"])
    led.other_seconds += time.perf_counter() - t0
    if col_bound is not None:
        t0 = time.perf_counter()
        n = layout.n_nodes
        cb = torch.zeros(n, dtype=torch.float64)
        cb[: op.N] = col_bound
        exact_rows_sq = torch.zeros(n, dtype=torch.bool)
        exact_rows_sq[n - layout.n_logits:] = bool(cfg["exact_logit_rows"])
        msq = square_mask(op.causal_mask().dense(), layout)
        Lo, Hi = entry_intervals(adj, cb, msq, exact_rows_sq)
        w = torch.zeros(n, dtype=torch.float64)
        w[n - layout.n_logits:] = layout.logit_probs
        lo, hi = influence_interval(Lo, Hi, w)
        thr = float(find_threshold(res.node_influence, pr["node_threshold"]))
        diag["influence_interval_width_median"] = float((hi - lo)[: layout.n_features].median()) \
            if layout.n_features else 0.0
        led.other_seconds += time.perf_counter() - t0
        if cfg["refine"]["enabled"]:
            rows = rows_to_refine(res.node_mask, lo, hi, thr, layout.n_features)
            c5 = op.counters.snapshot()
            A_hat, n_ref = refine_rows(op, A_hat, rows, layout.feature_node_to_target)
            got = op.counters.exact_rows - c5["exact_rows"]
            led.exact_rows += got
            led.refine_rows += got
            diag["n_refined"] = n_ref
            t0 = time.perf_counter()
            adj = to_square(A_hat, layout)
            res = prune_adjacency(adj, layout.n_features, layout.n_tokens, layout.logit_probs,
                                  pr["node_threshold"], pr["edge_threshold"])
            led.other_seconds += time.perf_counter() - t0
    c_end = op.counters.snapshot()
    led.pass_seconds = (c_end["seconds"]["rows"] - c0["seconds"]["rows"]) + \
        (c_end["seconds"]["cols"] - c0["seconds"]["cols"])
    led.wall_seconds = time.perf_counter() - t_wall
    return CCTResult(A_hat, adj, res, led, diag)
