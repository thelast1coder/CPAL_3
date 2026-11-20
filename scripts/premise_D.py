"""Gate 0, Stage D (GATE0_PREMISES.md Section 3; PREREG_GATE0.md): structure and recovery at trace
length from sampled exact columns. PR7 (sparsity survives length, hubs are rare), PR8 (screening is
safe and selective), PR9 (per-column count-sketch recovery at trace length).

Per committed trace (results/gate0/traces.json):
  - operator on the full trace: targets = logits at the last position + every active feature; sources =
    every active feature, error and token;
  - 300 sources stratified by kind (feature, error, token) and position (first, early, middle, late),
    exact columns over every target by forward mode (torch.autograd.forward_ad; fallback: double
    backward; which one is recorded); closed-form A_0 columns (cct.local_part); remainder A - A_0;
  - absolute s95 of the remainder over feature targets (k_eff, PR7);
  - attention-rollout candidate sets Omega_s at each tau, absolute-mass recall (energy-weighted over
    columns) and |Omega_s| / |causal set| (PR8);
  - count-sketch D2 simulated exactly on the trace's full feature-target set (CountSketch(T_f, m, R=5,
    seed); logit rows are exact, as in Stage C), applied to the exact remainder columns and decoded over Omega_s, for every tau, m and seed (PR9;
    the tau used for the verdict is chosen from PR8 after all traces are done);
  - timing at full length: seconds per backward row and kappa = forward-mode column cost / row cost.
Sampling weights: each sampled column stands for (stratum size / sampled in stratum) columns. Energy
shares, hub rates and k_eff are reported both design-weighted (primary: they estimate the quantity on
all sources) and unweighted.

Needs results/gate0/stage_C.json (epsilon_star, k_eff_short_qwen); without it PR7 and PR9 are
"not_run". Writes results/gate0/stage_D.json (shared contract plus the Stage E keys) and per-trace
files in results/gate0/stage_D/. Exit 2 if PR7 or PR9 is a kill (stop whole-trace work).

--model-set qwen3_4b_plt (SWITCH-MODEL) reads and writes results/gate0/qwen3_4b/ (cct.premise_traces paths).
--tiny: tiny traces (results/tiny/gate0/traces.json) and the tiny premise model
(cct.premise_traces.tiny_premise_model), smaller grid; numbers meaningless.
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gate0_columns import source_strata, stratified_sources, trace_tokens  # noqa: E402

from cct import premise_traces as pt  # noqa: E402
from cct import scriptutil as su  # noqa: E402
from cct.killtest import s95, weighted_median  # noqa: E402
from cct.local_part import from_circuit_tracer, from_tiny  # noqa: E402
from cct.operators.base import KIND_LOGIT  # noqa: E402
from cct.premise_df import (KIND_NAMES, abs_s95, causal_set, hub_flags, hub_rate_by_kind, mass_recall,  # noqa: E402
                            nanmedian, omega_fraction, pr8_select, ratio_status, select_m_trace, simulate_d2,
                            weighted_screen_count)
from cct.reach import candidate_mask, head_weights, rollout_layers, source_write_layers, tiny_rollout  # noqa: E402
from cct.utils.io import atomic_write_json, read_json  # noqa: E402
from cct.utils.logging import get_logger, run_record  # noqa: E402
from cct.utils.timing import median_time  # noqa: E402

CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")
PREREG = os.path.join(su.ROOT, "configs", "prereg_gate0.yaml")
TINY_D = {"n_sources": 40, "forward_batch": 16, "m_grid": [16, 32, 64, 128, 256], "row_batch": 16}
COL_MEM_BYTES = 4e9  # cap on the [b, T, d] read tensor of a forward-mode column batch


def find_key(d, key):
    """First value stored under `key` anywhere in a nested dict/list (depth first)."""
    if isinstance(d, dict):
        if key in d and d[key] is not None:
            return d[key]
        for v in d.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def kind_counts(kinds: torch.Tensor) -> dict:
    return {KIND_NAMES[int(k)]: int((kinds == k).sum()) for k in kinds.unique().tolist()}


def exact_columns(op, src: torch.Tensor, col_batch: int, log) -> tuple[torch.Tensor, str]:
    """[T, S] exact columns by forward mode, falling back to double backward."""
    old = op.batch_size
    op.batch_size = col_batch
    try:
        try:
            A = op.exact_cols(src).T.double().cpu()
            mode = "fwad"
        except Exception as e:  # noqa: BLE001
            log.info(f"forward AD failed ({e!r}); using double backward")
            op.forward_mode = "double_backward"
            A = op.exact_cols(src).T.double().cpu()
            mode = "double_backward"
    finally:
        op.batch_size = old
    return A, mode


def run_trace(args, cfg, prereg, tr: dict, model, gen, log) -> dict:
    d = cfg["stage_d"]
    toks = trace_tokens(tr)
    t0 = time.perf_counter()
    device = "cuda" if (torch.cuda.is_available() and not args.tiny) else "cpu"
    if args.tiny:
        from cct.operators.tiny import TinyOperator

        op = TinyOperator(model, toks, batch_size=d["row_batch"])
        lp = from_tiny(op)
        Rl = tiny_rollout(op)
        transcoder, L, dtype_name, d_model = model.cfg.transcoder, model.cfg.n_layers, "float64", model.cfg.d_model
    else:
        from cct.operators.circuit_tracer_backend import CircuitTracerOperator

        mc = su.model_config(args.model_set)
        dtype_name = mc["dtype"] or "float32"
        ex = cfg["exact"]
        op = CircuitTracerOperator(model, toks, max_n_logits=ex["max_n_logits"],
                                   desired_logit_prob=ex["desired_logit_prob"], batch_size=d["row_batch"])
        lp = from_circuit_tracer(op)
        Wv, Wo = op.ov_weights()
        Rl = rollout_layers(op.attention_patterns(), head_weights(Wv, Wo))
        transcoder, L, d_model = mc["transcoder_type"], op.n_layers, int(op.enc_vecs.shape[1])
    n_pos, T, N = int(op.n_pos), int(op.T), int(op.N)
    sm, tm = op.source_meta, op.target_meta
    kinds, buckets = d["strata_kinds"], d["strata_positions"]
    src = stratified_sources(sm, n_pos, d["n_sources"], kinds, buckets, gen)
    sid_all = source_strata(sm, n_pos, kinds, buckets)
    sid = sid_all[src]
    sizes = torch.bincount(sid_all[sid_all >= 0], minlength=len(kinds) * len(buckets))
    taken = torch.bincount(sid, minlength=len(kinds) * len(buckets))
    weights = sizes[sid].double() / taken[sid].double()
    col_batch = max(1, min(d["forward_batch"], int(COL_MEM_BYTES / max(1, T * d_model * 4))))
    t_cols = time.perf_counter()
    A, mode = exact_columns(op, src, col_batch, log)
    t_cols = time.perf_counter() - t_cols
    A0 = lp.cols(src).double().cpu()
    Rm = A - A0
    feat_t = tm.kind != KIND_LOGIT
    wl = source_write_layers(sm.subset(src), transcoder, L)
    min_wl = torch.tensor([min(w) for w in wl], dtype=torch.long)
    s_abs = abs_s95(Rm[feat_t])
    s_sq = s95(Rm[feat_t])
    w_feat = Rm[feat_t].pow(2).sum(0)
    w_all = Rm.pow(2).sum(0)
    causal = causal_set(tm.kind, tm.layer, tm.pos, min_wl, sm.pos[src])  # [T, S]
    causal_size = causal.sum(0)
    # -- PR8 and PR9 per tau ---------------------------------------------------------------------------
    pr8, pr9 = {}, {}
    for tau in d["taus"]:
        tk = str(float(tau))
        mask = candidate_mask(Rl, wl, sm.pos[src], tm.layer, tm.pos, float(tau)).cpu()
        mf = mask[feat_t]
        rec_rem = mass_recall(Rm[feat_t], mf)
        rec_full = mass_recall(A[feat_t], mf)
        frac = omega_fraction(mask, causal)
        pr8[tk] = {"recall": rec_rem["recall"], "recall_full_A": rec_full["recall"],
                   "per_column_recall": rec_rem["per_column"].tolist(), "omega_frac_causal": frac.tolist(),
                   "median_frac": nanmedian(frac), "omega_frac_T": float(mask.double().mean())}
        errs, dsec, energy = {}, [], {}
        for m in d["m_grid"]:
            per_seed = []
            for seed in d["seeds"]:
                r = simulate_d2(Rm[feat_t], int(m), int(seed), mf, R=d["d2_repetitions"], device=device)
                per_seed.append(r["err"].tolist())
                dsec.append(r["decode_seconds_per_column"])
                if seed == d["seeds"][0]:
                    energy[str(m)] = r["energy_est"].tolist()
            errs[str(m)] = per_seed
        pr9[tk] = {"err": errs, "decode_seconds_per_column": float(torch.tensor(dsec).median()),
                   "energy_est_seed0": energy}
        del mask
    # -- timing at full length ----------------------------------------------------------------------------
    rb = int(op.batch_size)
    ridx = torch.randperm(T, generator=gen)[:rb]
    tr_t, _ = median_time(lambda: op.exact_rows(ridx), 3)
    old = op.batch_size
    op.batch_size = col_batch
    cidx = src[:col_batch]
    tc_t, _ = median_time(lambda: op.exact_cols(cidx), 3)
    op.batch_size = old
    row_s = tr_t / ridx.numel()
    col_s = tc_t / cidx.numel()
    out = {
        "trace_id": tr["id"], "T": T, "N": N, "n_pos": n_pos, "n_sampled": int(src.numel()), "dtype": dtype_name,
        "forward_mode": mode, "col_batch": col_batch, "cols_seconds": t_cols,
        "sources": src.tolist(), "source_kind": sm.kind[src].tolist(), "source_pos": sm.pos[src].tolist(),
        "source_min_write_layer": min_wl.tolist(), "stratum": sid.tolist(), "weight": weights.tolist(),
        "stratum_sizes": sizes.tolist(), "n_sources_by_kind": kind_counts(sm.kind),
        "s95_abs_R": s_abs.tolist(), "s95_sq_R": s_sq.tolist(), "w_feat_R": w_feat.tolist(),
        "w_all_R": w_all.tolist(), "causal_size": causal_size.tolist(),
        "a0_energy_share": 1.0 - float(Rm.pow(2).sum()) / max(float(A.pow(2).sum()), 1e-300),
        "pr8": pr8, "pr9": pr9,
        "timing": {"row_seconds": row_s, "col_seconds": col_s, "kappa": col_s / row_s, "row_batch": rb,
                   "col_batch": col_batch, "forward_mode": mode},
        "seconds": time.perf_counter() - t0,
    }
    del op, lp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def cat(per: dict, key: str, sub: str | None = None, dtype=torch.float64) -> torch.Tensor:
    vals = []
    for v in per.values():
        x = v[key] if sub is None else v[key][sub]
        vals.extend(x)
    return torch.tensor(vals, dtype=dtype)


def pool(per: dict, cfg: dict, prereg: dict, stage_c: dict | None) -> dict:
    d, pd = cfg["stage_d"], prereg["stage_d"]
    W = cat(per, "weight")
    s_abs = cat(per, "s95_abs_R")
    w_feat = cat(per, "w_feat_R")
    w_all = cat(per, "w_all_R")
    kinds = cat(per, "source_kind", dtype=torch.long)
    k_eff_w = weighted_median(s_abs, w_feat * W)
    k_eff_u = weighted_median(s_abs, w_feat)
    k_eff_per_trace = {t: weighted_median(torch.tensor(v["s95_abs_R"], dtype=torch.float64),
                                          torch.tensor(v["w_feat_R"]) * torch.tensor(v["weight"]))
                       for t, v in per.items()}
    eps = find_key(stage_c, "epsilon_star") if stage_c else None
    k_short = find_key(stage_c, "k_eff_short_qwen") if stage_c else None
    notes, decisions = [], [
        "k_eff, energy shares and hub rates are design-weighted by stratum size / sampled count (primary); "
        "unweighted values are in detail",
        "mass recall and |Omega_s| / |causal set| use feature targets of the remainder column (A - A_0); the "
        "recall of the full column A is in detail",
        "as Stage C (premise_ac.d2_estimate), the count-sketch hashes the feature-target rows only (logit rows "
        "are computed exactly); relative errors and the energy for m_trace are over feature targets",
        "hub = error above epsilon_star at the largest m on a majority (2 of 3) of seeds, at the PR9 tau",
    ]
    # -- PR7 ---------------------------------------------------------------------------------------------
    pr7_t = pd["pr7"]
    if k_short is None:
        pr7 = {"status": "not_run", "values": {"k_eff_trace": k_eff_w, "k_eff_trace_unweighted": k_eff_u,
                                                "k_eff_per_trace": k_eff_per_trace},
               "thresholds": pr7_t, "notes": "stage_C.json (k_eff_short_qwen) missing"}
        ratio = float("nan")
    else:
        ratio = k_eff_w / float(k_short)
        pr7 = {"status": ratio_status(ratio, pr7_t["pass"]["k_eff_trace_over_short_at_most"],
                                      pr7_t["kill"]["k_eff_trace_over_short_above"]),
               "values": {"k_eff_trace": k_eff_w, "k_eff_short_qwen": float(k_short), "ratio": ratio,
                          "k_eff_trace_unweighted": k_eff_u, "ratio_unweighted": k_eff_u / float(k_short),
                          "k_eff_per_trace": k_eff_per_trace},
               "thresholds": pr7_t, "notes": "k_eff in entries (energy-weighted median of the remainder absolute s95)"}
    # -- PR8 ---------------------------------------------------------------------------------------------
    taus = [str(float(t)) for t in d["taus"]]
    per_tau = {}
    for tk in taus:
        rec = torch.tensor(sum((v["pr8"][tk]["per_column_recall"] for v in per.values()), []), dtype=torch.float64)
        fr = torch.tensor(sum((v["pr8"][tk]["omega_frac_causal"] for v in per.values()), []), dtype=torch.float64)
        ok = ~torch.isnan(rec) & (w_feat > 0)
        ww = (w_feat * W)[ok]
        recall_w = float((rec[ok] * ww).sum() / ww.sum()) if bool(ok.any()) else float("nan")
        recall_u = float((rec[ok] * w_feat[ok]).sum() / w_feat[ok].sum()) if bool(ok.any()) else float("nan")
        okf = ~torch.isnan(fr)
        med_w = weighted_median(fr[okf], W[okf]) if bool(okf.any()) else float("nan")
        per_tau[tk] = {"recall": recall_w, "median_frac": med_w, "recall_unweighted": recall_u,
                       "median_frac_unweighted": nanmedian(fr),
                       "recall_full_A_by_trace": {t: v["pr8"][tk]["recall_full_A"] for t, v in per.items()}}
    pr8_t = pd["pr8"]["pass"]
    sel = pr8_select({t: {"recall": v["recall"], "median_frac": v["median_frac"]} for t, v in per_tau.items()},
                     pr8_t["mass_recall_at_least"], pr8_t["median_omega_frac_causal_at_most"])
    tau_star = sel["tau"]
    pr8 = {"status": "pass" if sel["passed"] else "fail", "values": {"per_tau": per_tau, "tau_for_pr9": tau_star},
           "thresholds": pd["pr8"], "notes": sel["reason"]}
    # -- PR9 ---------------------------------------------------------------------------------------------
    grid = [str(m) for m in d["m_grid"]]
    m_max = grid[-1]
    errs = {m: torch.cat([torch.tensor(v["pr9"][tau_star]["err"][m], dtype=torch.float64) for v in per.values()], 1)
            for m in grid}  # [n_seeds, S_total]
    dec_s = float(torch.tensor([v["pr9"][tau_star]["decode_seconds_per_column"] for v in per.values()]).median())
    detail_pr9 = {"tau": tau_star, "m_grid": [int(m) for m in grid],
                  "median_err_by_m": {m: nanmedian(errs[m].median(0).values) for m in grid}}
    hub_rates, hub_rates_u, m_trace, m_trace_u, n_screen = None, None, None, None, {}
    if eps is None:
        pr9 = {"status": "not_run", "values": {"decode_seconds_per_column": dec_s}, "thresholds": pd["pr9"],
               "notes": "stage_C.json (epsilon_star) missing"}
    else:
        eps = float(eps)
        hub = hub_flags(errs[m_max], eps)
        hub_rates = {}
        for k in sorted(set(kinds.tolist())):
            s = kinds == k
            hub_rates[KIND_NAMES[int(k)]] = float((hub[s].double() * W[s]).sum() / W[s].sum())
        hub_rates_u = hub_rate_by_kind(hub, kinds)
        m_trace, table = select_m_trace({int(m): errs[m] for m in grid}, w_feat * W, hub, eps,
                                        pd["pr9"]["non_hub_energy_share_within_eps_star_at_least"],
                                        prereg["fidelity"]["seeds_required"])
        m_trace_u, table_u = select_m_trace({int(m): errs[m] for m in grid}, w_feat, hub, eps,
                                            pd["pr9"]["non_hub_energy_share_within_eps_star_at_least"],
                                            prereg["fidelity"]["seeds_required"])
        per_trace_m = {}
        off = 0
        for t, v in per.items():
            S = len(v["weight"])
            sl = slice(off, off + S)
            off += S
            mt, _ = select_m_trace({int(m): errs[m][:, sl] for m in grid}, (w_feat * W)[sl], hub[sl], eps,
                                   pd["pr9"]["non_hub_energy_share_within_eps_star_at_least"],
                                   prereg["fidelity"]["seeds_required"])
            per_trace_m[t] = mt
        detail_pr9.update({"share_table": table, "share_table_unweighted": table_u, "m_trace_unweighted": m_trace_u,
                           "m_trace_per_trace": per_trace_m, "hub_count": int(hub.sum()), "n_columns": int(hub.numel())})
        pr9 = {"status": "pass" if m_trace is not None else "kill",
               "values": {"m_trace": m_trace, "epsilon_star": eps, "tau": tau_star, "hub_rate_by_kind": hub_rates,
                          "decode_seconds_per_column": dec_s},
               "thresholds": pd["pr9"],
               "notes": ("decoded over Omega_s at the PR8 tau" if sel["passed"] else
                         "PR8 did not pass: decoded at the tau with the highest recall") +
                        f" (tau={tau_star})"}
        if pr7["status"] != "not_run":
            pr7["values"]["hub_rate_by_kind"] = hub_rates
    # -- screening count for Stage E -----------------------------------------------------------------------
    m_scr = str(m_trace) if m_trace is not None else m_max
    q, safety = cfg["stage_c"]["d1"]["screening"]["q"], cfg["stage_c"]["d1"]["screening"]["safety"]
    for t, v in per.items():
        e = torch.tensor(v["pr9"][tau_star]["energy_est_seed0"][m_scr], dtype=torch.float64)
        n_screen[t] = weighted_screen_count(e, torch.tensor(v["weight"], dtype=torch.float64), q, safety)
    return {
        "premises": {"PR7": pr7, "PR8": pr8, "PR9": pr9},
        "m_trace": m_trace, "hub_rate_by_kind": hub_rates, "decode_seconds_per_column": dec_s,
        "n_sources_by_kind": {t: v["n_sources_by_kind"] for t, v in per.items()},
        "kappa": float(torch.tensor([v["timing"]["kappa"] for v in per.values()]).median()),
        "row_seconds_full": {t: v["timing"]["row_seconds"] for t, v in per.items()},
        "n_cols_after_screening": n_screen,
        "detail": {"pr9": detail_pr9, "hub_rate_by_kind_unweighted": hub_rates_u, "k_eff_trace_unweighted": k_eff_u,
                   "screening": {"q": q, "safety": safety, "m": int(m_scr), "tau": tau_star,
                                 "estimate": "count-sketch column energy ||Y_s||^2 / R (seed 0), stratum-weighted"},
                   "decisions": decisions, "notes": notes,
                   "per_trace": {t: {k: v[k] for k in ("T", "N", "n_pos", "n_sampled", "dtype", "forward_mode",
                                                       "col_batch", "cols_seconds", "a0_energy_share", "timing",
                                                       "seconds", "n_sources_by_kind")}
                                 for t, v in per.items()}},
    }


def main() -> None:
    p = su.parser("Gate 0 Stage D: PR7, PR8, PR9 from sampled exact columns at trace length")
    p.set_defaults(config=CONFIG)
    p.add_argument("--model-set", default=None, help="model set (default: sets.qwen of the config)")
    p.add_argument("--stage-c", default=None, help="stage_C.json to read (default: <results>/gate0/stage_C.json)")
    p.add_argument("--out-dir", default=None, help="output directory (default: <results>/gate0; tests)")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    prereg = su.load_yaml(PREREG)
    args.model_set = args.model_set or cfg["sets"]["qwen"]
    log = get_logger("premise_D", os.path.join(su.ROOT, "logs"))
    tag = pt.model_tag(args.model_set, cfg["sets"]["qwen"])
    res_dir = pt.gate0_results_dir(su.ROOT, args.tiny, tag)
    out_dir = args.out_dir or res_dir
    tdir = os.path.join(out_dir, "stage_D")
    os.makedirs(tdir, exist_ok=True)
    d = cfg["stage_d"]
    d.setdefault("row_batch", cfg["exact"]["batch_size_trace"])
    if args.tiny:
        d.update(TINY_D)
    traces = pt.load_traces(pt.traces_file(su.ROOT, args.tiny, tag, cfg))
    c_path = args.stage_c or os.path.join(res_dir, "stage_C.json")
    stage_c = read_json(c_path) if os.path.exists(c_path) else None
    if stage_c is None:
        log.info(f"{c_path} missing: PR7 and PR9 will be not_run")
    gen = torch.Generator().manual_seed(cfg["seed"])
    t_start = time.perf_counter()
    model, per = None, {}
    for tr in traces:
        tid = tr["id"]
        path = os.path.join(tdir, f"{tid}.json")
        if os.path.exists(path) and not args.force:
            per[tid] = read_json(path)
            continue
        if model is None:
            if args.tiny:
                model = pt.tiny_premise_model()  # the model that generated the tiny premise traces
            else:
                from cct.operators.circuit_tracer_backend import load_replacement_model
                from cct.utils.gpu import cap_memory, nvidia_smi

                log.info(nvidia_smi())
                mc = su.model_config(args.model_set)
                cap_memory(mc["memory_cap_gb"])
                model = load_replacement_model(mc["model_name"], mc["transcoder_set"], dtype=su.torch_dtype(mc["dtype"]))
        gen.manual_seed(cfg["seed"] + sum(map(ord, tid)))  # per-trace seed: resumable and order independent
        res = run_trace(args, cfg, prereg, tr, model, gen, log)
        res["run"] = run_record(cfg, cfg["seed"], res["seconds"], trace_id=tid)
        atomic_write_json(path, res)
        per[tid] = res
        log.info(f"{tid}: T={res['T']} N={res['N']} mode={res['forward_mode']} "
                 f"row_s={res['timing']['row_seconds']:.3e} kappa={res['timing']['kappa']:.2f}")
    out = pool(per, cfg, prereg, stage_c)
    st = {k: v["status"] for k, v in out["premises"].items()}
    stop = st["PR7"] == "kill" or st["PR9"] == "kill"
    reason = None
    if stop:
        reason = "; ".join(f"{k} kill" for k in ("PR7", "PR9") if st[k] == "kill") + ": stop whole-trace work"
    wall = time.perf_counter() - t_start
    gpu_s = sum(v["seconds"] for v in per.values()) if (torch.cuda.is_available() and not args.tiny) else 0.0
    result = {"run": run_record(cfg, cfg["seed"], wall, model_set=args.model_set, tiny=args.tiny,
                                stage_c_present=stage_c is not None),
              "stage": "D", "premises": out.pop("premises"), "stop": stop, "stop_reason": reason,
              "gpu_seconds": gpu_s}
    result.update(out)
    atomic_write_json(os.path.join(out_dir, "stage_D.json"), result)
    log.info(f"Stage D: {st}; m_trace={result['m_trace']} kappa={result['kappa']:.2f}")
    if stop:
        log.error(f"STOP: {reason}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
