"""Gate 0 premise tests, Stage E (GATE0_PREMISES.md Section 3, PREREG_GATE0.md): PR10, does shared
sensing beat row methods?

Inputs: the reference graphs and index of Stage B (results/gate0/reference_graphs.json), the traces,
results/gate0/stage_C.json (epsilon_star, recorded only) and results/gate0/stage_D.json (keys
m_trace, kappa, hub_rate_by_kind {kind: rate}, n_sources_by_kind {trace_id: {kind: count}},
decode_seconds_per_column, row_seconds_full {trace_id: s}, n_cols_after_screening {trace_id: n};
m_trace and kappa may also be given per trace). Without Stage D, PR10 is "not_run" but the unions
and timings are still computed and written. Settings: configs/experiments/gate0_premises.yaml
stage_e; thresholds: configs/prereg_gate0.yaml stage_e.pr10.

Definitions
- Output order (fixed): per trace, the answer first, then the sentence outputs in sentence order.
- Realistic union |U_trunc(n)|: for each output, c = smallest c of the c grid (feature rows after
  the logit rows) whose truncation graph meets fidelity against the output's own reference graph
  (influence-weighted node recall >= 0.95 and node Jaccard >= 0.85, cct.gate0_union.smallest_c;
  the full budget if none); rows = the (layer, position, feature) of its first c visited features
  (positions in trace coordinates) plus its logit rows. Union over the first n outputs.
- Oracle union |U_oracle(n)|: the union of the exact pruned feature-node sets plus logit rows.
- Cost weights: on the GPU, per trace, the time of one batch of 64 one-hot rows (64 random active
  features, circuit-tracer's compute_batch) at prefix lengths 64, 128, 256 (those shorter than the
  trace) and the full trace, median of 3 after one warm-up. A row computed on prefix q costs
  t(q) / t(full) row-equivalents at full length, t interpolated linearly in length (proportional
  below 64). A distinct feature row costs its weight at the SHORTEST prefix of the outputs that
  selected it (the row is the same on every prefix that contains its position, PR2); logit rows
  cost their own output's weight (cct.premise_be.weighted_union). Unweighted counts are reported too.
- n grid. Per trace there are 17 outputs, so the spec's grid {1, 2, 4, 8, 16, 32, 51} cannot be
  applied per trace. We report (a) per-trace curves at n in {1, 2, 4, 5, 8, 11, 16, 17} with the
  local slope = the smaller of the log-log slopes 5 -> 11 and 11 -> 17 (the spec's 16 -> 32 and
  32 -> 51 scaled by 1/3, 51 -> 17; ratios 2.2 and 1.55 against 2 and 1.59), used for the pass rule
  "local slope >= 0.6 on >= 2 of 3 traces" and for each trace's projection; and (b) a pooled curve
  over all 51 outputs taken round-robin over the traces (output j of every trace before j + 1),
  at the spec grid, with its 16 -> 32 and 32 -> 51 slopes (reported). Rows of different traces are
  different rows (different sequences).
- Cost_CCT(L) per trace (row-equivalents at the trace's full length) = m_trace + kappa x projected
  hubs (sum over kinds of hub rate x sources of that kind) + 16 check probes + decode seconds per
  column x columns kept after screening / row seconds at full length (cct.premise_be.cost_cct).
- S_meas = cost-weighted |U_trunc(51)| / sum over traces of Cost_CCT(L), both converted to seconds
  with each trace's row time at full length (so traces of different lengths add up correctly):
  51 explained outputs for truncation against all L tokens of the three traces for CCT.
- S_proj per trace = |U_trunc(17)| (weighted) x (L_gen / 17)^(local slope) / Cost_CCT(L), with L_gen
  the trace's number of generated tokens; pooled S_proj = the same sums in seconds over traces.
  S_oracle_proj: the same with the oracle union and its own local slope.
- Pass (prereg): local slope >= 0.6 on >= 2 of 3 traces, pooled S_meas >= 3, pooled S_proj >= 20,
  pooled S_oracle_proj >= 5. Kill: S_proj < 3 on every trace (sets stop, exit code 2).

Output: results/gate0/stage_E.json (stage_E_<tag>.json for another model set). Per-output results
are cached next to the records as <q>.e.json (resumable; --force recomputes); timings in
results/gate0[/<tag>]/stage_E_timing.json.

--tiny: tiny model, tiny reference graphs and c grid; timing on the CPU with the tiny operator. If
stage_C.json / stage_D.json are absent in results/tiny/gate0, PR10 is not_run (the tests fabricate
them). Numbers are meaningless.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from cct import scriptutil as su
from cct.gate0_union import TruncationEvaluator, n_logit_rows, smallest_c, visited_features
from cct.premise_be import (PER_TRACE_N, PER_TRACE_SLOPE_PAIRS, POOLED_SLOPE_PAIRS, cost_cct, interp_seconds,
                            local_slope, oracle_rows, pooled_order, pr10_evaluate, projected_union, union_curve,
                            union_order, weighted_union)
from cct.premise_traces import gate0_results_dir, load_traces, model_tag, tiny_premise_model, traces_file
from cct.utils.io import atomic_write_json, load_pt, read_json
from cct.utils.logging import get_logger, run_record

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")
PREREG = os.path.join(su.ROOT, "configs", "prereg_gate0.yaml")
TINY_C_GRID = [4, 8, 16, 32, 48, 64, 96, 128]
TIMING_BATCH = 64
TIMING_REPEATS = 3
EXIT_STOP = 2
NAN = float("nan")


def per_trace(x, tid: str):
    """A Stage D value given either per trace ({trace_id: v}) or as one scalar."""
    if isinstance(x, dict):
        return x.get(tid)
    return x


# -------------------------------------------------------------------------------------------------
def analyse_output(pt: str, c_grid: list[int], fid: dict, force: bool) -> dict:
    cache = pt[:-3] + ".e.json"
    mt = os.path.getmtime(pt)
    if os.path.exists(cache) and not force:
        c = read_json(cache)
        if c.get("pt_mtime") == mt and c.get("c_grid") == list(c_grid):
            return c
    rec = load_pt(pt)
    off = int(rec.get("prefix_offset", rec.get("pos_offset", 0)))
    ev = TruncationEvaluator(rec)
    curve: list = []
    c = smallest_c(rec, c_grid, fid["recall_min"], fid["jaccard_min"], ev, curve)
    c_used = c if c is not None else ev.n_feature_rows
    out = {"trace_id": rec["trace_id"], "position": int(rec["position"]), "output_kind": rec["output_kind"],
           "sentence_index": rec.get("sentence_index"), "prefix_offset": off, "c": c, "c_used": c_used,
           "n_feature_rows": ev.n_feature_rows, "n_logits": n_logit_rows(rec), "budget": rec.get("budget"),
           "curve": curve, "realistic": visited_features(rec, c_used, off), "oracle": oracle_rows(rec, off),
           "pt_mtime": mt, "c_grid": list(c_grid)}
    atomic_write_json(cache, out)
    return out


def measure_timing(args, model_set: str, traces: list[dict], lengths_cfg: list, seed: int, log) -> dict:
    """{trace_id: {"lengths": [...], "seconds": [...], "all": [[...] per length]}} for one batch of 64
    one-hot rows at each prefix length."""
    from cct.utils.timing import median_time

    g = torch.Generator().manual_seed(seed)
    out = {}
    model = None
    t_load = time.perf_counter()
    if args.tiny:
        model = tiny_premise_model()
    else:
        from cct.operators.circuit_tracer_backend import load_replacement_model
        from cct.utils.gpu import cap_memory, nvidia_smi

        mc = su.model_config(model_set)
        log.info(nvidia_smi())
        cap_memory(mc["memory_cap_gb"])
        model = load_replacement_model(mc["model_name"], mc["transcoder_set"], dtype=su.torch_dtype(mc["dtype"]))
    load_s = time.perf_counter() - t_load
    for t in traces:
        L = int(t.get("n_total_tokens") or len(t["token_ids"]))
        lens = sorted({int(x) for x in lengths_cfg if x != "full" and int(x) < L} | {L})
        secs, alls = [], []
        for ell in lens:
            toks = torch.tensor(t["token_ids"][:ell], dtype=torch.long)
            if args.tiny:
                from cct.operators.tiny import TinyOperator

                op = TinyOperator(model, toks, batch_size=TIMING_BATCH)
                n_act = op.T - op.n_logits
                idx = op.n_logits + torch.randperm(n_act, generator=g)[: min(TIMING_BATCH, n_act)]
                fn = lambda: op.exact_rows(idx)  # noqa: E731
            else:
                from cct.operators.circuit_tracer_backend import CircuitTracerOperator

                op = CircuitTracerOperator(model, toks, batch_size=TIMING_BATCH)
                idx = op.n_logits + torch.randperm(op.n_active, generator=g)[: min(TIMING_BATCH, op.n_active)]
                fn = lambda: op.ct_rows(idx)  # noqa: E731
            fn()  # warm-up
            med, ts = median_time(fn, TIMING_REPEATS)
            secs.append(med)
            alls.append(ts)
            log.info(f"timing {t['id']} length {ell}: {med * 1000:.1f} ms per batch of {idx.numel()} rows")
            del op
            if not args.tiny:
                torch.cuda.empty_cache()
        out[t["id"]] = {"lengths": lens, "seconds": secs, "all": alls, "batch": TIMING_BATCH,
                        "repeats": TIMING_REPEATS}
    return {"traces": out, "model_load_seconds": load_s}


# -------------------------------------------------------------------------------------------------
def main() -> None:
    p = su.parser("Gate 0 Stage E: PR10 unions, cost weights and speed-ups")
    p.set_defaults(config=DEFAULT_CONFIG)
    p.add_argument("--model-set", default=None)
    p.add_argument("--force-timing", action="store_true", help="re-measure the pass timings")
    p.add_argument("--stage-c", default=None, help="default: <results>/stage_C.json")
    p.add_argument("--stage-d", default=None, help="default: <results>/stage_D.json")
    args = p.parse_args()
    t_start = time.perf_counter()
    cfg = su.load_yaml(args.config)
    prereg = su.load_yaml(PREREG)
    thr = prereg["stage_e"]["pr10"]
    se = cfg["stage_e"]
    default_set = cfg["sets"]["qwen"]
    model_set = args.model_set or default_set
    tag = model_tag(model_set, default_set)
    base = gate0_results_dir(su.ROOT, args.tiny)
    res_dir = gate0_results_dir(su.ROOT, args.tiny, tag)
    out_path = os.path.join(base, f"stage_E{'_' + tag if tag else ''}.json")
    log = get_logger("premise_E", os.path.join(su.ROOT, "logs"))
    f = prereg["fidelity"]
    fid = {"recall_min": float(f["influence_weighted_node_recall_at_least"]),
           "jaccard_min": float(f["node_jaccard_at_least"])}
    c_grid = TINY_C_GRID if args.tiny else list(se["c_grid"])

    index_path = os.path.join(res_dir, "reference_graphs.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"{index_path} missing: run scripts/premise_B.py first")
    index = read_json(index_path)
    traces = load_traces(traces_file(su.ROOT, args.tiny, tag, cfg))
    tr_by = {t["id"]: t for t in traces}
    c_path = args.stage_c or os.path.join(base, "stage_C.json")
    d_path = args.stage_d or os.path.join(base, "stage_D.json")
    stage_c = read_json(c_path) if os.path.exists(c_path) else None
    stage_d = read_json(d_path) if os.path.exists(d_path) else None

    # ---- per-output truncation and oracle rows (CPU) ---------------------------------------------
    per_out: dict[str, list[dict]] = {}
    for r in index["records"]:
        pt = os.path.join(su.ROOT, r["path"])
        if not os.path.exists(pt):
            log.warning(f"missing record {pt}")
            continue
        a = analyse_output(pt, c_grid, fid, args.force)
        per_out.setdefault(a["trace_id"], []).append(a)
        log.info(f"{a['trace_id']} q={a['position']} ({a['output_kind']}): c={a['c']} "
                 f"oracle={len(a['oracle'])} logits={a['n_logits']}")
    tids = [t["id"] for t in traces if t["id"] in per_out]
    ordered = {t: union_order(per_out[t]) for t in tids}

    # ---- per-length pass timing (GPU) -------------------------------------------------------------
    timing_path = os.path.join(res_dir, "stage_E_timing.json")
    timing = read_json(timing_path) if os.path.exists(timing_path) else None
    if timing is None or args.force or args.force_timing or set(tids) - set(timing["traces"]):
        t0 = time.perf_counter()
        timing = measure_timing(args, model_set, [tr_by[t] for t in tids], se["timing_lengths"], cfg["seed"], log)
        timing["gpu_seconds"] = time.perf_counter() - t0
        timing["run"] = run_record({"stage_e_timing": se, "model_set": model_set, "tiny": args.tiny}, cfg["seed"],
                                   timing["gpu_seconds"])
        atomic_write_json(timing_path, timing)

    def weight_fn(tid: str):
        tt = timing["traces"][tid]
        full = tt["seconds"][-1]
        return lambda q: interp_seconds(tt["lengths"], tt["seconds"], q) / full

    # ---- unions --------------------------------------------------------------------------------
    def entries(tid: str, kind: str, pooled: bool = False) -> list[dict]:
        return [{"features": [((tid,) if pooled else ()) + tuple(x) for x in a[kind]], "n_logits": a["n_logits"],
                 "q": a["position"], "tid": tid} for a in ordered[tid]]

    U = {"realistic": {}, "oracle": {}}
    slopes = {"realistic": {}, "oracle": {}}
    for tid in tids:
        w = weight_fn(tid)
        for kind in U:
            U[kind][tid] = union_curve(entries(tid, kind), PER_TRACE_N, w)
            slopes[kind][tid] = local_slope(U[kind][tid], PER_TRACE_SLOPE_PAIRS)
    # pooled: weights are per trace, so weight by each entry's own trace
    wfs = {t: weight_fn(t) for t in tids}

    def pooled_curve(kind: str) -> dict:
        ents = pooled_order({t: entries(t, kind, pooled=True) for t in tids})
        out = {}
        for n in se["n_grid"]:
            if 1 <= n <= len(ents):
                sub = ents[:n]
                # weighted union with per-trace weights: rows of different traces never coincide
                parts = {}
                for e in sub:
                    parts.setdefault(e["tid"], []).append(e)
                us = [weighted_union(v, wfs[t]) for t, v in parts.items()]
                out[int(n)] = {"count": sum(u["count"] for u in us), "weighted": sum(u["weighted"] for u in us)}
        return out

    pooled = {kind: pooled_curve(kind) for kind in U}
    pooled_slopes = {kind: local_slope(pooled[kind], POOLED_SLOPE_PAIRS) for kind in U}

    # ---- Cost_CCT and speed-ups ---------------------------------------------------------------
    notes = []
    cost, sp = {}, {}
    rowsec = {}
    for tid in tids:
        n_out = len(ordered[tid])
        L_gen = int(tr_by[tid].get("n_generated") or 0)
        u_end = U["realistic"][tid].get(n_out, {}).get("weighted", NAN)
        o_end = U["oracle"][tid].get(n_out, {}).get("weighted", NAN)
        s_r, s_o = slopes["realistic"][tid]["min"], slopes["oracle"][tid]["min"]
        meas_row = timing["traces"][tid]["seconds"][-1] / TIMING_BATCH
        c = None
        if stage_d is not None:
            try:
                rs = per_trace(stage_d.get("row_seconds_full"), tid)
                c = cost_cct(per_trace(stage_d["m_trace"], tid), per_trace(stage_d["kappa"], tid),
                             stage_d["hub_rate_by_kind"], stage_d["n_sources_by_kind"][tid], se["check_probes"],
                             stage_d["decode_seconds_per_column"], per_trace(stage_d["n_cols_after_screening"], tid),
                             rs)
                c["row_seconds_full"] = float(rs)
            except (KeyError, TypeError, ValueError, AttributeError, ZeroDivisionError) as e:
                notes.append(f"{tid}: Stage D inputs incomplete ({type(e).__name__}: {e})")
                c = None
        cost[tid] = c
        rowsec[tid] = c["row_seconds_full"] if c else meas_row
        u_proj = projected_union(u_end, n_out, L_gen, s_r)
        o_proj = projected_union(o_end, n_out, L_gen, s_o)
        cc = c["cost_cct"] if c else NAN
        sp[tid] = {"n_outputs": n_out, "L_gen": L_gen, "local_slope": s_r, "local_slope_oracle": s_o,
                   "U_trunc_end": u_end, "U_oracle_end": o_end, "U_trunc_proj": u_proj, "U_oracle_proj": o_proj,
                   "row_seconds_measured": meas_row,
                   "S_meas": u_end / cc if c else NAN, "S_proj": u_proj / cc if c else NAN,
                   "S_oracle_proj": o_proj / cc if c else NAN}
        if n_out < 17:
            notes.append(f"{tid}: only {n_out} of 17 outputs available")
    have = [t for t in tids if cost[t] is not None]

    def pooled_ratio(num_key: str) -> float:
        if not have:
            return NAN
        num = sum(sp[t][num_key] * rowsec[t] for t in have)
        den = sum(cost[t]["cost_cct"] * rowsec[t] for t in have)
        return num / den if den > 0 else NAN

    pooled_s = {"S_meas": pooled_ratio("U_trunc_end"), "S_proj": pooled_ratio("U_trunc_proj"),
                "S_oracle_proj": pooled_ratio("U_oracle_proj")}
    if stage_d is None:
        status = "not_run"
        notes.insert(0, "stage_D.json absent: Cost_CCT cannot be computed (unions and timings reported)")
        ev = {"conditions": None, "n_traces_slope_ok": None}
    elif not have:
        status = "not_run"
        notes.insert(0, "no trace has complete Stage D inputs")
        ev = {"conditions": None, "n_traces_slope_ok": None}
    else:
        ev = pr10_evaluate({t: sp[t] for t in have}, pooled_s, thr)
        status = ev["status"]
        if len(have) < len(tids):
            notes.append(f"Cost_CCT available for {len(have)} of {len(tids)} traces")
    stop = status == "kill"
    reason = "PR10 kill: S_proj < 3 on every trace; the whole-trace headline is dead" if stop else None

    values = {"local_slope_by_trace": {t: sp[t]["local_slope"] for t in tids},
              "local_slope_oracle_by_trace": {t: sp[t]["local_slope_oracle"] for t in tids},
              "n_traces_slope_ok": ev.get("n_traces_slope_ok"),
              "S_meas": pooled_s["S_meas"], "S_proj": pooled_s["S_proj"], "S_oracle_proj": pooled_s["S_oracle_proj"],
              "S_meas_by_trace": {t: sp[t]["S_meas"] for t in tids},
              "S_proj_by_trace": {t: sp[t]["S_proj"] for t in tids},
              "S_oracle_proj_by_trace": {t: sp[t]["S_oracle_proj"] for t in tids},
              "cost_cct_by_trace": {t: (cost[t] or {}).get("cost_cct") for t in tids},
              "U_trunc_51_weighted": pooled["realistic"].get(51, {}).get("weighted"),
              "U_oracle_51_weighted": pooled["oracle"].get(51, {}).get("weighted"),
              "pass_conditions": ev.get("conditions")}
    res = {
        "run": run_record({"stage_e": se, "c_grid": c_grid, "fidelity": fid, "model_set": model_set,
                           "tiny": args.tiny}, cfg["seed"], time.perf_counter() - t_start),
        "stage": "E",
        "premises": {"PR10": {"status": status, "values": values, "thresholds": thr, "notes": "; ".join(notes)}},
        "stop": stop, "stop_reason": reason,
        "gpu_seconds": float(timing.get("gpu_seconds", 0.0)),
        "detail": {
            "model_set": model_set,
            "U_trunc": {**{t: {str(n): v["weighted"] for n, v in U["realistic"][t].items()} for t in tids},
                        "pooled": {str(n): v["weighted"] for n, v in pooled["realistic"].items()}},
            "U_oracle": {**{t: {str(n): v["weighted"] for n, v in U["oracle"][t].items()} for t in tids},
                         "pooled": {str(n): v["weighted"] for n, v in pooled["oracle"].items()}},
            "U_trunc_count": {**{t: {str(n): v["count"] for n, v in U["realistic"][t].items()} for t in tids},
                              "pooled": {str(n): v["count"] for n, v in pooled["realistic"].items()}},
            "U_oracle_count": {**{t: {str(n): v["count"] for n, v in U["oracle"][t].items()} for t in tids},
                               "pooled": {str(n): v["count"] for n, v in pooled["oracle"].items()}},
            "local_slope": {t: slopes["realistic"][t] for t in tids},
            "local_slope_oracle": {t: slopes["oracle"][t] for t in tids},
            "pooled_local_slope": pooled_slopes["realistic"], "pooled_local_slope_oracle": pooled_slopes["oracle"],
            "per_trace": sp, "cost_cct": cost, "check_probes": se["check_probes"],
            "L": {t: sp[t]["L_gen"] for t in tids},
            "cost_weights": {t: {"lengths": timing["traces"][t]["lengths"], "seconds": timing["traces"][t]["seconds"],
                                 "row_equiv": [s / timing["traces"][t]["seconds"][-1]
                                               for s in timing["traces"][t]["seconds"]]} for t in tids},
            "epsilon_star": (stage_c or {}).get("epsilon_star") if isinstance(stage_c, dict) else None,
            "stage_d_present": stage_d is not None,
            "per_output": {t: [{k: a[k] for k in ("position", "output_kind", "sentence_index", "c", "c_used",
                                                  "n_feature_rows", "n_logits", "budget")}
                               | {"n_oracle": len(a["oracle"]), "curve": a["curve"]} for a in ordered[t]]
                           for t in tids},
            "definitions": {"per_trace_n": PER_TRACE_N, "per_trace_slope_pairs": PER_TRACE_SLOPE_PAIRS,
                            "pooled_n": se["n_grid"], "pooled_slope_pairs": POOLED_SLOPE_PAIRS,
                            "weighting": "row at its shortest selecting prefix; logit rows at their output's prefix",
                            "pooled_ratios": "seconds (row-equivalents x each trace's full-length row time)"},
        },
    }
    atomic_write_json(out_path, res)
    log.info(f"PR10 {status}: slopes {values['local_slope_by_trace']} S_meas={values['S_meas']} "
             f"S_proj={values['S_proj']} S_oracle_proj={values['S_oracle_proj']} -> {out_path}")
    if stop:
        log.warning(f"STOP: {reason}")
        sys.exit(EXIT_STOP)


if __name__ == "__main__":
    main()
