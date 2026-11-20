"""G0-B steps 3 and 4 (PROJECT_V2.md Section 7): sampled exact columns at trace length and screening
recall. For each committed trace (results/gate0/traces.json, written once by scripts/gen_traces.py):
  - operator on the full trace (targets: every active feature at every position, plus the logits at the
    last position; sources: every active feature, error and token);
  - n_sources sources stratified by kind (feature, error, token) and position (first token, early,
    middle, late), exact columns by forward mode;
  - closed-form A_0 columns, K0.7 on the trace (A_0 rows against attention-detached rows);
  - s95 of the columns of A and of A - A_0 over feature targets; k_eff_trace (energy-weighted median
    remainder s95, pooled over traces); hub fraction (remainder s95 > 10 k_eff_trace);
  - attention-rollout candidate sets Omega_s at each tau and their energy recall;
  - timing at trace length: row seconds and kappa.
Writes results/gate0/g0b_columns.json (tiny: results/tiny/gate0/), resumable per trace.
"""
from __future__ import annotations

import os
import time

import torch

from cct import scriptutil as su
from cct.gate0_structure import rel_err
from cct.killtest import s95, weighted_median
from cct.local_part import from_circuit_tracer, from_tiny
from cct.operators.base import KIND_ERROR, KIND_FEATURE, KIND_LOGIT, KIND_TOKEN
from cct.reach import (candidate_mask, head_weights, rollout_layers, screening_recall, source_write_layers,
                       tiny_rollout)
from cct.utils.io import atomic_write_json, read_json
from cct.utils.logging import get_logger, run_record
from cct.utils.timing import median_time

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0.yaml")
KINDS = {"feature": KIND_FEATURE, "error": KIND_ERROR, "token": KIND_TOKEN}


def trace_tokens(tr: dict) -> torch.Tensor:
    for k in ("token_ids", "tokens", "ids", "input_ids"):
        if k in tr:
            return torch.as_tensor(tr[k]).long().flatten()
    raise KeyError(f"no token ids in trace record (keys: {sorted(tr)})")


def source_strata(sm, n_pos: int, kinds: list[str], buckets: list[str]) -> torch.Tensor:
    """Stratum id per source (kind-major: kinds[i] x buckets[j] -> i * len(buckets) + j), -1 if none.
    Position buckets: first = position 0, then thirds of positions 1..n_pos-1."""
    edges = torch.linspace(1, n_pos, 4).floor().long().tolist()

    def bucket_of(pos: torch.Tensor, b: str) -> torch.Tensor:
        if b == "first":
            return pos == 0
        i = {"early": 0, "middle": 1, "late": 2}[b]
        return (pos >= edges[i]) & (pos < (edges[i + 1] if i < 2 else n_pos))

    out = torch.full((len(sm),), -1, dtype=torch.long)
    for ki, k in enumerate(kinds):
        for bi, b in enumerate(buckets):
            out[(sm.kind == KINDS[k]) & bucket_of(sm.pos, b)] = ki * len(buckets) + bi
    return out


def stratified_sources(sm, n_pos: int, n: int, kinds: list[str], buckets: list[str], gen: torch.Generator) -> torch.Tensor:
    """Equal share per (kind, position bucket) stratum; strata with too few nodes give their share to the
    others. Position buckets: first = position 0, then thirds of positions 1..n_pos-1."""
    sid = source_strata(sm, n_pos, kinds, buckets)
    strata = []
    for k in range(len(kinds) * len(buckets)):
        sel = torch.nonzero(sid == k).flatten()
        # zero-vector sources (position 0 features and errors are zeroed) are useless; keep tokens
        strata.append(sel[torch.randperm(sel.numel(), generator=gen)])
    picked: list[torch.Tensor] = [s[:0] for s in strata]
    remaining = n
    live = [i for i, s in enumerate(strata) if s.numel()]
    while remaining > 0 and live:
        share = max(1, remaining // len(live))
        nxt = []
        for i in live:
            have = picked[i].numel()
            take = strata[i][have: have + share]
            if take.numel() == 0:
                continue
            take = take[: remaining]
            picked[i] = torch.cat([picked[i], take])
            remaining -= take.numel()
            if picked[i].numel() < strata[i].numel():
                nxt.append(i)
            if remaining <= 0:
                break
        live = nxt
    return torch.sort(torch.cat(picked)).values


def column_stats(A_cols: torch.Tensor, A0_cols: torch.Tensor, feat_t: torch.Tensor) -> dict:
    A, R = A_cols.double(), (A_cols - A0_cols).double()
    sA, sR = s95(A[feat_t]), s95(R[feat_t])
    wA, wR = A[feat_t].pow(2).sum(0), R[feat_t].pow(2).sum(0)
    return {"s95_A": sA, "s95_R": sR, "wA": wA, "wR": wR,
            "a0_energy_share": 1.0 - float(R.pow(2).sum()) / max(float(A.pow(2).sum()), 1e-300)}


def run_trace(args, cfg, tr: dict, model, gen, log) -> dict:
    c = cfg["g0b"]["columns"]
    toks = trace_tokens(tr)
    t0 = time.perf_counter()
    if args.tiny:
        from cct.operators.tiny import TinyOperator

        op = TinyOperator(model, toks, batch_size=c["batch"])
        lp = from_tiny(op)
        R = tiny_rollout(op)
        transcoder = model.cfg.transcoder
        L = model.cfg.n_layers
        dtype_name = "float64"
        op_det = TinyOperator(model, toks, batch_size=c["batch"], detach_attn=True)
    else:
        from cct.operators.circuit_tracer_backend import CircuitTracerOperator

        mc = su.model_config(cfg["g0b"]["model_set"])
        dtype_name = mc["dtype"] or "float32"
        op = CircuitTracerOperator(model, toks, batch_size=c["forward_batch_size"])
        op.batch_size = c["batch"]
        lp = from_circuit_tracer(op)
        Wv, Wo = op.ov_weights()
        R = rollout_layers(op.attention_patterns(), head_weights(Wv, Wo))
        transcoder = mc["transcoder_type"]
        L = op.n_layers
        op_det = None
    n_pos = int(op.n_pos)
    src = stratified_sources(op.source_meta, n_pos, c["n_sources"], c["strata_kinds"], c["strata_positions"], gen)
    # exact columns (forward mode, fallback double backward)
    try:
        A_cols = op.exact_cols(src).T.double().cpu()  # [T, S]
        mode = "fwad"
    except Exception as e:  # noqa: BLE001
        log.info(f"forward AD failed ({e!r}); using double backward")
        op.forward_mode = "double_backward"
        A_cols = op.exact_cols(src).T.double().cpu()
        mode = "double_backward"
    A0_cols = lp.cols(src).double().cpu()
    feat_t = op.target_meta.kind != KIND_LOGIT
    st = column_stats(A_cols, A0_cols, feat_t)
    # screening sets
    wl_all = source_write_layers(op.source_meta.subset(src), transcoder, L)
    screen = {}
    for tau in c["taus"]:
        mask = candidate_mask(R, wl_all, op.source_meta.pos[src], op.target_meta.layer, op.target_meta.pos, float(tau))
        mask = mask.cpu()
        full = screening_recall(A_cols, mask)
        rem = screening_recall(A_cols - A0_cols, mask)
        screen[str(float(tau))] = {"recall": full["recall"], "recall_remainder": rem["recall"],
                                   "recall_median_per_column": full["recall_median_per_column"],
                                   "omega_frac_T": full["omega_frac_T"]}
        del mask
    # timing at trace length
    b = op.batch_size
    ridx = torch.randperm(op.T, generator=gen)[:b]
    tr_t, _ = median_time(lambda: op.exact_rows(ridx), 3)
    tc_t, _ = median_time(lambda: op.exact_cols(src[:b]), 3)
    nb_c = min(b, src.numel())
    timing = {"row_seconds": tr_t / ridx.numel(), "kappa": (tc_t / nb_c) / (tr_t / ridx.numel()), "batch": b,
              "forward_mode": mode}
    # K0.7 on the trace
    kidx = torch.randperm(op.T, generator=gen)[: c["k07_rows"]]
    A0_k = lp.rows(kidx).double().cpu()
    T, N = int(op.T), int(op.N)
    if not args.tiny:
        from cct.operators.circuit_tracer_backend import CircuitTracerOperator

        del lp, op
        torch.cuda.empty_cache()
        op_det = CircuitTracerOperator(model, toks, batch_size=c["k07_rows"], detach_attention=True)
    k07 = rel_err(A0_k, op_det.exact_rows(kidx))
    del op_det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"T": T, "N": N, "n_pos": n_pos, "n_generated": tr.get("n_generated"), "n_sampled": int(src.numel()),
            "sources": src.tolist(), "dtype": dtype_name, "k07_rel_err": k07, "timing": timing,
            "screen_recall": screen, "a0_energy_share": st["a0_energy_share"],
            "s95_A": st["s95_A"].tolist(), "s95_R": st["s95_R"].tolist(), "wA": st["wA"].tolist(),
            "wR": st["wR"].tolist(), "source_kind": op.source_meta.kind[src].tolist(),
            "source_write_layer": [min(w) for w in wl_all],
            "seconds": time.perf_counter() - t0}


def pool(per_trace: dict, prereg: dict) -> dict:
    sA = torch.tensor(sum((v["s95_A"] for v in per_trace.values()), []), dtype=torch.float64)
    sR = torch.tensor(sum((v["s95_R"] for v in per_trace.values()), []), dtype=torch.float64)
    wA = torch.tensor(sum((v["wA"] for v in per_trace.values()), []), dtype=torch.float64)
    wR = torch.tensor(sum((v["wR"] for v in per_trace.values()), []), dtype=torch.float64)
    k_eff = weighted_median(sR, wR)
    hub_k = prereg["g0b"]["hub_s95_over_k_eff_trace_above"]
    taus = sorted({t for v in per_trace.values() for t in v["screen_recall"]}, key=float, reverse=True)
    screen = {}
    for t in taus:
        rs = [v["screen_recall"][t] for v in per_trace.values() if t in v["screen_recall"]]
        # energy-weighted pooling across traces by the traces' column energy
        ws = [sum(v["wA"]) for v in per_trace.values() if t in v["screen_recall"]]
        wr = [sum(v["wR"]) for v in per_trace.values() if t in v["screen_recall"]]
        tot, totr = sum(ws) or 1.0, sum(wr) or 1.0
        screen[t] = {"recall": sum(r["recall"] * w for r, w in zip(rs, ws)) / tot,
                     "recall_remainder": sum(r["recall_remainder"] * w for r, w in zip(rs, wr)) / totr,
                     "omega_frac_T": sum(r["omega_frac_T"] for r in rs) / len(rs)}
    med = lambda xs: float(torch.tensor(xs, dtype=torch.float64).median())  # noqa: E731
    return {
        "T": int(med([v["T"] for v in per_trace.values()])), "N": int(med([v["N"] for v in per_trace.values()])),
        "n_sampled": int(sR.numel()), "k07_rel_err": max(v["k07_rel_err"] for v in per_trace.values()),
        "dtype": next(iter(per_trace.values()))["dtype"],
        "s95_A_ewm": weighted_median(sA, wA), "s95_rem_ewm": k_eff, "k_eff_trace": k_eff,
        "hub_fraction": float((sR > hub_k * k_eff).double().mean()) if sR.numel() else float("nan"),
        "a0_energy_share": med([v["a0_energy_share"] for v in per_trace.values()]),
        "screen_recall": screen,
        "timing": {"row_seconds": med([v["timing"]["row_seconds"] for v in per_trace.values()]),
                   "kappa": med([v["timing"]["kappa"] for v in per_trace.values()])},
    }


def main() -> None:
    p = su.parser("Gate 0, G0-B steps 3 and 4: sampled exact columns and screening recall")
    p.set_defaults(config=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    prereg = su.load_yaml(os.path.join(su.ROOT, "configs", "prereg_gate0.yaml"))
    log = get_logger("gate0_columns", os.path.join(su.ROOT, "logs"))
    out_dir = os.path.join(su.results_dir(args), "gate0")
    os.makedirs(os.path.join(out_dir, "g0b_columns"), exist_ok=True)
    traces_path = os.path.join(out_dir, "traces.json") if args.tiny else os.path.join(su.ROOT, cfg["g0b"]["traces_file"])
    traces = read_json(traces_path)
    traces = traces["traces"] if isinstance(traces, dict) and "traces" in traces else traces
    gen = torch.Generator().manual_seed(cfg["seed"])
    t_start = time.perf_counter()
    model = None
    if args.tiny:
        kt = su.load_yaml(os.path.join(su.ROOT, "configs", "experiments", "kill_test.yaml"))
        model = su.tiny_model(kt, "tiny_plt")
        cfg["g0b"]["columns"].update({"n_sources": 40, "batch": 16, "k07_rows": 16})
    per_trace = {}
    for tr in traces:
        tid = tr["id"]
        path = os.path.join(out_dir, "g0b_columns", f"{tid}.json")
        if os.path.exists(path) and not args.force:
            per_trace[tid] = read_json(path)
            continue
        if model is None:
            from cct.operators.circuit_tracer_backend import load_replacement_model
            from cct.utils.gpu import cap_memory, nvidia_smi

            log.info(nvidia_smi())
            mc = su.model_config(cfg["g0b"]["model_set"])
            cap_memory(mc["memory_cap_gb"])
            model = load_replacement_model(mc["model_name"], mc["transcoder_set"], dtype=su.torch_dtype(mc["dtype"]))
        res = run_trace(args, cfg, tr, model, gen, log)
        res["run"] = run_record(cfg, cfg["seed"], res["seconds"], trace_id=tid)
        atomic_write_json(path, res)
        tol = prereg["k07"]["fp32_rel_err_at_most" if res["dtype"] in ("float32", "float64") else "bf16_rel_err_at_most"]
        if not res["k07_rel_err"] <= tol:
            log.error(f"K0.7 FAILED on {tid} ({res['k07_rel_err']:.3e} > {tol}): Gate 0 stops (PREREG_GATE0)")
            raise SystemExit(3)
        per_trace[tid] = res
        log.info(f"{tid}: T={res['T']} N={res['N']} k07={res['k07_rel_err']:.2e} "
                 f"recall={ {t: round(v['recall'], 4) for t, v in res['screen_recall'].items()} }")
    out = pool(per_trace, prereg)
    out["per_trace"] = {k: {kk: v[kk] for kk in ("T", "N", "n_pos", "n_generated", "n_sampled", "k07_rel_err",
                                                 "timing", "screen_recall", "a0_energy_share", "seconds")}
                        for k, v in per_trace.items()}
    out["run"] = run_record(cfg, cfg["seed"], time.perf_counter() - t_start)
    atomic_write_json(os.path.join(out_dir, "g0b_columns.json"), out)
    log.info(f"k_eff_trace={out['k_eff_trace']:.1f} hub_fraction={out['hub_fraction']:.3f}")


if __name__ == "__main__":
    main()
