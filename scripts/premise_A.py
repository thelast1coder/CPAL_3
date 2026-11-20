"""Gate 0, Stage A: foundations (GATE0_PREMISES.md Section 3; thresholds in configs/prereg_gate0.yaml).

PR1  mixed passes: on Gemma PLT, 2 short prompts, 5 random-sign mixes over all targets against phi times
     the exact rows; on Qwen3-1.7B, the 128-token prefix of the first committed trace, 5 mixes over 256
     random targets; timing of one mixed pass against one single-target pass (same batch shape) at
     prefix lengths 128 and 300.
PR2  one table: for 3 prefix lengths of the first trace, exact rows of 50 random feature targets on the
     prefix against the same nodes' rows in the full-trace frozen pass (sources aligned by identity;
     sources only in the full trace must carry zero).
PR3  closed-form A_0 against every row of the attention-detached pass on 2 short prompts per set
     (Gemma PLT, Gemma CLT, Qwen3); A_0 energy share; reduction of the energy-weighted median s95.
     The full exact matrices computed here are saved where Stage C reads them
     (artifacts/<run>/<set>/premise_C/<prompt>.pt), so Stage C does not recompute them.

Resumable: one JSON per item under results/[tiny/]gate0/premise_A/<set>/; --force recomputes. Per set,
the model runs in the dtype of its config (fp32); if a set runs out of GPU memory in fp32 it is
restarted in bf16 (every item of the set, so dtypes are never mixed) and that is recorded.
Writes results/[tiny/]gate0/stage_A.json. Stop rule A: a PR1 or PR2 kill writes stop=true and exits
with code 2.
"""
from __future__ import annotations

import os
import sys
import time
import zlib

import torch

from cct import premise_ac as pa
from cct import scriptutil as su
from cct.utils.io import atomic_write_json, read_json, save_pt
from cct.utils.logging import get_logger, run_record

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")
# Tiny run only (numbers meaningless): the tiny trace has about 28 tokens.
TINY_A = {"pr1_trace_prefix": 16, "timing_lengths": [12, 24], "timing_repeats": 3}


def gen_for(seed: int, *key) -> torch.Generator:
    return torch.Generator().manual_seed(seed + zlib.crc32("/".join(map(str, key)).encode()) % 100_000)


def res_root(args) -> str:
    d = os.path.join(su.results_dir(args), "gate0")
    os.makedirs(d, exist_ok=True)
    return d


def pr2_prefix_lengths(n_prompt: int, n_total: int, k: int) -> list[int]:
    """k prefix lengths spread over the generated part (quartiles for k = 3)."""
    gen = max(1, n_total - n_prompt)
    qs = sorted({min(n_total - 1, max(n_prompt + 1, n_prompt + round(gen * (i + 1) / (k + 1)))) for i in range(k)})
    return qs


# -------------------------------------------------------------------------------------------------
# items
# -------------------------------------------------------------------------------------------------
def item_short(be: pa.Backend, args, cfg, p: dict, do_pr1: bool, art_dir: str, log) -> dict:
    """Full exact matrix, PR1 mixes (Gemma PLT), A_0 against the detached pass (PR3), structure."""
    A_ = cfg["stage_a"]
    ex = cfg["exact"]
    gen = gen_for(cfg["seed"], be.set_name, p["id"], "short")
    inp = pa.prompt_input(p)
    A, A0, tm, sm, info, op = pa.full_matrix_and_a0(be, inp, ex["batch_size_short"])
    out = {"prompt_id": p["id"], "prompt": p.get("prompt"), **info}
    if do_pr1:
        out["pr1"] = pa.mix_check(op, A_["pr1_mixes"], gen, A=A)
    del op
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t0 = time.perf_counter()
    op_det = be.op(inp, ex["batch_size_short"], detach=True)
    if (op_det.T, op_det.N) != (A.shape[0], A.shape[1]):
        raise RuntimeError(f"detached operator has a different node set {op_det.T, op_det.N} vs {tuple(A.shape)}")
    A_det = pa.all_rows(op_det)
    del op_det
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    row_err = ((A0.double() - A_det.double()).norm(dim=1) / A_det.double().norm(dim=1).clamp_min(1e-300))
    nz = A_det.double().norm(dim=1) > 0
    out["pr3"] = {"rel_err": pa.rel_err(A0, A_det), "max_row_rel_err": float(row_err[nz].max()) if bool(nz.any()) else 0.0,
                  "n_rows": int(A.shape[0]), "detached_seconds": time.perf_counter() - t0}
    st = pa.structure_AR(A, A0, tm, sm)
    out["pr3"].update({"a0_energy_share": st["a0_energy_share"], "s95_sq_reduction": st["s95_sq_reduction"],
                       "s95_abs_reduction": st["s95_abs_reduction"]})
    out["structure"] = st
    rec = pa.build_record(A, A0, tm, sm, {"node_threshold": ex["node_threshold"], "edge_threshold": ex["edge_threshold"]},
                          {"set": be.set_name, "prompt_id": p["id"], "prompt": p.get("prompt"), "dtype": be.dtype_name,
                           "tokens": inp if not isinstance(inp, str) else None, "source": "premise_A", **info})
    save_pt(os.path.join(art_dir, f"{p['id']}.pt"), rec)
    log.info(f"{be.set_name} {p['id']}: T={info['T']} N={info['N']} pr3_rel={out['pr3']['rel_err']:.2e} "
             f"a0_share={st['a0_energy_share']:.3f}" + (f" pr1_max={out['pr1']['rel_err_max']:.2e}" if do_pr1 else ""))
    return out


def item_trace_mix(be: pa.Backend, cfg, A_: dict, trace: dict, log) -> dict:
    ex = cfg["exact"]
    gen = gen_for(cfg["seed"], be.set_name, trace["id"], "trace_mix")
    q = min(A_["pr1_trace_prefix"], len(trace["token_ids"]))
    op = be.op(trace["token_ids"][:q], ex["batch_size_trace"])
    tgt = torch.randperm(op.T, generator=gen)[: min(A_["pr1_trace_targets"], op.T)]
    out = pa.mix_check(op, A_["pr1_mixes"], gen, targets=tgt)
    out.update({"prefix_tokens": q, "n_pos": int(op.n_pos), "T": int(op.T), "N": int(op.N), "trace": trace["id"]})
    del op
    log.info(f"{be.set_name} trace mix (prefix {q}): max rel err {out['rel_err_max']:.2e}")
    return out


def item_timing(be: pa.Backend, cfg, A_: dict, trace: dict, length: int, log) -> dict:
    ex = cfg["exact"]
    gen = gen_for(cfg["seed"], be.set_name, trace["id"], "timing", length)
    q = min(length, len(trace["token_ids"]))
    op = be.op(trace["token_ids"][:q], ex["batch_size_trace"])
    out = pa.time_ratio(op, ex["batch_size_trace"], A_["timing_repeats"], gen)
    out.update({"length_requested": length, "prefix_tokens": q, "n_pos": int(op.n_pos), "trace": trace["id"]})
    if torch.cuda.is_available():
        out["peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    del op
    log.info(f"{be.set_name} timing at {q} tokens: mixed {out['mixed_seconds']:.4f}s one-hot "
             f"{out['onehot_seconds']:.4f}s ratio {out['ratio']:.3f}")
    return out


def item_pr2(be: pa.Backend, cfg, A_: dict, trace: dict, log) -> dict:
    ex = cfg["exact"]
    ids = trace["token_ids"]
    qs = pr2_prefix_lengths(trace["n_prompt_tokens"], len(ids), A_["pr2_positions"])
    parts, offs = [], []
    for q in qs:
        gen = gen_for(cfg["seed"], be.set_name, trace["id"], "pr2", q)
        op = be.op(ids[:q], ex["batch_size_trace"])
        offs.append(int(op.n_pos) - q)
        part = pa.pr2_prefix_part(op, A_["pr2_targets"], gen)
        part["q"] = q
        parts.append(part)
        del op
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    op_full = be.op(ids, ex["batch_size_trace"])
    off_full = int(op_full.n_pos) - len(ids)
    if any(o != off_full for o in offs):
        raise RuntimeError(f"position offsets differ between prefixes {offs} and the full trace {off_full}")
    fulls = pa.pr2_full_rows(op_full, parts)
    comps = []
    for part, full in zip(parts, fulls):
        c = pa.pr2_compare(part, full, op_full.source_meta)
        c.update({"prefix_tokens": part["q"], "n_pos_prefix": part["n_pos"]})
        comps.append(c)
        log.info(f"PR2 prefix {part['q']}: rel err {c['rel_err']:.2e}, only-full mass {c['only_full_rel_mass']:.2e}, "
                 f"missing targets {c['n_targets_missing_in_full']}")
    out = {"trace": trace["id"], "n_total_tokens": len(ids), "n_pos_full": int(op_full.n_pos),
           "position_offset": off_full, "comparisons": comps, "T_full": int(op_full.T), "N_full": int(op_full.N)}
    del op_full
    return out


# -------------------------------------------------------------------------------------------------
# aggregation and statuses
# -------------------------------------------------------------------------------------------------
def aggregate(args, cfg, pre, items: dict, dtype_by_set: dict, load_seconds: float, t_start: float) -> dict:
    p1, p2, p3 = pre["stage_a"]["pr1"], pre["stage_a"]["pr2"], pre["stage_a"]["pr3"]
    sets = {r: pa.role_set(cfg, r, args.tiny) for r in pa.ROLES}
    dt = lambda r: dtype_by_set.get(sets[r], "float32")  # noqa: E731

    # PR1
    gp = sets["gemma_plt"]
    gemma = {k: v["pr1"]["rel_err"] for (s, k), v in items.items() if s == gp and k.startswith("short_") and "pr1" in v}
    tm_ = items.get((sets["qwen"], "trace_mix"))
    timing = {k: v for (s, k), v in items.items() if s == sets["qwen"] and k.startswith("timing_")}
    tol_g = pa.tol_for(dt("gemma_plt"), p1["rel_err_fp32_at_most"], p1["rel_err_bf16_at_most"])
    tol_q = pa.tol_for(dt("qwen"), p1["rel_err_fp32_at_most"], p1["rel_err_bf16_at_most"])
    g_errs = [e for v in gemma.values() for e in v]
    t_errs = tm_["rel_err"] if tm_ else []
    ratios = [v["ratio"] for v in timing.values()]
    s1, n1 = pa.pr1_status(g_errs, t_errs, ratios, tol_g, tol_q, p1["time_ratio_at_most"])
    if s1 != "not_run" and (len(gemma) < cfg["stage_a"]["pr1_prompts"] or len(timing) < len(p1["timing_lengths"])):
        n1 += " (some PR1 items missing)"
    PR1 = pa.premise(s1, {"gemma_plt_mix_rel_err": gemma, "gemma_plt_mix_rel_err_max": max(g_errs) if g_errs else None,
                          "qwen_trace_mix_rel_err": t_errs, "qwen_trace_mix_rel_err_max": max(t_errs) if t_errs else None,
                          "qwen_trace_prefix_tokens": tm_["prefix_tokens"] if tm_ else None,
                          "time_ratio": {v["prefix_tokens"]: v["ratio"] for v in timing.values()},
                          "time_ratio_max": max(ratios) if ratios else None,
                          "timing_batch": next(iter(timing.values()))["batch"] if timing else None},
                     {"rel_err_at_most_gemma": tol_g, "rel_err_at_most_qwen": tol_q,
                      "time_ratio_at_most": p1["time_ratio_at_most"]}, n1)

    # PR2
    it2 = items.get((sets["qwen"], "pr2"))
    comps = it2["comparisons"] if it2 else []
    tol2 = pa.tol_for(dt("qwen"), p2["rel_err_fp32_at_most"], p2["rel_err_bf16_at_most"])
    s2, n2 = pa.pr2_status(comps, tol2)
    PR2 = pa.premise(s2, {"rel_err": {c["prefix_tokens"]: c["rel_err"] for c in comps},
                          "rel_err_max": max((c["rel_err"] for c in comps), default=None),
                          "only_full_rel_mass_max": max((c["only_full_rel_mass"] for c in comps), default=None),
                          "targets_missing_in_full": sum(c["n_targets_missing_in_full"] for c in comps)},
                     {"rel_err_at_most": tol2}, n2)

    # PR3
    errs, shares, reds, reds_abs, per = [], [], [], [], {}
    tol3_by = {}
    for r in pa.ROLES:
        s = sets[r]
        tol3_by[s] = pa.tol_for(dt(r), p3["rel_err_at_most"], p2["rel_err_bf16_at_most"])
        for (ss, k), v in items.items():
            if ss == s and k.startswith("short_"):
                per.setdefault(s, {})[v["prompt_id"]] = {
                    "rel_err": v["pr3"]["rel_err"], "a0_energy_share": v["pr3"]["a0_energy_share"],
                    "s95_sq_reduction": v["pr3"]["s95_sq_reduction"], "s95_abs_reduction": v["pr3"]["s95_abs_reduction"]}
                errs.append(v["pr3"]["rel_err"] / tol3_by[s])  # normalised so one tolerance (1) applies
                shares.append(v["pr3"]["a0_energy_share"])
                reds.append(v["pr3"]["s95_sq_reduction"])
                reds_abs.append(v["pr3"]["s95_abs_reduction"])
    s3, n3, info3 = pa.pr3_status(errs, 1.0, shares, reds, p3["informative_drop_local_part_if"])
    if s3 != "not_run" and len(per) < 3:
        n3 += f" (only {len(per)} of 3 sets measured)"
    info3["s95_abs_reduction_median"] = float(torch.tensor(reds_abs).median()) if reds_abs else None
    PR3 = pa.premise(s3, {"by_set": per, **info3},
                     {"rel_err_at_most_by_set": tol3_by, "informative_drop_local_part_if": p3["informative_drop_local_part_if"]},
                     n3)

    stop = s1 == "kill" or s2 == "kill"
    reason = None
    if stop:
        reason = "; ".join(f"{n} kill: {t}" for n, s_, t in (("PR1", s1, n1), ("PR2", s2, n2)) if s_ == "kill")
    gpu_s = load_seconds + sum(v.get("seconds", 0.0) for v in items.values())
    detail = {f"{s}/{k}": v for (s, k), v in items.items()}
    return pa.stage_doc(run_record(cfg, cfg["seed"], time.perf_counter() - t_start, stage="A", tiny=args.tiny),
                        "A", {"PR1": PR1, "PR2": PR2, "PR3": PR3}, stop, reason, gpu_s, dtype_by_set,
                        pa.jsonable(detail))


# -------------------------------------------------------------------------------------------------
def main() -> int:
    p = su.parser("Gate 0, Stage A: PR1, PR2, PR3")
    p.set_defaults(config=DEFAULT_CONFIG)
    p.add_argument("--roles", default=",".join(pa.ROLES), help="subset of gemma_plt,gemma_clt,qwen")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    pre = su.load_yaml(os.path.join(su.ROOT, "configs", "prereg_gate0.yaml"))
    kt = su.load_yaml(os.path.join(su.ROOT, "configs", "experiments", "kill_test.yaml"))
    A_ = dict(cfg["stage_a"])
    # The prereg fixes the PR1 and PR2 sizes; the settings file must agree (the prereg wins).
    A_.update(pr1_mixes=pre["stage_a"]["pr1"]["n_mixes"], pr1_trace_prefix=pre["stage_a"]["pr1"]["trace_prefix_tokens"],
              pr1_trace_targets=pre["stage_a"]["pr1"]["trace_mix_targets"],
              timing_lengths=pre["stage_a"]["pr1"]["timing_lengths"], pr2_positions=pre["stage_a"]["pr2"]["n_positions"],
              pr2_targets=pre["stage_a"]["pr2"]["n_targets"])
    if args.tiny:
        A_.update(TINY_A)
    cfg["stage_a"] = A_
    log = get_logger("premise_A", os.path.join(su.ROOT, "logs"))
    root = res_root(args)
    item_dir = os.path.join(root, "premise_A")
    t_start = time.perf_counter()
    roles = [r for r in args.roles.split(",") if r]
    trace = pa.load_trace(cfg, args.tiny) if "qwen" in roles else None
    loads_path = os.path.join(item_dir, "model_loads.json")
    loads = read_json(loads_path) if os.path.exists(loads_path) and not args.force else {}
    dtype_path = os.path.join(item_dir, "dtype_by_set.json")
    dtype_by_set = read_json(dtype_path) if os.path.exists(dtype_path) else {}

    for role in roles:
        set_name = pa.role_set(cfg, role, args.tiny)
        prompts = pa.role_prompts(cfg, role, args.tiny, kt)[: A_["pr3_prompts"]]
        jobs = [(f"short_{pr['id']}", pr) for pr in prompts]
        if role == "qwen":
            jobs += [("trace_mix", None)] + [(f"timing_{L}", L) for L in A_["timing_lengths"]] + [("pr2", None)]
        dtypes = ["float64"] if args.tiny else [None, "bfloat16"]
        for attempt, dname in enumerate(dtypes):
            be = pa.Backend(set_name, args.tiny, cfg["exact"], kt, log, dtype_name=dname)
            if attempt > 0:
                log.warning(f"{set_name}: restarting every item in {be.dtype_name} after running out of memory")
            try:
                for name, arg in jobs:
                    path = os.path.join(item_dir, set_name, f"{name}.json")
                    if os.path.exists(path) and not args.force and read_json(path).get("dtype") == be.dtype_name:
                        log.info(f"skip {set_name} {name} (done)")
                        continue
                    t0 = time.perf_counter()
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    if name.startswith("short_"):
                        do_pr1 = role == "gemma_plt" and prompts.index(arg) < A_["pr1_prompts"]
                        out = item_short(be, args, cfg, arg, do_pr1, su.artifacts_dir(args, set_name, "premise_C"), log)
                    elif name == "trace_mix":
                        out = item_trace_mix(be, cfg, A_, trace, log)
                    elif name.startswith("timing_"):
                        out = item_timing(be, cfg, A_, trace, arg, log)
                    else:
                        out = item_pr2(be, cfg, A_, trace, log)
                    out.update(dtype=be.dtype_name, seconds=time.perf_counter() - t0, set=set_name, item=name,
                               peak_gb=(torch.cuda.max_memory_allocated() / 1024 ** 3) if torch.cuda.is_available() else 0.0,
                               run=run_record(cfg, cfg["seed"], time.perf_counter() - t0, set_name=set_name, item=name))
                    atomic_write_json(path, pa.jsonable(out))
                dtype_by_set[set_name] = be.dtype_name
                atomic_write_json(dtype_path, dtype_by_set)
                if be.notes:
                    dtype_by_set.setdefault("_notes", {})[set_name] = be.notes
                    atomic_write_json(dtype_path, dtype_by_set)
                loads[set_name] = loads.get(set_name, 0.0) + be.load_seconds
                atomic_write_json(loads_path, loads)
                be.free()
                break
            except Exception as e:  # noqa: BLE001
                loads[set_name] = loads.get(set_name, 0.0) + be.load_seconds
                be.free()
                if pa.is_oom(e) and attempt + 1 < len(dtypes):
                    log.error(f"{set_name}: out of memory in {be.dtype_name} ({e!r})")
                    continue
                raise

    # aggregate every finished item (also from earlier runs)
    items = {}
    for role in pa.ROLES:
        s = pa.role_set(cfg, role, args.tiny)
        d = os.path.join(item_dir, s)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith(".json"):
                v = read_json(os.path.join(d, fn))
                if v.get("dtype") == dtype_by_set.get(s, v.get("dtype")):
                    items[(s, fn[:-5])] = v
    doc = aggregate(args, cfg, pre, items, {k: v for k, v in dtype_by_set.items() if not k.startswith("_")},
                    sum(loads.values()), t_start)
    if dtype_by_set.get("_notes"):
        doc["notes"] = dtype_by_set["_notes"]
    out_path = os.path.join(root, "stage_A.json")
    atomic_write_json(out_path, doc)
    for k, v in doc["premises"].items():
        log.info(f"{k}: {v['status']} ({v['notes']})")
    log.info(f"wrote {out_path}")
    if doc["stop"]:
        log.error(f"STOP (rule A): {doc['stop_reason']}. Report and wait for Dawar.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
