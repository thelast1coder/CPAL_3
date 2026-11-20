"""Gate 0, Stage C: structure and recovery on short prompts with full exact matrices (GATE0_PREMISES.md
Section 3; thresholds in configs/prereg_gate0.yaml; definitions in PREREG_GATE0.md Section 2).

For each set (Gemma PLT, Gemma CLT-426k, Qwen3-1.7B PLT) and its 3 short prompts:
  1. the full exact matrix with every active feature as a target (all rows, batch 256) and the
     closed-form A_0, stored fp16 with per-row scale and checksum under
     artifacts/<run>/<set>/premise_C/<prompt>.pt (Stage A already wrote 2 of the 3 per set; they are
     reused when their dtype matches);
  2. PR5: per-column s95 (squared and absolute) of A and of A - A_0 as a fraction of each source's
     causal candidate set, energy-weighted medians, hub share by node kind;
  3. PR6: D1 (Rademacher + masked OMP, run_cct with local_part) and D2 (count-sketch, R = 5, decoded
     over the causal set, B // 4 kept) on the remainder, A_0 added back, logit rows exact; prune and
     score fidelity against the exact pruned graph; recovery time; per-column remainder errors.
Then epsilon* (Qwen prompts) and k_eff of the Qwen short prompts, which Stage D needs.

Resumable: one JSON per (set, prompt) under results/[tiny/]gate0/premise_C/<set>/; --force recomputes
(matrices too). dtype per set: the one Stage A recorded (results/.../stage_A.json), else the config.
Writes results/[tiny/]gate0/stage_C.json. Stop rule C: a PR5 or PR6 kill writes stop=true and exits
with code 2.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

from cct import premise_ac as pa
from cct import scriptutil as su
from cct.utils.io import atomic_write_json, load_pt, read_json, save_pt
from cct.utils.logging import get_logger, run_record

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")


def stage_a_dtypes(root: str) -> dict:
    p = os.path.join(root, "stage_A.json")
    if os.path.exists(p):
        return read_json(p).get("dtype_by_set") or {}
    return {}


def matrix_record(be: pa.Backend, p: dict, art: str, cfg: dict, force: bool, log) -> tuple[dict, float, bool]:
    """(record, GPU seconds spent, reused?)."""
    ex = cfg["exact"]
    if os.path.exists(art) and not force:
        rec = load_pt(art)
        if rec.get("dtype") == be.dtype_name:
            return rec, 0.0, True
        log.warning(f"{art}: stored dtype {rec.get('dtype')} != {be.dtype_name}; recomputing")
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    inp = pa.prompt_input(p)
    A, A0, tm, sm, info, op = pa.full_matrix_and_a0(be, inp, ex["batch_size_short"])
    del op
    if torch.cuda.is_available():
        info["peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        torch.cuda.empty_cache()
    rec = pa.build_record(A, A0, tm, sm, {"node_threshold": ex["node_threshold"], "edge_threshold": ex["edge_threshold"]},
                          {"set": be.set_name, "prompt_id": p["id"], "prompt": p.get("prompt"), "dtype": be.dtype_name,
                           "tokens": inp if not isinstance(inp, str) else None, "source": "premise_C", **info})
    save_pt(art, rec)
    return rec, time.perf_counter() - t0, False


def analyse(rec: dict, cfg: dict, pre: dict, ms: list[int], device: str, log) -> dict:
    C = cfg["stage_c"]
    A, A0, tm, sm = pa.record_arrays(rec)
    t0 = time.perf_counter()
    st = pa.structure_AR(A, A0, tm, sm)
    del A, A0
    runs = pa.recovery_runs(rec, ms, sorted(set(C["d1_seeds"]) | set(C["d2_seeds"])), C["d1"], C["d2_repetitions"],
                            rec["prune_cfg"], pre["fidelity"], device)
    # D1 and D2 use their own seed lists (identical in the settings file; kept general)
    runs = [r for r in runs if r["seed"] in (C["d1_seeds"] if r["decoder"] == "D1" else C["d2_seeds"])]
    table = pa.meets_table(runs, pre["fidelity"]["seeds_required"])
    rec_secs = {dec: {str(m): statistics.median([r["recovery_seconds"] for r in runs if r["decoder"] == dec and r["m"] == m])
                      for m in ms} for dec in ("D1", "D2")}
    return {"pr5": st, "pr6": {"runs": runs, "table": table, "median_recovery_seconds": rec_secs},
            "T_features": st["T_features"], "analysis_seconds": time.perf_counter() - t0}


def aggregate(args, cfg, pre, items: dict, sets: dict, dtype_by_set: dict, gpu_seconds: float, t_start: float,
              ms: list[int], stage_a_pr3: str | None) -> dict:
    p5, p6 = pre["stage_c"]["pr5"], pre["stage_c"]["pr6"]
    qset = sets["qwen"]
    # PR5
    fracs = {s: {pid: v["pr5"]["remainder"]["s95_abs_ewm_frac_causal"] for pid, v in d.items()} for s, d in items.items()}
    s5, n5 = pa.pr5_status(fracs, qset, p5)
    per5 = {s: {pid: {"A_s95_sq_frac": v["pr5"]["A"]["s95_sq_ewm_frac_causal"],
                      "A_s95_abs_frac": v["pr5"]["A"]["s95_abs_ewm_frac_causal"],
                      "rem_s95_sq_frac": v["pr5"]["remainder"]["s95_sq_ewm_frac_causal"],
                      "rem_s95_abs_frac": v["pr5"]["remainder"]["s95_abs_ewm_frac_causal"],
                      "rem_s95_abs_entries": v["pr5"]["remainder"]["s95_abs_ewm"],
                      "a0_energy_share": v["pr5"]["a0_energy_share"],
                      "T_features": v["T_features"], "N": v["pr5"]["N"],
                      "hub_share_remainder": v["pr5"]["hub_share_remainder"]}
                for pid, v in d.items()} for s, d in items.items()}
    if s5 != "not_run" and len(items) < 3:
        n5 += f" (only {len(items)} of 3 sets measured: the kill rule needs all three)"
    PR5 = pa.premise(s5, {"remainder_abs_s95_frac_causal": fracs, "by_set": per5},
                     {"pass": p5["pass"], "kill": p5["kill"]}, n5)
    # PR6 and epsilon*
    q = items.get(qset, {})
    tables = {pid: v["pr6"]["table"] for pid, v in q.items()}
    T_by = {pid: v["T_features"] for pid, v in q.items()}
    s6, n6, info6 = pa.pr6_status(tables, T_by, p6)
    if s6 != "not_run" and len(q) < 3:
        n6 += f" (only {len(q)} of 3 Qwen prompts measured)"
    runs_q = [dict(r, prompt=pid) for pid, v in q.items() for r in v["pr6"]["runs"]]
    eps = pa.epsilon_star(runs_q, pre["fidelity"]["seeds_required"], 2, "ew_mean")
    eps_alt = {k: pa.epsilon_star(runs_q, pre["fidelity"]["seeds_required"], 2, k)["epsilon_star"]
               for k in ("ew_median", "ew_q90")}
    fid_by = {s: {pid: v["pr6"]["table"] for pid, v in d.items()} for s, d in items.items()}
    rec_by = {s: {pid: v["pr6"]["median_recovery_seconds"] for pid, v in d.items()} for s, d in items.items()}
    PR6 = pa.premise(s6, {**info6, "fidelity_tables": fid_by, "median_recovery_seconds": rec_by,
                          "epsilon_star": eps["epsilon_star"], "epsilon_star_detail": eps,
                          "epsilon_star_alternatives": eps_alt, "m_grid": ms},
                     {"pass": p6["pass"], "kill": p6["kill"], "fidelity": pre["fidelity"]}, n6)
    k_eff_vals = [v["pr5"]["remainder"]["s95_abs_ewm"] for v in q.values()]
    k_eff = statistics.median(k_eff_vals) if k_eff_vals else None
    stop = s5 == "kill" or s6 == "kill"
    reason = "; ".join(f"{n} kill: {t}" for n, s_, t in (("PR5", s5, n5), ("PR6", s6, n6)) if s_ == "kill") or None
    detail = {f"{s}/{pid}": v for s, d in items.items() for pid, v in d.items()}
    return pa.stage_doc(run_record(cfg, cfg["seed"], time.perf_counter() - t_start, stage="C", tiny=args.tiny),
                        "C", {"PR5": PR5, "PR6": PR6}, stop, reason, gpu_seconds, dtype_by_set, pa.jsonable(detail),
                        epsilon_star=eps["epsilon_star"], k_eff_short_qwen=k_eff,
                        k_eff_short_qwen_by_prompt=dict(zip(q.keys(), k_eff_vals)),
                        stage_a_pr3_status=stage_a_pr3)


def main() -> int:
    p = su.parser("Gate 0, Stage C: full exact matrices on short prompts, PR5, PR6, epsilon*")
    p.set_defaults(config=DEFAULT_CONFIG)
    p.add_argument("--roles", default=",".join(pa.ROLES), help="subset of gemma_plt,gemma_clt,qwen")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    pre = su.load_yaml(os.path.join(su.ROOT, "configs", "prereg_gate0.yaml"))
    kt = su.load_yaml(os.path.join(su.ROOT, "configs", "experiments", "kill_test.yaml"))
    log = get_logger("premise_C", os.path.join(su.ROOT, "logs"))
    root = os.path.join(su.results_dir(args), "gate0")
    item_dir = os.path.join(root, "premise_C")
    os.makedirs(item_dir, exist_ok=True)
    t_start = time.perf_counter()
    ms = list(kt["tiny"]["cct_m"]) if args.tiny else list(pre["stage_c"]["pr6"]["m_grid"])
    if not args.tiny and ms != list(cfg["stage_c"]["m_grid"]):
        log.warning(f"settings m_grid {cfg['stage_c']['m_grid']} differs from the prereg {ms}; using the prereg")
    device = "cuda" if (torch.cuda.is_available() and not args.tiny) else "cpu"
    a_dt = stage_a_dtypes(root)
    a_path = os.path.join(root, "stage_A.json")
    stage_a_pr3 = read_json(a_path)["premises"]["PR3"]["status"] if os.path.exists(a_path) else None
    if stage_a_pr3 and stage_a_pr3 != "pass":
        log.warning(f"Stage A PR3 is {stage_a_pr3}: A_0 is subtracted and added back unchanged, so fidelity is "
                    "unaffected, but 'A_0 at zero passes' is not established")
    sets = {r: pa.role_set(cfg, r, args.tiny) for r in pa.ROLES}
    roles = [r for r in args.roles.split(",") if r]
    gpu_path = os.path.join(item_dir, "gpu_seconds.json")
    gpu = read_json(gpu_path) if os.path.exists(gpu_path) and not args.force else {}
    dtype_by_set: dict = {}

    for role in roles:
        set_name = sets[role]
        prompts = pa.role_prompts(cfg, role, args.tiny, kt)
        be = pa.Backend(set_name, args.tiny, cfg["exact"], kt, log, dtype_name=a_dt.get(set_name))
        dtype_by_set[set_name] = be.dtype_name
        todo = [pr for pr in prompts
                if args.force or not os.path.exists(os.path.join(item_dir, set_name, f"{pr['id']}.json"))]
        # 1. matrices (model loaded only if some matrix is missing), then free the model
        recs = {}
        for pr in todo:
            art = os.path.join(su.artifacts_dir(args, set_name, "premise_C"), f"{pr['id']}.pt")
            try:
                rec, secs, reused = matrix_record(be, pr, art, cfg, args.force, log)
            except Exception as e:  # noqa: BLE001
                if pa.is_oom(e):
                    log.error(f"{set_name} {pr['id']}: out of memory in {be.dtype_name} under the lane's memory cap. "
                              "Stage A's dtype is kept (never mixed): rerun Stage C for this set with a larger "
                              "CCT_MEMORY_CAP_GB.")
                raise
            gpu[set_name] = gpu.get(set_name, 0.0) + secs
            recs[pr["id"]] = (art, reused, secs)
            log.info(f"{set_name} {pr['id']}: matrix {'reused' if reused else f'computed in {secs:.1f}s'}")
        gpu[set_name] = gpu.get(set_name, 0.0) + be.load_seconds
        be.free()
        atomic_write_json(gpu_path, gpu)
        # 2. analyses (CPU pruning, GPU recovery)
        for pr in todo:
            art, reused, secs = recs[pr["id"]]
            rec = load_pt(art)
            t0 = time.perf_counter()
            res = analyse(rec, cfg, pre, ms, device, log)
            gpu[set_name] = gpu.get(set_name, 0.0) + time.perf_counter() - t0
            atomic_write_json(gpu_path, gpu)
            res.update(set=set_name, prompt_id=pr["id"], prompt=pr.get("prompt"), dtype=rec.get("dtype"),
                       matrix_reused=reused, matrix_seconds=secs, matrix_source=rec.get("source"),
                       n_pos=rec.get("n_pos"), artifact=os.path.relpath(art, su.ROOT),
                       run=run_record(cfg, cfg["seed"], time.perf_counter() - t0, set_name=set_name, prompt_id=pr["id"]))
            atomic_write_json(os.path.join(item_dir, set_name, f"{pr['id']}.json"), pa.jsonable(res))
            st = res["pr5"]
            tab = res["pr6"]["table"]
            log.info(f"{set_name} {pr['id']}: T_f={res['T_features']} rem_abs_s95_frac={st['remainder']['s95_abs_ewm_frac_causal']:.4f} "
                     f"D1 meets at m={[m for m, v in tab.get('D1', {}).items() if v['meets']]} "
                     f"D2 meets at m={[m for m, v in tab.get('D2', {}).items() if v['meets']]}")

    items: dict = {}
    for role in pa.ROLES:
        s = sets[role]
        d = os.path.join(item_dir, s)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".json"):
                    v = read_json(os.path.join(d, fn))
                    items.setdefault(s, {})[fn[:-5]] = v
                    dtype_by_set.setdefault(s, v.get("dtype"))
    doc = aggregate(args, cfg, pre, items, sets, dtype_by_set, sum(gpu.values()), t_start, ms, stage_a_pr3)
    out_path = os.path.join(root, "stage_C.json")
    atomic_write_json(out_path, doc)
    for k, v in doc["premises"].items():
        log.info(f"{k}: {v['status']} ({v['notes']})")
    log.info(f"epsilon* = {doc['epsilon_star']}, k_eff(short, Qwen) = {doc['k_eff_short_qwen']}; wrote {out_path}")
    if doc["stop"]:
        log.error(f"STOP (rule C): {doc['stop_reason']}. Report and wait for Dawar.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
