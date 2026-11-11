"""K2: sensing simulation on stored exact matrices (PHASE1 Section 7). Legitimate because mixed passes
are exactly linear (K0.2). Recovery uses the same code as the real pipeline and its time is measured.

Step 1 (validation prompts only): choose hyperparameters from the grid in the experiment config.
Step 2 (test prompts): cost-fidelity curves for CCT-fast, CCT-certified, B1, B2, B3; c75; S_exact,
S_B1; K2 verdict per set; regime F secondary statistics and the GO-F upgrade path.
Per-run results go to artifacts/<run>/<set>/k2_runs.jsonl (resumable).
"""
from __future__ import annotations

import glob
import os
import statistics

from cct import scriptutil as su
from cct.killtest import (c75, eval_cct, eval_lowrank, eval_random_rows, eval_truncation, go_f_upgrade,
                          grid_settings, k2_decision, prompt_meets)
from cct.pipeline import merge
from cct.utils.io import append_jsonl, atomic_write_json, load_pt, read_jsonl
from cct.utils.logging import get_logger, run_record


def budget_key(b):
    return float("inf") if b == "T_D" else float(b)


class Runs:
    """Resumable store of single evaluations."""

    def __init__(self, path: str, force: bool):
        self.path = path
        if force and os.path.exists(path):
            os.remove(path)
        self.done = {r["key"]: r for r in read_jsonl(path)}

    def get(self, key: str, fn):
        if key in self.done:
            return self.done[key]
        r = fn()
        r = {"key": key, "fidelity": r["fidelity"], "cost": r["cost"], "ledger": r.get("ledger"), "diag": r.get("diag")}
        append_jsonl(self.path, r)
        self.done[key] = r
        return r


def load_records(args, set_name: str, regime: str, ids: set) -> list[dict]:
    out = []
    for f in sorted(glob.glob(os.path.join(su.artifacts_dir(args, set_name, f"exact_{regime}"), "*.pt"))):
        rec = load_pt(f)
        if rec["prompt_id"] in ids:
            out.append(rec)
    return out


def tune(recs, settings, budgets, seeds, base_cfg, runs, rs, kappa, device, pre, log, tag) -> dict:
    best = None
    table = []
    for si, s in enumerate(settings):
        cfg = merge(base_cfg, s)
        for m in budgets:
            costs, ok = [], True
            for rec in recs:
                res = [runs.get(f"{tag}|tune|{si}|{m}|{rec['prompt_id']}|{sd}",
                                lambda: eval_cct(rec, cfg, m, sd, rs, kappa, device)) for sd in seeds]
                costs.append(statistics.median(r["cost"] for r in res))
                ok &= prompt_meets([r["fidelity"] for r in res], pre)
            if ok:
                c = statistics.median(costs)
                table.append({"setting": s, "m": m, "cost": c})
                if best is None or c < best["cost"]:
                    best = {"setting": s, "index": si, "m": m, "cost": c}
                break
    if best is None:
        log.info(f"{tag}: no setting met fidelity on all validation prompts; using the first setting")
        best = {"setting": settings[0], "index": 0, "m": None, "cost": None, "fallback": True}
    return {"chosen": best, "table": table}


def curves_for(recs, method: str, budgets, seeds, cfg, runs, rs, kappa, device, pre, tag) -> dict:
    curve = {}
    for b in budgets:
        meets, costs = [], []
        for rec in recs:
            if method in ("cct_fast", "cct_certified"):
                res = [runs.get(f"{tag}|{method}|{b}|{rec['prompt_id']}|{sd}",
                                lambda: eval_cct(rec, cfg, int(b), sd, rs, kappa, device)) for sd in seeds]
            elif method == "b1":
                c = rec["T_D"] if b == "T_D" else int(b)
                res = [runs.get(f"{tag}|b1|{b}|{rec['prompt_id']}", lambda: eval_truncation(rec, c))]
            elif method == "b2":
                res = [runs.get(f"{tag}|b2|{b}|{rec['prompt_id']}|{sd}",
                                lambda: eval_random_rows(rec, int(b), sd)) for sd in seeds]
            elif method == "b3":
                res = [runs.get(f"{tag}|b3|{b}|{rec['prompt_id']}|{sd}",
                                lambda: eval_lowrank(rec, int(b), sd, kappa)) for sd in seeds]
            else:
                raise ValueError(method)
            meets.append(prompt_meets([r["fidelity"] for r in res], pre))
            costs.append(statistics.median(r["cost"] for r in res))
        curve[str(b)] = {"meets": meets, "costs": costs, "share": sum(meets) / max(1, len(meets)),
                         "median_cost": statistics.median(costs) if costs else None}
    return curve


def main() -> None:
    p = su.parser("K2 sensing simulation")
    p.add_argument("--skip-secondary", action="store_true", help="skip B2, B3 and CCT-certified curves")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    pre = su.prereg()
    log = get_logger("simulate_sensing", os.path.join(su.ROOT, "logs"))
    prompts = su.prompts(args, cfg)
    seeds = pre["seeds"]
    if args.tiny:
        t = cfg["tiny"]
        budgets_cct, budgets_b1, grid = t["cct_m"], t["b1_rows"], t["grid"]
        device = t["recovery_device"]
    else:
        budgets_cct, budgets_b1, grid = pre["k2"]["cct_budgets_m"], pre["k2"]["b1_budgets_rows"], cfg["grid"]
        device = cfg["cct_fast"]["recovery"]["device"]
    base_fast = merge(cfg["cct_fast"], {"recovery": {"device": device}})
    settings = grid_settings(grid)
    val_ids = {x["id"] for x in prompts["validation"]}
    test_ids = {x["id"] for x in prompts["test"]}
    out = {"sets": {}, "regime_F": {}}
    for set_name in su.sets(args):
        rs, kappa = su.timing(args, cfg, set_name)
        runs = Runs(os.path.join(su.artifacts_dir(args, set_name), "k2_runs.jsonl"), args.force)
        val = load_records(args, set_name, "D", val_ids)
        tuning = tune(val, settings, budgets_cct, seeds, base_fast, runs, rs, kappa, device, pre, log, "D")
        chosen = merge(base_fast, tuning["chosen"]["setting"])
        log.info(f"{set_name}: chosen hyperparameters {tuning['chosen']}")
        test = load_records(args, set_name, "D", test_ids)
        cur = {"cct_fast": curves_for(test, "cct_fast", budgets_cct, seeds, chosen, runs, rs, kappa, device, pre, "D"),
               "b1": curves_for(test, "b1", budgets_b1, seeds, chosen, runs, rs, kappa, device, pre, "D")}
        if not args.skip_secondary:
            cert = merge(chosen, cfg.get("cct_certified", {}))
            cur["cct_certified"] = curves_for(test, "cct_certified", budgets_cct, seeds, cert, runs, rs, kappa, device, pre, "D")
            cur["b2"] = curves_for(test, "b2", budgets_cct, seeds, chosen, runs, rs, kappa, device, pre, "D")
            cur["b3"] = curves_for(test, "b3", budgets_cct, seeds, chosen, runs, rs, kappa, device, pre, "D")
        c75s = {k: c75(v, pre) for k, v in cur.items()}
        T_D_med = statistics.median(r["T_D"] for r in test)
        dec = k2_decision(c75s["cct_fast"], c75s["b1"], T_D_med, pre)
        out["sets"][set_name] = {"tuning": tuning, "chosen_config": chosen, "curves": cur, "c75": c75s,
                                 "T_D_median": T_D_med, "decision": dec, "row_seconds": rs, "kappa": kappa}
        log.info(f"K2 {set_name}: {dec}")
        # regime F (secondary): CCT-fast and B1 against exact full graphs
        recF = load_records(args, set_name, "F", test_ids | {x["id"] for x in prompts["long"]})
        if recF:
            b1F = sorted({*[b for b in budgets_b1 if b != "T_D"], "T_D"}, key=budget_key)
            curF = {"cct_fast": curves_for(recF, "cct_fast", budgets_cct, seeds, chosen, runs, rs, kappa, device, pre, "F"),
                    "b1": curves_for(recF, "b1", b1F, seeds, chosen, runs, rs, kappa, device, pre, "F")}
            c75F = {k: c75(v, pre) for k, v in curF.items()}
            T_F_med = statistics.median(r["T_D"] for r in recF)
            out["regime_F"][set_name] = {"curves": curF, "c75": c75F, "T_F_median": T_F_med,
                                         "decision": k2_decision(c75F["cct_fast"], c75F["b1"], T_F_med, pre)}
    final = go_f_upgrade({s: v["decision"] for s, v in out["sets"].items()},
                         {s: v["decision"] for s, v in out["regime_F"].items()})
    for s in out["sets"]:
        out["sets"][s]["decision_final"] = final[s]
    out["run_record"] = run_record({"grid": grid, "budgets_cct": budgets_cct, "budgets_b1": budgets_b1}, None)
    atomic_write_json(os.path.join(su.results_dir(args), "k2.json"), out)


if __name__ == "__main__":
    main()
