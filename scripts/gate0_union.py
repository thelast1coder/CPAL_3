"""Gate 0, G0-B step 2: union growth (PROJECT_V2.md Sections 2.3 and 7; settings in
configs/experiments/gate0.yaml, g0b.union; fidelity thresholds in configs/prereg_gate0.yaml).

GPU phase (--phase gpu): loads the replacement model once and, level by level (level 1 for every
trace, then the new slots of level 2 for every trace, and so on), builds the exact influence-ordered
graph of each explained output on its prefix tokens[:q] with a budget of reference_budget feature
rows. The convergence check (node Jaccard of the first convergence_budget rows against the
reference) runs straight away; if it fails and time remains, the output is recomputed with
raised_budget. One record per output is saved to
artifacts/<run>/<set>/gate0_union/<trace>/<q>.pt, with a JSONL progress line (seconds, peak GPU
memory) in results/[tiny/]gate0/g0b_union_progress.jsonl. It stops starting new outputs once
time_budget_minutes is used up, or before a level that the median seconds per output so far says
cannot finish. A DONE marker is written at the end.

CPU phase (--phase cpu): can run in a second process while the GPU phase continues. It polls for new
records until the DONE marker exists (or once, with --no-wait), and for each record computes c_i
(smallest c in the grid meeting fidelity against the record's own reference), the convergence
result and the (layer, position, feature) ids of the first c_i visited rows (cached next to the
record as <q>.cpu.json). Then per trace |U(n)| for every complete level and a beta fit, and overall
beta (mean over traces) and c_median. Writes results/[tiny/]gate0/g0b_union.json.

--phase all (default) runs the GPU phase, then the CPU phase. Resumable: finished outputs are skipped
unless --force.
"""
from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch

from cct import scriptutil as su
from cct.gate0_union import (TruncationEvaluator, beta_random_subsets, compute_order, convergence_check,
                             explained_positions, fit_beta, level_slots, n_logit_rows, smallest_c,
                             union_size, visited_features)
from cct.killtest import exact_graph_real, exact_graph_tiny
from cct.utils.io import append_jsonl, atomic_write_json, load_pt, read_json, read_jsonl, save_pt
from cct.utils.logging import get_logger, run_record

# Tiny run only (numbers meaningless): 20 generated tokens cannot hold 32 slots, and the tiny graphs
# have a few hundred active features, so the budgets are scaled down to exercise every code path
# (including the raised-budget recompute).
TINY_UNION = {"slots": 8, "levels": [1, 2, 4, 8], "reference_budget": 128, "convergence_budget": 64,
              "raised_budget": 256, "batch_size": 256, "c_grid": [8, 16, 32, 64, 96, 128, 192, 256]}
RANDOM_SUBSET_DRAWS = 200
POLL_SECONDS = 30


@dataclass
class Paths:
    art: str
    res: str
    traces: str
    progress: str
    done: str
    out: str


def get_paths(args, cfg: dict) -> Paths:
    set_name = "tiny_plt" if args.tiny else cfg["g0b"]["model_set"]
    art = su.artifacts_dir(args, set_name, "gate0_union")
    res = os.path.join(su.ROOT, "results", "tiny", "gate0") if args.tiny else os.path.join(su.ROOT, "results",
                                                                                            "gate0")
    os.makedirs(res, exist_ok=True)
    traces = (os.path.join(res, "traces.json") if args.tiny else os.path.join(su.ROOT, cfg["g0b"]["traces_file"]))
    return Paths(art, res, traces, os.path.join(res, "g0b_union_progress.jsonl"), os.path.join(art, "DONE"),
                 os.path.join(res, "g0b_union.json"))


def union_settings(args, cfg: dict) -> dict:
    u = dict(cfg["g0b"]["union"])
    if args.tiny:
        u.update(TINY_UNION)
    return u


def fidelity_thresholds(args) -> dict:
    path = os.path.join(su.ROOT, "configs", "prereg_gate0.yaml")
    if os.path.exists(path):
        f = su.load_yaml(path)["fidelity"]
        src = "configs/prereg_gate0.yaml"
    elif args.tiny:  # the Gate 0 prereg may not exist yet on the laptop; same values as the kill test
        f = su.prereg()["fidelity"]
        src = "configs/prereg_kill_test.yaml (tiny fallback)"
    else:
        raise RuntimeError("configs/prereg_gate0.yaml is missing: commit the Gate 0 prereg first")
    return {"recall_min": float(f["influence_weighted_node_recall_at_least"]),
            "jaccard_min": float(f["node_jaccard_at_least"]), "source": src}


def plan(traces: list[dict], u: dict) -> tuple[dict, list[dict]]:
    """Per trace the slot positions; and the global compute order, level-major across traces."""
    pos = {t["id"]: explained_positions(t["n_prompt_tokens"], t["end_position"], u["slots"]) for t in traces}
    order = []
    for level, slot in compute_order(u["levels"], u["slots"]):
        for t in traces:
            order.append({"level": level, "slot": slot, "trace": t["id"], "q": pos[t["id"]][slot]})
    return pos, order


def pt_path(P: Paths, tid: str, q: int) -> str:
    return os.path.join(P.art, tid, f"{q}.pt")


# -------------------------------------------------------------------------------------------------
# GPU phase
# -------------------------------------------------------------------------------------------------
class Builder:
    def __init__(self, args, cfg: dict, u: dict, log):
        self.args, self.cfg, self.u, self.log = args, cfg, u, log
        self.model = None
        self.load_seconds = 0.0
        self.prune_cfg = {"node_threshold": u["node_threshold"], "edge_threshold": u["edge_threshold"]}

    def ensure_model(self):
        if self.model is not None:
            return
        t0 = time.perf_counter()
        if self.args.tiny:
            from cct.premise_traces import tiny_premise_model  # same tiny model as gen_traces --tiny

            self.model = tiny_premise_model()
        else:
            from cct.operators.circuit_tracer_backend import load_replacement_model
            from cct.utils.gpu import cap_memory, nvidia_smi

            mc = su.model_config(self.cfg["g0b"]["model_set"])
            self.log.info(nvidia_smi())
            cap_memory(mc["memory_cap_gb"])
            self.model = load_replacement_model(mc["model_name"], mc["transcoder_set"],
                                                dtype=su.torch_dtype(mc["dtype"]))
        self.load_seconds = time.perf_counter() - t0

    def graph(self, ids: list[int], q: int, budget: int) -> dict:
        """Exact influence-ordered graph of the output at position q (prefix tokens[:q])."""
        self.ensure_model()
        u = self.u
        prefix = torch.tensor(ids[:q], dtype=torch.long)
        t0 = time.perf_counter()
        if self.args.tiny:
            rec = exact_graph_tiny(self.model, prefix, budget, u["batch_size"], "D", self.prune_cfg)
            rec.update(pos_offset=0, peak_gb=0.0, n_pos=q)
        else:
            from cct.operators.circuit_tracer_backend import CircuitTracerOperator

            torch.cuda.reset_peak_memory_stats()
            op = CircuitTracerOperator(self.model, prefix, max_n_logits=u["max_n_logits"],
                                       desired_logit_prob=u["desired_logit_prob"], batch_size=u["batch_size"])
            off = int(op.n_pos) - q  # 1 if circuit-tracer prepended a special token (ensure_tokenized)
            if off not in (0, 1):
                raise RuntimeError(f"unexpected prefix length {op.n_pos} for q={q}")
            rec = exact_graph_real(op, budget, "D", u["update_interval"], self.prune_cfg)
            rec.update(pos_offset=off, n_pos=int(op.n_pos), peak_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
            del op
            torch.cuda.empty_cache()
        rec["graph_seconds"] = time.perf_counter() - t0
        rec["budget"] = int(budget)
        return rec


def gpu_phase(args, cfg: dict, u: dict, P: Paths, log) -> None:
    traces = read_json(P.traces)["traces"]
    _, order = plan(traces, u)
    by_id = {t["id"]: t for t in traces}
    if os.path.exists(P.done):
        os.remove(P.done)
    budget_s = 60.0 * u["time_budget_minutes"]
    # outputs finished by an earlier (interrupted) run count towards the time budget
    prior = {} if args.force else {(r["trace"], r["q"]): r["seconds"] for r in read_jsonl(P.progress)
                                   if os.path.exists(pt_path(P, r["trace"], r["q"]))}
    secs = list(prior.values())
    spent_before = sum(secs)
    b = Builder(args, cfg, u, log)
    t_run = None  # starts after the model is loaded
    stop_reason = "all levels done"

    def elapsed() -> float:
        return spent_before + (0.0 if t_run is None else time.perf_counter() - t_run)

    for level in sorted(u["levels"]):
        todo = [o for o in order if o["level"] == level
                and (args.force or not os.path.exists(pt_path(P, o["trace"], o["q"])))]
        if not todo:
            continue
        if secs and elapsed() + len(todo) * statistics.median(secs) > budget_s:
            stop_reason = (f"level {level} cannot finish: {len(todo)} outputs x median "
                           f"{statistics.median(secs):.0f}s > remaining {budget_s - elapsed():.0f}s")
            break
        stopped = False
        for o in todo:
            if elapsed() > budget_s:
                stop_reason = f"time budget used up during level {level}"
                stopped = True
                break
            b.ensure_model()
            if t_run is None:
                t_run = time.perf_counter()
            t0 = time.perf_counter()
            tr = by_id[o["trace"]]
            rec = b.graph(tr["token_ids"], o["q"], u["reference_budget"])
            conv = convergence_check(rec, u["convergence_budget"], u["convergence_jaccard"])
            first = {"budget": rec["budget"], "conv": conv, "seconds": rec["graph_seconds"]}
            raised = False
            if not conv["ok"]:
                need = 2.0 * (time.perf_counter() - t0)  # twice the rows
                if elapsed() + need <= budget_s:
                    del rec
                    rec = b.graph(tr["token_ids"], o["q"], u["raised_budget"])
                    conv = convergence_check(rec, u["reference_budget"], u["convergence_jaccard"])
                    raised = True
                else:
                    log.info(f"{o['trace']} q={o['q']}: convergence failed, no time to raise the budget")
            s = time.perf_counter() - t0
            rec["gate0"] = dict(o, budget=rec["budget"], raised=raised, conv=conv, first_attempt=first,
                                gpu_seconds=s)
            rec["run_record"] = run_record({"g0b_union": u, "model_set": cfg["g0b"]["model_set"],
                                            "tiny": args.tiny}, cfg["seed"], s)
            save_pt(pt_path(P, o["trace"], o["q"]), rec)
            append_jsonl(P.progress, {"trace": o["trace"], "q": o["q"], "level": level, "slot": o["slot"],
                                      "budget": rec["budget"], "raised": raised, "conv_ok": conv["ok"],
                                      "conv_jaccard": conv["node_jaccard"], "T_D": rec["T_D"], "N": rec["N"],
                                      "n_active": rec["n_active"], "n_pos": rec["n_pos"], "seconds": s,
                                      "peak_gb": rec["peak_gb"]})
            secs.append(s)
            log.info(f"{o['trace']} q={o['q']} level {level}: T={rec['T_D']} n_active={rec['n_active']} "
                     f"budget={rec['budget']} conv={conv['node_jaccard']:.3f} {s:.1f}s")
            del rec
        if stopped:
            break
    atomic_write_json(P.done, {"stop_reason": stop_reason, "gpu_seconds_total": elapsed(),
                               "model_load_seconds": b.load_seconds,
                               "run": run_record({"g0b_union": u, "tiny": args.tiny}, cfg["seed"], elapsed())})
    log.info(f"GPU phase finished: {stop_reason}")


# -------------------------------------------------------------------------------------------------
# CPU phase
# -------------------------------------------------------------------------------------------------
def analyse(path: str, u: dict, fid: dict) -> dict:
    rec = load_pt(path)
    g = rec["gate0"]
    ev = TruncationEvaluator(rec)
    curve: list = []
    c = smallest_c(rec, u["c_grid"], fid["recall_min"], fid["jaccard_min"], ev, curve)
    conv_budget = u["convergence_budget"] if rec["budget"] <= u["reference_budget"] else u["reference_budget"]
    conv = convergence_check(rec, conv_budget, u["convergence_jaccard"], ev)
    c_used = c if c is not None else ev.n_feature_rows
    return {"trace": g["trace"], "q": g["q"], "slot": g["slot"], "level": g["level"], "budget": rec["budget"],
            "raised": g["raised"], "T_D": rec["T_D"], "n_active": rec["n_active"], "n_logits": n_logit_rows(rec),
            "c": c, "c_used": c_used, "conv": conv, "curve": curve, "gpu_seconds": g["gpu_seconds"],
            "pos_offset": rec.get("pos_offset", 0),
            "features": visited_features(rec, c_used, rec.get("pos_offset", 0)), "pt_mtime": os.path.getmtime(path)}


def cpu_phase(args, cfg: dict, u: dict, fid: dict, P: Paths, log, wait: bool) -> dict:
    traces = read_json(P.traces)["traces"]
    pos, order = plan(traces, u)
    done_here: set = set()
    t0 = time.perf_counter()
    while True:
        gpu_done = os.path.exists(P.done)
        for o in order:
            pt = pt_path(P, o["trace"], o["q"])
            cache = pt[:-3] + ".cpu.json"
            if not os.path.exists(pt):
                continue
            stale = (not os.path.exists(cache) or read_json(cache).get("pt_mtime") != os.path.getmtime(pt)
                     or (args.force and pt not in done_here))
            if stale:
                a = analyse(pt, u, fid)
                atomic_write_json(cache, a)
                log.info(f"cpu: {o['trace']} q={o['q']} c={a['c']} conv_ok={a['conv']['ok']}")
            done_here.add(pt)
        if gpu_done or not wait:
            break
        time.sleep(POLL_SECONDS)
    return finalise(args, cfg, u, fid, P, traces, pos, order, time.perf_counter() - t0)


def finalise(args, cfg, u, fid, P, traces, pos, order, cpu_seconds) -> dict:
    out_traces = {}
    all_c, n_without, gpu_s = [], 0, 0.0
    for t in traces:
        tid = t["id"]
        an = {}
        for o in order:
            cache = pt_path(P, tid, o["q"])[:-3] + ".cpu.json"
            if o["trace"] == tid and os.path.exists(cache):
                an[o["slot"]] = read_json(cache)
        entries = {s: {"features": a["features"], "n_logits": a["n_logits"]} for s, a in an.items()}
        levels_done = [n for n in sorted(u["levels"]) if all(s in an for s in level_slots(n, u["slots"]))]
        U = {n: union_size([entries[s] for s in level_slots(n, u["slots"])]) for n in levels_done}
        fit = fit_beta(list(U), list(U.values()))
        rnd = {n: beta_random_subsets(list(entries.values()), n, RANDOM_SUBSET_DRAWS, cfg["seed"])
               for n in sorted(u["levels"]) if n <= len(entries)}
        rfit = fit_beta(list(rnd), list(rnd.values()))
        for a in an.values():
            gpu_s += a["gpu_seconds"]
            if a["c"] is None:
                n_without += 1
            else:
                all_c.append(a["c"])
        q_of = {a["q"]: a for a in an.values()}
        out_traces[tid] = {
            "L": t["n_total_tokens"], "n_generated": t["n_generated"], "n_prompt_tokens": t["n_prompt_tokens"],
            "end_position": t["end_position"], "answer_found": t.get("answer_found"),
            "positions": pos[tid], "levels_done": levels_done, "U_by_n": {str(n): v for n, v in U.items()},
            "c_per_output": {str(q): a["c"] for q, a in sorted(q_of.items())},
            "budget_per_output": {str(q): a["budget"] for q, a in sorted(q_of.items())},
            "conv_ok": {str(q): a["conv"]["ok"] for q, a in sorted(q_of.items())},
            "conv_jaccard": {str(q): a["conv"]["node_jaccard"] for q, a in sorted(q_of.items())},
            "n_active_per_output": {str(q): a["n_active"] for q, a in sorted(q_of.items())},
            "beta": fit["beta"], "beta_se": fit["beta_se"], "c_fit": fit["c_fit"], "fit": fit,
            "U_random_subsets": {"draws": RANDOM_SUBSET_DRAWS, "U_by_n": {str(n): v for n, v in rnd.items()},
                                 "beta": rfit["beta"], "beta_se": rfit["beta_se"]},
        }
    betas = [v["beta"] for v in out_traces.values() if v["beta"] == v["beta"]]
    done = read_json(P.done) if os.path.exists(P.done) else {}
    res = {
        "run": run_record({"g0b_union": u, "fidelity": fid, "model_set": cfg["g0b"]["model_set"],
                           "tiny": args.tiny}, cfg["seed"], cpu_seconds),
        "traces": out_traces,
        "beta": statistics.mean(betas) if betas else float("nan"),
        "n_traces_with_beta": len(betas),
        "c_median": statistics.median(all_c) if all_c else float("nan"),
        "c_median_note": "median over outputs with a c in the grid; outputs without c count with their full "
                         "budget in the union and are counted in n_outputs_without_c",
        "n_outputs_without_c": n_without,
        "gpu_seconds": gpu_s,
        "gpu_stop_reason": done.get("stop_reason"),
        "model_load_seconds": done.get("model_load_seconds"),
    }
    atomic_write_json(P.out, res)
    return res


def main() -> None:
    p = su.parser("Gate 0 G0-B: union growth over whole traces")
    p.set_defaults(config=os.path.join(su.ROOT, "configs", "experiments", "gate0.yaml"))
    p.add_argument("--phase", default="all", choices=["gpu", "cpu", "all"])
    p.add_argument("--no-wait", action="store_true", help="CPU phase: do not poll for the GPU DONE marker")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    u = union_settings(args, cfg)
    P = get_paths(args, cfg)
    log = get_logger("gate0_union", os.path.join(su.ROOT, "logs"))
    if not os.path.exists(P.traces):
        raise SystemExit(f"{P.traces} missing: run scripts/gen_traces.py{' --tiny' if args.tiny else ''} first")
    fid = fidelity_thresholds(args)
    if args.phase in ("gpu", "all"):
        if args.force and os.path.exists(P.progress):
            os.remove(P.progress)
        gpu_phase(args, cfg, u, P, log)
    if args.phase in ("cpu", "all"):
        res = cpu_phase(args, cfg, u, fid, P, log, wait=not args.no_wait)
        log.info(f"beta={res['beta']:.3f} c_median={res['c_median']} without_c={res['n_outputs_without_c']} "
                 f"-> {P.out}")


if __name__ == "__main__":
    main()
