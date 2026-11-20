"""G0-A: short prompts (PROJECT_V2.md Section 7). Gemma-2-2B with PLT and CLT-426k, the 3 canonical
prompts. Per prompt:
  1. exact regime-D graph (circuit-tracer's visiting loop, cap 7,500), all rows stored;
  2. closed-form A_0 (cct.local_part) and K0.7: A_0 rows equal attention-detached backward rows;
     forward-mode check (columns equal the stored rows) and, on the first prompt, K0.4 (our pruning
     equals circuit-tracer's prune_graph);
  3. structure of A and of the remainder A - A_0;
  4. truncation curve (smallest c meeting fidelity);
  5. simulated sensing of A - A_0 at m in {128, 256, 512, 1024}, 3 seeds (A_0 added back).
Timing (row seconds, kappa) is measured on the first prompt of each set.

Resumable: one JSON per (set, prompt) under results/gate0/g0a/<set>/ (tiny: results/tiny/gate0/...);
the aggregate results/gate0/g0a.json is rebuilt at the end. Use --transcoders plt|clt|both to run the
two sets in separate processes (lanes).
"""
from __future__ import annotations

import os
import time

import torch

from cct import scriptutil as su
from cct.gate0_structure import record_index_map, rel_err, sensing_curve, structure, truncation_curve
from cct.killtest import exact_graph_real, exact_graph_tiny, record_matrix
from cct.local_part import from_circuit_tracer, from_tiny
from cct.operators.base import NodeMeta
from cct.utils.io import atomic_write_json, pack_matrix, read_json, save_pt
from cct.utils.logging import get_logger, run_record
from cct.utils.timing import median_time

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0.yaml")


def prereg_gate0() -> dict:
    return su.load_yaml(os.path.join(su.ROOT, "configs", "prereg_gate0.yaml"))


def gate0_results_dir(args) -> str:
    d = os.path.join(su.results_dir(args), "gate0")
    os.makedirs(d, exist_ok=True)
    return d


# -------------------------------------------------------------------------------------------------
# per-prompt GPU (or tiny) work: returns the exact record, A_0 in record layout and checks
# -------------------------------------------------------------------------------------------------
def tiny_prompt_work(model, tokens, cfg, first: bool, gen) -> tuple[dict, torch.Tensor, dict]:
    from cct.operators.tiny import TinyOperator

    t = cfg["tiny_exact"]
    rec = exact_graph_tiny(model, tokens, t["max_feature_nodes"], t["batch_size"], "D",
                           {"node_threshold": 0.8, "edge_threshold": 0.98})
    op = TinyOperator(model, tokens, batch_size=t["batch_size"])
    lp = from_tiny(op)
    rows, cols = record_index_map(rec, op.n_logits, op.active.shape[0])
    A0 = lp.rows(rows)[:, cols].double()
    idx = torch.randperm(op.T, generator=gen)[: min(op.T, 64)]
    op_det = TinyOperator(model, tokens, batch_size=t["batch_size"], detach_attn=True)
    checks = {"k07_rel_err": rel_err(lp.rows(idx), op_det.exact_rows(idx)), "dtype": "float64"}
    A = record_matrix(rec)
    src = torch.randperm(cols.numel(), generator=gen)[:16]
    checks["fwd_rel_err"] = rel_err(op.exact_cols(cols[src])[:, rows].T, A[:, src])
    checks["forward_mode"] = "fwad"
    if first:
        b = t["batch_size"]
        tr, _ = median_time(lambda: op.exact_rows(torch.arange(min(b, op.T))), 3)
        tc, _ = median_time(lambda: op.exact_cols(torch.arange(min(b, op.N))), 3)
        checks["timing"] = {"row_seconds": tr / min(b, op.T), "kappa": (tc / min(b, op.N)) / (tr / min(b, op.T))}
    return rec, A0, checks


def real_prompt_work(model, prompt: str, cfg, first: bool, gen, dtype_name: str, log) -> tuple[dict, torch.Tensor, dict]:
    from cct.operators.circuit_tracer_backend import CircuitTracerOperator

    ex = cfg["g0a"]["exact"]
    kw = dict(max_n_logits=ex["max_n_logits"], desired_logit_prob=ex["desired_logit_prob"], batch_size=ex["batch_size"])
    op = CircuitTracerOperator(model, prompt, **kw)
    rec = exact_graph_real(op, ex["max_feature_nodes"], "D", ex["update_interval"],
                           {"node_threshold": ex["node_threshold"], "edge_threshold": ex["edge_threshold"]})
    checks: dict = {"dtype": dtype_name, "n_tokens": int(op.n_pos), "n_active": int(op.n_active)}
    rows, cols = record_index_map(rec, op.n_logits, op.n_active)
    A = record_matrix(rec)
    # forward mode: 16 columns against the stored rows (fallback to double backward)
    src = torch.randperm(cols.numel(), generator=gen)[:16]
    try:
        C = op.exact_cols(cols[src])
        checks["forward_mode"] = "fwad"
    except Exception as e:  # noqa: BLE001  (a hook without a forward-AD formula)
        log.info(f"forward AD failed ({e!r}); using double backward")
        op.forward_mode = "double_backward"
        C = op.exact_cols(cols[src])
        checks["forward_mode"] = "double_backward"
    checks["fwd_rel_err"] = rel_err(C[:, rows.to(C.device)].T, A[:, src])
    if first:
        b = ex["batch_size"]
        ridx = torch.arange(op.n_logits, op.n_logits + b) % op.T
        tr, _ = median_time(lambda: op.exact_rows(ridx), cfg["g0a"]["timing_repeats"])
        tc, _ = median_time(lambda: op.exact_cols(torch.arange(b) % op.N), cfg["g0a"]["timing_repeats"])
        checks["timing"] = {"row_seconds": tr / b, "kappa": tc / tr, "batch": b,
                            "forward_mode": checks["forward_mode"]}
    # closed-form A_0 and K0.7
    t0 = time.perf_counter()
    lp = from_circuit_tracer(op)
    A0 = lp.rows(rows)[:, cols.to(lp.device)].double().cpu()
    checks["a0_seconds"] = time.perf_counter() - t0
    checks["read_map_kind"] = sorted(set(op.read_map_kind))
    idx = torch.randperm(op.T, generator=gen)[: cfg["g0a"]["k07_rows"]]
    A0_k = lp.rows(idx).double().cpu()
    del lp, op
    torch.cuda.empty_cache()
    op_det = CircuitTracerOperator(model, prompt, detach_attention=True, **kw)
    checks["k07_rel_err"] = rel_err(A0_k, op_det.exact_rows(idx))
    del op_det
    torch.cuda.empty_cache()
    if first:  # K0.4: our pruning equals circuit-tracer's on the same graph
        try:
            from circuit_tracer import attribute
            from circuit_tracer.graph import prune_graph

            from cct.graph.prune import prune_adjacency

            g = attribute(prompt, model, max_n_logits=ex["max_n_logits"], desired_logit_prob=ex["desired_logit_prob"],
                          batch_size=ex["batch_size"], max_feature_nodes=ex["max_feature_nodes"])
            ct = prune_graph(g, ex["node_threshold"], ex["edge_threshold"])
            ours = prune_adjacency(g.adjacency_matrix.double(), len(g.selected_features), len(g.input_tokens),
                                   g.logit_probabilities.double().cpu(), ex["node_threshold"], ex["edge_threshold"])
            checks["k04_pass"] = bool(torch.equal(ct.node_mask.cpu(), ours.node_mask) and
                                      torch.equal(ct.edge_mask.cpu(), ours.edge_mask))
            del g
        except Exception as e:  # noqa: BLE001
            checks["k04_pass"] = None
            checks["k04_error"] = repr(e)
        torch.cuda.empty_cache()
    return rec, A0, checks


def main() -> None:
    p = su.parser("Gate 0, G0-A: short prompts")
    p.set_defaults(config=DEFAULT_CONFIG)
    p.add_argument("--skip-sensing", action="store_true", help="structure and truncation only")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    pre = prereg_gate0()
    fid = pre["fidelity"]
    log = get_logger("gate0_short", os.path.join(su.ROOT, "logs"))
    out_root = gate0_results_dir(args)
    gen = torch.Generator().manual_seed(cfg["seed"])
    g = cfg["g0a"]
    t_start = time.perf_counter()

    if args.tiny:
        kt = su.load_yaml(os.path.join(su.ROOT, "configs", "experiments", "kill_test.yaml"))
        cfg["tiny_exact"] = {"max_feature_nodes": kt["tiny"]["max_feature_nodes"], "batch_size": kt["tiny"]["batch_size"]}
        prompts = su.tiny_prompts(kt)["test"][:3]
        set_names = su.sets(args)
        device = "cpu"
        ms = kt["tiny"]["cct_m"]
        seeds = g["sensing_seeds"]
    else:
        from cct.killtest import load_prompts

        allp = {x["id"]: x for x in load_prompts(os.path.join(su.ROOT, g["prompts_file"]))}
        prompts = [allp[i] for i in g["prompt_ids"]]
        set_names = su.sets(args)
        device = "cuda"
        ms = g["sensing_m"]
        seeds = g["sensing_seeds"]

    for set_name in set_names:
        set_dir = os.path.join(out_root, "g0a", set_name)
        os.makedirs(set_dir, exist_ok=True)
        model, dtype_name = None, "float64"
        for i, pr in enumerate(prompts):
            path = os.path.join(set_dir, f"{pr['id']}.json")
            if os.path.exists(path) and not args.force:
                log.info(f"skip {set_name} {pr['id']} (done)")
                continue
            t0 = time.perf_counter()
            if args.tiny:
                model = su.tiny_model(su.load_yaml(os.path.join(su.ROOT, "configs", "experiments", "kill_test.yaml")),
                                      set_name)
                rec, A0, checks = tiny_prompt_work(model, torch.tensor(pr["tokens"]), cfg, i == 0, gen)
            else:
                if model is None:
                    from cct.operators.circuit_tracer_backend import load_replacement_model
                    from cct.utils.gpu import cap_memory, nvidia_smi

                    log.info(nvidia_smi())
                    mc = su.model_config(set_name)
                    dtype_name = mc["dtype"] or "float32"
                    cap_memory(mc["memory_cap_gb"])
                    model = load_replacement_model(mc["model_name"], mc["transcoder_set"], dtype=su.torch_dtype(dtype_name))
                rec, A0, checks = real_prompt_work(model, pr["prompt"], cfg, i == 0, gen, dtype_name, log)
            gpu_seconds = time.perf_counter() - t0
            art = su.artifacts_dir(args, set_name, "gate0_short")
            save_pt(os.path.join(art, f"{pr['id']}.pt"), {**rec, "A0": pack_matrix(A0)})
            # analysis (CPU or GPU recovery)
            A = record_matrix(rec)
            st = structure(A, A0, NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"]))
            tr = truncation_curve(rec, g["truncation_budgets"] if not args.tiny else [8, 16, 32, 64, 128, "T_D"], fid)
            timing_path = os.path.join(set_dir, "timing.json")
            if "timing" in checks:
                atomic_write_json(timing_path, checks["timing"])
            timing = read_json(timing_path) if os.path.exists(timing_path) else {"row_seconds": 1.0, "kappa": 1.0}
            sens = None
            if not args.skip_sensing:
                sens = sensing_curve(rec, A0, ms, seeds, g["cct"], fid, timing["row_seconds"], timing["kappa"], device)
            res = {**st, **{k: v for k, v in checks.items() if k != "timing"},
                   "c_trunc": tr["c_trunc"], "truncation_curve": tr["curve"],
                   "m_min": sens["m_min"] if sens else None,
                   "fidelity_by_m": {m: {"n_meet": v["n_meet"], "meets": v["meets"],
                                         "median_cost": v["median_cost"]}
                                     for m, v in sens["by_m"].items()} if sens else None,
                   "sensing_runs": {m: v["runs"] for m, v in sens["by_m"].items()} if sens else None,
                   "recovery_seconds_by_m": {m: v["median_recovery_seconds"] for m, v in sens["by_m"].items()} if sens else None,
                   "prompt": pr.get("prompt"), "gpu_seconds": gpu_seconds,
                   "run": run_record(cfg, cfg["seed"], time.perf_counter() - t0, set_name=set_name, prompt_id=pr["id"])}
            atomic_write_json(path, res)
            log.info(f"{set_name} {pr['id']}: T={st['T']} a0_share={st['a0_energy_share']:.3f} "
                     f"s95_rem_frac={st['s95_rem_frac_T']:.4f} k07={checks['k07_rel_err']:.2e} "
                     f"c_trunc={tr['c_trunc']} m_min={res['m_min']}")
            # The A_0 check is PR3 in prereg version 2 (stage_a.pr3; bf16 tolerance as PR1/PR2). PR3 is not a
            # kill: log the failure and carry on (GATE0_PREMISES Section 3, PR3).
            sa = pre["stage_a"]
            tol = (sa["pr3"]["rel_err_at_most"] if checks["dtype"] in ("float32", "float64")
                   else sa["pr2"]["rel_err_bf16_at_most"])
            res["k07_pass"] = bool(checks["k07_rel_err"] <= tol)
            atomic_write_json(path, res)
            if not res["k07_pass"]:
                log.error(f"A_0 check (PR3) FAILED ({checks['k07_rel_err']:.3e} > {tol}): not a kill; fix the "
                          "formula or drop A_0 subtraction and record it")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # aggregate every finished (set, prompt)
    agg: dict = {"sets": {}}
    for set_name in sorted(os.listdir(os.path.join(out_root, "g0a"))):
        sd = os.path.join(out_root, "g0a", set_name)
        if not os.path.isdir(sd):
            continue
        entry = {"prompts": {}, "timing": read_json(os.path.join(sd, "timing.json"))
                 if os.path.exists(os.path.join(sd, "timing.json")) else None}
        for fn in sorted(os.listdir(sd)):
            if fn.endswith(".json") and fn != "timing.json":
                entry["prompts"][fn[:-5]] = read_json(os.path.join(sd, fn))
        agg["sets"][set_name] = entry
    agg["run"] = run_record(cfg, cfg["seed"], time.perf_counter() - t_start)
    atomic_write_json(os.path.join(out_root, "g0a.json"), agg)
    log.info(f"wrote {os.path.join(out_root, 'g0a.json')}")


if __name__ == "__main__":
    main()
