"""K3: real mixed passes and real clocks (PHASE1 Section 8).

Per set, on the first prompt of each category: run the full CCT-fast pipeline with real passes through
the backend at CCT's c75 budget with the hyperparameters chosen in K2 Step 1, three times (sensing
seeds 0, 1, 2, which are also the three timing repeats), and rerun the exact default graph three
times in the same session. CCT wall time includes building the operator (forward pass and cached
state), sensing, recovery, hubs and certification; the exact time is circuit-tracer's attribute()
with defaults (tiny: building the operator and every exact row with the visiting loop).
"""
from __future__ import annotations

import os
import statistics
import time

import torch

from cct import scriptutil as su
from cct.graph.export import make_layout, to_square
from cct.graph.metrics import fidelity
from cct.killtest import (eval_cct, exact_graph_tiny, exact_prune_result, first_of_each_category, k3_decision,
                          prompt_meets, record_matrix)
from cct.pipeline import run_cct
from cct.utils.io import atomic_write_json, load_pt, read_json
from cct.utils.logging import get_logger, run_record


def main() -> None:
    args = su.parser("K3 real mixed passes").parse_args()
    cfg = su.load_yaml(args.config)
    pre = su.prereg()
    log = get_logger("run_real_cct", os.path.join(su.ROOT, "logs"))
    k2 = read_json(os.path.join(su.results_dir(args), "k2.json"))
    prompts = su.prompts(args, cfg)
    seeds = pre["seeds"]
    out = {"sets": {}}
    for set_name in su.sets(args):
        s2 = k2["sets"][set_name]
        regime = "F" if s2["decision_final"]["verdict"] == "GO-F" else "D"
        c = s2["c75"]["cct_fast"] if regime == "D" else k2["regime_F"][set_name]["c75"]["cct_fast"]
        budgets = cfg["tiny"]["cct_m"] if args.tiny else pre["k2"]["cct_budgets_m"]
        m = int(c["budget"]) if c else int(max(budgets))
        note = None if c else "CCT never reached c75 in K2; K3 run at the largest budget for information only"
        chosen = s2["chosen_config"]
        rs, kappa = su.timing(args, cfg, set_name)
        k3p = first_of_each_category(prompts["test"])
        model = None
        per_prompt = {}
        dtype_name = "float64" if args.tiny else su.model_config(set_name)["dtype"]
        for pr in k3p:
            rec = load_pt(os.path.join(su.artifacts_dir(args, set_name, f"exact_{regime}"), f"{pr['id']}.pt"))
            visit = rec["visit_features"]
            sel = torch.sort(visit).values
            A_ex = record_matrix(rec)
            ex_pr = exact_prune_result(rec)
            fids, cct_times, A_real0 = [], [], None
            for sd in seeds:
                t0 = time.perf_counter()
                if args.tiny:
                    from cct.operators.tiny import TinyOperator

                    model = su.tiny_model(cfg, set_name)
                    op = TinyOperator(model, torch.tensor(pr["tokens"]), target_features=visit,
                                      source_features=sel, batch_size=cfg["tiny"]["batch_size"])
                else:
                    from cct.operators.circuit_tracer_backend import CircuitTracerOperator, load_replacement_model
                    from cct.utils.gpu import cap_memory

                    mc = su.model_config(set_name)
                    if model is None:
                        cap_memory(mc["memory_cap_gb"])
                        model = load_replacement_model(mc["model_name"], mc["transcoder_set"],
                                                       dtype=su.torch_dtype(mc["dtype"]))
                    ex = cfg["exact"]
                    op = CircuitTracerOperator(model, pr["prompt"], max_n_logits=ex["max_n_logits"],
                                               desired_logit_prob=ex["desired_logit_prob"], batch_size=ex["batch_size"],
                                               target_features=visit, source_features=sel)
                lay = make_layout(op.target_meta, op.source_meta)
                res = run_cct(op, dict(chosen, m=m), seed=sd, row_seconds=rs, kappa=kappa, layout=lay)
                if op.device.type == "cuda":
                    torch.cuda.synchronize()
                cct_times.append(time.perf_counter() - t0)
                fids.append(fidelity(to_square(A_ex, lay), ex_pr, res.adjacency, res.prune, lay.n_features,
                                     lay.n_errors, lay.n_tokens, lay.logit_probs))
                if sd == seeds[0]:
                    A_real0 = res.A_hat.double()
                    ledger0 = res.ledger.to_dict()
                del op
            sim = eval_cct(rec, chosen, m, seeds[0], rs, kappa, "cpu" if args.tiny else "cuda")
            rel_frob = float((A_real0 - sim["A_hat"].double()).norm() / sim["A_hat"].double().norm().clamp_min(1e-30))
            ex_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                if args.tiny:
                    exact_graph_tiny(su.tiny_model(cfg, set_name), torch.tensor(pr["tokens"]),
                                     cfg["tiny"]["max_feature_nodes"], cfg["tiny"]["batch_size"], regime)
                else:
                    from circuit_tracer import attribute

                    ex = cfg["exact"]
                    attribute(pr["prompt"], model, max_n_logits=ex["max_n_logits"],
                              desired_logit_prob=ex["desired_logit_prob"], batch_size=ex["batch_size"],
                              max_feature_nodes=ex["max_feature_nodes"] if regime == "D" else None)
                    torch.cuda.synchronize()
                ex_times.append(time.perf_counter() - t0)
            t_cct, t_ex = statistics.median(cct_times), statistics.median(ex_times)
            per_prompt[pr["id"]] = {"meets": prompt_meets(fids, pre), "fidelity": fids, "cct_seconds": cct_times,
                                    "exact_seconds": ex_times, "speedup": t_ex / t_cct, "rel_frob_real_vs_sim": rel_frob,
                                    "ledger_seed0": ledger0}
            log.info(f"K3 {set_name} {pr['id']}: speedup {t_ex / t_cct:.2f} meets {per_prompt[pr['id']]['meets']}")
        n_meet = sum(v["meets"] for v in per_prompt.values())
        speed = statistics.median(v["speedup"] for v in per_prompt.values())
        rf = max(v["rel_frob_real_vs_sim"] for v in per_prompt.values())
        out["sets"][set_name] = {"regime": regime, "m": m, "note": note, "per_prompt": per_prompt,
                                 "decision": k3_decision(n_meet, speed, rf, dtype_name, pre)}
        log.info(f"K3 {set_name}: {out['sets'][set_name]['decision']}")
    out["run_record"] = run_record({"k3": pre["k3"]}, None)
    atomic_write_json(os.path.join(su.results_dir(args), "k3.json"), out)


if __name__ == "__main__":
    main()
