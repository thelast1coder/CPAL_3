"""Exact graphs (PHASE1 Section 5). Regime D: all test and validation prompts with circuit-tracer's
defaults (cap 7,500). Regime F: the first prompt of each category with all active features as
targets, skipped (and logged) if it would exceed the 30-minute budget.

Saves one record per prompt under artifacts/<run>/<set>/exact_<regime>/<prompt id>.pt (see
cct.killtest), resumable: finished prompts are skipped unless --force.
"""
from __future__ import annotations

import os
import time

import torch

from cct import scriptutil as su
from cct.killtest import exact_graph_real, exact_graph_tiny, first_of_each_category
from cct.utils.io import append_jsonl, load_pt, save_pt
from cct.utils.logging import get_logger, run_record


def main() -> None:
    p = su.parser("Exact graphs, regimes D and F")
    p.add_argument("--regimes", default="D,F")
    args = p.parse_args()
    cfg = su.load_yaml(args.config)
    log = get_logger("run_exact_graphs", os.path.join(su.ROOT, "logs"))
    prompts = su.prompts(args, cfg)
    ex = cfg["exact"]
    prune_cfg = {"node_threshold": ex["node_threshold"], "edge_threshold": ex["edge_threshold"]}
    for set_name in su.sets(args):
        model = None
        for regime in args.regimes.split(","):
            if regime == "D":
                todo = prompts["test"] + prompts["validation"]
            else:
                todo = first_of_each_category(prompts["test"]) + (prompts["long"] if args.tiny else [])
            out_dir = su.artifacts_dir(args, set_name, f"exact_{regime}")
            index = os.path.join(out_dir, "index.jsonl")
            for pr in todo:
                path = os.path.join(out_dir, f"{pr['id']}.pt")
                if os.path.exists(path) and not args.force:
                    continue
                t0 = time.perf_counter()
                if args.tiny:
                    model = su.tiny_model(cfg, set_name)
                    rec = exact_graph_tiny(model, torch.tensor(pr["tokens"]), cfg["tiny"]["max_feature_nodes"],
                                           cfg["tiny"]["batch_size"], regime, prune_cfg)
                else:
                    from cct.operators.circuit_tracer_backend import CircuitTracerOperator, load_replacement_model
                    from cct.utils.gpu import cap_memory, nvidia_smi

                    mc = su.model_config(set_name)
                    if model is None:
                        log.info(nvidia_smi())
                        cap_memory(mc["memory_cap_gb"])
                        model = load_replacement_model(mc["model_name"], mc["transcoder_set"],
                                                       dtype=su.torch_dtype(mc["dtype"]))
                    op = CircuitTracerOperator(model, pr["prompt"], max_n_logits=ex["max_n_logits"],
                                               desired_logit_prob=ex["desired_logit_prob"], batch_size=ex["batch_size"])
                    if regime == "F":
                        est = op.n_active * su.model_config(set_name)["timing"]["row_seconds"]
                        if est > 60 * ex["regime_f_time_budget_minutes"]:
                            log.info(f"skip regime F {pr['id']}: estimated {est/60:.1f} min > budget")
                            append_jsonl(index, {"key": pr["id"], "skipped": True, "estimated_minutes": est / 60})
                            continue
                    torch.cuda.reset_peak_memory_stats()
                    rec = exact_graph_real(op, ex["max_feature_nodes"], regime, ex["update_interval"], prune_cfg)
                    rec["peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
                    rec["prompt"] = pr["prompt"]
                    del op
                    torch.cuda.empty_cache()
                rec["prompt_id"] = pr["id"]
                rec["category"] = pr["category"]
                rec["run_record"] = run_record({"exact": ex, "set": set_name, "regime": regime}, args.seed,
                                               time.perf_counter() - t0)
                save_pt(path, rec)
                append_jsonl(index, {"key": pr["id"], "T_D": rec["T_D"], "N": rec["N"],
                                     "n_active": rec["n_active"], "seconds": time.perf_counter() - t0})
                log.info(f"{set_name} {regime} {pr['id']}: T={rec['T_D']} N={rec['N']} "
                         f"{time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
