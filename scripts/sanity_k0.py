"""K0 sanity checks (PHASE1_KILL_TEST.md Section 3). All must pass before any exact-graph run.

Tiny mode: K0.1 against brute_force_A (stand-in for circuit-tracer's adjacency), K0.2 linearity,
K0.3 forward mode, K0.4 pruning against the slow reference, K0.5 timing on the tiny backend, K0.6
prefix property of the influence-ordered visiting.
Real mode (VM): the same checks against circuit-tracer 0.5.0 (attribute, prune_graph), plus prompt
token lengths with the real tokenizer.
"""
from __future__ import annotations

import os
import sys
import time

import torch

from cct import scriptutil as su
from cct.graph.prune import prune_adjacency, prune_reference
from cct.killtest import visit_order_from_full
from cct.operators.tiny import TinyOperator
from cct.recovery.batched import recover_columns
from cct.sensing.matrices import sensing_matrix
from cct.utils.io import atomic_write_json, read_json
from cct.utils.logging import get_logger, run_record
from cct.utils.status import log_status
from cct.utils.timing import median_time


def rel(a, b) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def max_rel_rowwise(a, b) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float(((a - b).abs().amax(1) / b.abs().amax(1).clamp_min(1e-30)).max())


def recovery_throughput(T: int, m: int = 256, C: int = 1024, k: int = 32, device: str = "cpu") -> float:
    Phi = sensing_matrix("rademacher", m, T, 0)
    Y = torch.randn(m, C)
    _, secs = recover_columns(Phi, Y, torch.arange(C), None, "omp", k_max=k, tol=0.0, device=device)
    return C / secs


def tiny_checks(args, cfg) -> dict:
    out = {}
    for set_name in su.sets(args):
        model = su.tiny_model(cfg, set_name)
        prompts = su.tiny_prompts(cfg)["test"][:3]
        res = {"K0.1": [], "K0.2": [], "K0.3": [], "K0.4": [], "K0.6": []}
        for p in prompts:
            tok = torch.tensor(p["tokens"])
            op = TinyOperator(model, tok, batch_size=cfg["tiny"]["batch_size"])
            A = op.exact_matrix()
            res["K0.1"].append(max_rel_rowwise(A, op.brute_force_A()))
            g = torch.Generator().manual_seed(0)
            S = (torch.randint(0, 2, (5, op.T), generator=g) * 2 - 1).double()
            res["K0.2"].append(rel(op.rows(S), S @ A))
            src = torch.randperm(op.N, generator=g)[:20]
            res["K0.3"].append(rel(op.exact_cols(src), A[:, src].T))
            # K0.4 pruning against the slow reference on a small square graph
            from cct.graph.export import make_layout, to_square
            lay = make_layout(op.target_meta, op.source_meta)
            sq = to_square(A, lay)
            n_keep = 40  # keep the reference fast: features subset plus all other nodes
            nf = lay.n_features
            idx = torch.cat([torch.arange(min(n_keep, nf)), torch.arange(nf, sq.shape[0])])
            sub = sq[idx][:, idx]
            fast = prune_adjacency(sub, min(n_keep, nf), lay.n_tokens, lay.logit_probs)
            ref_nodes, ref_edges = prune_reference(sub, min(n_keep, nf), lay.n_tokens, lay.logit_probs)
            res["K0.4"].append(bool(torch.equal(fast.node_mask, ref_nodes) and torch.equal(fast.edge_mask, ref_edges)))
            # K0.6 prefix property: the capped visit equals the prefix of the uncapped visit
            n_act = op.active.shape[0]
            full = visit_order_from_full(A, op.n_logits, n_act, op.logit_probs, n_act, cfg["tiny"]["batch_size"])
            capped = visit_order_from_full(A, op.n_logits, n_act, op.logit_probs, min(32, n_act), cfg["tiny"]["batch_size"])
            res["K0.6"].append(bool(torch.equal(set_sorted(capped), set_sorted(full[: capped.numel()]))))
        # K0.5 timing on the tiny backend
        op = TinyOperator(model, torch.tensor(prompts[0]["tokens"]), batch_size=cfg["tiny"]["batch_size"])
        b = min(cfg["tiny"]["batch_size"], op.T)
        seeds = torch.randn(b, op.T, dtype=torch.float64)
        tang = torch.randn(b, op.N, dtype=torch.float64)
        t_rows, _ = median_time(lambda: op.rows(seeds), 3)
        t_cols, _ = median_time(lambda: op.cols(tang), 3)
        timing = {"row_seconds": t_rows / b, "kappa": t_cols / t_rows, "batch": b,
                  "recovery_columns_per_second": recovery_throughput(op.T)}
        tol = 1e-6
        checks = {
            "K0.1": {"value": max(res["K0.1"]), "pass": max(res["K0.1"]) <= tol},
            "K0.2": {"value": max(res["K0.2"]), "pass": max(res["K0.2"]) <= tol},
            "K0.3": {"value": max(res["K0.3"]), "pass": max(res["K0.3"]) <= tol, "mode": "forward_ad"},
            "K0.4": {"value": all(res["K0.4"]), "pass": all(res["K0.4"])},
            "K0.5": {"value": timing, "pass": True},
            "K0.6": {"value": all(res["K0.6"]), "pass": all(res["K0.6"])},
        }
        out[set_name] = {"checks": checks, "timing": timing, "dtype": "float64"}
    return out


def set_sorted(x):
    return torch.sort(x).values


def real_checks(args, cfg, log) -> dict:
    from circuit_tracer import attribute  # VM only
    from circuit_tracer.graph import prune_graph

    from cct.graph.export import make_layout
    from cct.operators.circuit_tracer_backend import CircuitTracerOperator, load_replacement_model
    from cct.utils.gpu import cap_memory, nvidia_smi

    log.info(nvidia_smi())
    ex = cfg["exact"]
    out = {}
    test_prompts = su.real_prompts(cfg)
    for set_name in su.sets(args):
        mc = su.model_config(set_name)
        if mc["dtype"] is None:
            raise SystemExit(f"D10 dtype not set in configs/models/{set_name}.yaml; decide and record it first")
        cap_memory(mc["memory_cap_gb"])
        dt = su.torch_dtype(mc["dtype"])
        fp32 = mc["dtype"] == "float32"
        tol_rows, tol_lin = (1e-5, 1e-5) if fp32 else (1e-2, 2e-2)
        model = load_replacement_model(mc["model_name"], mc["transcoder_set"], dtype=dt)
        # prompt lengths with the real tokenizer
        lengths = {k: [int(model.ensure_tokenized(p["prompt"]).numel()) for p in v] for k, v in test_prompts.items()}
        res = {"K0.1": [], "K0.2": [], "K0.3": [], "K0.4": [], "K0.6": None}
        fwd_mode = "fwad"
        for p in test_prompts["test"][:3]:
            g = attribute(p["prompt"], model, max_n_logits=ex["max_n_logits"], desired_logit_prob=ex["desired_logit_prob"],
                          batch_size=ex["batch_size"], max_feature_nodes=ex["max_feature_nodes"])
            n_log = len(g.logit_targets)
            adj = g.adjacency_matrix.double()
            nf = len(g.selected_features)
            op = CircuitTracerOperator(model, p["prompt"], max_n_logits=ex["max_n_logits"],
                                       desired_logit_prob=ex["desired_logit_prob"], batch_size=ex["batch_size"],
                                       target_features=g.selected_features, source_features=g.selected_features)
            N = op.N
            # K0.1: our one-hot rows (additive injection) vs circuit-tracer's adjacency rows (subset of rows)
            rows_idx = torch.cat([torch.arange(n_log), n_log + torch.arange(min(512, nf))])
            ours = op.exact_rows(rows_idx).double().cpu()
            ref = torch.cat([adj[-n_log:, :N], adj[: min(512, nf), :N]])
            res["K0.1"].append(max_rel_rowwise(ours, ref))
            # K0.2: linearity over all targets of the default graph
            A_ex = torch.cat([adj[-n_log:, :N], adj[:nf, :N]])
            gen = torch.Generator().manual_seed(0)
            S = (torch.randint(0, 2, (5, op.T), generator=gen) * 2 - 1).double()
            res["K0.2"].append(rel(op.rows(S.to(op.dtype)), S @ A_ex))
            # K0.3: 20 sources including tokens and error nodes
            kinds = op.source_meta.kind
            pick = torch.cat([torch.nonzero(kinds == k).flatten()[torch.randperm(int((kinds == k).sum()), generator=gen)[:7]]
                              for k in kinds.unique()])[:20]
            try:
                cols = op.exact_cols(pick)
            except Exception as e:  # forward AD blocked by a hook: fall back and record it
                log.info(f"forward AD failed ({e!r}); using double backward")
                op.forward_mode = fwd_mode = "double_backward"
                cols = op.exact_cols(pick)
            res["K0.3"].append(rel(cols, A_ex[:, pick].T))
            # K0.4: our pruning vs circuit-tracer's prune_graph on the same graph
            ct = prune_graph(g, ex["node_threshold"], ex["edge_threshold"])
            ours_pr = prune_adjacency(adj, nf, len(g.input_tokens), g.logit_probabilities.double().cpu(),
                                      ex["node_threshold"], ex["edge_threshold"])
            res["K0.4"].append(bool(torch.equal(ct.node_mask.cpu(), ours_pr.node_mask) and
                                    torch.equal(ct.edge_mask.cpu(), ours_pr.edge_mask)))
            del op
            torch.cuda.empty_cache()
        # K0.5 timing (first prompt)
        p0 = test_prompts["test"][0]["prompt"]
        op = CircuitTracerOperator(model, p0, batch_size=ex["batch_size"], forward_mode=fwd_mode)
        b = ex["batch_size"]
        idx = torch.arange(op.n_logits, op.n_logits + b) % op.T
        seeds = torch.randn(b, op.T)
        t_rows, _ = median_time(lambda: op.rows(seeds.to(op.dtype)), 3)
        t_cols, _ = median_time(lambda: op.cols(torch.randn(b, op.N).to(op.dtype)), 3)
        torch.cuda.reset_peak_memory_stats()
        t_exact, all_t = median_time(lambda: attribute(p0, model, max_n_logits=ex["max_n_logits"],
                                                       desired_logit_prob=ex["desired_logit_prob"],
                                                       batch_size=b, max_feature_nodes=ex["max_feature_nodes"]), 3)
        timing = {"row_seconds": t_rows / b, "kappa": t_cols / t_rows, "batch": b,
                  "recovery_columns_per_second": recovery_throughput(min(op.T, ex["max_feature_nodes"] + 10), device="cuda"),
                  "exact_default_wall_seconds": t_exact, "exact_default_wall_all": all_t,
                  "exact_default_peak_gb": torch.cuda.max_memory_allocated() / 1024 ** 3}
        # K0.6 truncation: real capped run vs prefix of the full influence-ordered run
        full = op.exact_attribution(ex["max_feature_nodes"], ex["update_interval"])
        g1024 = attribute(p0, model, max_n_logits=ex["max_n_logits"], desired_logit_prob=ex["desired_logit_prob"],
                          batch_size=b, max_feature_nodes=1024)
        prefix = torch.sort(full["visit_order_features"][:1024]).values
        same_set = bool(torch.equal(prefix.cpu(), torch.sort(g1024.selected_features.cpu()).values))
        res["K0.6"] = same_set
        checks = {
            "K0.1": {"value": max(res["K0.1"]), "pass": max(res["K0.1"]) <= tol_rows},
            "K0.2": {"value": max(res["K0.2"]), "pass": max(res["K0.2"]) <= tol_lin},
            "K0.3": {"value": max(res["K0.3"]), "pass": max(res["K0.3"]) <= tol_lin, "mode": fwd_mode},
            "K0.4": {"value": all(res["K0.4"]), "pass": all(res["K0.4"])},
            "K0.5": {"value": timing, "pass": True},
            "K0.6": {"value": same_set, "pass": True,
                     "note": "if False, B1 uses real capped runs at each budget (logged deviation)"},
        }
        out[set_name] = {"checks": checks, "timing": timing, "dtype": mc["dtype"], "forward_mode": fwd_mode,
                         "prompt_token_lengths": lengths}
        mc["timing"].update({k: timing[k] for k in ("row_seconds", "kappa", "recovery_columns_per_second",
                                                     "exact_default_wall_seconds", "exact_default_peak_gb")})
        su.save_yaml(os.path.join(su.ROOT, "configs", "models", f"{set_name}.yaml"), mc)
        del model
        torch.cuda.empty_cache()
    return out


def main() -> None:
    args = su.parser("K0 sanity checks").parse_args()
    cfg = su.load_yaml(args.config)
    log = get_logger("sanity_k0", os.path.join(su.ROOT, "logs"))
    path = os.path.join(su.results_dir(args), "k0.json")
    if os.path.exists(path) and not args.force:
        log.info(f"{path} exists; use --force to recompute")
        k0 = read_json(path)
    else:
        t0 = time.perf_counter()
        sets = tiny_checks(args, cfg) if args.tiny else real_checks(args, cfg, log)
        k0 = run_record(cfg if not args.tiny else cfg["tiny"], args.seed, time.perf_counter() - t0, sets=sets,
                        tiny=args.tiny)
        atomic_write_json(path, k0)
    ok = all(c["pass"] for s in k0["sets"].values() for c in s["checks"].values())
    for s, v in k0["sets"].items():
        for name, c in v["checks"].items():
            log.info(f"{s} {name}: pass={c['pass']} value={c['value']}")
    log_status(f"K0 ({'tiny' if args.tiny else 'real'}): {'all pass' if ok else 'FAILED'}",
               os.path.join(su.ROOT, "results", "STATUS.md")) if not args.tiny else None
    if not ok:
        log.info("K0 FAILED: stop, fix and rerun before any exact-graph run")
        sys.exit(1)


if __name__ == "__main__":
    main()
