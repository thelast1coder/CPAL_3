"""Gate 0 premise tests, Stage B (GATE0_PREMISES.md Section 3, PREREG_GATE0.md): the 51 reference
graphs and PR4 (transcoder quality on thinking text).

Settings: configs/experiments/gate0_premises.yaml (exact, reference_graphs, stage_b); thresholds:
configs/prereg_gate0.yaml (stage_b.pr4, reference_graphs). Traces: results/gate0/traces.json from
scripts/gen_traces.py (committed before this runs).

Reference graphs. For every trace and each of its 17 sampled outputs (cct.premise_traces), the
exact influence-ordered graph of the output at position q on the prefix tokens[:q], with
circuit-tracer's own machinery (CircuitTracerOperator + cct.killtest.exact_graph_real: compute_batch
rows, compute_partial_influences visiting order, update_interval 4), feature-row budget
reference_graphs.budget (16384), batch exact.batch_size_trace (64), max_n_logits 10, desired logit
probability 0.95, pruning 0.8 / 0.98. Convergence: node Jaccard between the pruned graph of the
first convergence_budget (8192) visited feature rows and the full-budget pruned graph must be
>= 0.95 (a graph that visited every active feature passes); otherwise the output is recomputed with
raised_budget (32768) on the same cached forward pass, checked against 16384, and logged. Each
record (packed rows over [logits, visited features] x [visited features, errors, tokens], visiting
order, exact prune) is saved to artifacts/<run>/<set>/premise_refs/<trace_id>/<q>.pt with the extra
keys position, prefix_offset, sentence_index, output_kind, budget, conv_ok, conv_jaccard, trace_id
(plus logit_token_ids and model_tokens). Index: results/gate0/reference_graphs.json.

Compute order (partial runs stay balanced): every trace's answer, then the other faithfulness
outputs, then the remaining sentence outputs round-robin over traces. Resumable: an output whose
.pt and per-output JSON (<q>.b.json next to it) exist is skipped unless --force.

PR4, on the 51 trace outputs and the 3 Qwen short prompts (configs/prompts/gate0_qwen_short.json,
graph at the circuit-tracer default cap 7500 features, batch 256, saved under premise_short/):
1. completeness and replacement scores: circuit-tracer 0.5.0 compute_graph_scores
   (cct.graph.metrics.graph_scores) on the graph of the record (all visited features, all error and
   token nodes, the logits);
2. pruned graph size: kept nodes (all kinds, as circuit-tracer's pruned graph) and kept edges;
3. faithfulness, on 8 trace outputs (every answer, plus sentence outputs spread over the traces,
   cct.premise_be.faithfulness_selection) and the 3 short prompts: each of the top 10 feature nodes
   by node influence is set to 0 one at a time in the real model with circuit-tracer's
   ReplacementModel.feature_intervention (freeze_attention=True, its default), and the change of
   the output metric m = sum_t p_t (logit_t - mean logit) over the graph's logit targets (the
   prob-weighted demeaned logits that circuit-tracer's logit nodes represent) is compared with the
   graph-predicted change. Decision (PREREG_GATE0 Section 2): predicted = minus the direct-path
   attribution, -sum_t p_t A[logit_t, f]. Reported too: the direct + indirect prediction through the
   graph's features (cct.premise_be.predicted_logit_change). Spearman per output over the 10
   features; median over outputs.
Status per the prereg: kill (median trace completeness < 0.50 or median trace faithfulness < 0.2),
pass (all four pass conditions), else fail. Informative flag: median pruned nodes on traces > 2000.

Stop rule B. On Qwen3-1.7B, a kill sets stop (exit code 2): switch model before anything else
(scripts/run_premises.sh runs `premise_B.py --model-set qwen3_4b_plt`). On Qwen3-4B (SWITCH-MODEL
path; traces regenerated under results/gate0/qwen3_4b/ by gen_traces.py if missing; default time cap
from prereg stage_b.pr4.switch_model), a kill when Qwen3-1.7B was also killed sets stop: whole-trace
science is not viable on available transcoders.

Output: results/gate0/stage_B.json (stage_B_qwen3_4b.json) = {"run", "stage": "B", "premises":
{"PR4": {...}}, "stop", "stop_reason", "gpu_seconds", "detail"}. Every graph logs its seconds and
peak GPU memory.

--tiny: tiny pure-PyTorch model (cct.premise_traces.TINY_PREMISE_MODEL), tiny traces, budgets
scaled down (TINY below) to run every code path; the "real model" ablation is
TinyOperator.replacement_forward with feature_scale (live features, frozen errors, attention not
frozen). Numbers are meaningless.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import torch

from cct import scriptutil as su
from cct.gate0_union import convergence_check
from cct.killtest import load_prompts, record_matrix
from cct.premise_be import (faithfulness_selection, graph_stats, pr4_evaluate, predicted_logit_change,
                            reference_order, spearman_or_nan, top_influence_features)
from cct.premise_traces import (TINY_PREMISE_MODEL, TINY_PROMPT_LENGTHS, gate0_results_dir, load_traces, model_tag,
                                tiny_premise_model, trace_outputs, traces_file)
from cct.utils.io import atomic_write_json, load_pt, read_json, save_pt
from cct.utils.logging import get_logger, run_record

DEFAULT_CONFIG = os.path.join(su.ROOT, "configs", "experiments", "gate0_premises.yaml")
PREREG = os.path.join(su.ROOT, "configs", "prereg_gate0.yaml")
TINY = {"budget": 64, "convergence_budget": 32, "raised_budget": 128, "batch_size_trace": 64,
        "short_cap": 7500, "batch_size_short": 256}
SHORT_CAP = 7500  # circuit-tracer CLI default max_feature_nodes
EXIT_STOP = 2


def settings(args, cfg: dict, prereg: dict) -> dict:
    ex, rg = cfg["exact"], cfg["reference_graphs"]
    s = {"max_n_logits": ex["max_n_logits"], "desired_logit_prob": ex["desired_logit_prob"],
         "update_interval": ex["update_interval"],
         "prune_cfg": {"node_threshold": ex["node_threshold"], "edge_threshold": ex["edge_threshold"]},
         "budget": rg["budget"], "convergence_budget": rg["convergence_budget"], "raised_budget": rg["raised_budget"],
         "convergence_jaccard": prereg["reference_graphs"]["convergence_jaccard_at_least"],
         "batch_size_trace": ex["batch_size_trace"], "short_cap": SHORT_CAP, "batch_size_short": ex["batch_size_short"],
         "n_sentences": cfg["traces"]["sentences_per_trace"], "tokens_per_sentence": cfg["traces"]["tokens_per_sentence"],
         "faithfulness_outputs": cfg["stage_b"]["faithfulness_trace_outputs"],
         "top_k": cfg["stage_b"]["top_k_features"]}
    if args.tiny:
        s.update(TINY)
    return s


class Paths:
    def __init__(self, args, cfg: dict, model_set: str, tag: str):
        self.refs = su.artifacts_dir(args, model_set, "premise_refs")
        self.short = su.artifacts_dir(args, model_set, "premise_short")
        base = gate0_results_dir(su.ROOT, args.tiny)
        self.res = gate0_results_dir(su.ROOT, args.tiny, tag)
        os.makedirs(self.res, exist_ok=True)
        self.traces = traces_file(su.ROOT, args.tiny, tag, cfg)
        self.index = os.path.join(self.res, "reference_graphs.json")
        self.out = os.path.join(base, f"stage_B{'_' + tag if tag else ''}.json")
        self.stage_b_default = os.path.join(base, "stage_B.json")

    def pt(self, tid: str, q: int) -> str:
        return os.path.join(self.refs, tid, f"{q}.pt")

    @staticmethod
    def js(pt: str) -> str:
        return pt[:-3] + ".b.json"


# -------------------------------------------------------------------------------------------------
class Engine:
    """Loads the model once; builds graphs; runs ablations."""

    def __init__(self, args, cfg: dict, model_set: str, s: dict, log):
        self.args, self.cfg, self.model_set, self.s, self.log = args, cfg, model_set, s, log
        self.model = None
        self.load_seconds = 0.0

    def ensure_model(self):
        if self.model is not None:
            return
        t0 = time.perf_counter()
        if self.args.tiny:
            self.model = tiny_premise_model()
        else:
            from cct.operators.circuit_tracer_backend import load_replacement_model
            from cct.utils.gpu import cap_memory, nvidia_smi

            mc = su.model_config(self.model_set)
            self.log.info(nvidia_smi())
            cap_memory(mc["memory_cap_gb"])
            self.model = load_replacement_model(mc["model_name"], mc["transcoder_set"],
                                                dtype=su.torch_dtype(mc["dtype"]))
        self.load_seconds += time.perf_counter() - t0

    def build(self, prompt, budget: int, batch: int, q: int | None, op=None):
        """(record, handle). prompt: token list (traces) or string (real short prompts) or token
        list (tiny short prompts). handle: the operator (real) or the token tensor (tiny); pass it
        back as op to rebuild with a larger budget on the same cached forward pass."""
        self.ensure_model()
        s = self.s
        t0 = time.perf_counter()
        if self.args.tiny:
            from cct.killtest import exact_graph_tiny

            toks = torch.as_tensor(prompt, dtype=torch.long)
            rec = exact_graph_tiny(self.model, toks, budget, batch, "D", s["prune_cfg"])
            rec.update(pos_offset=0, peak_gb=0.0, n_pos=int(toks.numel()), model_tokens=toks)
            handle = toks
        else:
            from cct.killtest import exact_graph_real
            from cct.operators.circuit_tracer_backend import CircuitTracerOperator

            torch.cuda.reset_peak_memory_stats()
            if op is None:
                if not isinstance(prompt, str):
                    prompt = torch.as_tensor(prompt, dtype=torch.long)
                op = CircuitTracerOperator(self.model, prompt, max_n_logits=s["max_n_logits"],
                                           desired_logit_prob=s["desired_logit_prob"], batch_size=batch)
            off = int(op.n_pos) - q if q is not None else 0
            if off not in (0, 1):
                raise RuntimeError(f"unexpected prefix length {op.n_pos} for q={q}")
            rec = exact_graph_real(op, budget, "D", s["update_interval"], s["prune_cfg"])
            rec.update(pos_offset=off, n_pos=int(op.n_pos), model_tokens=op.tokens.detach().cpu(),
                       logit_token_ids=op.targets.token_ids.detach().cpu(),
                       peak_gb=torch.cuda.max_memory_allocated() / 1024 ** 3)
            handle = op
        rec["graph_seconds"] = time.perf_counter() - t0
        rec["budget"] = int(budget)
        return rec, handle

    def measure(self, rec: dict, src: list[int]) -> dict:
        """Measured metric change for zeroing each source feature (one at a time) in the model."""
        self.ensure_model()
        sm = rec["source_meta"]
        t0 = time.perf_counter()
        if self.args.tiny:
            from cct.operators.tiny import TinyOperator

            top = TinyOperator(self.model, torch.as_tensor(rec["model_tokens"]).long())
            key = {tuple(int(x) for x in top.active[i]): i for i in range(top.active.shape[0])}
            with torch.no_grad():
                base = float(top.replacement_forward())
                out = []
                for s in src:
                    fs = torch.ones(top.active.shape[0], dtype=self.model.dtype)
                    fs[key[(int(sm["layer"][s]), int(sm["pos"][s]), int(sm["feature"][s]))]] = 0.0
                    out.append(float(top.replacement_forward(fs)) - base)
        else:
            dev = self.model.cfg.device
            toks = torch.as_tensor(rec["model_tokens"]).long().to(dev)
            ids = torch.as_tensor(rec["logit_token_ids"]).long().to(dev)
            p = torch.as_tensor(rec["target_meta"]["extra"][: ids.numel()]).double().to(dev)

            def metric(logits) -> float:
                lg = logits[0, -1].double()
                return float((p * (lg[ids] - lg.mean())).sum())

            with torch.no_grad():
                base = metric(self.model(toks))
                out = []
                for s in src:
                    iv = [(int(sm["layer"][s]), int(sm["pos"][s]), int(sm["feature"][s]), 0.0)]
                    logits, _ = self.model.feature_intervention(toks, iv, freeze_attention=True,
                                                                return_activations=False)
                    out.append(metric(logits) - base)
                torch.cuda.synchronize()
        return {"metric_base": base, "measured": out, "seconds": time.perf_counter() - t0}


def faithfulness(eng: Engine, rec: dict, k: int, A=None) -> dict:
    src = top_influence_features(rec, k)
    pred = predicted_logit_change(rec, src, A)
    m = eng.measure(rec, src)
    sm = rec["source_meta"]
    off = int(rec.get("pos_offset", 0))
    return {"features": [[int(sm["layer"][s]), int(sm["pos"][s]) - off, int(sm["feature"][s])] for s in src],
            "feature_activation": [float(sm["extra"][s]) for s in src],
            "predicted_direct": pred["direct"], "predicted_total": pred["total"], "measured": m["measured"],
            "metric_base": m["metric_base"], "spearman_direct": spearman_or_nan(pred["direct"], m["measured"]),
            "spearman_total": spearman_or_nan(pred["total"], m["measured"]), "seconds": m["seconds"],
            "metric": "sum_t p_t (logit_t - mean logit), graph logit targets",
            "intervention": "tiny: replacement_forward feature_scale 0" if eng.args.tiny
            else "ReplacementModel.feature_intervention value 0, freeze_attention=True"}


def build_with_convergence(eng: Engine, prompt, q: int, s: dict, log, tag: str) -> dict:
    rec, h = eng.build(prompt, s["budget"], s["batch_size_trace"], q)
    conv = convergence_check(rec, s["convergence_budget"], s["convergence_jaccard"])
    first = {"budget": rec["budget"], "conv": conv, "seconds": rec["graph_seconds"], "peak_gb": rec["peak_gb"],
             "n_active": rec["n_active"]}
    raised = False
    gsec = rec["graph_seconds"]
    if not conv["ok"]:
        log.info(f"{tag}: convergence failed (Jaccard {conv['node_jaccard']:.3f}); raising the budget to "
                 f"{s['raised_budget']}")
        del rec
        rec, h = eng.build(prompt, s["raised_budget"], s["batch_size_trace"], q, op=None if eng.args.tiny else h)
        conv = convergence_check(rec, s["budget"], s["convergence_jaccard"])
        raised = True
        gsec += rec["graph_seconds"]
    rec["_handle"] = h
    rec["premise_b"] = {"first_attempt": first, "raised": raised, "conv": conv, "graph_seconds_total": gsec}
    return rec


def entry_from(rec: dict, extra: dict) -> dict:
    pb = rec["premise_b"]
    e = {k: rec.get(k) for k in ("T_D", "N", "n_active", "n_pos", "budget", "peak_gb")}
    e.update(extra)
    e.update(conv_ok=bool(pb["conv"]["ok"]), conv_jaccard=float(pb["conv"]["node_jaccard"]), conv=pb["conv"],
             raised=pb["raised"], first_attempt=pb["first_attempt"], graph_seconds=pb["graph_seconds_total"],
             prefix_offset=int(rec.get("pos_offset", 0)))
    return e


# -------------------------------------------------------------------------------------------------
def ensure_traces(args, P: Paths, model_set: str, log) -> None:
    if os.path.exists(P.traces):
        return
    cmd = [sys.executable, os.path.join(su.ROOT, "scripts", "gen_traces.py"), "--model-set", model_set]
    if args.tiny:
        cmd.append("--tiny")
    if args.tiny or model_set != su.load_yaml(DEFAULT_CONFIG)["sets"]["qwen"]:
        log.info(f"{P.traces} missing: generating ({' '.join(cmd[1:])})")
        r = subprocess.run(cmd, cwd=su.ROOT)
        if r.returncode not in (0, 3) or not os.path.exists(P.traces):
            raise SystemExit(f"trace generation failed (exit {r.returncode})")
        if not args.tiny:
            log.warning(f"generated {P.traces}: commit it before reading the results")
    else:
        raise SystemExit(f"{P.traces} missing: run scripts/gen_traces.py{' --tiny' if args.tiny else ''} first")


def short_prompts(args, cfg: dict) -> list[dict]:
    if args.tiny:
        from cct.operators.tiny import random_prompts

        toks = random_prompts(3, TINY_PREMISE_MODEL["vocab"], TINY_PROMPT_LENGTHS, seed=cfg["seed"] + 200)
        return [{"id": f"tiny_short_{i:02d}", "prompt": t.tolist()} for i, t in enumerate(toks)]
    q = cfg["prompts"]["qwen"]
    ps = {p["id"]: p for p in load_prompts(os.path.join(su.ROOT, q["file"]))}
    return [ps[i] for i in q["ids"]]


def main() -> None:
    p = su.parser("Gate 0 Stage B: reference graphs and PR4")
    p.set_defaults(config=DEFAULT_CONFIG)
    p.add_argument("--model-set", default=None, help="default sets.qwen; qwen3_4b_plt for SWITCH-MODEL")
    p.add_argument("--max-minutes", type=float, default=None,
                   help="stop starting new graphs after this many minutes (default: none for Qwen3-1.7B; "
                        "the prereg switch-model cap for other sets)")
    args = p.parse_args()
    t_start = time.perf_counter()
    cfg = su.load_yaml(args.config)
    prereg = su.load_yaml(PREREG)
    default_set = cfg["sets"]["qwen"]
    model_set = args.model_set or default_set
    tag = model_tag(model_set, default_set)
    thr = prereg["stage_b"]["pr4"]
    max_min = args.max_minutes
    if max_min is None and tag:
        max_min = 60.0 * thr["switch_model"]["time_hours_at_most"]
    s = settings(args, cfg, prereg)
    P = Paths(args, cfg, model_set, tag)
    log = get_logger("premise_B", os.path.join(su.ROOT, "logs"))
    ensure_traces(args, P, model_set, log)
    traces = load_traces(P.traces)
    outs = {t["id"]: trace_outputs(t, s["n_sentences"], s["tokens_per_sentence"]) for t in traces}
    by_id = {t["id"]: t for t in traces}
    faithful = set(faithfulness_selection(outs, s["faithfulness_outputs"]))
    order = reference_order(outs, faithfulness_selection(outs, s["faithfulness_outputs"]))
    eng = Engine(args, cfg, model_set, s, log)
    per_graph_secs: list[float] = []
    stopped_by_time = None

    def over_time() -> bool:
        if max_min is None:
            return False
        el = time.perf_counter() - t_start
        nxt = sorted(per_graph_secs)[len(per_graph_secs) // 2] if per_graph_secs else 0.0
        return el + nxt > 60.0 * max_min

    # ---- short prompts ----------------------------------------------------------------------
    short_entries = []
    for sp in short_prompts(args, cfg):
        pt = os.path.join(P.short, f"{sp['id']}.pt")
        js = Paths.js(pt)
        if os.path.exists(js) and os.path.exists(pt) and not args.force:
            short_entries.append(read_json(js))
            continue
        if over_time():
            stopped_by_time = f"time cap {max_min} min reached before short prompt {sp['id']}"
            break
        t0 = time.perf_counter()
        rec, h = eng.build(sp["prompt"], s["short_cap"], s["batch_size_short"], None)
        A = record_matrix(rec)
        st = graph_stats(rec, A)
        f = faithfulness(eng, rec, s["top_k"], A)
        del h
        rec.update(prompt_id=sp["id"])
        save_pt(pt, rec)
        e = {k: rec.get(k) for k in ("T_D", "N", "n_active", "n_pos", "budget", "peak_gb", "graph_seconds")}
        e.update(st, id=sp["id"], faithfulness=f, gpu_seconds=time.perf_counter() - t0,
                 path=os.path.relpath(pt, su.ROOT).replace(os.sep, "/"))
        atomic_write_json(js, e)
        short_entries.append(e)
        per_graph_secs.append(e["gpu_seconds"])
        log.info(f"short {sp['id']}: n_active={rec['n_active']} T={rec['T_D']} completeness="
                 f"{st['completeness_score']:.3f} faith={f['spearman_direct']:.3f} {e['gpu_seconds']:.1f}s "
                 f"peak {rec['peak_gb']:.1f} GB")
        del rec, A
        if not args.tiny:
            torch.cuda.empty_cache()

    # ---- trace outputs ----------------------------------------------------------------------
    trace_entries = []
    if stopped_by_time is None:
        for o in order:
            tid, q = o["trace_id"], int(o["position"])
            pt, tag_o = P.pt(tid, q), f"{tid} q={q}"
            js = Paths.js(pt)
            want_f = (tid, q) in faithful
            if os.path.exists(js) and os.path.exists(pt) and not args.force:
                e = read_json(js)
                if not want_f or e.get("faithfulness"):
                    trace_entries.append(e)
                    continue
            if over_time():
                stopped_by_time = f"time cap {max_min} min reached; {len(trace_entries)} of {len(order)} outputs done"
                log.warning(stopped_by_time)
                break
            t0 = time.perf_counter()
            meta = {"trace_id": tid, "position": q, "sentence_index": o["sentence_index"], "output_kind": o["kind"]}
            if os.path.exists(pt) and not args.force:
                rec = load_pt(pt)
                h = None
            else:
                rec = build_with_convergence(eng, by_id[tid]["token_ids"][:q], q, s, log, tag_o)
                h = rec.pop("_handle")
                rec.update(meta, prefix_offset=int(rec["pos_offset"]), budget=int(rec["budget"]),
                           conv_ok=bool(rec["premise_b"]["conv"]["ok"]),
                           conv_jaccard=float(rec["premise_b"]["conv"]["node_jaccard"]))
                rec["run_record"] = run_record({"premise_B": s, "model_set": model_set, "tiny": args.tiny},
                                               cfg["seed"], rec["premise_b"]["graph_seconds_total"])
                save_pt(pt, rec)
            A = record_matrix(rec)
            st = graph_stats(rec, A)
            f = faithfulness(eng, rec, s["top_k"], A) if want_f else None
            del h
            e = entry_from(rec, meta)
            e.update(st, faithfulness=f, faithfulness_selected=want_f, gpu_seconds=time.perf_counter() - t0,
                     path=os.path.relpath(pt, su.ROOT).replace(os.sep, "/"))
            atomic_write_json(js, e)
            trace_entries.append(e)
            per_graph_secs.append(e["gpu_seconds"])
            log.info(f"{tag_o} ({o['kind']}): n_active={e['n_active']} T={e['T_D']} budget={e['budget']} "
                     f"conv={e['conv_jaccard']:.3f} completeness={st['completeness_score']:.3f} "
                     f"pruned nodes={st['n_nodes_pruned']} "
                     + (f"faith={f['spearman_direct']:.3f} " if f else "")
                     + f"{e['gpu_seconds']:.1f}s peak {e.get('peak_gb') or 0:.1f} GB")
            del rec, A
            if not args.tiny:
                torch.cuda.empty_cache()

    # ---- index --------------------------------------------------------------------------------
    keys = ("trace_id", "position", "prefix_offset", "sentence_index", "output_kind", "budget", "conv_ok",
            "conv_jaccard", "raised", "T_D", "N", "n_active", "n_pos", "graph_seconds", "peak_gb", "path")
    pos_rank = {(o["trace_id"], o["position"]): i for i, o in enumerate(o2 for t in outs for o2 in outs[t])}
    idx = sorted(({k: e.get(k) for k in keys} for e in trace_entries),
                 key=lambda r: pos_rank.get((r["trace_id"], r["position"]), 1 << 30))
    atomic_write_json(P.index, {"model_set": model_set, "tiny": args.tiny, "n_expected": len(order),
                                "n_done": len(idx), "settings": s, "records": idx})

    # ---- PR4 ---------------------------------------------------------------------------------
    ev = pr4_evaluate(trace_entries, short_entries, thr)
    notes = [ev["notes"]] if ev["notes"] else []
    if stopped_by_time:
        notes.append(f"partial: {stopped_by_time}")
    if len(trace_entries) < len(order):
        notes.append(f"{len(trace_entries)} of {len(order)} trace outputs measured")
    n_conv_fail = sum(not e["conv_ok"] for e in trace_entries)
    if n_conv_fail:
        notes.append(f"{n_conv_fail} reference graphs did not converge even at the raised budget")
    if ev["values"].get("informative_graphs_too_big"):
        notes.append("informative: median pruned trace graph above "
                     f"{thr['informative_median_pruned_nodes_above']} nodes (too big to read)")
    status = ev["status"]
    stop, reason = False, None
    if status == "kill" and not tag:
        stop = True
        reason = ("PR4 kill on Qwen3-1.7B: switch model before anything else (premise_B.py --model-set "
                  "qwen3_4b_plt); stop rule B applies if Qwen3-4B also fails")
    elif status == "kill" and tag:
        st17 = read_json(P.stage_b_default)["premises"]["PR4"]["status"] if os.path.exists(P.stage_b_default) else None
        if st17 == "kill":
            stop = True
            reason = ("Stop rule B: PR4 kill on Qwen3-1.7B and Qwen3-4B; whole-trace science is not viable on "
                      "available transcoders (structure-only path remains open)")
    gpu_s = sum(float(e.get("gpu_seconds") or 0) for e in trace_entries + short_entries) + eng.load_seconds
    secs = [float(e.get("graph_seconds") or 0) for e in trace_entries]
    peaks = [float(e.get("peak_gb") or 0) for e in trace_entries]
    res = {
        "run": run_record({"premise_B": s, "model_set": model_set, "tiny": args.tiny, "max_minutes": max_min},
                          cfg["seed"], time.perf_counter() - t_start),
        "stage": "B",
        "premises": {"PR4": {"status": status, "values": ev["values"], "thresholds": thr, "notes": "; ".join(notes)}},
        "stop": stop, "stop_reason": reason, "gpu_seconds": gpu_s,
        "detail": {
            "model_set": model_set, "settings": s, "model_load_seconds": eng.load_seconds,
            "faithfulness_outputs": [list(x) for x in faithfulness_selection(outs, s["faithfulness_outputs"])],
            "faithfulness_definition": "Spearman over the top-10 influence features between the measured change of "
                                       "sum_t p_t (logit_t - mean logit) when the feature is set to 0 "
                                       "(feature_intervention, freeze_attention=True) and the predicted change "
                                       "-sum_t p_t A[logit_t, f] (direct path; decision). spearman_total: direct + "
                                       "indirect paths through graph features (reported only)",
            "graph_seconds": {"median": sorted(secs)[len(secs) // 2] if secs else None, "total": sum(secs)},
            "peak_gb_max": max(peaks) if peaks else None,
            "n_convergence_failures": n_conv_fail, "stopped_by_time": stopped_by_time,
            "reference_index": os.path.relpath(P.index, su.ROOT).replace(os.sep, "/"),
            "trace_outputs": trace_entries, "short_prompts": short_entries,
        },
    }
    atomic_write_json(P.out, res)
    log.info(f"PR4 {status}: completeness {ev['values']['median_completeness_traces']:.3f} (short "
             f"{ev['values']['median_completeness_short']:.3f}), faithfulness "
             f"{ev['values']['median_faithfulness_spearman_traces']:.3f} (short "
             f"{ev['values']['median_faithfulness_spearman_short']:.3f}) -> {P.out}")
    if stop:
        log.warning(f"STOP: {reason}")
        sys.exit(EXIT_STOP)


if __name__ == "__main__":
    main()
