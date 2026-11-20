"""Gate 0 premise tests, Stages B (PR4) and E (PR10): pure logic, CPU-testable. The scripts are
scripts/premise_B.py and scripts/premise_E.py; definitions are in their docstrings and here.

Conventions shared with cct.gate0_union: an output at position q explains the prediction of token q,
so its graph is circuit-tracer on tokens[:q]; record positions are in the model's token coordinates,
which are trace coordinates plus `prefix_offset` (1 if circuit-tracer prepended a special token).
"""
from __future__ import annotations

import math
import statistics
from typing import Callable, Iterable

import torch
from torch import Tensor

from .gate0_union import n_logit_rows, visited_features
from .graph.export import make_layout, to_square
from .graph.metrics import graph_scores
from .killtest import record_matrix
from .operators.base import NodeMeta

NAN = float("nan")


def _finite(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def median_finite(xs: Iterable) -> float:
    v = [float(x) for x in xs if _finite(x)]
    return statistics.median(v) if v else NAN


# =================================================================================================
# Stage B
# =================================================================================================
def first_token_outputs(outputs: list[dict]) -> list[dict]:
    """The first sampled token of each sampled sentence, in sentence order."""
    seen, out = set(), []
    for o in sorted((o for o in outputs if o["kind"] == "sentence_token"), key=lambda o: o["position"]):
        if o["sentence_index"] not in seen:
            seen.add(o["sentence_index"])
            out.append(o)
    return out


def faithfulness_selection(outputs_by_trace: dict[str, list[dict]], n: int = 8) -> list[tuple[str, int]]:
    """(trace_id, position) of the n trace outputs used for ablation faithfulness: the answer of
    every trace (trace order), then n - n_traces sentence outputs spread over the traces round-robin
    (trace i gets k_i of them); within a trace, the first token of its sampled sentences at evenly
    spaced slots floor((2j + 1) S / (2 k_i)), j < k_i, of its S sampled sentences."""
    tids = list(outputs_by_trace)
    sel = [(t, o["position"]) for t in tids for o in outputs_by_trace[t] if o["kind"] == "answer"][:n]
    rest = n - len(sel)
    k = {t: 0 for t in tids}
    for j in range(max(0, rest)):
        k[tids[j % len(tids)]] += 1
    picks = {}
    for t in tids:
        firsts = first_token_outputs(outputs_by_trace[t])
        S = len(firsts)
        kt = min(k[t], S)
        picks[t] = [firsts[((2 * j + 1) * S) // (2 * kt)]["position"] for j in range(kt)] if kt else []
    for j in range(max((len(v) for v in picks.values()), default=0)):
        for t in tids:
            if j < len(picks[t]):
                sel.append((t, picks[t][j]))
    return sel


def reference_order(outputs_by_trace: dict[str, list[dict]], faithful: list[tuple[str, int]]) -> list[dict]:
    """Compute order of the reference graphs, balanced for partial runs: every trace's answer, then
    the faithfulness outputs, then the remaining sentence outputs round-robin over traces (each
    trace in sentence order)."""
    tids = list(outputs_by_trace)
    by = {(t, o["position"]): dict(o, trace_id=t) for t in tids for o in outputs_by_trace[t]}
    order, seen = [], set()

    def add(key):
        if key in by and key not in seen:
            seen.add(key)
            order.append(by[key])

    for t in tids:
        for o in outputs_by_trace[t]:
            if o["kind"] == "answer":
                add((t, o["position"]))
    for key in faithful:
        add(key)
    queues = {t: sorted((o for o in outputs_by_trace[t] if o["kind"] != "answer"), key=lambda o: o["position"])
              for t in tids}
    while any(queues.values()):
        for t in tids:
            while queues[t]:
                o = queues[t].pop(0)
                if (t, o["position"]) not in seen:
                    add((t, o["position"]))
                    break
    return order


def record_layout(rec: dict):
    return make_layout(NodeMeta.from_dict(rec["target_meta"]), NodeMeta.from_dict(rec["source_meta"]))


def graph_stats(rec: dict, A: Tensor | None = None) -> dict:
    """Replacement and completeness scores (circuit-tracer compute_graph_scores, on the graph of the
    record: every visited feature, all error and token nodes, the logits), and the size of the exact
    pruned graph at the record's pruning settings. Node counts: all kept nodes (features, errors,
    tokens, logits, as in circuit-tracer's pruned graph) and kept feature nodes."""
    A = record_matrix(rec) if A is None else A
    lay = record_layout(rec)
    sq = to_square(A, lay)
    repl, comp = graph_scores(sq, lay.n_features, lay.n_errors, lay.n_tokens, lay.logit_probs)
    del sq
    e = rec["exact_prune"]
    nm = e["node_mask"]
    em = e["edge_mask"]
    n_edges = int(em.coalesce().values().bool().sum()) if em.is_sparse else int(em.sum())
    nf = lay.n_features
    return {"replacement_score": repl, "completeness_score": comp, "n_nodes_pruned": int(nm.sum()),
            "n_feature_nodes_pruned": int(nm[:nf].sum()), "n_error_nodes_pruned": int(nm[nf:nf + lay.n_errors].sum()),
            "n_edges_pruned": n_edges, "n_features_graph": nf, "n_errors": lay.n_errors, "n_tokens": lay.n_tokens,
            "n_logits": lay.n_logits}


def top_influence_features(rec: dict, k: int = 10) -> list[int]:
    """Source indices of the k feature nodes with the largest node influence of the exact graph."""
    lay = record_layout(rec)
    infl = rec["exact_prune"]["node_influence"][: lay.n_features].double()
    k = min(k, infl.numel())
    return torch.topk(infl, k).indices.tolist()


def predicted_logit_change(rec: dict, src: list[int], A: Tensor | None = None) -> dict:
    """Graph-predicted change of the output metric when feature source s is set to zero.

    Output metric: m = sum_t p_t (logit_t - mean over the vocabulary of the logits), over the graph's
    logit targets t with their probabilities p_t (the prob-weighted logit nodes of circuit-tracer,
    whose vectors are demeaned unembeddings).

    - "direct" (PREREG_GATE0 Section 2, the decision): minus the direct-path attribution of the
      feature to the metric, -sum_t p_t A[logit_t, s]. A[logit_t, s] is the per-unit direct effect
      times the feature's activation, so this is "direct-path attribution times minus activation".
    - "total" (reported only): direct plus indirect paths through other feature nodes of the graph
      in the linearised graph: y = d + sum over paths, where a feature t whose input changes by
      delta passes delta / a_t * A[u, t] to each target u (A's entries scale with the source's
      activation a_t); predicted change = -y_s. Uses the unnormalised adjacency of the record
      (visited features only; error and token nodes are fixed).
    """
    A = record_matrix(rec) if A is None else A
    A = A.double()
    lay = record_layout(rec)
    k, nf = lay.n_logits, lay.n_features
    p = lay.logit_probs.double()
    d = p @ A[lay.logit_targets][:, :nf]
    act = NodeMeta.from_dict(rec["source_meta"]).extra[:nf].double()
    act = torch.where(act.abs() > 0, act, torch.ones_like(act))
    F = torch.zeros(nf, nf, dtype=torch.float64)
    has = lay.feature_node_to_target >= 0
    F[has] = A[lay.feature_node_to_target[has], :nf]
    tot, cur = d.clone(), d.clone()
    for _ in range(10_000):
        cur = (cur / act) @ F
        if not bool(cur.any()):
            break
        tot += cur
    else:
        raise RuntimeError("feature graph is not a DAG")
    idx = torch.as_tensor(src, dtype=torch.long)
    return {"direct": (-d[idx]).tolist(), "total": (-tot[idx]).tolist()}


def spearman_or_nan(a: list[float], b: list[float]) -> float:
    """Spearman correlation with average ranks for ties; nan if fewer than 3 points or either side is
    constant (circuit-tracer-free; cct.graph.metrics.spearman returns 1.0 for a constant side)."""
    if len(a) != len(b) or len(a) < 3:
        return NAN

    def rank(x):
        x = list(map(float, x))
        o = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and x[o[j + 1]] == x[o[i]]:
                j += 1
            for m in range(i, j + 1):
                r[o[m]] = (i + j) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va == 0 or vb == 0:
        return NAN
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)


def pr4_evaluate(trace_entries: list[dict], short_entries: list[dict], thr: dict) -> dict:
    """PR4 status from per-output entries ({"completeness_score", "faithfulness": {"spearman_direct"}
    or absent}). thr = prereg stage_b.pr4. Kill: median trace completeness < kill bound, or median
    trace faithfulness Spearman < kill bound. Pass: all four pass conditions. Else fail. Gaps are
    short minus trace (positive = traces worse). not_run if no trace completeness is available."""
    def fa(e, key="spearman_direct"):
        f = e.get("faithfulness")
        return f.get(key) if isinstance(f, dict) else None

    v = {
        "median_completeness_traces": median_finite(e.get("completeness_score") for e in trace_entries),
        "median_completeness_short": median_finite(e.get("completeness_score") for e in short_entries),
        "median_replacement_traces": median_finite(e.get("replacement_score") for e in trace_entries),
        "median_replacement_short": median_finite(e.get("replacement_score") for e in short_entries),
        "median_faithfulness_spearman_traces": median_finite(fa(e) for e in trace_entries),
        "median_faithfulness_spearman_short": median_finite(fa(e) for e in short_entries),
        "median_faithfulness_spearman_total_traces": median_finite(fa(e, "spearman_total") for e in trace_entries),
        "median_faithfulness_spearman_total_short": median_finite(fa(e, "spearman_total") for e in short_entries),
        "median_pruned_nodes_traces": median_finite(e.get("n_nodes_pruned") for e in trace_entries),
        "median_pruned_feature_nodes_traces": median_finite(e.get("n_feature_nodes_pruned") for e in trace_entries),
        "median_pruned_edges_traces": median_finite(e.get("n_edges_pruned") for e in trace_entries),
        "median_pruned_nodes_short": median_finite(e.get("n_nodes_pruned") for e in short_entries),
        "median_pruned_edges_short": median_finite(e.get("n_edges_pruned") for e in short_entries),
        "n_trace_outputs": len(trace_entries),
        "n_trace_faithfulness_outputs": sum(_finite(fa(e)) for e in trace_entries),
        "n_short_prompts": len(short_entries),
    }
    v["completeness_gap_vs_short"] = v["median_completeness_short"] - v["median_completeness_traces"]
    v["faithfulness_gap_vs_short"] = (v["median_faithfulness_spearman_short"]
                                      - v["median_faithfulness_spearman_traces"])
    v["informative_graphs_too_big"] = (bool(v["median_pruned_nodes_traces"] > thr["informative_median_pruned_nodes_above"])
                                       if _finite(v["median_pruned_nodes_traces"]) else None)
    P, K = thr["pass"], thr["kill"]
    comp, faith = v["median_completeness_traces"], v["median_faithfulness_spearman_traces"]
    notes = []
    if not _finite(comp):
        status = "not_run"
        notes.append("no trace completeness available")
    elif comp < K["median_completeness_below"] or (_finite(faith) and faith < K["faithfulness_spearman_below"]):
        status = "kill"
    else:
        conds = {"completeness": comp >= P["median_completeness_traces_at_least"],
                 "completeness_gap": _finite(v["completeness_gap_vs_short"])
                 and v["completeness_gap_vs_short"] <= P["completeness_gap_vs_short_at_most"],
                 "faithfulness": _finite(faith) and faith >= P["median_faithfulness_spearman_traces_at_least"],
                 "faithfulness_gap": _finite(v["faithfulness_gap_vs_short"])
                 and v["faithfulness_gap_vs_short"] <= P["faithfulness_gap_vs_short_at_most"]}
        v["pass_conditions"] = conds
        status = "pass" if all(conds.values()) else "fail"
        if not _finite(faith):
            notes.append("faithfulness not measured: cannot pass")
    return {"status": status, "values": v, "notes": "; ".join(notes)}


# =================================================================================================
# Stage E
# =================================================================================================
PER_TRACE_N = [1, 2, 4, 5, 8, 11, 16, 17]
PER_TRACE_SLOPE_PAIRS = [(5, 11), (11, 17)]   # the spec's 16->32 and 32->51 scaled by 1/3 (51 -> 17)
POOLED_SLOPE_PAIRS = [(16, 32), (32, 51)]


def union_order(outputs: list[dict]) -> list[dict]:
    """Fixed order of one trace's outputs for the union: the answer first, then the sentence outputs
    in sentence (position) order."""
    kind = lambda o: o.get("kind", o.get("output_kind"))  # noqa: E731
    ans = [o for o in outputs if kind(o) == "answer"]
    rest = sorted((o for o in outputs if kind(o) != "answer"), key=lambda o: o["position"])
    return ans + rest


def pooled_order(ordered_by_trace: dict[str, list]) -> list:
    """Round-robin over traces of their ordered outputs (output j of every trace before j + 1)."""
    out = []
    m = max((len(v) for v in ordered_by_trace.values()), default=0)
    for j in range(m):
        for t in ordered_by_trace:
            if j < len(ordered_by_trace[t]):
                out.append(ordered_by_trace[t][j])
    return out


def realistic_rows(rec: dict, c: int, offset: int) -> list[tuple]:
    """(layer, position in trace coordinates, feature) of the first c visited feature rows."""
    return visited_features(rec, c, offset)


def oracle_rows(rec: dict, offset: int) -> list[tuple]:
    """(layer, position in trace coordinates, feature) of the feature nodes kept by the exact prune."""
    sm = rec["source_meta"]
    lay = record_layout(rec)
    nm = rec["exact_prune"]["node_mask"][: lay.n_features]
    idx = torch.nonzero(nm).flatten()
    return list(zip(sm["layer"][idx].tolist(), (sm["pos"][idx] - offset).tolist(), sm["feature"][idx].tolist()))


def interp_seconds(lengths: list[float], seconds: list[float], q: float) -> float:
    """Pass time at prefix length q by linear interpolation in length between measured points; below
    the shortest measured length, proportional to length (t_min * q / l_min, which favours
    truncation); above the longest, the longest's time."""
    pts = sorted(zip(map(float, lengths), map(float, seconds)))
    if not pts:
        raise ValueError("no timing points")
    if q <= pts[0][0]:
        return pts[0][1] * q / pts[0][0]
    for (a, ta), (b, tb) in zip(pts, pts[1:]):
        if q <= b:
            return ta + (tb - ta) * (q - a) / (b - a)
    return pts[-1][1]


def weighted_union(entries: list[dict], weight: Callable[[float], float]) -> dict:
    """entries: {"features": iterable of hashable row keys, "n_logits": int, "q": prefix length}.
    count = distinct feature rows + sum of logit rows (logit rows of different outputs differ).
    weighted: each distinct feature row costs weight(q) at the SHORTEST prefix among the outputs
    that selected it (it is the same row on every prefix that contains its position, PR2, and that
    pass is run anyway); each output's logit rows cost weight(q) at its own prefix. Note: adding an
    output with a shorter prefix can lower the weight of rows already in the union, so the weighted
    union need not be monotone in n (the count is)."""
    best: dict = {}
    logit_w, n_log = 0.0, 0
    for e in entries:
        q = float(e["q"])
        for key in e["features"]:
            key = tuple(key)
            if key not in best or q < best[key]:
                best[key] = q
        n_log += int(e["n_logits"])
        logit_w += int(e["n_logits"]) * weight(q)
    return {"count": len(best) + n_log, "weighted": sum(weight(q) for q in best.values()) + logit_w}


def union_curve(entries: list[dict], ns: list[int], weight: Callable[[float], float]) -> dict:
    """{n: weighted_union(first n entries)} for n <= len(entries)."""
    return {int(n): weighted_union(entries[:n], weight) for n in ns if 1 <= n <= len(entries)}


def slope(ua: float, ub: float, a: float, b: float) -> float:
    if not (_finite(ua) and _finite(ub)) or ua <= 0 or ub <= 0 or a <= 0 or b <= a:
        return NAN
    return math.log(ub / ua) / math.log(b / a)


def local_slope(curve: dict, pairs, key: str = "weighted") -> dict:
    """Slopes of log U against log n over each pair (a, b) present in the curve, and their minimum."""
    out = {}
    for a, b in pairs:
        if a in curve and b in curve:
            out[f"{a}-{b}"] = slope(curve[a][key], curve[b][key], a, b)
    fin = [x for x in out.values() if _finite(x)]
    out["min"] = min(fin) if fin else NAN
    return out


def projected_union(u_end: float, n_end: int, L: int, s: float) -> float:
    """U extrapolated from (n_end, u_end) to L explained outputs with log-log slope s."""
    if not (_finite(u_end) and _finite(s)) or n_end <= 0:
        return NAN
    return u_end * (L / n_end) ** s


def cost_cct(m_trace: float, kappa: float, hub_rates: dict, n_sources: dict, check_probes: int,
             decode_seconds_per_column: float, n_cols: float, row_seconds_full: float) -> dict:
    """Cost_CCT(L) in row-equivalents at full length (GATE0_PREMISES.md Stage E): m_trace mixed passes
    + kappa * projected hubs (sum over node kinds of hub rate x number of sources of that kind)
    + check probes + recovery time (decode seconds per column x columns kept after screening, divided
    by the row time at full length)."""
    hubs = sum(float(hub_rates.get(k, 0.0)) * float(n) for k, n in n_sources.items())
    rec = float(decode_seconds_per_column) * float(n_cols) / float(row_seconds_full)
    total = float(m_trace) + float(kappa) * hubs + float(check_probes) + rec
    return {"m_trace": float(m_trace), "kappa": float(kappa), "projected_hubs": hubs,
            "hub_row_equiv": float(kappa) * hubs, "check_probes": int(check_probes), "recovery_row_equiv": rec,
            "cost_cct": total}


def pr10_evaluate(per_trace: dict, pooled: dict, thr: dict) -> dict:
    """per_trace: {tid: {"local_slope", "S_proj"}}; pooled: {"S_meas", "S_proj", "S_oracle_proj"}.
    thr = prereg stage_e.pr10. Pass: local slope >= bound on >= traces_at_least traces, and pooled
    S_meas, S_proj, S_oracle_proj above their bounds. Kill: S_proj < bound on every trace."""
    P, K = thr["pass"], thr["kill"]
    slopes = {t: v.get("local_slope") for t, v in per_trace.items()}
    sproj = {t: v.get("S_proj") for t, v in per_trace.items()}
    n_slope_ok = sum(_finite(s) and s >= P["local_slope_at_least"] for s in slopes.values())
    conds = {"local_slope": n_slope_ok >= P["traces_at_least"],
             "S_meas": _finite(pooled.get("S_meas")) and pooled["S_meas"] >= P["s_meas_at_least"],
             "S_proj": _finite(pooled.get("S_proj")) and pooled["S_proj"] >= P["s_proj_at_least"],
             "S_oracle_proj": _finite(pooled.get("S_oracle_proj")) and pooled["S_oracle_proj"] >= P["s_oracle_proj_at_least"]}
    kill = bool(sproj) and all(_finite(s) and s < K["s_proj_below"] for s in sproj.values())
    status = "kill" if kill else ("pass" if all(conds.values()) else "fail")
    return {"status": status, "conditions": conds, "n_traces_slope_ok": n_slope_ok}
