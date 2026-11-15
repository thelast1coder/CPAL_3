"""Gate 0, G0-B step 2: union growth over the explained outputs of a whole trace (PROJECT_V2.md
Sections 2.3, 3 and 7). Pure logic, CPU-testable; the scripts live in scripts/gate0_union.py and
scripts/gen_traces.py.

Conventions
- An "output at position q" is the model's prediction of token q, so its graph is circuit-tracer on
  the prefix tokens[:q] (the logits are read at position q - 1).
- Explained positions: `slots` positions evenly spaced over the generated part, end-aligned so the
  last slot is the answer position. Level n uses the end-aligned nested subset
  {j * (slots / n) + slots / n - 1 : j = 0 .. n - 1}, so n = 1 is the answer alone and the levels are
  nested.
- c (truncation budget) counts FEATURE rows after the logit rows (the logit rows are always computed
  first), capped at the record's number of feature rows. Note that killtest.eval_truncation counts
  all rows, logits included; TruncationEvaluator converts.
- |U(n)| = distinct (layer, position, feature) among the first c_i visited features of each output in
  the level, plus the sum of the outputs' logit row counts (logit rows of different outputs are
  different targets).
"""
from __future__ import annotations

import math
import random
import re
from typing import Callable, Iterable

import torch

from .baselines.truncation import truncation_adjacency
from .graph.export import make_layout, to_square
from .graph.metrics import iw_node_recall, node_jaccard
from .graph.prune import prune_adjacency
from .killtest import exact_prune_result, record_matrix
from .operators.base import KIND_LOGIT, NodeMeta


# -------------------------------------------------------------------------------------------------
# positions
# -------------------------------------------------------------------------------------------------
def _check_slots(slots: int) -> None:
    if slots < 1 or slots & (slots - 1):
        raise ValueError(f"slots must be a power of two, got {slots}")


def explained_positions(n_prompt: int, end_pos: int, slots: int = 32) -> list[int]:
    """Slot positions over the generated part [n_prompt, end_pos], evenly spaced and end-aligned
    (slot slots - 1 == end_pos). end_pos is the answer position, or the last generated position if the
    answer was not found."""
    _check_slots(slots)
    G = end_pos - n_prompt + 1
    if G < slots:
        raise ValueError(f"only {G} generated positions up to {end_pos}; need at least {slots}")
    return [n_prompt + ((j + 1) * G) // slots - 1 for j in range(slots)]


def level_slots(n: int, slots: int = 32) -> list[int]:
    """Slot indices of level n (end-aligned, nested)."""
    _check_slots(slots)
    if n < 1 or slots % n:
        raise ValueError(f"level {n} does not divide {slots} slots")
    s = slots // n
    return [j * s + s - 1 for j in range(n)]


def compute_order(levels: Iterable[int], slots: int = 32) -> list[tuple[int, int]]:
    """[(level, slot)]: the slots of level 1 first, then the new slots of each next level, so a
    time-limited run always ends with complete nested levels."""
    out, seen = [], set()
    for n in sorted(levels):
        for s in level_slots(n, slots):
            if s not in seen:
                seen.add(s)
                out.append((n, s))
    return out


# -------------------------------------------------------------------------------------------------
# answer position (used by scripts/gen_traces.py)
# -------------------------------------------------------------------------------------------------
def find_answer_position(token_ids: list[int], start: int, answer: str,
                         decode: Callable[[list[int]], str]) -> int | None:
    """Token index of the first token of the LAST occurrence of `answer` in tokens[start:], found by
    decoding token spans (character offsets of each token from decoding growing prefixes). An
    occurrence must not be glued to letters or digits on either side (so "29" does not match "290").
    Case-sensitive. Returns None if absent."""
    seg = list(token_ids[start:])
    if not seg or not answer:
        return None
    offsets = [len(decode(seg[:k])) for k in range(len(seg) + 1)]
    text = decode(seg)
    last = None
    for m in re.finditer(re.escape(answer), text):
        i, j = m.start(), m.end()
        if i > 0 and text[i - 1].isalnum():
            continue
        if j < len(text) and text[j].isalnum():
            continue
        last = i
    if last is None:
        return None
    k = max(k for k in range(len(seg)) if offsets[k] <= last)
    return start + k


# -------------------------------------------------------------------------------------------------
# truncation fidelity against a record's own reference graph
# -------------------------------------------------------------------------------------------------
def n_logit_rows(rec: dict) -> int:
    return int((rec["target_meta"]["kind"] == KIND_LOGIT).sum())


class TruncationEvaluator:
    """Same computation as killtest.eval_truncation (truncation_adjacency, prune_adjacency, then the
    fidelity metrics against the record's exact prune), with the matrix, layout and reference unpacked
    once and only the two decision metrics computed. c is in FEATURE rows."""

    def __init__(self, rec: dict):
        self.rec = rec
        self.A = record_matrix(rec)
        self.tm = NodeMeta.from_dict(rec["target_meta"])
        self.sm = NodeMeta.from_dict(rec["source_meta"])
        self.lay = make_layout(self.tm, self.sm)
        self.ref = exact_prune_result(rec)
        self.n_logits = n_logit_rows(rec)
        self.T_D = int(rec["T_D"])
        self.n_feature_rows = self.T_D - self.n_logits
        self.order = torch.arange(self.T_D)
        self._cache: dict[int, dict] = {}

    def cap(self, c: int) -> int:
        return max(0, min(int(c), self.n_feature_rows))

    def prune(self, c: int):
        c = self.cap(c)
        adj = truncation_adjacency(self.A, self.order, self.n_logits + c, self.lay)
        pc = self.rec["prune_cfg"]
        return prune_adjacency(adj, self.lay.n_features, self.lay.n_tokens, self.lay.logit_probs,
                               pc["node_threshold"], pc["edge_threshold"])

    def scores(self, c: int) -> dict:
        c = self.cap(c)
        if c not in self._cache:
            pr = self.prune(c)
            nf = self.lay.n_features
            self._cache[c] = {"c": c, "iw_node_recall": iw_node_recall(self.ref, pr.node_mask, nf),
                              "node_jaccard": node_jaccard(self.ref.node_mask, pr.node_mask, nf),
                              "n_kept_exact": int(self.ref.node_mask[:nf].sum()),
                              "n_kept_approx": int(pr.node_mask[:nf].sum())}
        return self._cache[c]


def smallest_c(record: dict, c_grid: list[int], recall_min: float, jaccard_min: float,
               evaluator: TruncationEvaluator | None = None, curve: list | None = None) -> int | None:
    """Smallest c in the grid (feature rows, capped at the record's feature rows) whose truncation
    graph meets fidelity against the record's own reference graph; None if none. Linear scan in grid
    order (fidelity need not be monotone in c, so no bisection). The returned value is the capped c.
    `curve`, if given, receives the scores of every grid value evaluated."""
    ev = evaluator or TruncationEvaluator(record)
    for c in sorted(c_grid):
        s = ev.scores(c)
        if curve is not None:
            curve.append(dict(s, grid_c=int(c)))
        if s["iw_node_recall"] >= recall_min and s["node_jaccard"] >= jaccard_min:
            return s["c"]
        if ev.cap(c) >= ev.n_feature_rows:  # larger grid values give the same graph
            break
    return None


def convergence_check(record: dict, convergence_budget: int, jaccard_min: float,
                      evaluator: TruncationEvaluator | None = None) -> dict:
    """Node Jaccard between the pruned graph of the first `convergence_budget` feature rows and the
    record's reference graph (the full-budget graph). A reference that visited every active feature is
    the uncapped graph, so it passes whatever the Jaccard."""
    ev = evaluator or TruncationEvaluator(record)
    s = ev.scores(convergence_budget)
    complete = int(record.get("n_active", -1)) == ev.n_feature_rows
    return {"convergence_budget": int(convergence_budget), "node_jaccard": s["node_jaccard"],
            "complete": complete, "ok": bool(complete or s["node_jaccard"] >= jaccard_min)}


def visited_features(record: dict, c: int, pos_offset: int = 0) -> list[tuple[int, int, int]]:
    """(layer, position, feature) of the first c visited feature rows (target rows after the logits),
    with positions shifted by -pos_offset (e.g. a BOS prepended by circuit-tracer)."""
    tm = record["target_meta"]
    k = n_logit_rows(record)
    c = max(0, min(int(c), int(record["T_D"]) - k))
    sl = slice(k, k + c)
    return list(zip(tm["layer"][sl].tolist(), (tm["pos"][sl] - pos_offset).tolist(), tm["feature"][sl].tolist()))


# -------------------------------------------------------------------------------------------------
# union growth
# -------------------------------------------------------------------------------------------------
def union_size(selected_rows_per_output: Iterable[dict]) -> int:
    """Each entry: {"features": iterable of (layer, pos, feature), "n_logits": int}."""
    feats: set = set()
    n_log = 0
    for o in selected_rows_per_output:
        feats.update(tuple(x) for x in o["features"])
        n_log += int(o["n_logits"])
    return len(feats) + n_log


def fit_beta(ns: list[float], Us: list[float]) -> dict:
    """OLS of log U on log n: slope beta, intercept, the usual (bootstrap-free) standard error of the
    slope (nan with fewer than 3 points), and c_fit = exp(intercept)."""
    pts = [(math.log(n), math.log(u)) for n, u in zip(ns, Us) if n > 0 and u > 0]
    m = len(pts)
    nan = float("nan")
    if m < 2:
        return {"beta": nan, "intercept": nan, "beta_se": nan, "c_fit": nan, "n_points": m}
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    mx, my = sum(x) / m, sum(y) / m
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return {"beta": nan, "intercept": nan, "beta_se": nan, "c_fit": nan, "n_points": m}
    beta = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    icpt = my - beta * mx
    if m > 2:
        ssr = sum((b - icpt - beta * a) ** 2 for a, b in zip(x, y))
        se = math.sqrt(ssr / (m - 2) / sxx)
    else:
        se = nan
    return {"beta": beta, "intercept": icpt, "beta_se": se, "c_fit": math.exp(icpt), "n_points": m}


def beta_random_subsets(sets: list[dict], n: int, n_draws: int = 200, seed: int = 0) -> float:
    """Mean |U| over `n_draws` random size-n subsets of all computed outputs (entries as in
    union_size). Reported only; not used for the decision. nan if n exceeds the number of outputs."""
    if n > len(sets) or n < 1:
        return float("nan")
    rng = random.Random(seed)
    tot = 0
    for _ in range(n_draws):
        tot += union_size(rng.sample(sets, n))
    return tot / n_draws
