"""Gate 0 G0-B union growth (src/cct/gate0_union.py, scripts/gen_traces.py, scripts/gate0_union.py)."""
import json
import math
import os
import subprocess
import sys

import pytest
import torch

from cct.gate0_union import (TruncationEvaluator, beta_random_subsets, compute_order, convergence_check,
                             explained_positions, find_answer_position, fit_beta, level_slots, smallest_c,
                             union_size, visited_features)
from cct.killtest import eval_truncation, exact_graph_tiny
from cct.operators.tiny import TinyConfig, TinyModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_positions_nested_and_end_aligned():
    pos = explained_positions(50, 299, 32)
    assert len(pos) == 32 and len(set(pos)) == 32
    assert pos[-1] == 299 and pos[0] >= 50 and pos == sorted(pos)
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    assert max(gaps) - min(gaps) <= 1  # evenly spaced
    assert level_slots(1) == [31]
    assert level_slots(2) == [15, 31]
    assert level_slots(4) == [7, 15, 23, 31]
    for n in [1, 2, 4, 8, 16]:
        assert set(level_slots(n)) <= set(level_slots(2 * n))
    assert level_slots(32) == list(range(32))
    with pytest.raises(ValueError):
        explained_positions(10, 20, 32)  # fewer generated positions than slots


def test_compute_order():
    order = compute_order([1, 2, 4, 8, 16, 32], 32)
    slots = [s for _, s in order]
    assert len(slots) == 32 and sorted(slots) == list(range(32))
    assert order[0] == (1, 31) and order[1] == (2, 15)
    assert [lv for lv, _ in order] == sorted(lv for lv, _ in order)  # levels come in order
    # every prefix that ends at a level boundary is exactly that level's slot set
    k = 0
    for n in [1, 2, 4, 8, 16, 32]:
        k = len(level_slots(n))
        assert set(slots[:k]) == set(level_slots(n))


def test_union_size_hand_example():
    outs = [{"features": [(0, 1, 5), (1, 2, 3)], "n_logits": 3},
            {"features": [(1, 2, 3), (2, 4, 7), (0, 1, 6)], "n_logits": 2},
            {"features": [], "n_logits": 1}]
    # distinct features: (0,1,5), (1,2,3), (2,4,7), (0,1,6) = 4; logits 3 + 2 + 1 = 6
    assert union_size(outs) == 10
    assert union_size(outs[:1]) == 5
    assert beta_random_subsets(outs, 3, 10, 0) == 10
    assert math.isnan(beta_random_subsets(outs, 4, 10, 0))
    m = beta_random_subsets(outs, 1, 500, 0)
    assert min(union_size([o]) for o in outs) <= m <= max(union_size([o]) for o in outs)


def test_fit_beta_recovers_exponent():
    ns = [1, 2, 4, 8, 16, 32]
    fit = fit_beta(ns, [50 * n ** 0.7 for n in ns])
    assert abs(fit["beta"] - 0.7) < 1e-9 and abs(fit["c_fit"] - 50) < 1e-6 and fit["beta_se"] < 1e-9
    g = torch.Generator().manual_seed(0)
    noisy = [50 * n ** 0.7 * math.exp(0.05 * float(torch.randn(1, generator=g))) for n in ns]
    f2 = fit_beta(ns, noisy)
    assert abs(f2["beta"] - 0.7) < 3 * f2["beta_se"] + 0.05 and f2["beta_se"] > 0
    assert math.isnan(fit_beta([1, 2], [50, 80])["beta_se"])
    assert math.isnan(fit_beta([1], [50])["beta"])


def test_find_answer_position_last_occurrence():
    vocab = ["<think>", "</think>", "The", " answer", " is", " 2", "9", " and", " 290", ".", " 29", "29"]
    decode = lambda ids: "".join(vocab[i] for i in ids)  # noqa: E731
    ids = [0, 2, 3, 4, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # "...</think>The answer is 29 and 290."
    assert find_answer_position(ids, 6, "29", decode) == 9  # " 2" starts the last whole "29"
    assert find_answer_position(ids, 6, "31", decode) is None
    ids2 = [1, 11, 9, 2, 10]  # "</think>29.The 29"
    assert find_answer_position(ids2, 1, "29", decode) == 4


@pytest.fixture(scope="module")
def tiny_record():
    model = TinyModel(TinyConfig(transcoder="plt", n_layers=3, n_features=32, seed=0))
    tokens = torch.randint(0, 64, (10,), generator=torch.Generator().manual_seed(3))
    return exact_graph_tiny(model, tokens, cap=10_000, batch_size=64)


def test_evaluator_matches_eval_truncation(tiny_record):
    ev = TruncationEvaluator(tiny_record)
    for c in [4, 16, 40, ev.n_feature_rows]:
        s = ev.scores(c)
        ref = eval_truncation(tiny_record, ev.n_logits + c)["fidelity"]
        assert s["iw_node_recall"] == pytest.approx(ref["iw_node_recall"], abs=1e-12)
        assert s["node_jaccard"] == pytest.approx(ref["node_jaccard"], abs=1e-12)
    full = ev.scores(ev.n_feature_rows)
    assert full["iw_node_recall"] == pytest.approx(1.0) and full["node_jaccard"] == pytest.approx(1.0)


def test_smallest_c_on_tiny_record(tiny_record):
    ev = TruncationEvaluator(tiny_record)
    grid = list(range(1, ev.n_feature_rows + 1, 2)) + [ev.n_feature_rows]
    rmin, jmin = 0.95, 0.85
    c = smallest_c(tiny_record, grid, rmin, jmin)
    assert c is not None and c != grid[0]
    meets = lambda f: f["iw_node_recall"] >= rmin and f["node_jaccard"] >= jmin  # noqa: E731
    assert meets(eval_truncation(tiny_record, ev.n_logits + c)["fidelity"])
    prev = max(g for g in grid if g < c)
    assert not meets(eval_truncation(tiny_record, ev.n_logits + prev)["fidelity"])
    # a grid that never reaches the full graph and never meets fidelity gives None
    assert smallest_c(tiny_record, [1], 1.01, 1.01) is None
    feats = visited_features(tiny_record, c)
    assert len(feats) == c and len(set(feats)) == c
    conv = convergence_check(tiny_record, ev.n_feature_rows, 0.95)
    assert conv["ok"] and conv["node_jaccard"] == pytest.approx(1.0)


@pytest.mark.slow
def test_gate0_union_tiny_end_to_end():
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    for script, extra in [("gen_traces.py", []), ("gate0_union.py", [])]:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), "--tiny", "--force", *extra],
                           cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        assert r.returncode == 0, r.stderr[-3000:]
    # gen_traces refuses to overwrite without --force
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_traces.py"), "--tiny"], cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode != 0
    with open(os.path.join(ROOT, "results", "tiny", "gate0", "g0b_union.json"), encoding="utf-8") as f:
        res = json.load(f)
    assert set(res) >= {"run", "traces", "beta", "c_median", "n_outputs_without_c", "gpu_seconds"}
    assert len(res["traces"]) == 2 and math.isfinite(res["beta"])
    for t in res["traces"].values():
        assert t["levels_done"] == [1, 2, 4, 8]
        U = [t["U_by_n"][str(n)] for n in t["levels_done"]]
        assert U == sorted(U)  # nested levels: the union never shrinks
        assert len(t["c_per_output"]) == 8 and set(t["budget_per_output"]) == set(t["c_per_output"])
        assert t["positions"][-1] == t["end_position"]
    # the CPU phase alone reproduces the same result from the cached records
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gate0_union.py"), "--tiny", "--phase", "cpu",
                        "--no-wait"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    with open(os.path.join(ROOT, "results", "tiny", "gate0", "g0b_union.json"), encoding="utf-8") as f:
        res2 = json.load(f)
    assert res2["beta"] == res["beta"] and res2["c_median"] == res["c_median"]
