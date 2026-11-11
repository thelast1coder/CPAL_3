"""T14 baselines; T15 group testing; T16 prereg consistency; T17 tiny end to end; plus storage and
decision-logic checks."""
import os
import re
import subprocess
import sys

import pytest
import torch
import yaml

from cct.baselines.colouring import colouring_rows, greedy_colouring
from cct.baselines.lowrank import lowrank_sketch
from cct.baselines.truncation import truncation_adjacency
from cct.graph.export import make_layout, to_square
from cct.interventions.group_testing import (SyntheticAblator, bernoulli_design, decode_topk, default_p,
                                             n_tests_for, recall, run_design)
from cct.killtest import k1_decision, k2_decision, k3_decision, overall_decision
from cct.operators.dense import DenseOperator
from cct.utils.io import pack_matrix, unpack_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_T14_truncation_full_equals_exact(tiny_op):
    A = tiny_op.exact_matrix()
    lay = make_layout(tiny_op.target_meta, tiny_op.source_meta)
    S = truncation_adjacency(A, torch.arange(tiny_op.T), tiny_op.T, lay)
    assert torch.equal(S, to_square(A, lay))
    S2 = truncation_adjacency(A, torch.arange(tiny_op.T), tiny_op.n_logits + 5, lay)
    assert int((S2[: lay.n_features].abs().sum(1) > 0).sum()) <= 5


def test_T14_lowrank_exact_on_low_rank():
    g = torch.Generator().manual_seed(0)
    T, N, r = 60, 50, 4
    A = torch.randn(T, r, generator=g, dtype=torch.float64) @ torch.randn(r, N, generator=g, dtype=torch.float64)
    op = DenseOperator.from_arrays(A, torch.ones(T), torch.zeros(T), torch.zeros(N), torch.zeros(N), dtype=torch.float64)
    A_hat, cost = lowrank_sketch(op, m=12, seed=0)
    assert float((A_hat - A).norm() / A.norm()) < 1e-8
    assert cost["forward_passes"] == 6 and cost["mixed_rows"] == 6


def test_T14_colouring_valid(tiny_op):
    M = tiny_op.causal_mask().dense()
    colours = greedy_colouring(M.T)  # colour targets
    for c in colours.unique():
        rows = M[colours == c]
        assert int(rows.sum(0).max()) <= 1  # no two targets of a colour share a source
    A_col, n_col = colouring_rows(tiny_op, M)
    assert torch.allclose(A_col, tiny_op.exact_matrix(), atol=1e-10)
    pat = torch.rand(40, 30, generator=torch.Generator().manual_seed(1)) < 0.1
    col = greedy_colouring(pat)
    for c in col.unique():
        assert int(pat[:, col == c].sum(1).max()) <= 1


def test_T15_group_testing():
    k, N, trials, ok = 3, 200, 40, 0
    n_tests = n_tests_for(k, N, C=4.0)
    for s in range(trials):
        ab = SyntheticAblator(N, k, seed=s)
        design = bernoulli_design(n_tests, N, default_p(k), seed=1000 + s)
        y = run_design(ab, design)
        found = decode_topk(design, y, k, "nnls")
        found2 = decode_topk(design, y, k, "comp_dd", tau=0.5)
        ok += recall(found, ab.true) == 1.0 and recall(found2, ab.true) == 1.0
    assert ok / trials >= 0.95
    assert n_tests < N


def test_T16_prereg_consistency():
    with open(os.path.join(ROOT, "PREREG_KILL_TEST.md"), encoding="utf-8") as f:
        md = f.read()
    block = re.search(r"```yaml\n(.*?)```", md, re.S).group(1)
    with open(os.path.join(ROOT, "configs", "prereg_kill_test.yaml"), encoding="utf-8") as f:
        cfg_text = f.read()
    assert yaml.safe_load(block) == yaml.safe_load(cfg_text)
    assert block.strip() == cfg_text.strip()


def test_storage_roundtrip():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(50, 40, generator=g) * torch.logspace(-9, 0, 40)[None]
    B = unpack_matrix(pack_matrix(A))
    nz = A != 0
    assert float(((B - A).abs() / A.abs().amax(1, keepdim=True))[nz].max()) < 1e-3
    assert int((B[A.abs() > 1e-6 * A.abs().amax(1, keepdim=True)] == 0).sum()) == 0  # no flushing to zero


def test_decision_logic():
    pre = yaml.safe_load(open(os.path.join(ROOT, "configs", "prereg_kill_test.yaml"), encoding="utf-8"))
    assert k1_decision([0.005] * 8, pre)["verdict"] == "GO"
    assert k1_decision([0.2] * 8, pre)["verdict"] == "NO-GO"
    assert k1_decision([0.05] * 8, pre)["verdict"] == "AMBER"
    assert k2_decision({"cost": 100}, {"cost": 400}, 1500, pre)["verdict"] == "GO"
    assert k2_decision({"cost": 300}, {"cost": 500}, 1500, pre)["verdict"] == "AMBER"
    assert k2_decision(None, {"cost": 500}, 1500, pre)["verdict"] == "NO-GO"
    assert k3_decision(6, 5.0, 0.001, "float32", pre)["verdict"] == "GO"
    assert k3_decision(6, 5.0, 0.03, "float32", pre)["verdict"] == "AMBER"
    assert k3_decision(4, 9.0, 0.0, "bfloat16", pre)["verdict"] == "NO-GO"
    G, A, N = {"verdict": "GO"}, {"verdict": "AMBER"}, {"verdict": "NO-GO"}
    assert overall_decision({"p": G, "c": G}, {"p": G, "c": A}, {"p": G, "c": A})["verdict"] == "GO"
    assert overall_decision({"p": N, "c": N}, {"p": G, "c": G}, {"p": G})["verdict"] == "NO-GO"
    assert overall_decision({"p": G, "c": G}, {"p": G, "c": N}, {"p": G})["verdict"] == "AMBER"
    assert overall_decision({"p": G, "c": G}, {"p": A, "c": A}, {"p": G})["verdict"] == "AMBER"


@pytest.mark.slow
def test_T17_end_to_end_tiny():
    env = dict(os.environ, PYTHON=sys.executable, PYTHONWARNINGS="ignore")
    r = subprocess.run(["bash", os.path.join(ROOT, "scripts", "run_kill_test.sh"), "--tiny", "--transcoders", "plt",
                        "--force"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stderr[-3000:]
    rd = os.path.join(ROOT, "results", "tiny")
    for f in ["k0.json", "k1.json", "k2.json", "k3.json", "KILL_REPORT.md", "kill_test.json"]:
        assert os.path.exists(os.path.join(rd, f)), f
    rep = open(os.path.join(rd, "KILL_REPORT.md"), encoding="utf-8").read()
    assert rep.startswith("# TINY RUN, NUMBERS ARE MEANINGLESS")
    assert "## Decision table" in rep and re.search(r"Overall: \*\*(GO|AMBER|NO-GO)\*\*", rep)
