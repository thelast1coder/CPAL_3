"""Gate 0 Stages D and F: pure logic (cct.premise_df) and end-to-end tiny runs of scripts/premise_D.py and
scripts/premise_F.py."""
import json
import math
import os
import subprocess
import sys
import time

import pytest
import torch

from cct import premise_df as pdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -------------------------------------------------------------------------------------------------
# Stage F: dependency aggregation
# -------------------------------------------------------------------------------------------------
def _chain_record(prefix_offset: int = 0):
    """Toy DAG in circuit-tracer node order [features, errors, tokens, logits]:
    token t0 (pos 0) -> feature f1 (pos 1) -> feature f2 (pos 3) -> logit, plus token t2 (pos 2) -> f2 with
    weight 3 (so f2's normalised in-edges are 1/4 from f1 and 3/4 from t2). Positions are model
    coordinates = trace + prefix_offset."""
    from cct.graph.influence import node_influence

    # nodes: 0 f1, 1 f2, 2 t0, 3 t2, 4 logit
    A = torch.zeros(5, 5, dtype=torch.float64)
    A[0, 2] = 2.0   # f1 <- t0
    A[1, 0] = 1.0   # f2 <- f1
    A[1, 3] = 3.0   # f2 <- t2
    A[4, 1] = 5.0   # logit <- f2
    w = torch.tensor([0, 0, 0, 0, 1.0], dtype=torch.float64)
    ni = node_influence(A, w)
    sm = {"kind": torch.tensor([1, 1, 3, 3]), "layer": torch.tensor([0, 1, -1, -1]),
          "pos": torch.tensor([1, 3, 0, 2]) + prefix_offset, "feature": torch.tensor([7, 9, -1, -1])}
    return {"source_meta": sm, "exact_prune": {"node_influence": ni}, "prefix_offset": prefix_offset}, ni


def test_toy_dag_influence_known():
    _, ni = _chain_record()
    # f2 = 1, f1 = 1/4, t2 = 3/4, t0 = 1/4 (f1's only in-edge), logit 0
    assert torch.allclose(ni, torch.tensor([0.25, 1.0, 0.25, 0.75, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("off", [0, 1])
def test_dependency_aggregation_toy_dag(off):
    rec, ni = _chain_record(off)
    pos, kind = pdf.graph_node_positions(rec)
    assert pos.tolist() == [1, 3, 0, 2]  # back in trace coordinates
    spans = [(0, 2), (2, 3), (3, 4)]  # s0 = {t0, f1}, s1 = {t2}, s2 = {f2}
    by_s = pdf.output_sentence_influence(ni, pos, spans)
    assert torch.allclose(by_s, torch.tensor([0.5, 0.75, 1.0], dtype=torch.float64))
    # two sampled outputs of sentence 2 (same graph twice) and one of sentence 1 (only s0 counts)
    outs = [{"sentence_index": 2, "influence_by_sentence": by_s}, {"sentence_index": 2, "influence_by_sentence": by_s},
            {"sentence_index": 1, "influence_by_sentence": by_s}]
    D, raw = pdf.dependency_matrix(outs, spans)
    assert torch.allclose(raw[2, :2], torch.tensor([1.0, 1.5], dtype=torch.float64))
    assert torch.allclose(D[2, :2], torch.tensor([0.4, 0.6], dtype=torch.float64))
    assert float(D[1, 0]) == pytest.approx(1.0)
    assert torch.isnan(D[0]).all() and torch.isnan(D[2, 2])
    assert pdf.pairs_from(D) == [(1, 0), (2, 0), (2, 1)]


# -------------------------------------------------------------------------------------------------
# Stage F: metrics, interventions, Thought Anchors' measure
# -------------------------------------------------------------------------------------------------
def test_auroc_hand_values():
    s = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    y = torch.tensor([1, 1, 0, 1, 0, 0])
    # positives 0.9, 0.8, 0.6 against negatives 0.7, 0.5, 0.4: 3 + 3 + 2 = 8 of 9
    assert pdf.auroc(s, y) == pytest.approx(8 / 9)
    assert pdf.auroc(torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1, 0, 0])) == pytest.approx(0.75)  # ties = 1/2
    assert math.isnan(pdf.auroc(s, torch.zeros(6)))
    assert pdf.auroc(-s, y) == pytest.approx(1 / 9)


def test_top_quartile():
    a = torch.arange(8, dtype=torch.float64)  # 75th percentile (linear) = 5.25
    assert pdf.top_quartile(a).tolist() == [False] * 6 + [True, True]
    b = torch.tensor([1.0, 1.0, 1.0, 5.0])
    assert pdf.top_quartile(b).tolist() == [False, False, False, True]


def test_filler_keeps_length_and_deletion_alignment():
    t = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
    f = pdf.filler_replace(t, (2, 5), 99)
    assert f.numel() == t.numel() and f.tolist() == [10, 11, 99, 99, 99, 15, 16, 17]
    assert t.tolist()[2] == 12  # input untouched
    d = pdf.delete_span(t, (2, 5))
    assert d.tolist() == [10, 11, 15, 16, 17]
    assert pdf.align_after_deletion(t, d, [1, 5, 7], (2, 5)) == [1, 2, 4]
    with pytest.raises(ValueError):
        pdf.align_after_deletion(t, d, [3], (2, 5))
    with pytest.raises(ValueError):  # content check
        pdf.align_after_deletion(t, torch.tensor([10, 11, 0, 16, 17]), [5], (2, 5))


def test_ta_kl_matches_full_kl_and_zero_on_identical():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(4, 12, generator=g), torch.randn(4, 12, generator=g)
    kl = pdf.ta_sparse_kl(a, b, p_nucleus=1.0, max_k=100)  # nucleus keeps all 12 tokens
    P = torch.log_softmax(a.double() / 0.6, -1)
    Q = torch.log_softmax(b.double() / 0.6, -1)
    ref = (P.exp() * (P - Q)).sum(-1)
    assert torch.allclose(kl, ref, atol=1e-5)
    assert torch.allclose(pdf.ta_sparse_kl(a, a), torch.zeros(4, dtype=torch.float64), atol=1e-7)
    # logits and log-probabilities give the same measure
    assert torch.allclose(pdf.ta_sparse_kl(torch.log_softmax(a, -1), b), pdf.ta_sparse_kl(a, b), atol=1e-5)
    # sentence effect: mean of log(KL + 1e-9) over the span
    k = torch.tensor([0.0, 1.0, math.e - 1e-9, 0.5])
    assert pdf.ta_sentence_effect(k, (1, 3)) == pytest.approx(0.5, abs=1e-6)


def test_suppress_scores_keeps_causal_minus_inf():
    s = torch.zeros(1, 1, 4, 4).masked_fill(~torch.ones(4, 4, dtype=torch.bool).tril(), float("-inf"))
    o = pdf.suppress_scores(s, [(1, 3)])
    p = torch.softmax(o, -1)[0, 0]
    assert torch.isinf(o[0, 0, 0, 1]) and o[0, 0, 0, 1] < 0  # future key stays -inf
    assert float(p[1:, 1:3].sum()) == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(p.sum(-1), torch.ones(4, dtype=p.dtype))


def test_tiny_forward_equals_clean_forward_and_suppresses():
    from cct.premise_traces import tiny_premise_model

    m = tiny_premise_model()
    toks = torch.randint(0, m.cfg.vocab, (12,), generator=torch.Generator().manual_seed(3))
    ref = m.clean_forward(toks)["logits"][0]
    assert torch.allclose(pdf.tiny_forward(m, toks), ref, atol=1e-10)
    lg, pats = pdf.tiny_forward(m, toks, suppress=[(3, 6)], return_patterns=True)
    assert float(pats[..., 3:6].sum()) == pytest.approx(0.0, abs=1e-12)
    assert torch.allclose(lg[:3], ref[:3], atol=1e-10)  # earlier positions unaffected
    assert not torch.allclose(lg[6:], ref[6:])


def test_pr11_status_branches():
    pr = {"pass": {"auroc_minus_s4_at_least": -0.02, "auroc_at_least": 0.65, "nameable_at_least": 9},
          "weak": {"auroc_minus_s4_below": -0.10, "nameable_below": 5}}
    assert pdf.pr11_status(0.80, 0.70, 10, pr)["status"] == "pass"
    assert pdf.pr11_status(0.80, 0.70, None, pr)["status"] == "pending"
    assert pdf.pr11_status(0.69, 0.70, 9, pr)["status"] == "pass"  # within -0.02 of S4
    assert pdf.pr11_status(0.64, 0.60, 9, pr)["status"] == "fail"  # below 0.65 absolute
    assert pdf.pr11_status(0.75, 0.80, 9, pr)["status"] == "fail"  # 0.05 below S4: neither pass nor weak
    assert pdf.pr11_status(0.69, 0.80, 12, pr)["status"] == "weak"  # more than 0.10 below S4
    assert pdf.pr11_status(0.80, 0.70, 4, pr)["status"] == "weak"
    assert pdf.pr11_status(0.80, 0.70, 7, pr)["status"] == "fail"
    assert pdf.pr11_status(0.69, 0.80, None, pr)["status"] == "weak"
    assert pdf.pr11_status(0.70, 0.75, None, pr)["status"] == "fail"
    assert pdf.pr11_status(float("nan"), 0.7, None, pr)["status"] == "not_run"


# -------------------------------------------------------------------------------------------------
# Stage D
# -------------------------------------------------------------------------------------------------
def test_abs_s95_mass_recall_and_causal_fraction():
    cols = torch.tensor([[10.0, 0.0], [5.0, 1.0], [3.0, 1.0], [2.0, 0.0]])
    # column 0: L1 = 20, 95% = 19 -> needs 10 + 5 + 3 + 2 = all 4 (18 < 19); column 1: 2 entries
    assert pdf.abs_s95(cols).tolist() == [4, 2]
    assert pdf.abs_s95(torch.tensor([[60.0], [36.0], [4.0]])).tolist() == [2]  # 96 >= 95
    mask = torch.tensor([[1, 0], [1, 1], [0, 0], [0, 0]], dtype=torch.bool)
    r = pdf.mass_recall(cols, mask)
    assert r["per_column"].tolist() == pytest.approx([0.75, 0.5])
    w0, w1 = 100 + 25 + 9 + 4, 2.0
    assert r["recall"] == pytest.approx((0.75 * w0 + 0.5 * w1) / (w0 + w1))
    # causal set: feature targets (kind 1) at a later layer and same or later position
    t_kind = torch.tensor([0, 1, 1, 1, 1])
    t_layer = torch.tensor([9, 2, 1, 3, 3])
    t_pos = torch.tensor([5, 4, 4, 2, 5])
    c = pdf.causal_set(t_kind, t_layer, t_pos, torch.tensor([1, -1]), torch.tensor([4, 3]))
    assert c[:, 0].tolist() == [False, True, False, False, True]
    assert c[:, 1].tolist() == [False, True, True, False, True]
    om = torch.tensor([[1, 1], [1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.bool)
    assert pdf.omega_fraction(om, c).tolist() == pytest.approx([0.5, 1 / 3])


def test_pr8_select():
    pt = {"0.01": {"recall": 0.97, "median_frac": 0.02}, "0.001": {"recall": 0.995, "median_frac": 0.08},
          "0.0001": {"recall": 0.999, "median_frac": 0.30}}
    s = pdf.pr8_select(pt, 0.99, 0.10)
    assert s["passed"] and s["tau"] == "0.001"
    pt["0.001"]["median_frac"] = 0.2
    s = pdf.pr8_select(pt, 0.99, 0.10)
    assert not s["passed"] and s["tau"] == "0.0001"


def test_hub_flags_and_m_trace_on_fabricated_errors():
    eps = 0.1
    # 4 columns x 3 seeds at the largest m; column 3 is a hub (above eps on 2 of 3 seeds)
    at_max = torch.tensor([[0.01, 0.05, 0.2, 0.5], [0.02, 0.05, 0.05, 0.5], [0.01, 0.2, 0.05, 0.05]])
    hub = pdf.hub_flags(at_max, eps)
    assert hub.tolist() == [False, False, False, True]
    assert pdf.hub_flags(torch.tensor([[float("nan")], [float("nan")], [float("nan")]]), eps).tolist() == [False]
    w = torch.tensor([6.0, 2.0, 1.0, 100.0])  # the hub's energy is excluded
    errs = {
        512: torch.tensor([[0.5, 0.5, 0.5, 0.9]] * 3),
        1024: torch.tensor([[0.05, 0.5, 0.5, 0.9], [0.5, 0.05, 0.05, 0.9], [0.5, 0.05, 0.05, 0.9]]),  # 6/9, 3/9, 3/9
        2048: torch.tensor([[0.05, 0.05, 0.5, 0.9], [0.05, 0.05, 0.05, 0.9], [0.5, 0.05, 0.05, 0.9]]),  # 8/9, 1, 3/9
        4096: torch.tensor([[0.05, 0.05, 0.5, 0.9], [0.05, 0.05, 0.05, 0.9], [0.05, 0.5, 0.05, 0.9]]),  # 8/9, 1, 7/9
        8192: at_max,
    }
    assert pdf.non_hub_share_within(errs[1024][0], w, hub, eps) == pytest.approx(6 / 9)
    m, table = pdf.select_m_trace(errs, w, hub, eps, 0.85, 2)
    assert m == 2048 and table["1024"]["seeds_ok"] == 0 and table["2048"]["seeds_ok"] == 2
    m, _ = pdf.select_m_trace(errs, w, hub, eps, 0.95, 2)
    assert m is None  # only seed 1 ever reaches 0.95
    m, _ = pdf.select_m_trace(errs, w, hub, eps, 0.30, 2)
    assert m == 1024
    m, _ = pdf.select_m_trace({512: errs[512]}, w, hub, eps, 0.9, 2)
    assert m is None
    rates = pdf.hub_rate_by_kind(hub, torch.tensor([1, 1, 2, 2]))
    assert rates == {"feature": 0.0, "error": 0.5}


def test_weighted_screen_count_matches_screen_columns():
    from cct.sensing.screening import screen_columns

    g = torch.Generator().manual_seed(1)
    Y = torch.randn(40, 200, generator=g) * torch.rand(200, generator=g).pow(4)
    e = Y.double().pow(2).sum(0)
    keep = screen_columns(Y, 0.995, 2.0)
    assert pdf.weighted_screen_count(e, torch.ones(200), 0.995, 2.0) == pytest.approx(float(keep.sum()))
    # duplicating every column (weight 2) doubles the count
    assert pdf.weighted_screen_count(e, torch.full((200,), 2.0), 0.995, 2.0) == pytest.approx(2 * float(keep.sum()))


def test_simulate_d2_recovers_sparse_columns():
    g = torch.Generator().manual_seed(0)
    T, S = 500, 6
    A = torch.zeros(T, S, dtype=torch.float64)
    for s in range(S):
        idx = torch.randperm(T, generator=g)[:4]
        A[idx, s] = torch.randn(4, generator=g, dtype=torch.float64) * 10
    r = pdf.simulate_d2(A, 2000, 0, None)
    assert float(r["err"].max()) < 1e-5  # exact recovery of 4-sparse columns with B = 400 buckets, R = 5
    assert r["decode_seconds_per_column"] >= 0
    assert torch.allclose(r["energy_est"], A.pow(2).sum(0), rtol=0.5)


# -------------------------------------------------------------------------------------------------
# end-to-end tiny runs
# -------------------------------------------------------------------------------------------------
def _ensure_tiny_traces():
    path = os.path.join(ROOT, "results", "tiny", "gate0", "traces.json")
    if not os.path.exists(path):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gen_traces.py"), "--tiny"], cwd=ROOT,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.slow
def test_premise_D_tiny_end_to_end(tmp_path):
    _ensure_tiny_traces()
    stage_c = {"stage": "C", "epsilon_star": 0.5, "k_eff_short_qwen": 1.0e6,
               "premises": {"PR5": {"status": "pass"}, "PR6": {"status": "pass"}}}
    cp = tmp_path / "stage_C.json"
    cp.write_text(json.dumps(stage_c))
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "premise_D.py"), "--tiny", "--force",
                        "--stage-c", str(cp), "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode in (0, 2), r.stderr[-3000:]
    d = json.loads((tmp_path / "stage_D.json").read_text())
    assert d["stage"] == "D" and set(d["premises"]) == {"PR7", "PR8", "PR9"}
    for k in ("run", "stop", "stop_reason", "gpu_seconds", "detail", "m_trace", "hub_rate_by_kind",
              "decode_seconds_per_column", "n_sources_by_kind", "kappa", "row_seconds_full", "n_cols_after_screening"):
        assert k in d
    assert d["premises"]["PR7"]["status"] == "pass"  # k_eff_short huge: ratio tiny
    assert d["premises"]["PR9"]["status"] in ("pass", "kill")
    assert (r.returncode == 2) == d["stop"]
    for tid, n in d["n_cols_after_screening"].items():
        assert 0 < n <= sum(d["n_sources_by_kind"][tid].values())
    assert d["kappa"] > 0 and all(v > 0 for v in d["row_seconds_full"].values())
    assert set(d["hub_rate_by_kind"]) <= {"feature", "error", "token"}
    # without stage C: dependent statuses not_run, no stop
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "premise_D.py"), "--tiny",
                        "--stage-c", str(tmp_path / "absent.json"), "--out-dir", str(tmp_path)], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads((tmp_path / "stage_D.json").read_text())
    assert d["premises"]["PR7"]["status"] == "not_run" and d["premises"]["PR9"]["status"] == "not_run"
    assert d["premises"]["PR8"]["status"] in ("pass", "fail") and d["stop"] is False
    assert time.time() - t0 < 120


@pytest.mark.slow
def test_premise_F_tiny_end_to_end(tmp_path):
    _ensure_tiny_traces()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "premise_F.py"), "--tiny",
                        "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d = json.loads((tmp_path / "stage_F.json").read_text())
    assert d["stage"] == "F" and d["stop"] is False
    pr = d["premises"]["PR11"]
    assert pr["status"] in ("pending", "weak", "fail")  # no labels yet
    assert pr["values"]["n_top_dependencies"] == 15
    met = d["detail"]["metrics"]
    for t in ("filler", "deletion"):
        for p in ("D", "S4", "S1", "S2"):
            a = met["pooled"][t][p]["auroc"]
            assert a is None or 0.0 <= a <= 1.0
    assert d["detail"]["S5"]["status"] == "skipped"
    for v in d["detail"]["per_trace"].values():
        assert v["suppression_max_mass"] == pytest.approx(0.0, abs=1e-9)
        assert len(v["carriers"]) == 5 and all(len(c["features"]) <= 5 for c in v["carriers"])
        D = torch.tensor(v["D"], dtype=torch.float64)
        for j, i in v["pairs"]:
            assert i < j and not math.isnan(float(D[j, i]))
        rows = {j for j, _ in v["pairs"]}
        for j in rows:
            assert float(torch.nansum(D[j, :j])) == pytest.approx(1.0, abs=1e-9) or float(torch.nansum(D[j, :j])) == 0.0
    md = (tmp_path / "carriers_for_labelling.md").read_text(encoding="utf-8")
    keys = [c["key"] for v in d["detail"]["per_trace"].values() for c in v["carriers"]]
    assert all(k in md for k in keys)
    # Dawar's labels are read back: all nameable
    (tmp_path / "carrier_labels.json").write_text(json.dumps({k: True for k in keys}))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "premise_F.py"), "--tiny",
                        "--out-dir", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    d2 = json.loads((tmp_path / "stage_F.json").read_text())
    v = d2["premises"]["PR11"]["values"]
    assert v["n_nameable"] == 15 and v["nameable_condition"] == "pass"
    assert d2["premises"]["PR11"]["status"] != "pending"
