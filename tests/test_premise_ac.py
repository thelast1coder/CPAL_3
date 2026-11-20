"""Gate 0 Stages A and C (cct.premise_ac, scripts/premise_A.py, scripts/premise_C.py) on the tiny backend
and on fabricated inputs."""
import json
import os
import subprocess
import sys

import pytest
import torch

from cct import premise_ac as pa
from cct.operators.base import KIND_ERROR, KIND_FEATURE, KIND_LOGIT, KIND_TOKEN, CausalMask, NodeMeta
from cct.operators.tiny import TinyConfig, TinyModel, TinyOperator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -- PR1: mixed rows are exactly linear ------------------------------------------------------------
def test_pr1_mixed_rows_linear_all_targets(tiny_op):
    g = torch.Generator().manual_seed(0)
    r = pa.mix_check(tiny_op, 5, g)
    assert r["n_mixes"] == 5 and r["n_targets"] == tiny_op.T
    assert r["rel_err_max"] < 1e-10


def test_pr1_mixed_rows_linear_subset_and_precomputed(tiny_op):
    g = torch.Generator().manual_seed(1)
    tgt = torch.randperm(tiny_op.T, generator=g)[:20]
    assert pa.mix_check(tiny_op, 3, g, targets=tgt)["rel_err_max"] < 1e-10
    A = pa.all_rows(tiny_op)
    assert pa.mix_check(tiny_op, 3, g, A=A)["rel_err_max"] < 1e-10


def test_pr1_status_rules():
    assert pa.pr1_status([1e-7], [1e-7], [1.1], 1e-5, 1e-5, 1.2)[0] == "pass"
    assert pa.pr1_status([1e-7], [1e-7], [1.3], 1e-5, 1e-5, 1.2)[0] == "fail"
    assert pa.pr1_status([1e-3], [1e-7], [1.0], 1e-5, 1e-5, 1.2)[0] == "kill"
    assert pa.pr1_status([], [1e-7], [1.0], 1e-5, 1e-5, 1.2)[0] == "not_run"


def test_time_ratio_runs(tiny_op):
    r = pa.time_ratio(tiny_op, 16, 2, torch.Generator().manual_seed(0))
    assert r["batch"] == 16 and r["ratio"] > 0


# -- PR2: a prefix's rows are a slice of the full-trace pass ---------------------------------------
@pytest.mark.parametrize("tc", ["plt", "clt"])
def test_pr2_prefix_equals_full_slice(tc):
    model = TinyModel(TinyConfig(transcoder=tc, n_layers=3, n_features=32, seed=0))
    ids = torch.randint(0, 64, (14,), generator=torch.Generator().manual_seed(3))
    full = TinyOperator(model, ids, batch_size=64)
    comps = []
    for q in (6, 9, 12):
        part = pa.pr2_prefix_part(TinyOperator(model, ids[:q], batch_size=64), 50, torch.Generator().manual_seed(q))
        f = pa.pr2_full_rows(full, [part])[0]
        c = pa.pr2_compare(part, f, full.source_meta)
        assert c["n_targets_missing_in_full"] == 0 and c["n_prefix_sources_missing_in_full"] == 0
        assert c["n_sources_only_in_full"] > 0
        assert c["rel_err"] < 1e-12
        assert c["only_full_rel_mass"] == 0.0
        comps.append(c)
    assert pa.pr2_status(comps, 1e-4)[0] == "pass"
    bad = dict(comps[0], rel_err=1e-2)
    assert pa.pr2_status([bad], 1e-4)[0] == "kill"


def test_align_sources_by_identity():
    sp = NodeMeta(torch.tensor([KIND_FEATURE, KIND_ERROR, KIND_TOKEN]), torch.tensor([0, 0, -1]),
                  torch.tensor([1, 1, 1]), torch.tensor([7, -1, -1]))
    sf = NodeMeta(torch.tensor([KIND_FEATURE, KIND_FEATURE, KIND_ERROR, KIND_ERROR, KIND_TOKEN, KIND_TOKEN]),
                  torch.tensor([0, 1, 0, 0, -1, -1]), torch.tensor([1, 2, 1, 2, 1, 2]), torch.tensor([7, 3, -1, -1, -1, -1]))
    m, only = pa.align_sources(sp, sf)
    assert m.tolist() == [0, 2, 4]
    assert only.tolist() == [False, True, False, True, False, True]


# -- PR5: causal candidate sets and s95 ------------------------------------------------------------
def test_causal_set_sizes_hand_example():
    # feature targets (layer, pos): (1,0) (1,1) (2,0) (2,2) (3,1); one logit (ignored)
    tm = NodeMeta(torch.tensor([KIND_LOGIT] + [KIND_FEATURE] * 5), torch.tensor([4, 1, 1, 2, 2, 3]),
                  torch.tensor([2, 0, 1, 0, 2, 1]), torch.full((6,), -1))
    # sources: token pos 0 (layer -1), feature (1, 1), error (0, 2), feature (2, 0), feature (3, 0)
    sm = NodeMeta(torch.tensor([KIND_TOKEN, KIND_FEATURE, KIND_ERROR, KIND_FEATURE, KIND_FEATURE]),
                  torch.tensor([-1, 1, 0, 2, 3]), torch.tensor([0, 1, 2, 0, 0]), torch.full((5,), -1))
    c = pa.causal_set_sizes(tm, sm)
    # token@0: every feature target (5); feature (1,1): layer > 1 and pos >= 1 -> (2,2), (3,1) = 2;
    # error (0,2): layer > 0, pos >= 2 -> (2,2) = 1; feature (2,0): layer > 2 -> (3,1) = 1; (3,0): none
    assert c.tolist() == [5, 2, 1, 1, 0]
    # brute force agrees
    f = tm.kind != KIND_LOGIT
    bf = CausalMask(tm.layer[f], tm.pos[f], sm.layer, sm.pos).dense().sum(0)
    assert torch.equal(bf, c)


def test_s95_abs_hand_example():
    v = torch.tensor([[10.0, 0.0], [5.0, 0.0], [3.0, 0.0], [2.0, 0.0]])
    # L1 mass 20: 95% = 19 -> 10 + 5 + 3 = 18 < 19 -> 4 entries; zero column -> 0
    assert pa.s95_abs(v).tolist() == [4, 0]
    v2 = torch.tensor([[100.0, 10.0], [1.0, 5.0], [1.0, 5.0]])
    # 100 of 102 already holds 95%; 10 + 5 = 15 < 19 so the second column needs all 3
    assert pa.s95_abs(v2).tolist() == [1, 3]


# -- D2 with A_0 round trip -------------------------------------------------------------------------
def _synthetic(seed=0, k=3, T_layers=4, n_pos=6, per=40):
    g = torch.Generator().manual_seed(seed)
    tl = torch.randint(1, T_layers + 1, (per * n_pos,), generator=g)
    tp = torch.arange(n_pos).repeat_interleave(per)
    n_log = 2
    tm = NodeMeta(torch.cat([torch.full((n_log,), KIND_LOGIT), torch.full((tl.numel(),), KIND_FEATURE)]),
                  torch.cat([torch.full((n_log,), T_layers + 1), tl]), torch.cat([torch.full((n_log,), n_pos - 1), tp]),
                  torch.full((n_log + tl.numel(),), -1), torch.cat([torch.tensor([0.6, 0.3]), torch.zeros(tl.numel())]))
    S = 60
    sm = NodeMeta(torch.full((S,), KIND_ERROR), torch.randint(0, T_layers, (S,), generator=g),
                  torch.randint(0, n_pos, (S,), generator=g), torch.full((S,), -1))
    mask = CausalMask(tm.layer, tm.pos, sm.layer, sm.pos).dense()
    A0 = torch.randn(tm.kind.numel(), S, generator=g, dtype=torch.float64) * (tm.pos[:, None] == sm.pos[None]) * mask
    R = torch.zeros_like(A0)
    for s in range(S):
        cand = torch.nonzero(mask[n_log:, s]).flatten() + n_log
        if cand.numel():
            pick = cand[torch.randperm(cand.numel(), generator=g)[:k]]
            R[pick, s] = torch.randn(pick.numel(), generator=g, dtype=torch.float64) * 3
    return A0 + R, A0, tm, sm


def test_d2_round_trip_adds_a0_back():
    A, A0, tm, sm = _synthetic()
    # remainder zero: the estimate is exactly A_0 on feature rows and A on logit rows
    A_hat, secs = pa.d2_estimate(A0.clone(), A0, tm, sm, m=100, seed=0)
    assert torch.allclose(A_hat, A0) and secs >= 0
    # sparse remainder and m >> k: recovered almost exactly, logit rows untouched
    A_hat, _ = pa.d2_estimate(A, A0, tm, sm, m=400, seed=1)
    err = pa.remainder_col_errors(A_hat, A, A0, tm)
    # float32 sketch; a rare collision in 3 of 5 repetitions can leave a small false entry
    assert err["ew_mean"] < 1e-2 and err["ew_median"] < 1e-5 and err["n_cols"] > 0
    assert torch.equal(A_hat[:2], A[:2])
    # decoded entries never leave the causal set
    f = tm.kind != KIND_LOGIT
    outside = ~CausalMask(tm.layer[f], tm.pos[f], sm.layer, sm.pos).dense()
    assert float((A_hat[f] - A0[f])[outside].abs().max()) == 0.0
    # too few buckets: worse
    A_bad, _ = pa.d2_estimate(A, A0, tm, sm, m=20, seed=1)
    assert pa.remainder_col_errors(A_bad, A, A0, tm)["ew_mean"] > err["ew_mean"]


# -- epsilon*, PR5 and PR6 statuses on fabricated inputs ----------------------------------------------
def _run(prompt, dec, m, seed, meets, e):
    return {"prompt": prompt, "decoder": dec, "m": m, "seed": seed, "meets": meets,
            "col_err": {"ew_mean": e, "ew_median": e / 2, "ew_q90": 2 * e}}


def test_epsilon_star_selection():
    runs = []
    for p in ("a", "b", "c"):
        for sd in range(3):
            runs.append(_run(p, "D2", 256, sd, False, 0.5))
            # at 512: prompts a and b meet on 2 of 3 seeds, c on 1
            meets = (sd < 2) if p in ("a", "b") else (sd == 0)
            runs.append(_run(p, "D2", 512, sd, meets, 0.1 + 0.01 * sd + (0.05 if p == "b" else 0)))
            runs.append(_run(p, "D2", 1024, sd, True, 0.05))
            runs.append(_run(p, "D1", 256, sd, False, 0.4))
            runs.append(_run(p, "D1", 512, sd, False, 0.3))
            runs.append(_run(p, "D1", 1024, sd, p != "c", 0.08 + 0.001 * sd))
    e = pa.epsilon_star(runs)
    # D2: m_min = 512; largest error among meeting runs there: prompt b seed 1 -> 0.1 + 0.01 + 0.05;
    # prompt c seed 0 meets individually (0.1) and counts too
    assert e["by_decoder"]["D2"]["m_min"] == 512
    assert e["by_decoder"]["D2"]["epsilon"] == pytest.approx(0.16)
    # D1: m_min = 1024, max error among meeting runs 0.082; epsilon* = min over decoders
    assert e["by_decoder"]["D1"]["m_min"] == 1024
    assert e["by_decoder"]["D1"]["epsilon"] == pytest.approx(0.082)
    assert e["epsilon_star"] == pytest.approx(0.082) and e["decoder"] == "D1"
    assert pa.epsilon_star(runs, err_key="ew_q90")["by_decoder"]["D2"]["epsilon"] == pytest.approx(0.32)
    none = [dict(r, meets=False) for r in runs]
    assert pa.epsilon_star(none)["epsilon_star"] is None


def test_pr6_status_rules():
    pre6 = {"pass": {"m_at_most_T_over": 10, "prompts_at_least": 2, "decoder": "D2"},
            "kill": {"m_at_most_T_over": 4}}

    def tab(d2, d1):
        return {"D2": {str(m): {"meets": m in d2} for m in (256, 512, 1024, 2048)},
                "D1": {str(m): {"meets": m in d1} for m in (256, 512, 1024, 2048)}}

    T = {"a": 6000, "b": 6000, "c": 3000}
    # D2 meets at 512 on a, b (512 <= 600): pass
    s, _, info = pa.pr6_status({"a": tab({512}, set()), "b": tab({512}, set()), "c": tab(set(), set())}, T, pre6)
    assert s == "pass" and info["passing_m"] == [512]
    # D2 only at 1024 (> T/10 = 600) but <= T/4: fail
    s, _, _ = pa.pr6_status({"a": tab({1024}, set()), "b": tab({1024}, set()), "c": tab(set(), set())}, T, pre6)
    assert s == "fail"
    # only D1 at 512 on one prompt: fail, not kill
    s, _, _ = pa.pr6_status({"a": tab(set(), {512}), "b": tab(set(), set()), "c": tab(set(), set())}, T, pre6)
    assert s == "fail"
    # only at 2048 > T/4 = 1500 everywhere: kill
    s, _, _ = pa.pr6_status({"a": tab({2048}, {2048}), "b": tab({2048}, set()), "c": tab(set(), set())}, T, pre6)
    assert s == "kill"


def test_pr5_status_rules():
    pre5 = {"pass": {"remainder_abs_s95_frac_causal_at_most": 0.02, "prompts_at_least": 2},
            "kill": {"remainder_abs_s95_frac_causal_above": 0.10}}
    good = {"g1": {"p": 0.5}, "g2": {"p": 0.5}, "q": {"a": 0.01, "b": 0.015, "c": 0.5}}
    assert pa.pr5_status(good, "q", pre5)[0] == "pass"
    assert pa.pr5_status({"g1": {"p": 0.5}, "g2": {"p": 0.5}, "q": {"a": 0.01, "b": 0.05}}, "q", pre5)[0] == "fail"
    assert pa.pr5_status({"g1": {"p": 0.5}, "g2": {"p": 0.2}, "q": {"a": 0.11, "b": 0.3}}, "q", pre5)[0] == "kill"
    # a single set below the kill level somewhere is not a kill
    assert pa.pr5_status({"g1": {"p": 0.05}, "g2": {"p": 0.2}, "q": {"a": 0.11}}, "q", pre5)[0] == "fail"


def test_meets_table_seed_rule():
    runs = [{"decoder": "D2", "m": 256, "meets": x} for x in (True, False, True)] + \
           [{"decoder": "D2", "m": 512, "meets": x} for x in (True, False, False)]
    t = pa.meets_table(runs, 2)
    assert t["D2"]["256"]["meets"] and not t["D2"]["512"]["meets"]


def test_structure_on_tiny_record(tiny_op):
    from cct.local_part import from_tiny

    A = pa.all_rows(tiny_op)
    A0 = from_tiny(tiny_op).dense()
    st = pa.structure_AR(A, A0, tiny_op.target_meta, tiny_op.source_meta)
    assert 0 < st["a0_energy_share"] < 1
    for k in ("s95_sq_ewm_frac_causal", "s95_abs_ewm_frac_causal"):
        assert 0 < st["remainder"][k] <= 1
    assert set(st["hub_share_remainder"]) == {"feature", "error", "token", "first_token", "all"}


# -- end to end ---------------------------------------------------------------------------------------
def _check_stage(path, stage, prs):
    d = json.load(open(path, encoding="utf-8"))
    assert d["stage"] == stage
    for k in ("run", "premises", "stop", "stop_reason", "gpu_seconds", "dtype_by_set", "detail"):
        assert k in d
    for pr in prs:
        e = d["premises"][pr]
        assert e["status"] in ("pass", "fail", "kill", "not_run")
        assert set(e) >= {"status", "values", "thresholds", "notes"}
    assert d["run"]["git"]["hash"]
    return d


@pytest.mark.slow
def test_premise_A_and_C_tiny_end_to_end():
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    ra = subprocess.run([py, os.path.join(ROOT, "scripts", "premise_A.py"), "--tiny", "--force"], cwd=ROOT,
                        env=env, capture_output=True, text=True, timeout=900)
    assert ra.returncode in (0, 2), ra.stderr[-3000:]
    a = _check_stage(os.path.join(ROOT, "results", "tiny", "gate0", "stage_A.json"), "A", ("PR1", "PR2", "PR3"))
    assert ra.returncode == (2 if a["stop"] else 0)
    # float64 tiny model: linearity, prefix slices and A_0 hold to rounding
    for pr in ("PR1", "PR2", "PR3"):
        assert a["premises"][pr]["status"] in ("pass", "fail")
    assert a["premises"]["PR2"]["status"] == "pass" and a["premises"]["PR3"]["status"] == "pass"
    assert a["premises"]["PR1"]["values"]["gemma_plt_mix_rel_err_max"] < 1e-10
    rc = subprocess.run([py, os.path.join(ROOT, "scripts", "premise_C.py"), "--tiny", "--force"], cwd=ROOT,
                        env=env, capture_output=True, text=True, timeout=1800)
    assert rc.returncode in (0, 2), rc.stderr[-3000:]
    c = _check_stage(os.path.join(ROOT, "results", "tiny", "gate0", "stage_C.json"), "C", ("PR5", "PR6"))
    assert rc.returncode == (2 if c["stop"] else 0)
    assert "epsilon_star" in c and c["k_eff_short_qwen"] is not None
    assert os.path.exists(os.path.join(ROOT, "artifacts", "tiny", "tiny_qwen_plt", "premise_C", "tiny_test_03.pt"))
