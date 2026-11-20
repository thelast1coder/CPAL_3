"""Gate 0 Stages B and E: sentences, sampled outputs, PR4 and PR10 logic (src/cct/premise_traces.py,
src/cct/premise_be.py, scripts/premise_B.py, scripts/premise_E.py)."""
import json
import math
import os
import subprocess
import sys
import warnings

import pytest
import torch

from cct.killtest import exact_graph_tiny
from cct.operators.base import KIND_ERROR, KIND_FEATURE, KIND_LOGIT, KIND_TOKEN, NodeMeta
from cct.operators.tiny import TinyConfig, TinyModel
from cct.premise_be import (cost_cct, faithfulness_selection, interp_seconds, local_slope, oracle_rows, pooled_order,
                            pr4_evaluate, pr10_evaluate, predicted_logit_change, projected_union, reference_order,
                            slope, spearman_or_nan, union_curve, union_order, weighted_union)
from cct.premise_traces import sampled_outputs, split_sentences, ta_chunk_spans, tiny_sentences
from cct.utils.io import pack_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# -------------------------------------------------------------------------------------------------
# sentences
# -------------------------------------------------------------------------------------------------
VOCAB = ["<p0>", "<p1>", "<think>", "</think>", "\n", "Okay", ",", " so", " 4", " x", " 9", " =", " 36", ".",
         " Then", " -", " 23", " 13", ".\n\n", "So", " yes", "!", " Hi", " Done"]
V = {w: i for i, w in enumerate(VOCAB)}


def dec(ids):
    return "".join(VOCAB[i] for i in ids)


def test_ta_chunks_hand_example():
    t = "Okay, so 4 x 9 = 36. Then 36 - 23 = 13.\n\nSo yes! Hi. Done"
    spans = ta_chunk_spans(t)
    # "So yes!" (7 chars) merges into "Hi." (11 chars), then the trailing "Done" (4) merges back
    assert [t[a:b] for a, b in spans] == ["Okay, so 4 x 9 = 36.", "Then 36 - 23 = 13.", "So yes! Hi. Done"]
    # Thought Anchors drops text after the last boundary
    assert [t[a:b] for a, b in ta_chunk_spans(t, keep_trailing=False)][-1] == "So yes! Hi."
    # "3.5" is not a boundary (no space after the dot); short first chunk merges into the next
    assert [t2[a:b] for t2 in ["Ok. x is 3.5 here. End of it."] for a, b in ta_chunk_spans(t2)] == \
        ["Ok. x is 3.5 here.", "End of it."]


def test_split_sentences_token_spans():
    words = ["<p0>", "<p1>", "<think>", "\n", "Okay", ",", " so", " 4", " x", " 9", " =", " 36", ".", " Then", " 36",
             " -", " 23", " =", " 13", ".\n\n", "So", " yes", "!", " Hi", ".", " Done", "</think>"]
    ids = [V[w] for w in words]
    spans = split_sentences(ids, dec, 3, 26)
    # " Then" starts sentence 1 (its leading space is the gap before the chunk); ".\n\n" belongs to
    # sentence 1 (its first character is the full stop)
    assert spans == [(3, 13), (13, 20), (20, 26)]
    assert spans[0][0] == 3 and spans[-1][1] == 26
    assert all(a < b for a, b in spans) and all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    assert split_sentences(ids, dec, 5, 5) == []


def test_tiny_sentences():
    assert tiny_sentences(10, 22) == [(10, 14), (14, 18), (18, 22)]
    assert tiny_sentences(10, 24) == [(10, 14), (14, 18), (18, 24)]  # remainder joins the last


def _trace(n_prompt=10, n_gen=100, answer=105):
    return {"id": "t", "n_prompt_tokens": n_prompt, "n_generated": n_gen, "n_total_tokens": n_prompt + n_gen,
            "token_ids": list(range(n_prompt + n_gen)), "answer_position": answer}


def test_sampled_outputs_17_in_second_half():
    tr = _trace()
    sents = tiny_sentences(11, 100, 4) + [(100, 101)]  # last one is too short to sample 2 tokens
    outs = sampled_outputs(tr, sents)
    assert len(outs) == 17
    assert outs[0] == {"position": 105, "sentence_index": None, "kind": "answer"}
    mid = 10 + 100 // 2
    sel = [o for o in outs if o["kind"] == "sentence_token"]
    assert len({o["sentence_index"] for o in sel}) == 8
    for o in sel:
        a, b = sents[o["sentence_index"]]
        assert a >= mid and a <= o["position"] < min(a + 2, b)
    pos = [o["position"] for o in outs]
    assert len(set(pos)) == 17
    idx = [o["sentence_index"] for o in sel[::2]]
    assert idx == sorted(idx)
    elig = [i for i, (a, b) in enumerate(sents) if a >= mid and b - a >= 2]
    assert idx[0] == elig[0] and idx[-1] == elig[-1]  # spread includes the first and the last
    # no answer position: the last generated token
    tr2 = dict(tr, answer_position=None)
    assert sampled_outputs(tr2, sents)[0]["position"] == 109


def test_sampled_outputs_fallback_and_error():
    tr = _trace()
    sents = tiny_sentences(11, 100, 10)  # 9 sentences, only 4 start in the second half
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        outs = sampled_outputs(tr, sents)
    assert any("second half" in str(x.message) for x in w)
    assert len(outs) == 17
    with pytest.raises(ValueError):
        sampled_outputs(tr, tiny_sentences(11, 100, 30))


# -------------------------------------------------------------------------------------------------
# Stage B logic
# -------------------------------------------------------------------------------------------------
def _outs(tid, n_sent=8, base=100):
    o = [{"position": base + 99, "sentence_index": None, "kind": "answer"}]
    for i in range(n_sent):
        o += [{"position": base + 10 * i, "sentence_index": 3 + i, "kind": "sentence_token"},
              {"position": base + 10 * i + 1, "sentence_index": 3 + i, "kind": "sentence_token"}]
    return o


def test_faithfulness_selection_and_order():
    outs = {"a": _outs("a"), "b": _outs("b"), "c": _outs("c")}
    sel = faithfulness_selection(outs, 8)
    assert len(sel) == 8 and len(set(sel)) == 8
    assert sel[:3] == [("a", 199), ("b", 199), ("c", 199)]
    # 5 sentence outputs: a and b get 2 each (slots 2 and 6 of 8), c gets 1 (slot 4); first tokens only
    assert sorted(sel[3:]) == [("a", 120), ("a", 160), ("b", 120), ("b", 160), ("c", 140)]
    order = reference_order(outs, sel)
    assert len(order) == 51 and len({(o["trace_id"], o["position"]) for o in order}) == 51
    assert [(o["trace_id"], o["position"]) for o in order[:8]] == sel
    rest = order[8:]
    assert [o["trace_id"] for o in rest[:3]] == ["a", "b", "c"]  # round-robin


def test_pr4_evaluate_branches():
    thr = {"pass": {"median_completeness_traces_at_least": 0.70, "completeness_gap_vs_short_at_most": 0.10,
                    "median_faithfulness_spearman_traces_at_least": 0.5, "faithfulness_gap_vs_short_at_most": 0.15},
           "kill": {"median_completeness_below": 0.50, "faithfulness_spearman_below": 0.2},
           "informative_median_pruned_nodes_above": 2000}

    def ent(c, f=None, n=100):
        return {"completeness_score": c, "replacement_score": c - 0.1, "n_nodes_pruned": n,
                "faithfulness": None if f is None else {"spearman_direct": f, "spearman_total": f}}

    short = [ent(0.8, 0.7), ent(0.82, 0.6), ent(0.78, 0.65)]
    good = [ent(0.75, 0.6), ent(0.76), ent(0.74, 0.55), ent(0.8, 0.7)]
    r = pr4_evaluate(good, short, thr)
    assert r["status"] == "pass" and r["values"]["informative_graphs_too_big"] is False
    assert r["values"]["completeness_gap_vs_short"] == pytest.approx(0.8 - 0.755)
    assert pr4_evaluate([ent(0.75, 0.4)], short, thr)["status"] == "fail"
    assert pr4_evaluate([ent(0.65, 0.6)], short, thr)["status"] == "fail"  # below 0.70
    assert pr4_evaluate([ent(0.45, 0.6)], short, thr)["status"] == "kill"
    assert pr4_evaluate([ent(0.75, 0.1)], short, thr)["status"] == "kill"
    assert pr4_evaluate([ent(0.75)], short, thr)["status"] == "fail"  # faithfulness missing: cannot pass
    assert pr4_evaluate([], short, thr)["status"] == "not_run"
    assert pr4_evaluate([ent(0.75, 0.6, 3000)], short, thr)["values"]["informative_graphs_too_big"] is True


def test_spearman_or_nan():
    assert spearman_or_nan([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman_or_nan([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman_or_nan([1, 2, 3], [5, 5, 5]))
    assert math.isnan(spearman_or_nan([1, 2], [1, 2]))


def _toy_record():
    """Sources [f0 (layer 0), f1 (layer 1), error, token] at position 0; targets [2 logits, f0, f1].
    f0 -> f1 edge w = 0.8; activations a0 = 2, a1 = 4."""
    sm = NodeMeta(kind=torch.tensor([KIND_FEATURE, KIND_FEATURE, KIND_ERROR, KIND_TOKEN]),
                  layer=torch.tensor([0, 1, 0, -1]), pos=torch.tensor([0, 0, 0, 0]),
                  feature=torch.tensor([5, 7, -1, -1]), extra=torch.tensor([2.0, 4.0, 1.0, 1.0]))
    tm = NodeMeta(kind=torch.tensor([KIND_LOGIT, KIND_LOGIT, KIND_FEATURE, KIND_FEATURE]),
                  layer=torch.tensor([2, 2, 0, 1]), pos=torch.tensor([0, 0, 0, 0]),
                  feature=torch.tensor([-1, -1, 5, 7]), extra=torch.tensor([0.75, 0.25, 0.0, 0.0]))
    A = torch.tensor([[0.3, 0.5, 0.1, 0.2],
                      [-0.2, 0.4, 0.0, 0.1],
                      [0.0, 0.0, 0.3, 0.6],
                      [0.8, 0.0, 0.2, 0.4]], dtype=torch.float64)
    return {"A": pack_matrix(A, fp32=True), "target_meta": tm.to_dict(), "source_meta": sm.to_dict(), "T_D": 4,
            "N": 4, "exact_prune": {"node_mask": torch.tensor([True, False, True, True, True, True]),
                                    "node_influence": torch.tensor([0.5, 0.3, 0.1, 0.0, 0.0, 0.0])}}


def test_predicted_logit_change_hand_example():
    rec = _toy_record()
    p = predicted_logit_change(rec, [0, 1])
    d0 = 0.75 * 0.3 + 0.25 * -0.2
    d1 = 0.75 * 0.5 + 0.25 * 0.4
    assert p["direct"] == pytest.approx([-d0, -d1])
    # f0 changes f1's input by -0.8, i.e. f1 by -0.8 / a1 of its activation scale: d1 * 0.8 / 4
    assert p["total"] == pytest.approx([-(d0 + d1 * 0.8 / 4.0), -d1])


def test_oracle_rows_toy_and_tiny():
    rec = _toy_record()
    assert oracle_rows(rec, 0) == [(0, 0, 5)]  # f1 not kept
    model = TinyModel(TinyConfig(transcoder="plt", n_layers=3, n_features=32, seed=0))
    toks = torch.randint(0, 64, (10,), generator=torch.Generator().manual_seed(3))
    r = exact_graph_tiny(model, toks, cap=10_000, batch_size=64)
    nf = int((torch.as_tensor(r["source_meta"]["kind"]) == KIND_FEATURE).sum())
    kept = oracle_rows(r, 1)
    assert len(kept) == int(r["exact_prune"]["node_mask"][:nf].sum()) == len(set(kept))
    assert all(p >= -1 for _, p, _ in kept)


# -------------------------------------------------------------------------------------------------
# Stage E logic
# -------------------------------------------------------------------------------------------------
def test_union_and_oracle_on_toy_graphs():
    w = lambda q: q / 100.0  # noqa: E731
    e = [{"features": [(0, 1, 5), (1, 2, 3)], "n_logits": 2, "q": 100},
         {"features": [(1, 2, 3), (2, 4, 7)], "n_logits": 1, "q": 50},
         {"features": [(0, 1, 5)], "n_logits": 3, "q": 80}]
    u = weighted_union(e, w)
    # distinct rows 3 + logits 6; (0,1,5) min q 80, (1,2,3) min q 50, (2,4,7) q 50; logits 2*1 + 1*0.5 + 3*0.8
    assert u["count"] == 9
    assert u["weighted"] == pytest.approx(0.8 + 0.5 + 0.5 + 2.0 + 0.5 + 2.4)
    cur = union_curve(e, [1, 2, 3, 4], w)
    assert set(cur) == {1, 2, 3} and cur[1]["count"] == 4 and cur[2]["count"] == 6
    outs = [{"kind": "sentence_token", "position": 30}, {"kind": "answer", "position": 90},
            {"output_kind": "sentence_token", "position": 10}]
    assert [o["position"] for o in union_order(outs)] == [90, 10, 30]
    assert pooled_order({"a": [1, 2, 3], "b": [4, 5]}) == [1, 4, 2, 5, 3]


def test_cost_weights_slopes_and_projection():
    lengths, secs = [64, 128, 256, 300], [1.0, 1.5, 2.5, 3.0]
    assert interp_seconds(lengths, secs, 32) == pytest.approx(0.5)  # proportional below 64
    assert interp_seconds(lengths, secs, 96) == pytest.approx(1.25)
    assert interp_seconds(lengths, secs, 278) == pytest.approx(2.75)
    assert interp_seconds(lengths, secs, 400) == pytest.approx(3.0)
    curve = {n: {"weighted": 10 * n ** 0.8, "count": 10 * n} for n in [1, 2, 4, 5, 8, 11, 16, 17]}
    sl = local_slope(curve, [(5, 11), (11, 17)])
    assert sl["5-11"] == pytest.approx(0.8) and sl["min"] == pytest.approx(0.8)
    assert local_slope(curve, [(5, 11)], key="count")["min"] == pytest.approx(1.0)
    assert slope(100, 200, 16, 32) == pytest.approx(1.0)
    assert math.isnan(slope(0, 10, 1, 2))
    assert projected_union(100.0, 17, 170, 0.5) == pytest.approx(100 * 10 ** 0.5)
    c = cost_cct(512, 2.0, {"feature": 0.01, "error": 0.1}, {"feature": 1000, "error": 50, "token": 30}, 16,
                 0.002, 4000, 0.04)
    assert c["projected_hubs"] == pytest.approx(15.0)
    assert c["recovery_row_equiv"] == pytest.approx(200.0)
    assert c["cost_cct"] == pytest.approx(512 + 30 + 16 + 200)
    # S_proj on a hand example: U(17) = 2000 row-equivalents, slope 0.8, L = 250 generated tokens
    s_proj = projected_union(2000, 17, 250, 0.8) / c["cost_cct"]
    assert s_proj == pytest.approx(2000 * (250 / 17) ** 0.8 / 758)


def test_pr10_evaluate_branches():
    thr = {"pass": {"local_slope_at_least": 0.6, "traces_at_least": 2, "s_meas_at_least": 3.0,
                    "s_proj_at_least": 20.0, "s_oracle_proj_at_least": 5.0},
           "kill": {"s_proj_below": 3.0}}
    per = {"a": {"local_slope": 0.7, "S_proj": 30}, "b": {"local_slope": 0.65, "S_proj": 25},
           "c": {"local_slope": 0.4, "S_proj": 10}}
    good = {"S_meas": 4.0, "S_proj": 22.0, "S_oracle_proj": 6.0}
    assert pr10_evaluate(per, good, thr)["status"] == "pass"
    assert pr10_evaluate(per, dict(good, S_meas=2.0), thr)["status"] == "fail"
    per2 = dict(per, b={"local_slope": 0.5, "S_proj": 25})
    assert pr10_evaluate(per2, good, thr)["status"] == "fail"
    low = {t: dict(v, S_proj=2.0) for t, v in per.items()}
    assert pr10_evaluate(low, good, thr)["status"] == "kill"
    assert pr10_evaluate(dict(low, a={"local_slope": 0.7, "S_proj": 3.5}), good, thr)["status"] != "kill"


# -------------------------------------------------------------------------------------------------
# end to end (tiny)
# -------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_premise_b_and_e_tiny_end_to_end(tmp_path):
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=ROOT, env=env, capture_output=True,  # noqa: E731
                                    text=True, timeout=3600)
    r = run(os.path.join(ROOT, "scripts", "gen_traces.py"), "--tiny", "--force")
    assert r.returncode == 0, r.stderr[-3000:]
    res = os.path.join(ROOT, "results", "tiny", "gate0")
    with open(os.path.join(res, "traces.json"), encoding="utf-8") as f:
        traces = json.load(f)["traces"]
    assert [t["category"] for t in traces] == ["arithmetic", "multi_hop", "logic"]
    assert all(len(t["sampled_outputs"]) == 17 for t in traces)

    r = run(os.path.join(ROOT, "scripts", "premise_B.py"), "--tiny", "--force")
    assert r.returncode in (0, 2), r.stderr[-3000:]
    with open(os.path.join(res, "stage_B.json"), encoding="utf-8") as f:
        b = json.load(f)
    assert b["stage"] == "B" and set(b) >= {"run", "premises", "stop", "stop_reason", "gpu_seconds", "detail"}
    pr4 = b["premises"]["PR4"]
    assert pr4["status"] in ("pass", "fail", "kill")
    assert (r.returncode == 2) == b["stop"] == (pr4["status"] == "kill")
    assert pr4["values"]["n_trace_outputs"] == 51 and pr4["values"]["n_trace_faithfulness_outputs"] == 8
    assert pr4["values"]["n_short_prompts"] == 3
    with open(os.path.join(res, "reference_graphs.json"), encoding="utf-8") as f:
        idx = json.load(f)
    assert idx["n_done"] == 51 and len(idx["records"]) == 51
    rec = torch.load(os.path.join(ROOT, idx["records"][0]["path"]), weights_only=False)
    for k in ("position", "prefix_offset", "sentence_index", "output_kind", "budget", "conv_ok", "conv_jaccard",
              "trace_id"):
        assert k in rec
    assert rec["trace_id"] == traces[0]["id"] and rec["output_kind"] == "answer"

    # resumable: a second run recomputes nothing and gives the same PR4 values
    r = run(os.path.join(ROOT, "scripts", "premise_B.py"), "--tiny")
    assert r.returncode in (0, 2), r.stderr[-3000:]
    with open(os.path.join(res, "stage_B.json"), encoding="utf-8") as f:
        b2 = json.load(f)
    assert b2["premises"]["PR4"]["values"]["median_completeness_traces"] == pr4["values"]["median_completeness_traces"]

    # Stage E without Stage D: not_run; with a fabricated Stage D: a verdict
    r = run(os.path.join(ROOT, "scripts", "premise_E.py"), "--tiny", "--force", "--stage-d",
            str(tmp_path / "absent.json"))
    assert r.returncode == 0, r.stderr[-3000:]
    with open(os.path.join(res, "stage_E.json"), encoding="utf-8") as f:
        e = json.load(f)
    assert e["premises"]["PR10"]["status"] == "not_run"
    tids = [t["id"] for t in traces]
    d = {"m_trace": 64, "kappa": 1.5, "hub_rate_by_kind": {"feature": 0.01, "error": 0.05, "token": 0.1},
         "decode_seconds_per_column": 1e-4, "n_sources_by_kind": {t: {"feature": 400, "error": 150, "token": 76}
                                                                  for t in tids},
         "row_seconds_full": {t: 0.01 for t in tids}, "n_cols_after_screening": {t: 500 for t in tids}}
    (tmp_path / "stage_D.json").write_text(json.dumps(d), encoding="utf-8")
    r = run(os.path.join(ROOT, "scripts", "premise_E.py"), "--tiny", "--stage-d", str(tmp_path / "stage_D.json"))
    assert r.returncode in (0, 2), r.stderr[-3000:]
    with open(os.path.join(res, "stage_E.json"), encoding="utf-8") as f:
        e = json.load(f)
    p = e["premises"]["PR10"]
    assert p["status"] in ("pass", "fail", "kill")
    assert set(e["detail"]["U_trunc"]) == set(tids) | {"pooled"}
    for t in tids:
        U = e["detail"]["U_trunc_count"][t]
        assert [int(n) for n in U] == [1, 2, 4, 5, 8, 11, 16, 17]
        assert list(U.values()) == sorted(U.values())  # counts are nested
        assert e["detail"]["cost_cct"][t]["cost_cct"] == pytest.approx(64 + 1.5 * 19.1 + 16 + 5.0)
    assert set(e["detail"]["U_trunc"]["pooled"]) == {"1", "2", "4", "8", "16", "32", "51"}
    pooled51 = e["detail"]["U_trunc"]["pooled"]["51"]
    assert pooled51 == pytest.approx(sum(e["detail"]["U_trunc"][t]["17"] for t in tids))
    assert p["values"]["S_meas"] == pytest.approx(pooled51 / (3 * (64 + 1.5 * 19.1 + 16 + 5.0)))
