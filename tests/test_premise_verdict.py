"""Gate 0 premise verdict (GATE0_PREMISES.md Section 4): prereg consistency, every branch of the decision
table on fabricated stage files, and scripts/premise_report.py on fabricated files (complete and partial)."""
import json
import os
import re
import subprocess
import sys

import pytest
import yaml

from cct.premise import PREMISES, STAGE_FILES, premise_verdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE_PRS = {"A": ["PR1", "PR2", "PR3"], "B": ["PR4"], "C": ["PR5", "PR6"], "D": ["PR7", "PR8", "PR9"],
             "E": ["PR10"], "F": ["PR11"]}


@pytest.fixture(scope="module")
def pre():
    with open(os.path.join(ROOT, "configs", "prereg_gate0.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_prereg_gate0_yaml_block_identical():
    with open(os.path.join(ROOT, "PREREG_GATE0.md"), "rb") as f:
        md = f.read().decode("utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", md, re.S)
    assert len(blocks) == 1
    with open(os.path.join(ROOT, "configs", "prereg_gate0.yaml"), "rb") as f:
        cfg = f.read().decode("utf-8")
    assert blocks[0] == cfg  # byte-identical (both LF)
    assert yaml.safe_load(blocks[0]) == yaml.safe_load(cfg)


def test_prereg_verdict_order(pre):
    assert pre["verdicts"]["order"] == ["NO-GO", "SWITCH-MODEL", "GO-BIG", "GO-METHOD", "GO-STRUCTURE",
                                        "UNDECIDED"]


# -------------------------------------------------------------------------------------------------
# fabricated stage files
# -------------------------------------------------------------------------------------------------
def stage(name, statuses=None, stop=False, stop_reason=None, gpu=360.0, detail=None, model="qwen3_1p7b_plt"):
    statuses = statuses or {}
    prs = STAGE_PRS["B" if name.startswith("B") else name]
    return {"run": {"config": {"model_set": model}, "seed": 0, "git": {"hash": "abc123def4567890", "dirty": False},
                    "packages": {}, "gpu": "NVIDIA H100", "wall_seconds": gpu + 5, "utc": "2026-11-16T00:00:00+00:00"},
            "stage": name,
            "premises": {p: {"status": statuses.get(p, "pass"), "values": {"x": 1.0}, "thresholds": {},
                             "notes": ""} for p in prs},
            "stop": stop, "stop_reason": stop_reason, "gpu_seconds": gpu, "detail": detail or {}}


def all_pass(**override):
    """{stage: stage dict} with every premise passing; override PRk=status (PR4_4b=status adds the 4B file)."""
    st = {k: v for k, v in override.items() if k.startswith("PR") and k != "PR4_4b"}
    stages = {s: stage(s, st) for s in STAGE_PRS}
    if "PR4_4b" in override:
        stages["B_qwen3_4b"] = stage("B_qwen3_4b", {"PR4": override["PR4_4b"]}, model="qwen3_4b_plt")
    return stages


def V(stages, pre):
    return premise_verdict(stages, pre)


def test_go_big(pre):
    v = V(all_pass(PR3="fail"), pre)  # PR3 informative only
    assert v["verdict"] == "GO-BIG" and not v["missing"] and not v["pending"]
    assert set(v["premises"]) >= set(PREMISES)


@pytest.mark.parametrize("pr", ["PR1", "PR5", "PR6"])
def test_nogo_single_kill(pre, pr):
    v = V(all_pass(**{pr: "kill"}), pre)
    assert v["verdict"] == "NO-GO" and any(pr in r for r in v["reasons"])


def test_nogo_pr10_and_pr4_kill(pre):
    v = V(all_pass(PR10="kill", PR4="kill", PR4_4b="kill"), pre)
    assert v["verdict"] == "NO-GO"
    v = V(all_pass(PR10="kill", PR4="kill"), pre)  # 4B never run: the 1.7B kill stands
    assert v["verdict"] == "NO-GO"


def test_pr10_kill_alone_is_go_structure(pre):
    v = V(all_pass(PR10="kill"), pre)
    assert v["verdict"] == "GO-STRUCTURE"


def test_switch_model(pre):
    for s in ("fail", "kill"):
        v = V(all_pass(PR4=s, PR4_4b="pass"), pre)
        assert v["verdict"] == "SWITCH-MODEL", s
    # PR4 killed on 1.7B, PR10 killed, but 4B passes: not NO-GO, SWITCH-MODEL
    assert V(all_pass(PR4="kill", PR10="kill", PR4_4b="pass"), pre)["verdict"] == "SWITCH-MODEL"


def test_nogo_beats_switch_model(pre):
    assert V(all_pass(PR1="kill", PR4="fail", PR4_4b="pass"), pre)["verdict"] == "NO-GO"


@pytest.mark.parametrize("pr11", ["fail", "weak", "kill"])
def test_go_method(pre, pr11):
    assert V(all_pass(PR11=pr11), pre)["verdict"] == "GO-METHOD"


@pytest.mark.parametrize("pr,status", [("PR7", "fail"), ("PR9", "fail"), ("PR10", "fail"), ("PR7", "kill"),
                                       ("PR9", "kill")])
def test_go_structure(pre, pr, status):
    v = V(all_pass(**{pr: status}), pre)
    assert v["verdict"] == "GO-STRUCTURE" and any(pr in r for r in v["reasons"])


def test_go_structure_with_later_stages_missing(pre):
    stages = all_pass(PR7="kill")
    stages["D"]["stop"], stages["D"]["stop_reason"] = True, "PR7 kill: stop whole-trace work"
    stages["E"] = stages["F"] = None
    v = V(stages, pre)
    assert v["verdict"] == "GO-STRUCTURE" and len(v["missing"]) == 2


def test_undecided(pre):
    v = V(all_pass(PR8="fail"), pre)  # PR8 fails alone: no row covers it
    assert v["verdict"] == "UNDECIDED" and any("PR8" in r for r in v["reasons"])
    v = V(all_pass(PR4="fail"), pre)  # no Qwen3-4B result yet
    assert v["verdict"] == "UNDECIDED" and any("B_qwen3_4b" in m for m in v["missing"])


def test_stopped_early_pr2(pre):
    stages = {s: None for s in STAGE_PRS}
    stages["A"] = stage("A", {"PR2": "kill"}, stop=True, stop_reason="PR2 kill: chat template differs")
    v = V(stages, pre)
    assert v["verdict"] == "STOPPED-EARLY" and "chat template" in v["reasons"][0]
    # stop flag not set by the stage: implied by the kill
    stages["A"]["stop"] = False
    v = V(stages, pre)
    assert v["verdict"] == "STOPPED-EARLY" and "PR2 kill" in v["reasons"][0]


def test_stopped_early_pr4_both_models(pre):
    stages = all_pass(PR4="kill", PR4_4b="kill")
    stages["B"]["stop"], stages["B"]["stop_reason"] = True, "PR4 kill on both models"
    stages["E"] = stages["F"] = None
    v = V(stages, pre)
    assert v["verdict"] == "STOPPED-EARLY" and any("both models" in r for r in v["reasons"])


def test_stopped_early_pr9_without_structure(pre):
    stages = all_pass(PR9="kill", PR6="fail")  # GO-STRUCTURE needs PR6 pass
    stages["D"]["stop"], stages["D"]["stop_reason"] = True, "PR9 kill: no m_trace"
    stages["E"] = stages["F"] = None
    assert V(stages, pre)["verdict"] == "STOPPED-EARLY"


def test_pending_labels(pre):
    v = V(all_pass(PR11="pending"), pre)
    assert v["verdict"] == "PENDING-LABELS" and v["candidates"] == ["GO-BIG", "GO-METHOD"]
    assert v["pending"] == ["PR11"] and "pending" in v["label"]


def test_missing_stages_never_go_big(pre):
    for s in STAGE_PRS:
        stages = all_pass()
        stages[s] = None
        v = V(stages, pre)
        assert v["verdict"] not in ("GO-BIG", "PENDING-LABELS")
        assert any(STAGE_FILES[s] in m for m in v["missing"])
    v = V({}, pre)
    assert v["verdict"] == "UNDECIDED" and len(v["missing"]) == 6
    stages = all_pass()
    del stages["D"]["premises"]["PR8"]
    v = V(stages, pre)
    assert v["verdict"] == "UNDECIDED" and any("PR8" in m for m in v["missing"])


def test_unknown_status_is_not_pass(pre):
    v = V(all_pass(PR8="maybe"), pre)
    assert v["verdict"] == "UNDECIDED" and any("unrecognised" in r for r in v["reasons"])


# -------------------------------------------------------------------------------------------------
# report script
# -------------------------------------------------------------------------------------------------
def rich_details(stages):
    stages["C"]["detail"] = {
        "s95_abs_frac_causal": {"qwen3_1p7b_plt": {"p0": [0.01, 0.02, 0.005], "p1": [0.03, 0.01]}},
        "fidelity_vs_m": {"qwen3_1p7b_plt": {"p0": {"D1": {"256": 0.9, "512": 0.97},
                                                    "D2": {"256": 0.85, "512": 0.96}}}},
        "epsilon_star": 0.12, "k_eff_short_qwen": 40.0}
    stages["D"]["detail"] = {"s95_abs_frac_causal": [0.02, 0.04, 0.01], "m_trace": 2048, "kappa": 2.5}
    stages["E"]["detail"] = {
        "union": {"trace_0": {"U_trunc": {"1": 100, "16": 900, "32": 1500, "51": 2100},
                              "U_oracle": {"1": 50, "16": 400, "32": 700, "51": 1000}}},
        "cost_cct": 3000.0, "cost_weights": {"64": 0.3, "128": 0.5, "256": 0.8, "full": 1.0},
        "S_meas": 4.0, "S_proj": 25.0, "S_oracle_proj": 6.0, "check_probes": 16}
    stages["F"]["detail"] = {
        "auroc": {"feature": 0.71, "S4": 0.70, "S1": 0.6, "S2": 0.62},
        "top_dependencies": [{"trace": "t0", "i": 1, "j": 4, "D": 0.3,
                              "carriers": [{"feature": 123, "layer": 10, "top_tokens": ["plus", "sum"]}]}]}
    return stages


def run_report(rd):
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONPATH=os.path.join(ROOT, "src"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "premise_report.py"), "--tiny",
                        "--results-dir", str(rd)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    with open(os.path.join(rd, "PREMISE_REPORT.md"), encoding="utf-8") as f:
        md = f.read()
    with open(os.path.join(rd, "gate0.json"), encoding="utf-8") as f:
        js = json.load(f)
    return md, js


def write_stages(rd, stages):
    for s, d in stages.items():
        if d is not None:
            with open(os.path.join(rd, STAGE_FILES[s]), "w", encoding="utf-8") as f:
                json.dump(d, f)


def check_summary(md, verdict):
    lines = md.split("\n")
    assert all(lines[i].strip() for i in range(3)) and lines[3] == ""
    assert lines[0].startswith("TINY: numbers meaningless. Gate 0 verdict: " + verdict)
    assert lines[1].startswith("Passed:") and lines[2].startswith("Why:")
    assert md.rstrip().endswith("Stop: waiting for Dawar's decision.")


def test_report_complete(tmp_path):
    write_stages(tmp_path, rich_details(all_pass()))
    md, js = run_report(tmp_path)
    check_summary(md, "GO-BIG")
    assert js["verdict"]["verdict"] == "GO-BIG"
    assert js["gpu_seconds_total"] == pytest.approx(6 * 360.0)
    assert js["git_hashes"]["A"] == "abc123def4567890"
    for h in ("## Premises", "## Plots", "## Cost table", "## Top 15 dependencies", "## Deviations",
              "## GPU time per stage", "## What surprised us"):
        assert h in md
    assert "| S_proj | 25 |" in md and "L10 123 (plus, sum)" in md
    if __import__("importlib").util.find_spec("matplotlib") is not None:
        for p in ("s95_hist.png", "fidelity_vs_m.png", "union_growth.png", "auroc.png"):
            assert os.path.getsize(tmp_path / "plots" / p) > 0
            assert f"plots/{p}" in md
    else:
        assert "matplotlib is not installed" in md


def test_report_partial(tmp_path):
    stages = all_pass(PR2="kill")
    stages["A"]["stop"], stages["A"]["stop_reason"] = True, "PR2 kill"
    write_stages(tmp_path, {"A": stages["A"]})
    (tmp_path / "stage_C.json").write_text("{not json", encoding="utf-8")
    md, js = run_report(tmp_path)
    check_summary(md, "STOPPED-EARLY")
    assert "data not available" in md and "unreadable" in md
    assert "| S_proj | not available |" in md
    assert not (tmp_path / "plots").exists()
    assert js["gpu_seconds"]["B"] is None and len(js["verdict"]["missing"]) == 5


def test_report_pending_labels(tmp_path):
    write_stages(tmp_path, all_pass(PR11="pending"))
    (tmp_path / "carriers_for_labelling.md").write_text("# carriers\n", encoding="utf-8")
    md, _ = run_report(tmp_path)
    check_summary(md, "PENDING-LABELS")
    assert "GO-BIG or GO-METHOD" in md.split("\n")[0]
    assert "carriers_for_labelling.md" in md
