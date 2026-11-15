"""Gate 0: prereg consistency (like T16), every branch of the decision table, S_proj arithmetic, and the
report script on synthetic inputs."""
import copy
import json
import math
import os
import re
import subprocess
import sys

import pytest
import yaml

from cct.gate0 import c_constant, gate0_verdict, s_proj, screening_pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def pre():
    with open(os.path.join(ROOT, "configs", "prereg_gate0.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_prereg_gate0_consistency():
    with open(os.path.join(ROOT, "PREREG_GATE0.md"), encoding="utf-8") as f:
        md = f.read()
    blocks = re.findall(r"```yaml\n(.*?)```", md, re.S)
    assert len(blocks) == 1
    with open(os.path.join(ROOT, "configs", "prereg_gate0.yaml"), encoding="utf-8") as f:
        cfg_text = f.read()
    assert yaml.safe_load(blocks[0]) == yaml.safe_load(cfg_text)
    assert blocks[0].strip() == cfg_text.strip()


def test_prereg_gate0_keys(pre):
    # keys read by the other Gate 0 scripts
    assert pre["fidelity"] == {"influence_weighted_node_recall_at_least": 0.95, "node_jaccard_at_least": 0.85,
                               "seeds_required": 2}
    assert pre["k07"]["fp32_rel_err_at_most"] == 1e-4 and pre["k07"]["bf16_rel_err_at_most"] == 2e-2
    assert pre["g0b"]["convergence_jaccard_at_least"] == 0.95


# -------------------------------------------------------------------------------------------------
# hand-made inputs. PLT prompt: T = 1000, k_eff = 10, m_min = 256, so C = 256 / (10 ln 100) = 5.559.
# Trace: L = 250, beta = 0.8, c = 200, T = 1e5, N = 2e5, k_eff_trace = 20 (ratio 2), f_hub = 0.001
# (H = 200), kappa = 2: S_proj = 200 * 250^0.8 / (5.559 * 20 * ln 5000 + 400 + 16) = about 12.2.
# -------------------------------------------------------------------------------------------------
def make_g0a(frac=0.01, k07=1e-6, dtype="float32"):
    def prompt():
        return {"T": 1000, "N": 2000, "k07_rel_err": k07, "dtype": dtype, "a0_energy_share": 0.6,
                "s95_A_ewm": 40.0, "s95_rem_ewm": 10.0, "s95_rem_frac_T": frac, "hub_share_by_kind": {"0": 0.1},
                "c_trunc": 512, "m_min": 256, "fidelity_by_m": {"256": {"recall": 0.97, "jaccard": 0.9}},
                "recovery_seconds_by_m": {"256": 1.5}}
    pids = ["kill_test_00", "kill_test_04", "kill_test_09"]
    return {"run": {}, "sets": {s: {"timing": {"row_seconds": 0.01, "kappa": 2.0}, "prompts": {p: prompt() for p in pids}}
                                for s in ["gemma2_2b_plt", "gemma2_2b_clt426k"]}}


def make_union(beta=0.8, c_median=200.0, n_traces=2):
    tr = {f"trace_{i}": {"L": 300, "n_generated": 250, "beta": beta,
                         "U_by_n": {"1": 100, "2": 170, "4": 300}, "c_per_output": {"10": 200}}
          for i in range(n_traces)}
    return {"traces": tr, "beta": beta, "c_median": c_median, "n_outputs_without_c": 0}


def make_columns(k_eff=20.0, recall=0.995, k07=1e-6):
    return {"T": 100000, "N": 200000, "n_sampled": 300, "k07_rel_err": k07, "dtype": "float32",
            "s95_A_ewm": 80.0, "s95_rem_ewm": k_eff, "k_eff_trace": k_eff, "hub_fraction": 0.001,
            "screen_recall": {"0.01": {"recall": recall - 0.05, "omega_frac_T": 0.01},
                              "0.001": {"recall": recall, "omega_frac_T": 0.05},
                              "0.0001": {"recall": recall, "omega_frac_T": 0.2}},
            "timing": {"row_seconds": 0.02, "kappa": 2.0}}


def expected_s_proj():
    C = 256 / (10 * math.log(100))
    return 200 * 250 ** 0.8 / (C * 20 * math.log(100000 / 20) + 2.0 * 200 + 16)


def test_s_proj_hand_example():
    # T = 10 e so ln(T / k_eff) = 1: denominator 2 * 10 * 1 + 3 * 10 + 16 = 66, numerator 100 * 100^0.5 = 1000
    assert s_proj(100, 100, 0.5, 2.0, 10, 10 * math.e, 3.0, 10, 16) == pytest.approx(1000 / 66)
    assert c_constant(256, 10, 1000) == pytest.approx(256 / (10 * math.log(100)))
    assert math.isnan(c_constant(256, 10, 5))


def test_screening_pass(pre):
    r = screening_pass(make_columns()["screen_recall"], pre)
    assert r["pass"] and r["tau"] == pytest.approx(1e-3) and r["omega_frac_T"] == 0.05
    assert screening_pass({"0.01": {"recall": 0.5}}, pre)["pass"] is False
    assert screening_pass({"0.5": {"recall": 1.0}}, pre)["pass"] is None  # tau not preregistered


def test_go_big(pre):
    v = gate0_verdict(make_g0a(), make_union(), make_columns(), pre)
    assert v["verdict"] == "GO-BIG" and not v["partial"] and v["missing"] == []
    assert v["s_proj"] == pytest.approx(expected_s_proj())
    assert expected_s_proj() > 10
    assert all(c["pass"] for c in v["conditions"].values())


def test_go_structure_via_beta(pre):
    v = gate0_verdict(make_g0a(), make_union(beta=0.5, c_median=5000.0), make_columns(), pre)
    assert v["verdict"] == "GO-STRUCTURE" and not v["partial"]
    assert v["conditions"]["beta"]["pass"] is False and v["conditions"]["s_proj"]["pass"] is True


def test_go_structure_via_s_proj(pre):
    v = gate0_verdict(make_g0a(), make_union(c_median=10.0), make_columns(), pre)
    assert v["verdict"] == "GO-STRUCTURE"
    assert v["conditions"]["beta"]["pass"] is True and v["conditions"]["s_proj"]["pass"] is False


def test_nogo_via_s95(pre):
    v = gate0_verdict(make_g0a(frac=0.2), make_union(), make_columns(), pre)
    assert v["verdict"] == "NO-GO" and v["conditions"]["not_nogo_s95"]["pass"] is False


def test_nogo_via_k_eff(pre):
    v = gate0_verdict(make_g0a(), make_union(), make_columns(k_eff=150.0), pre)
    assert v["verdict"] == "NO-GO" and v["conditions"]["not_nogo_k_eff"]["pass"] is False


def test_s95_between_thresholds_is_not_nogo(pre):
    # 0.05 on every prompt: not compressible, not NO-GO either
    v = gate0_verdict(make_g0a(frac=0.05), make_union(), make_columns(), pre)
    assert v["verdict"] == "UNDECIDED" and "compressible" in v["derived"]["failed_conditions"]


def test_undecided_screening(pre):
    v = gate0_verdict(make_g0a(), make_union(), make_columns(recall=0.9), pre)
    assert v["verdict"] == "UNDECIDED" and v["derived"]["failed_conditions"] == ["screening"]


def test_compressible_on_one_set_only(pre):
    g = make_g0a()
    for p in g["sets"]["gemma2_2b_clt426k"]["prompts"].values():
        p["s95_rem_frac_T"] = 0.2
    assert gate0_verdict(g, make_union(), make_columns(), pre)["verdict"] == "GO-BIG"
    g["sets"]["gemma2_2b_plt"]["prompts"]["kill_test_00"]["s95_rem_frac_T"] = 0.03
    g["sets"]["gemma2_2b_plt"]["prompts"]["kill_test_04"]["s95_rem_frac_T"] = 0.03
    v = gate0_verdict(g, make_union(), make_columns(), pre)
    assert v["verdict"] == "UNDECIDED" and v["conditions"]["compressible"]["pass"] is False


def test_partial(pre):
    v = gate0_verdict(make_g0a(), make_union(), None, pre)
    assert v["partial"] and v["verdict"] == "UNDECIDED"  # GO-BIG never on partial data
    assert any("g0b_columns" in m for m in v["missing"]) and v["s_proj"] is None
    v = gate0_verdict(make_g0a(), make_union(n_traces=1), make_columns(), pre)
    assert v["partial"] and v["verdict"] == "UNDECIDED"
    v = gate0_verdict(make_g0a(), make_union(beta=0.4), None, pre)
    assert v["partial"] and v["verdict"] == "GO-STRUCTURE"
    v = gate0_verdict(make_g0a(frac=0.2), None, None, pre)
    assert v["partial"] and v["verdict"] == "NO-GO"
    v = gate0_verdict(None, None, None, pre)
    assert v["partial"] and v["verdict"] == "UNDECIDED"


def test_k07_failure_stops(pre):
    v = gate0_verdict(make_g0a(k07=1e-2), make_union(), make_columns(), pre)
    assert v["verdict"] == "STOPPED"
    assert gate0_verdict(make_g0a(k07=1e-2, dtype="bfloat16"), make_union(), make_columns(), pre)["verdict"] == "GO-BIG"
    assert gate0_verdict(make_g0a(), make_union(), make_columns(k07=1e-3), pre)["verdict"] == "STOPPED"


# -------------------------------------------------------------------------------------------------
# report script
# -------------------------------------------------------------------------------------------------
def _run_report(rd):
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "gate0_report.py"), "--tiny",
                        "--results-dir", str(rd)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    with open(os.path.join(rd, "GATE0_REPORT.md"), encoding="utf-8") as f:
        md = f.read()
    with open(os.path.join(rd, "gate0.json"), encoding="utf-8") as f:
        js = json.load(f)
    return md, js


def test_gate0_report_tiny(tmp_path):
    for name, obj in [("g0a.json", make_g0a()), ("g0b_union.json", make_union()), ("g0b_columns.json", make_columns())]:
        (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    md, js = _run_report(tmp_path)
    assert md.startswith("# Gate 0 report (TINY: numbers meaningless)")
    assert "Verdict: **GO-BIG**" in md and "Stop: waiting for Dawar's decision (PD-G0)." in md
    assert "| s_proj |" in md and "U(n) trace_0" in md
    assert "—" not in md  # no em-dashes
    assert js["verdict"]["verdict"] == "GO-BIG" and js["tiny"] is True
    assert js["verdict"]["s_proj"] == pytest.approx(expected_s_proj())
    for k in ("config", "seed", "git", "packages", "gpu", "wall_seconds"):
        assert k in js["run_record"]


def test_gate0_report_partial(tmp_path):
    (tmp_path / "g0a.json").write_text(json.dumps(make_g0a()), encoding="utf-8")
    md, js = _run_report(tmp_path)
    assert js["verdict"]["partial"] and js["verdict"]["verdict"] == "UNDECIDED"
    assert "(PARTIAL)" in md and "Not run." in md
