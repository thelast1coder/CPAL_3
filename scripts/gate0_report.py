"""Gate 0 report (PROJECT_V2.md Section 7, PREREG_GATE0.md): applies the preregistered decision table to
g0a.json, g0b_union.json and g0b_columns.json (whichever exist) and writes gate0.json and GATE0_REPORT.md
into the same results directory. Then stop and wait for Dawar (PD-G0)."""
from __future__ import annotations

import argparse
import os
import time

import yaml

from cct.gate0 import gate0_verdict, verdict_label
from cct.utils.io import atomic_write_json, atomic_write_text, read_json
from cct.utils.logging import git_info, run_record

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "configs", "prereg_gate0.yaml")


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    if isinstance(x, dict):
        return "; ".join(f"{k}: {fmt(v, nd)}" for k, v in x.items())
    if isinstance(x, (list, tuple)):
        return ", ".join(fmt(v, nd) for v in x)
    return str(x)


def load(path: str):
    return read_json(path) if os.path.exists(path) else None


def _n_key(k) -> float:
    try:
        return float(k)
    except (TypeError, ValueError):
        return float("inf")


def build_report(g0a, union, columns, v: dict, pre: dict, tiny: bool) -> str:
    d = v["derived"]
    c = v["conditions"]
    plt = d.get("plt_set")
    L = []
    L += ["# Gate 0 report" + (" (TINY: numbers meaningless)" if tiny else ""), ""]
    # three-line plain summary
    comp = c["compressible"]["value"]
    L += [f"Verdict: **{verdict_label(v)}**" + (f"; missing: {'; '.join(v['missing'])}." if v["missing"] else "."),
          f"Short prompts (G0-A): remainder compressible on {fmt(comp)} prompts per set; "
          f"k_eff_short(PLT) = {fmt((d.get('k_eff_short') or {}).get(plt))}, "
          f"C_short = {fmt((d.get('C') or {}).get('C_short'))}.",
          f"Whole traces (G0-B): beta = {fmt(d.get('beta'))}, k_eff ratio = {fmt(d.get('k_eff_ratio'))}, "
          f"best screening recall = {fmt(c['screening']['value'])}, S_proj(L) = {fmt(v['s_proj'])}.", ""]
    if tiny:
        L += ["TINY: this run uses a tiny random transformer and synthetic inputs. It tests the code and the "
              "decision table only; no number below says anything about real attribution graphs.", ""]
    gi = git_info(ROOT)
    L += ["## Setup", "", f"- git: {gi['hash']} (dirty: {gi['dirty']})",
          f"- prereg: configs/prereg_gate0.yaml, version {pre['prereg_version']} ({pre.get('status', '')})", ""]

    # ---- G0-A ----------------------------------------------------------------------------------
    L += ["## G0-A: short prompts", ""]
    if g0a is None:
        L += ["Not run.", ""]
    else:
        for s, sv in g0a.get("sets", {}).items():
            t = sv.get("timing", {})
            L += [f"### {s}" + (" (PLT)" if s == plt else ""), "",
                  f"Row time {fmt(t.get('row_seconds'))} s; kappa {fmt(t.get('kappa'))}; "
                  f"k_eff_short {fmt((d.get('k_eff_short') or {}).get(s))}.", "",
                  "| prompt | T | N | K0.7 error | dtype | A_0 energy share | s95 A | s95 remainder | s95 rem / T | c_trunc | m_min |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for pid, p in sv.get("prompts", {}).items():
                L.append(f"| {pid} | {fmt(p.get('T'))} | {fmt(p.get('N'))} | {fmt(p.get('k07_rel_err'))} | "
                         f"{fmt(p.get('dtype'))} | {fmt(p.get('a0_energy_share'))} | {fmt(p.get('s95_A_ewm'))} | "
                         f"{fmt(p.get('s95_rem_ewm'))} | {fmt(p.get('s95_rem_frac_T'))} | {fmt(p.get('c_trunc'))} | "
                         f"{fmt(p.get('m_min'))} |")
            L += ["", "| prompt | hub share by kind | fidelity by m | recovery seconds by m |", "|---|---|---|---|"]
            for pid, p in sv.get("prompts", {}).items():
                L.append(f"| {pid} | {fmt(p.get('hub_share_by_kind'))} | {fmt(p.get('fidelity_by_m'))} | "
                         f"{fmt(p.get('recovery_seconds_by_m'))} |")
            L.append("")
        if d.get("C"):
            cc = d["C"]
            L += [f"Sensing constant on {plt}: C by prompt {fmt(cc['C_by_prompt'])}; C_short = {fmt(cc['C_short'])}."
                  + (f" Flagged (no m on the grid met fidelity, 1024 used): {', '.join(cc['flagged_no_m'])}."
                     if cc["flagged_no_m"] else ""), ""]

    # ---- G0-B union -----------------------------------------------------------------------------
    L += ["## G0-B: union growth", ""]
    if union is None:
        L += ["Not run.", ""]
    else:
        tr = union.get("traces", {})
        ns = sorted({k for t in tr.values() for k in (t.get("U_by_n") or {})}, key=_n_key)
        L += ["| n | " + " | ".join(f"U(n) {tid}" for tid in tr) + " |", "|---|" + "---|" * len(tr)]
        for n in ns:
            L.append(f"| {n} | " + " | ".join(fmt((t.get("U_by_n") or {}).get(n)) for t in tr.values()) + " |")
        L += ["", "| trace | L | generated tokens | beta | outputs with c |", "|---|---|---|---|---|"]
        for tid, t in tr.items():
            L.append(f"| {tid} | {fmt(t.get('L'))} | {fmt(t.get('n_generated'))} | {fmt(t.get('beta'))} | "
                     f"{len(t.get('c_per_output') or {})} |")
        L += ["", f"beta (mean over traces) = {fmt(d.get('beta'))}; c (median c_i) = {fmt(d.get('c_median'))}; "
                  f"outputs without c = {fmt(union.get('n_outputs_without_c'))}; L (median generated tokens) = "
                  f"{fmt(d.get('L'))}.", ""]

    # ---- G0-B columns ---------------------------------------------------------------------------
    L += ["## G0-B: sampled exact columns", ""]
    if columns is None:
        L += ["Not run.", ""]
    else:
        t = columns.get("timing") or {}
        L += ["| T_trace | N_trace | sampled | K0.7 error | dtype | s95 A | s95 remainder | k_eff_trace | hub fraction | row seconds | kappa |",
              "|---|---|---|---|---|---|---|---|---|---|---|",
              f"| {fmt(columns.get('T'))} | {fmt(columns.get('N'))} | {fmt(columns.get('n_sampled'))} | "
              f"{fmt(columns.get('k07_rel_err'))} | {fmt(columns.get('dtype'))} | {fmt(columns.get('s95_A_ewm'))} | "
              f"{fmt(columns.get('s95_rem_ewm'))} | {fmt(columns.get('k_eff_trace'))} | "
              f"{fmt(columns.get('hub_fraction'))} | {fmt(t.get('row_seconds'))} | {fmt(t.get('kappa'))} |",
              "", "| tau | energy recall | mean Omega_s size / T |", "|---|---|---|"]
        for tau, r in (columns.get("screen_recall") or {}).items():
            r = r if isinstance(r, dict) else {"recall": r}
            L.append(f"| {tau} | {fmt(r.get('recall'), 4)} | {fmt(r.get('omega_frac_T'))} |")
        scr = d.get("screening") or {}
        L += ["", f"Screening: {'passes at tau = ' + fmt(scr.get('tau')) if scr.get('pass') else 'does not pass'}"
                  + (f", |Omega_s| / T = {fmt(scr.get('omega_frac_T'))}." if scr.get("pass") else "."), ""]

    # ---- S_proj ---------------------------------------------------------------------------------
    L += ["## Projected speed-up", "",
          "S_proj(L) = c L^beta / (C_short k_eff_trace ln(T_trace / k_eff_trace) + kappa H_est + m')", ""]
    terms = d.get("s_proj_terms")
    if terms is None:
        L += ["Not computable: a G0-B part is missing.", ""]
    else:
        L += ["| term | value |", "|---|---|"]
        for k in ("c", "L", "beta", "C_short", "k_eff_trace", "T_trace", "kappa", "H_est", "m_prime",
                  "trunc_cost", "sensing_cost", "hub_cost", "s_proj"):
            L.append(f"| {k} | {fmt(terms.get(k))} |")
        L.append("")

    # ---- meaning --------------------------------------------------------------------------------
    L += ["## What the numbers mean", ""]
    if tiny:
        L += ["Nothing about real models: this is a tiny run. It shows that the report reads every input, "
              "applies the decision table and writes its outputs.", ""]
    words = {
        "k07": ("the closed-form local part matches the attention-detached backward pass",
                "the closed-form local part does not match the attention-detached backward pass, so Gate 0 stops"),
        "compressible": ("after removing the local part, the remainder is compressible on short prompts",
                         "the remainder is not compressible enough on short prompts"),
        "beta": ("union-truncation cost grows fast enough with the number of explained outputs",
                 "union-truncation cost grows too slowly with the number of explained outputs for a big win"),
        "k_eff_ratio": ("the remainder stays about as sparse at trace length as on short prompts",
                        "the remainder is much less sparse at trace length than on short prompts"),
        "screening": ("attention reachability keeps almost all column energy at some threshold",
                      "attention reachability screening loses more than 1% of column energy at every threshold"),
        "s_proj": ("the cost law projects at least a 10x saving over union-truncation at trace length",
                   "the cost law projects less than a 10x saving over union-truncation at trace length"),
    }
    for name, (yes, no) in words.items():
        p = c[name]["pass"]
        L.append(f"- {name}: " + (yes if p is True else no if p is False else "not evaluable (missing input)") + ".")
    if not tiny:
        L += ["", "(To be expanded by hand after the real run: surprises, caveats, next steps.)"]
    L += ["", "## Verdict", "", "| condition | value | threshold | pass |", "|---|---|---|---|"]
    for name, x in c.items():
        val, thr = x["value"], x["threshold"]
        if name == "k07" and isinstance(val, dict):  # one entry per check: show the worst
            val = f"max {fmt(max(val.values()))} over {len(val)} checks"
            thr = " or ".join(sorted({fmt(t) for t in x["threshold"].values()})) + " (by dtype)"
        L.append(f"| {name} | {fmt(val)} | {fmt(thr)} | {fmt(x['pass'])} |")
    L += ["", f"Verdict: **{verdict_label(v)}**."]
    if v["missing"]:
        L += ["", "Missing: " + "; ".join(v["missing"]) + "."]
    if v["verdict"] == "UNDECIDED" and d.get("failed_conditions"):
        L += ["", "Failed conditions: " + ", ".join(d["failed_conditions"]) + "."]
    if v["verdict"] == "STOPPED":
        L += ["", "K0.7 failed on: " + ", ".join(d.get("k07_failures", [])) + ". No verdict is issued."]
    L += ["", "Stop: waiting for Dawar's decision (PD-G0).", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Gate 0 report")
    ap.add_argument("--tiny", action="store_true", help="read results/tiny/gate0/; numbers are meaningless")
    ap.add_argument("--results-dir", default=None, help="override the results directory (tests)")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    rd = args.results_dir or os.path.join(ROOT, "results", *(["tiny"] if args.tiny else []), "gate0")
    os.makedirs(rd, exist_ok=True)
    with open(PREREG, encoding="utf-8") as f:
        pre = yaml.safe_load(f)
    g0a = load(os.path.join(rd, "g0a.json"))
    union = load(os.path.join(rd, "g0b_union.json"))
    columns = load(os.path.join(rd, "g0b_columns.json"))
    v = gate0_verdict(g0a, union, columns, pre)
    atomic_write_text(os.path.join(rd, "GATE0_REPORT.md"), build_report(g0a, union, columns, v, pre, args.tiny))
    summary = {
        "g0a": None if g0a is None else {
            s: {pid: {k: p.get(k) for k in ("T", "N", "k07_rel_err", "s95_rem_ewm", "s95_rem_frac_T", "c_trunc", "m_min")}
                for pid, p in sv.get("prompts", {}).items()} for s, sv in g0a.get("sets", {}).items()},
        "union": None if union is None else {k: union.get(k) for k in ("beta", "c_median", "n_outputs_without_c")},
        "columns": None if columns is None else {k: columns.get(k) for k in
                                                 ("T", "N", "n_sampled", "k_eff_trace", "hub_fraction", "screen_recall")},
    }
    atomic_write_json(os.path.join(rd, "gate0.json"),
                      {"verdict": v, "inputs": summary, "tiny": args.tiny,
                       "run_record": run_record({"prereg": pre, "results_dir": rd}, None,
                                                time.perf_counter() - t0)})
    print(f"verdict: {verdict_label(v)}")


if __name__ == "__main__":
    main()
