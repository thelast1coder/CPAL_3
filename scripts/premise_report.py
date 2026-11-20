"""Gate 0 premise report (GATE0_PREMISES.md Sections 4 and 5, PREREG_GATE0.md).

Reads stage_A.json, stage_B.json, stage_B_qwen3_4b.json, stage_C.json ... stage_F.json (whichever
exist) from the results directory, applies the preregistered verdict (cct.premise.premise_verdict),
and writes into the same directory:
  gate0.json          verdict, every premise status and values, GPU seconds per stage and in total,
                      run records and git hashes of every stage
  PREMISE_REPORT.md   the Section 5 report
  plots/*.png         plots (matplotlib, Agg backend); a plot whose data is missing is skipped and the
                      report says so

    python scripts/premise_report.py            # results/gate0
    python scripts/premise_report.py --tiny     # results/tiny/gate0 (numbers meaningless)
    python scripts/premise_report.py --results-dir DIR

Stage detail keys are read defensively: the report searches for any of a few accepted names (listed
in the *_KEYS constants below) anywhere in a stage's "premises" values and "detail", and prints
"not available" when none is found. Then stop and wait for Dawar.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cct.premise import (PR4_4B, PREMISE_STAGE, PREMISES, STAGE_FILES, STAGES,  # noqa: E402
                         premise_entry, premise_verdict)

PREREG = os.path.join(ROOT, "configs", "prereg_gate0.yaml")
PREREG_MD = os.path.join(ROOT, "PREREG_GATE0.md")
NA = "not available"
# Stage GPU budgets in H100-hours (GATE0_PREMISES.md Section 3 stage headings).
STAGE_BUDGET_H = {"A": 0.3, "B": 0.5, "B_qwen3_4b": 0.5, "C": 1.0, "D": 1.0, "E": 1.5, "F": 1.5}

# accepted key names for the plots and tables (first match wins, searched recursively)
S95_KEYS = ("s95_abs_frac_causal", "s95_rem_abs_frac_causal", "remainder_abs_s95_frac_causal", "s95_hist", "s95_values", "s95_abs",
            "s95_rem_abs", "s95_columns")
FID_KEYS = ("fidelity_vs_m", "fidelity_by_m", "fidelity_tables")
U_REAL_KEYS = ("U_trunc", "u_trunc", "union_realistic", "realistic")
U_ORACLE_KEYS = ("U_oracle", "u_oracle", "union_oracle", "oracle")
AUROC_KEYS = ("auroc_by_method", "auroc")
COST_ROWS = [  # (label, accepted keys)
    ("m_trace (PR9)", ("m_trace",)),
    ("kappa (hub cost in row-equivalents)", ("kappa",)),
    ("projected hub count", ("hub_count_projected", "projected_hubs", "H_proj", "hub_count")),
    ("check probes", ("check_probes",)),
    ("recovery cost (row-equivalents)", ("recovery_row_equiv", "recovery_cost_rows", "recovery_rows")),
    ("decode time per column (s)", ("decode_seconds_per_column", "decode_time_per_column")),
    ("columns kept after energy screening", ("n_cols_after_screening", "columns_kept", "n_columns_kept")),
    ("row time at full length (s)", ("row_seconds_full", "row_seconds", "row_time_full")),
    ("Cost_CCT(L) (row-equivalents)", ("cost_cct", "Cost_CCT", "cost_cct_L")),
    ("L (trace tokens)", ("L", "trace_tokens")),
    ("truncation cost weights by prefix length", ("cost_weights", "truncation_cost_weights",
                                                  "pass_seconds_by_length")),
    ("local slope", ("local_slope", "local_slopes")),
    ("U_trunc(51) size (cost-weighted)", ("U_trunc_51_weighted", "u_trunc_51_weighted")),
    ("S_meas", ("S_meas", "s_meas")),
    ("S_proj", ("S_proj", "s_proj")),
    ("S_oracle_proj", ("S_oracle_proj", "s_oracle_proj")),
]
DEP_KEYS = ("top_dependencies", "dependencies_top", "carriers")

MEASURED = {
    "PR1": "mixed-pass rows against Phi times exact rows (rel. error); mixed/single pass time ratio",
    "PR2": "prefix rows against full-trace frozen-pass rows, 3 positions x 50 targets",
    "PR3": "closed-form A_0 against frozen backward with attention stop-gradiented (informative)",
    "PR4": "Qwen3-1.7B trace completeness and ablation faithfulness against short prompts",
    PR4_4B: "the same on Qwen3-4B (SWITCH-MODEL path)",
    "PR5": "remainder abs. s95 as share of the causal set, Qwen3-1.7B PLT short prompts",
    "PR6": "pruned-graph fidelity from simulated D2 recovery, smallest m",
    "PR7": "k_eff(trace) / k_eff(short), hub rate",
    "PR8": "reachability mass recall and median size of Omega_s as share of the causal set",
    "PR9": "m_trace: smallest m with 90% of non-hub energy within epsilon*",
    "PR10": "union growth slope, S_meas, S_proj, S_oracle_proj",
    "PR11": "feature-level AUROC against S4; nameable carriers",
}


# -------------------------------------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------------------------------------
def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def fmt(x, nd=3, limit=220) -> str:
    if x is None:
        return NA
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, float):
        s = f"{x:.{nd}g}"
    elif isinstance(x, dict):
        s = "; ".join(f"{k}: {fmt(v, nd, limit)}" for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        s = ", ".join(fmt(v, nd, limit) for v in x)
    else:
        s = str(x)
    s = s.replace("|", "/").replace("\n", " ")
    return s if len(s) <= limit else s[: limit - 3] + "..."


def load_stage(path: str):
    """(stage dict or None, note or None). Unreadable or malformed files count as missing."""
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"{os.path.basename(path)} unreadable ({type(e).__name__}); treated as missing"
    if not isinstance(d, dict):
        return None, f"{os.path.basename(path)} is not a JSON object; treated as missing"
    return d, None


def find(obj, keys, depth=0):
    """First value under any of `keys`, searching dicts breadth-first (lists of dicts too)."""
    if depth > 8:
        return None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        children = list(obj.values())
    elif isinstance(obj, list):
        children = [c for c in obj if isinstance(c, (dict, list))]
    else:
        return None
    for c in children:
        if isinstance(c, (dict, list)):
            r = find(c, keys, depth + 1)
            if r is not None:
                return r
    return None


def stage_search_space(stage: dict | None) -> list:
    if not isinstance(stage, dict):
        return []
    space = [{k: v for k, v in stage.items() if k not in ("run", "premises", "detail")}]
    for e in (stage.get("premises") or {}).values():
        if isinstance(e, dict) and isinstance(e.get("values"), dict):
            space.append(e["values"])
    if isinstance(stage.get("detail"), dict):
        space.append(stage["detail"])
    return space


def lookup(stage, keys):
    for space in stage_search_space(stage):
        r = find(space, keys)
        if r is not None:
            return r
    return None


def numeric_lists(obj, prefix="", depth=0) -> dict[str, list[float]]:
    """{path: [numbers]} for every list of numbers inside obj (a bare list gives path prefix)."""
    out = {}
    if depth > 6:
        return out
    if isinstance(obj, list):
        nums = [float(v) for v in obj if _num(v)]
        if nums and len(nums) == len(obj):
            out[prefix or "all"] = nums
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(numeric_lists(v, f"{prefix}/{k}" if prefix else str(k), depth + 1))
    return out


def numeric_leaves(obj, depth=0) -> list[float]:
    """Every finite number inside nested dicts and lists."""
    if _num(obj):
        return [float(obj)]
    if depth > 8:
        return []
    items = obj.values() if isinstance(obj, dict) else obj if isinstance(obj, list) else []
    return [x for c in items for x in numeric_leaves(c, depth + 1)]


def xy_series(obj, prefix="", depth=0) -> dict[str, list[tuple[float, float]]]:
    """{path: [(x, y)]} for every dict mapping numeric-like keys to numbers (or to dicts holding a
    recall-like number)."""
    out = {}
    if not isinstance(obj, dict) or depth > 6:
        return out
    pts = []
    for k, v in obj.items():
        try:
            x = float(k)
        except (TypeError, ValueError):
            pts = None
            break
        y = v if _num(v) else (find(v, ("recall", "node_recall", "influence_weighted_node_recall",
                                        "fidelity", "frac_meeting", "value", "size"))
                               if isinstance(v, dict) else None)
        if y is None and isinstance(v, dict) and _num(v.get("n_meet")) and _num(v.get("n_runs")) and v["n_runs"]:
            y = v["n_meet"] / v["n_runs"]  # share of seeds meeting fidelity
        if isinstance(y, bool):
            y = float(y)
        if not _num(y):
            pts = None
            break
        pts.append((x, float(y)))
    if pts:
        out[prefix or "all"] = sorted(pts)
        return out
    for k, v in obj.items():
        out.update(xy_series(v, f"{prefix}/{k}" if prefix else str(k), depth + 1))
    return out


def local_slopes(pts: list[tuple[float, float]]) -> dict:
    d = {x: y for x, y in pts}
    out = {}
    for a, b in ((16, 32), (32, 51)):
        if d.get(a, 0) > 0 and d.get(b, 0) > 0:
            out[f"{a}-{b}"] = math.log(d[b] / d[a]) / math.log(b / a)
    if out:
        out["min"] = min(out.values())
    return out


# -------------------------------------------------------------------------------------------------
# thresholds text (from the YAML, so the report always matches the prereg)
# -------------------------------------------------------------------------------------------------
def threshold_text(pr: str, pre: dict) -> str:
    try:
        a, b, c = pre["stage_a"], pre["stage_b"]["pr4"], pre["stage_c"]
        d, e, f = pre["stage_d"], pre["stage_e"]["pr10"], pre["stage_f"]["pr11"]
        t = {
            "PR1": f"rel. err <= {a['pr1']['rel_err_fp32_at_most']:g} (fp32) or <= {a['pr1']['rel_err_bf16_at_most']:g} "
                   f"(bf16); time ratio <= {a['pr1']['time_ratio_at_most']:g}",
            "PR2": f"rel. err <= {a['pr2']['rel_err_fp32_at_most']:g} (fp32) or <= {a['pr2']['rel_err_bf16_at_most']:g} (bf16)",
            "PR3": f"rel. err <= {a['pr3']['rel_err_at_most']:g} (not a kill)",
            "PR4": f"median completeness >= {b['pass']['median_completeness_traces_at_least']:g} (gap <= "
                   f"{b['pass']['completeness_gap_vs_short_at_most']:g}); faithfulness >= "
                   f"{b['pass']['median_faithfulness_spearman_traces_at_least']:g} (gap <= "
                   f"{b['pass']['faithfulness_gap_vs_short_at_most']:g}); kill: completeness < "
                   f"{b['kill']['median_completeness_below']:g} or Spearman < {b['kill']['faithfulness_spearman_below']:g}",
            "PR5": f"<= {c['pr5']['pass']['remainder_abs_s95_frac_causal_at_most']:g} on >= "
                   f"{c['pr5']['pass']['prompts_at_least']} prompts; kill > "
                   f"{c['pr5']['kill']['remainder_abs_s95_frac_causal_above']:g} everywhere",
            "PR6": f"some m <= T/{c['pr6']['pass']['m_at_most_T_over']} meets fidelity on >= "
                   f"{c['pr6']['pass']['prompts_at_least']} prompts (D2); kill: none at m <= T/{c['pr6']['kill']['m_at_most_T_over']}",
            "PR7": f"ratio <= {d['pr7']['pass']['k_eff_trace_over_short_at_most']:g}; kill > "
                   f"{d['pr7']['kill']['k_eff_trace_over_short_above']:g}",
            "PR8": f"some tau: recall >= {d['pr8']['pass']['mass_recall_at_least']:g} and median size of Omega_s <= "
                   f"{d['pr8']['pass']['median_omega_frac_causal_at_most']:g} of causal set",
            "PR9": f"m_trace in {d['pr9']['m_grid']}",
            "PR10": f"slope >= {e['pass']['local_slope_at_least']:g} on >= {e['pass']['traces_at_least']} traces; "
                    f"S_meas >= {e['pass']['s_meas_at_least']:g}; S_proj >= {e['pass']['s_proj_at_least']:g}; "
                    f"S_oracle_proj >= {e['pass']['s_oracle_proj_at_least']:g}; kill: S_proj < {e['kill']['s_proj_below']:g} everywhere",
            "PR11": f"AUROC - S4 >= {f['pass']['auroc_minus_s4_at_least']:g} and >= {f['pass']['auroc_at_least']:g}; "
                    f">= {f['pass']['nameable_at_least']} of 15 nameable; weak: AUROC - S4 < {f['weak']['auroc_minus_s4_below']:g} "
                    f"or < {f['weak']['nameable_below']} nameable",
        }
        t[PR4_4B] = t["PR4"]
        return t.get(pr, NA)
    except (KeyError, TypeError, ValueError):
        return NA


# -------------------------------------------------------------------------------------------------
# plots
# -------------------------------------------------------------------------------------------------
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def make_plots(stages: dict, plot_dir: str) -> list[tuple[str, str | None, str]]:
    """[(title, relative png path or None, note)]."""
    plt = _plt()
    specs = []

    # 1. s95 histograms
    s95 = {}
    for s, lab in (("C", "short prompts"), ("D", "traces")):
        v = lookup(stages.get(s), S95_KEYS)
        vals = numeric_leaves(v)
        if vals:
            s95[lab] = vals
    specs.append(("s95 histograms (remainder, absolute version)", "s95_hist.png", s95, "s95"))

    # 2. fidelity vs m
    fid = lookup(stages.get("C"), FID_KEYS)
    series = {k: v for k, v in xy_series(fid).items() if re.search(r"D1|D2", k)} if fid is not None else {}
    specs.append(("Fidelity against m (D1 and D2)", "fidelity_vs_m.png", series, "fid"))

    # 3. union growth
    union = {}
    for lab, keys in (("realistic", U_REAL_KEYS), ("oracle", U_ORACLE_KEYS)):
        v = lookup(stages.get("E"), keys)
        for k, pts in (xy_series(v).items() if v is not None else []):
            union[f"{lab}: {k}" if k != "all" else lab] = pts
    specs.append(("Union growth, realistic and oracle (log-log)", "union_growth.png", union, "union"))

    # 4. AUROC
    au = lookup(stages.get("F"), AUROC_KEYS)
    bars = {}
    if isinstance(au, dict):
        for k, v in au.items():
            if _num(v):
                bars[str(k)] = float(v)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if _num(v2):
                        bars[f"{k}/{k2}"] = float(v2)
    specs.append(("Feature-level against baseline AUROC", "auroc.png", bars, "auroc"))

    out = []
    for title, fname, data, kind in specs:
        if not data:
            out.append((title, None, "skipped: data not available in the stage files"))
            continue
        if plt is None:
            out.append((title, None, "skipped: matplotlib is not installed here (it is on the VM)"))
            continue
        os.makedirs(plot_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        note = ""
        if kind == "s95":
            for lab, vals in data.items():
                ax.hist(vals, bins=40, alpha=0.6, label=f"{lab} (n = {len(vals)})")
            ax.set_xlabel("s95 (share of causal set or entries, as stored)")
            ax.set_ylabel("count (columns, or prompts when only per-prompt medians are stored)")
        elif kind == "fid":
            for k, pts in list(data.items())[:24]:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                        linestyle="-" if "D2" in k else "--", label=k)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("m (mixed passes)")
            ax.set_ylabel("fidelity (recall, or share of seeds meeting it)")
        elif kind == "union":
            slopes = []
            for k, pts in data.items():
                pts = [p for p in pts if p[0] > 0 and p[1] > 0]
                if not pts:
                    continue
                sl = local_slopes(pts)
                lab = k + (f" (slope {sl['min']:.2f})" if "min" in sl else "")
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                        linestyle="--" if k.startswith("oracle") else "-", label=lab)
                if "min" in sl:
                    slopes.append(f"{k}: {sl['min']:.3f}")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("n explained outputs")
            ax.set_ylabel("|U(n)| rows")
            note = "local slopes (min of 16-32 and 32-51): " + ("; ".join(slopes) if slopes else NA)
        else:
            ax.bar(range(len(data)), list(data.values()))
            ax.set_xticks(range(len(data)))
            ax.set_xticklabels(list(data.keys()), rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("AUROC (top-quartile effects)")
            ax.axhline(0.5, color="grey", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        if kind != "auroc":
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, fname), dpi=120)
        plt.close(fig)
        out.append((title, f"plots/{fname}", note))
    return out


# -------------------------------------------------------------------------------------------------
# report sections
# -------------------------------------------------------------------------------------------------
def deviations(path: str = PREREG_MD) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return []
    m = re.search(r"##\s*\d*\.?\s*Deviations log\s*\n(.*)", text, re.S)
    if not m:
        return []
    rows = []
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if rows or line.startswith("#"):
                break
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c) or cells[:1] == ["Date"]:
            continue
        if any(cells):
            rows.append(line)
    return rows


def dependencies_section(stages: dict, results_dir: str) -> list[str]:
    deps = lookup(stages.get("F"), DEP_KEYS)
    L = []
    if isinstance(deps, dict):  # {trace: [deps]}
        deps = [dict(d, trace=d.get("trace", t)) for t, ds in deps.items() if isinstance(ds, list)
                for d in ds if isinstance(d, dict)]
    if isinstance(deps, list) and deps and all(isinstance(d, dict) for d in deps):
        L += ["| # | trace | from sentence i | to sentence j | dependency | carrier features (top tokens) | "
              "nameable? (Dawar) |", "|---|---|---|---|---|---|---|"]
        for n, d in enumerate(deps[:15], 1):
            carriers = d.get("carriers") or d.get("carrier_features") or []
            cs = []
            for c in carriers if isinstance(carriers, list) else []:
                if isinstance(c, dict):
                    name = c.get("feature") or c.get("id") or c.get("node") or "?"
                    lay = c.get("layer")
                    toks = c.get("top_tokens") or c.get("top_activating_tokens")
                    cs.append(f"{'L' + str(lay) + ' ' if lay is not None else ''}{name}"
                              + (f" ({fmt(toks, limit=60)})" if toks else ""))
                else:
                    cs.append(fmt(c, limit=60))
            L.append(f"| {n} | {fmt(d.get('trace'))} | {fmt(d.get('i', d.get('source_sentence')))} | "
                     f"{fmt(d.get('j', d.get('target_sentence')))} | "
                     f"{fmt(d.get('D', d.get('dependency', d.get('strength'))))} | "
                     f"{fmt(cs, limit=600) if cs else NA} | {fmt(d.get('nameable')) if 'nameable' in d else ''} |")
        if len(deps) < 15:
            L.append(f"\nOnly {len(deps)} of 15 dependencies are in stage F.")
        return L
    md = os.path.join(results_dir, "carriers_for_labelling.md")
    if os.path.exists(md):
        return [f"The dependencies and carriers are in [carriers_for_labelling.md](carriers_for_labelling.md) "
                "(stage F detail has no dependency list)."]
    return [f"Top dependencies and carriers: {NA} (no stage F list and no carriers_for_labelling.md)."]


def gpu_rows(stages: dict, cap: float):
    rows, total = [], 0.0
    for s in STAGES:
        d = stages.get(s)
        g = d.get("gpu_seconds") if isinstance(d, dict) else None
        if _num(g):
            total += g
        rows.append((s, g if _num(g) else None, STAGE_BUDGET_H.get(s)))
    return rows, total


def run_info(stages: dict) -> dict:
    out = {}
    for s in STAGES:
        d = stages.get(s)
        if isinstance(d, dict):
            run = d.get("run") if isinstance(d.get("run"), dict) else {}
            out[s] = {"run": run, "git": run.get("git"), "gpu": run.get("gpu"),
                      "wall_seconds": run.get("wall_seconds"), "utc": run.get("utc")}
    return out


def build_report(stages, v, pre, tiny, plots, results_dir, notes) -> str:
    st = v["premises"]
    cap = float(pre.get("gpu_cap_h100_hours", 6))
    rows, total = gpu_rows(stages, cap)
    passed = [p for p in PREMISES if st[p] == "pass"]
    bad = [f"{p} ({st[p]})" for p in PREMISES if st[p] in ("fail", "kill", "weak")]
    other = [f"{p} ({st[p]})" for p in PREMISES if p not in passed and st[p] not in ("fail", "kill", "weak")]
    head = "TINY: numbers meaningless. " if tiny else ""
    cand = f" Candidates: {' or '.join(v['candidates'])}." if v["candidates"] else ""
    L = [
        f"{head}Gate 0 verdict: {v['verdict']}. {v['label']}.{cand}",
        f"Passed: {', '.join(passed) or 'none'}. Failed, killed or weak: {', '.join(bad) or 'none'}. "
        f"Pending, not run or missing: {', '.join(other) or 'none'}.",
        f"Why: {'; '.join(v['reasons']) or NA}. GPU time used: {total / 3600:.2f} of {cap:g} H100-hours.",
        "",
        "# Gate 0 premise report" + (" (TINY: numbers meaningless)" if tiny else ""),
        "",
    ]
    if tiny:
        L += ["TINY: this run uses the tiny random model and synthetic inputs. It tests the code path only; "
              "no number here says anything about real models.", ""]
    L += ["## Verdict", "", f"- Verdict: **{v['verdict']}** ({v['label']}).",
          f"- Applied from configs/prereg_gate0.yaml, order: {', '.join(pre.get('verdicts', {}).get('order', []))}."]
    for r in v["reasons"]:
        L.append(f"- Reason: {r}")
    if v["missing"]:
        L.append(f"- Missing: {'; '.join(v['missing'])}. No GO-BIG is issued while anything is missing.")
    if v["pending"]:
        L.append(f"- Pending: {', '.join(v['pending'])}.")
    if v["stop_reasons"]:
        L.append(f"- Stop rules triggered: {'; '.join(v['stop_reasons'])}.")
    for n in notes:
        L.append(f"- Note: {n}")
    L += ["", "## Premises", "",
          "| Premise | What was measured | Threshold | Value | Result |", "|---|---|---|---|---|"]
    for pr in list(PREMISES[:4]) + [PR4_4B] + list(PREMISES[4:]):
        stage = stages.get("B_qwen3_4b" if pr == PR4_4B else PREMISE_STAGE[pr])
        e = premise_entry(stage, "PR4" if pr == PR4_4B else pr)
        if pr == PR4_4B and e is None:
            continue
        thr = threshold_text(pr, pre)
        if e is not None and e.get("thresholds"):
            thr = fmt(e["thresholds"])
        val = fmt(e.get("values")) if e is not None and e.get("values") else NA
        if e is not None and e.get("notes"):
            val += f" (note: {fmt(e['notes'], limit=160)})"
        L.append(f"| {pr} | {MEASURED[pr]} | {thr} | {val} | {st[pr]} |")
    L += ["", "## Plots", ""]
    for title, path, note in plots:
        if path:
            L.append(f"**{title}**" + (f" ({note})" if note else ""))
            L += ["", f"![{title}]({path})", ""]
        else:
            L.append(f"- {title}: {note}.")
    L += ["", "## Cost table (PR10)", "", "| Quantity | Value |", "|---|---|"]
    for lab, keys in COST_ROWS:
        val = lookup(stages.get("E"), keys)
        if val is None:
            val = lookup(stages.get("D"), keys)
        if val is None and lab == "local slope":
            u = lookup(stages.get("E"), U_REAL_KEYS)
            sl = {k: local_slopes(p).get("min") for k, p in (xy_series(u).items() if u is not None else [])}
            sl = {k: x for k, x in sl.items() if x is not None}
            if sl:
                L.append(f"| local slope (computed here from U_trunc, min of 16-32 and 32-51) | {fmt(sl)} |")
                continue
        L.append(f"| {lab} | {fmt(val)} |")
    eps = lookup(stages.get("C"), ("epsilon_star", "eps_star"))
    L += [f"| epsilon* (PR6 calibration) | {fmt(eps)} |", ""]
    L += ["## Top 15 dependencies and carrier features (for Dawar to label)", ""]
    L += dependencies_section(stages, results_dir)
    dev = deviations()
    L += ["", "## Deviations from the prereg", ""]
    if dev:
        L += ["| Date | What | Why |", "|---|---|---|"] + dev
    else:
        L.append("None recorded in the PREREG_GATE0.md deviations log.")
    for s in STAGES:
        d = stages.get(s)
        extra = find(d.get("detail"), ("deviations",)) if isinstance(d, dict) else None
        if extra:
            L.append(f"- Stage {s} reports: {fmt(extra, limit=400)}")
    L += ["", "## GPU time per stage", "", "| Stage | GPU hours | Stage budget (H100-hours) |", "|---|---|---|"]
    for s, g, b in rows:
        L.append(f"| {s} | {fmt(g / 3600) if g is not None else NA} | {fmt(b)} |")
    L += [f"| total | {total / 3600:.3f} | cap {cap:g} |", ""]
    if total / 3600 > cap:
        L += [f"Over the {cap:g} H100-hour cap.", ""]
    L += ["## Provenance", "", "| Stage | git hash | dirty | GPU | wall seconds | UTC |", "|---|---|---|---|---|---|"]
    for s, r in run_info(stages).items():
        g = r["git"] or {}
        L.append(f"| {s} | {fmt(str(g.get('hash', NA))[:12] if g else None)} | {fmt(g.get('dirty'))} | "
                 f"{fmt(r['gpu'])} | {fmt(r['wall_seconds'])} | {fmt(r['utc'])} |")
    L += ["", "## What surprised us", "", "(To be filled in by hand after reading the results.)", "",
          "Stop: waiting for Dawar's decision.", ""]
    return "\n".join(L)


# -------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="results/tiny/gate0; numbers meaningless")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    rd = args.results_dir or os.path.join(ROOT, "results", "tiny" if args.tiny else "", "gate0")
    rd = os.path.normpath(rd)
    with open(PREREG, encoding="utf-8") as f:
        pre = yaml.safe_load(f)
    stages, notes = {}, []
    for s in STAGES:
        d, note = load_stage(os.path.join(rd, STAGE_FILES[s]))
        stages[s] = d
        if note:
            notes.append(note)
    v = premise_verdict(stages, pre)
    plots = make_plots(stages, os.path.join(rd, "plots"))
    report = build_report(stages, v, pre, args.tiny, plots, rd, notes)

    from cct.utils.io import atomic_write_json, atomic_write_text
    from cct.utils.logging import run_record

    rows, total = gpu_rows(stages, float(pre.get("gpu_cap_h100_hours", 6)))
    out = {
        "tiny": args.tiny,
        "verdict": v,
        "premises": {pr: premise_entry(stages.get(PREMISE_STAGE[pr]), pr) for pr in PREMISES},
        "premise_PR4_qwen3_4b": premise_entry(stages.get("B_qwen3_4b"), "PR4"),
        "gpu_seconds": {s: g for s, g, _ in rows},
        "gpu_seconds_total": total,
        "gpu_hours_total": total / 3600,
        "gpu_cap_h100_hours": pre.get("gpu_cap_h100_hours"),
        "stop": {s: {"stop": d.get("stop"), "stop_reason": d.get("stop_reason")}
                 for s, d in stages.items() if isinstance(d, dict)},
        "runs": {s: r["run"] for s, r in run_info(stages).items()},
        "git_hashes": {s: (r["git"] or {}).get("hash") for s, r in run_info(stages).items()},
        "plots": [{"title": t, "path": p, "note": n} for t, p, n in plots],
        "notes": notes,
        "report_run": run_record({"results_dir": rd, "tiny": args.tiny, "prereg": "configs/prereg_gate0.yaml"},
                                 None, wall_seconds=time.time() - t0),
    }
    atomic_write_json(os.path.join(rd, "gate0.json"), out)
    atomic_write_text(os.path.join(rd, "PREMISE_REPORT.md"), report)
    print(f"verdict: {v['verdict']} ({v['label']})")
    print(f"wrote {os.path.join(rd, 'PREMISE_REPORT.md')} and gate0.json")
    print("Stop: waiting for Dawar's decision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
