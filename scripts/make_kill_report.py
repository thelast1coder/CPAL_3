"""Kill report (PHASE1 Sections 9 and 11): applies the prereg decision table to k0..k3.json and writes
KILL_REPORT.md and kill_test.json. The generated report is then edited by hand for clarity (real run)."""
from __future__ import annotations

import os

from cct import scriptutil as su
from cct.killtest import overall_decision
from cct.utils.io import atomic_write_json, atomic_write_text, read_json
from cct.utils.logging import git_info, run_record


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def k2_plot(set_name: str, curves: dict, plot_dir: str) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, cur in curves.items():
        xs = [v["median_cost"] for v in cur.values()]
        ys = [v["share"] for v in cur.values()]
        ax.plot(xs, ys, marker="o", label=name)
    ax.axhline(0.75, ls="--", lw=0.8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("median cost (row-equivalents)")
    ax.set_ylabel("share of prompts meeting fidelity")
    ax.legend(fontsize=7)
    p = os.path.join(plot_dir, f"k2_curves_{set_name}.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main() -> None:
    args = su.parser("Kill report").parse_args()
    rd = su.results_dir(args)
    pre = su.prereg()
    k0 = read_json(os.path.join(rd, "k0.json"))
    k1 = read_json(os.path.join(rd, "k1.json"))
    k2 = read_json(os.path.join(rd, "k2.json"))
    k3 = read_json(os.path.join(rd, "k3.json"))
    sets = list(k2["sets"])
    d1 = {s: k1["sets"][s]["decision"] for s in sets}
    d2 = {s: k2["sets"][s]["decision_final"] for s in sets}
    d3 = {s: k3["sets"][s]["decision"] for s in sets if s in k3["sets"]}
    final = overall_decision(d1, d2, d3)
    plot_dir = os.path.join(su.ROOT, "plots", "tiny") if args.tiny else os.path.join(su.ROOT, "plots")
    L = []
    if args.tiny:
        L += ["# TINY RUN, NUMBERS ARE MEANINGLESS", "",
              "This report exercises the whole kill-test pipeline on a tiny random transformer. It tests the",
              "code and the decision logic only. No number below says anything about real attribution graphs.", ""]
    L += ["# Kill test report", "", "## Summary", "",
          f"Verdict: **{final['verdict']}**" + (f" (on {final['set']})" if final["set"] else "") + ".",
          "K1 asks whether attribution matrices are compressible, K2 whether a few mixed passes rebuild the graph "
          "more cheaply than exact enumeration and truncation, K3 whether this holds with real passes and clocks.",
          "Per-set verdicts are in the decision table below; thresholds are those of PREREG_KILL_TEST.md.", "",
          "## Setup", "",
          f"- git: {git_info()['hash']} (dirty: {git_info()['dirty']})",
          f"- prereg version: {pre['prereg_version']}; sets: {', '.join(sets)}; seeds: {pre['seeds']}",
          f"- packages: {k0.get('packages')}", f"- device: {k0.get('gpu')}", "",
          "## K0 sanity", "", "| set | check | pass | value |", "|---|---|---|---|"]
    for s, v in k0["sets"].items():
        for name, c in v["checks"].items():
            val = c["value"]
            if isinstance(val, dict):
                val = ", ".join(f"{k}={fmt(x)}" for k, x in val.items() if not isinstance(x, list))
            L.append(f"| {s} | {name} | {c['pass']} | {fmt(val) if not isinstance(val, str) else val} |")
    L += ["", "## K1 structure", "", "| set | median col s95 / T | median row s95 / N | median hub energy share | share GO | share NO-GO | verdict |",
          "|---|---|---|---|---|---|---|"]
    for s in sets:
        x = k1["sets"][s]
        L.append(f"| {s} | {fmt(x['median_col_s95_frac'])} | {fmt(x['median_row_s95_frac'])} | "
                 f"{fmt(x['median_hub_energy_share'])} | {fmt(x['decision']['share_go'])} | "
                 f"{fmt(x['decision']['share_nogo'])} | {x['decision']['verdict']} |")
    L += ["", "## K2 simulation", ""]
    for s in sets:
        x = k2["sets"][s]
        L += [f"### {s}", "", f"Chosen on validation: `{x['tuning']['chosen']}`", "",
              "| method | c75 budget | c75 cost | share at c75 |", "|---|---|---|---|"]
        for meth, c in x["c75"].items():
            L.append(f"| {meth} | {fmt(c['budget']) if c else 'never'} | {fmt(c['cost']) if c else 'n/a'} | "
                     f"{fmt(c['share']) if c else 'n/a'} |")
        dd = x["decision_final"]
        L += ["", f"Median T_D = {fmt(x['T_D_median'])}; S_exact = {fmt(dd.get('s_exact'))}; "
                  f"S_B1 = {fmt(dd.get('s_b1'))}; verdict {dd['verdict']}.", ""]
        L += ["| method | budget | share meeting fidelity | median cost |", "|---|---|---|---|"]
        for meth, cur in x["curves"].items():
            for b, v in cur.items():
                L.append(f"| {meth} | {b} | {fmt(v['share'])} | {fmt(v['median_cost'])} |")
        p = k2_plot(s, x["curves"], plot_dir)
        if p:
            L += ["", f"Plot: `{os.path.relpath(p, su.ROOT)}`"]
        if s in k2.get("regime_F", {}):
            f = k2["regime_F"][s]
            L += ["", f"Regime F (secondary): S_exact = {fmt(f['decision'].get('s_exact'))}, "
                      f"S_B1 = {fmt(f['decision'].get('s_b1'))}, {f['decision']['verdict']}."]
        L.append("")
    L += ["## K3 real passes", "", "| set | regime | m | prompts meeting | median speed-up | max rel. Frobenius real vs sim | verdict |",
          "|---|---|---|---|---|---|---|"]
    for s, x in k3["sets"].items():
        d = x["decision"]
        L.append(f"| {s} | {x['regime']} | {x['m']} | {d['n_meet']} | {fmt(d['speedup'])} | {fmt(d['rel_frob'])} | {d['verdict']} |")
    L += ["", "## Decision table", "", "| set | K1 | K2 | K3 |", "|---|---|---|---|"]
    for s in sets:
        L.append(f"| {s} | {d1[s]['verdict']} | {d2[s]['verdict']} | {(d3.get(s) or {}).get('verdict', 'not run')} |")
    L += ["", f"Overall: **{final['verdict']}**.", "", "## What the numbers mean", "",
          "(To be written by hand after the real run: plain-language reading, surprises, next steps.)" if not args.tiny
          else "Nothing: this is a tiny random model. The run shows that every stage executes and the decision "
               "table is applied.", ""]
    atomic_write_text(os.path.join(rd, "KILL_REPORT.md"), "\n".join(L))
    atomic_write_json(os.path.join(rd, "kill_test.json"),
                      {"verdict": final, "k1": d1, "k2": d2, "k3": d3, "tiny": args.tiny,
                       "run_record": run_record({"prereg": pre}, None)})
    print(f"verdict: {final}")


if __name__ == "__main__":
    main()
