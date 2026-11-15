"""Gate 0 decision logic (PROJECT_V2.md Section 7, PREREG_GATE0.md). Pure functions over the JSON results
of the Gate 0 scripts; every threshold comes from configs/prereg_gate0.yaml.

Inputs (None when the part did not run):
  g0a      results/gate0/g0a.json          {"run", "sets": {set: {"timing", "prompts": {pid: {...}}}}}
  union    results/gate0/g0b_union.json    {"traces": {tid: {...}}, "beta", "c_median", "n_outputs_without_c"}
  columns  results/gate0/g0b_columns.json  {"T", "N", "k_eff_trace", "hub_fraction", "screen_recall", ...}

Conditions carry "pass": True, False, or None (not evaluable from what exists). For the two NO-GO
triggers the condition is phrased positively ("not_nogo_*"): pass False means that NO-GO trigger fired.
"""
from __future__ import annotations

import math
import statistics

GO_BIG_CONDITIONS = ("compressible", "beta", "k_eff_ratio", "screening", "s_proj")
STRUCTURE_ONLY_FAILURES = {"beta", "s_proj"}


# -------------------------------------------------------------------------------------------------
# small helpers
# -------------------------------------------------------------------------------------------------
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _median(xs) -> float | None:
    xs = [float(x) for x in xs if _finite(x)]
    return statistics.median(xs) if xs else None


def _cond(value, threshold, ok) -> dict:
    return {"value": value, "threshold": threshold, "pass": ok}


def k07_threshold(dtype: str | None, prereg: dict) -> float:
    """Relative Frobenius tolerance for K0.7 by dtype (bfloat16 and float16 use the bf16 tolerance,
    everything else the float32 one)."""
    k = prereg["k07"]
    d = str(dtype or "float32").lower()
    if d in ("bfloat16", "bf16", "float16", "fp16", "half", "torch.bfloat16", "torch.float16"):
        return float(k["bf16_rel_err_at_most"])
    return float(k["fp32_rel_err_at_most"])


def plt_set(set_names, prereg: dict) -> str | None:
    suffix = prereg["g0a"].get("plt_set_suffix", "plt")
    for s in set_names:
        if s.endswith(suffix):
            return s
    return None


# -------------------------------------------------------------------------------------------------
# G0-A
# -------------------------------------------------------------------------------------------------
def fracs_by_set(g0a: dict) -> dict[str, dict[str, float]]:
    return {s: {pid: p.get("s95_rem_frac_T") for pid, p in v.get("prompts", {}).items()}
            for s, v in g0a.get("sets", {}).items()}


def g0a_compressible(per_prompt_fracs_by_set: dict[str, dict[str, float]], prereg: dict) -> dict:
    """Compressibility: s95_rem_frac_T <= 0.02 on at least 2 prompts, for at least 1 transcoder set.
    Input: {set: {prompt_id: s95_rem_frac_T}} (a list of fractions per set is also accepted)."""
    c = prereg["g0a"]["compressible"]
    by_set = {}
    for s, fr in per_prompt_fracs_by_set.items():
        vals = list(fr.values()) if isinstance(fr, dict) else list(fr)
        n_ok = sum(1 for f in vals if _finite(f) and f <= c["s95_rem_frac_T_at_most"])
        by_set[s] = {"n_ok": n_ok, "n": len(vals), "pass": n_ok >= c["prompts_required_at_least"]}
    passing = [s for s, v in by_set.items() if v["pass"]]
    return {"pass": len(passing) >= c["sets_required_at_least"], "by_set": by_set, "sets_passing": passing}


def g0a_nogo_s95(per_prompt_fracs_by_set: dict[str, dict[str, float]], prereg: dict) -> bool:
    """NO-GO trigger: s95_rem_frac_T above 0.10 on every prompt of every set present."""
    thr = prereg["verdicts"]["nogo"]["s95_rem_frac_T_above"]
    vals = [f for fr in per_prompt_fracs_by_set.values()
            for f in (fr.values() if isinstance(fr, dict) else fr)]
    return bool(vals) and all(_finite(f) and f > thr for f in vals)


def k_eff_short(set_result: dict) -> float | None:
    """Median over prompts of the energy-weighted median remainder s95."""
    return _median(p.get("s95_rem_ewm") for p in set_result.get("prompts", {}).values())


def c_constant(m_min: float, k_eff: float, T: float) -> float:
    """Practical sensing constant C = m_min / (k_eff ln(T / k_eff)); NaN when undefined (k_eff <= 0 or
    k_eff >= T)."""
    if not (_finite(m_min) and _finite(k_eff) and _finite(T)) or k_eff <= 0 or T <= k_eff:
        return float("nan")
    return float(m_min) / (float(k_eff) * math.log(float(T) / float(k_eff)))


def c_short(set_result: dict, prereg: dict) -> dict:
    """C per prompt and its median (C_short). A prompt with no working m uses m_min_fallback and is
    flagged."""
    fb = prereg["s_proj"]["m_min_fallback"]
    per, flagged = {}, []
    for pid, p in set_result.get("prompts", {}).items():
        m = p.get("m_min")
        if m is None:
            m = fb
            flagged.append(pid)
        per[pid] = c_constant(m, p.get("s95_rem_ewm"), p.get("T"))
    return {"C_short": _median(per.values()), "C_by_prompt": per, "flagged_no_m": flagged}


# -------------------------------------------------------------------------------------------------
# G0-B
# -------------------------------------------------------------------------------------------------
def s_proj(c: float, L: float, beta: float, C: float, k_eff: float, T: float, kappa: float, H: float,
           m_prime: float) -> float:
    """S_proj(L) = c L^beta / (C k_eff ln(T / k_eff) + kappa H + m')."""
    denom = C * k_eff * math.log(T / k_eff) + kappa * H + m_prime
    return c * L ** beta / denom


def s_proj_terms(c, L, beta, C, k_eff, T, kappa, H, m_prime) -> dict:
    terms = {"c": c, "L": L, "beta": beta, "C_short": C, "k_eff_trace": k_eff, "T_trace": T,
             "kappa": kappa, "H_est": H, "m_prime": m_prime}
    if not all(_finite(v) for v in terms.values()) or k_eff <= 0 or T <= k_eff:
        return {**terms, "trunc_cost": None, "sensing_cost": None, "hub_cost": None, "s_proj": None}
    return {**terms, "trunc_cost": c * L ** beta, "sensing_cost": C * k_eff * math.log(T / k_eff),
            "hub_cost": kappa * H, "s_proj": s_proj(c, L, beta, C, k_eff, T, kappa, H, m_prime)}


def screening_pass(recall_by_tau: dict, prereg: dict) -> dict:
    """Screening passes if energy recall >= 0.99 at some preregistered tau. Input: {tau (str or float):
    {"recall", "omega_frac_T"}} or {tau: recall}. Reports the largest passing tau (smallest Omega_s)."""
    sc = prereg["g0b"]["screening"]
    rows = {}
    for k, v in (recall_by_tau or {}).items():
        tau = float(k)
        if not any(math.isclose(tau, float(t), rel_tol=1e-9) for t in sc["taus"]):
            continue
        rec = v.get("recall") if isinstance(v, dict) else v
        om = v.get("omega_frac_T") if isinstance(v, dict) else None
        rows[tau] = (rec, om)
    passing = sorted((t for t, (r, _) in rows.items() if _finite(r) and r >= sc["energy_recall_at_least"]),
                     reverse=True)
    best = max((r for r, _ in rows.values() if _finite(r)), default=None)
    if not rows:
        return {"pass": None, "tau": None, "recall": None, "omega_frac_T": None, "best_recall": None}
    if passing:
        t = passing[0]
        return {"pass": True, "tau": t, "recall": rows[t][0], "omega_frac_T": rows[t][1], "best_recall": best}
    return {"pass": False, "tau": None, "recall": None, "omega_frac_T": None, "best_recall": best}


def union_beta(union: dict) -> float | None:
    b = union.get("beta")
    if _finite(b):
        return float(b)
    bs = [t.get("beta") for t in union.get("traces", {}).values() if _finite(t.get("beta"))]
    return sum(bs) / len(bs) if bs else None


def union_L(union: dict) -> float | None:
    return _median(t.get("n_generated") for t in union.get("traces", {}).values())


# -------------------------------------------------------------------------------------------------
# verdict
# -------------------------------------------------------------------------------------------------
def gate0_verdict(g0a: dict | None, union: dict | None, columns: dict | None, prereg: dict) -> dict:
    """Apply the PREREG_GATE0.md decision table. Returns {"verdict", "partial", "missing", "conditions",
    "s_proj", "derived"}. verdict is GO-BIG, GO-STRUCTURE, NO-GO, UNDECIDED, or STOPPED (K0.7 failed:
    no verdict)."""
    pa, pb = prereg["g0a"], prereg["g0b"]
    missing: list[str] = []
    conds: dict[str, dict] = {}
    derived: dict = {}

    # ---- completeness ------------------------------------------------------------------------
    if g0a is None:
        missing.append("G0-A results (g0a.json)")
    else:
        sets = g0a.get("sets", {})
        if len(sets) < len(pa["transcoder_sets"]):
            missing.append(f"G0-A: {len(sets)} of {len(pa['transcoder_sets'])} transcoder sets present")
        for s, v in sets.items():
            n = len(v.get("prompts", {}))
            if n < len(pa["prompt_ids"]):
                missing.append(f"G0-A {s}: {n} of {len(pa['prompt_ids'])} prompts present")
    if union is None:
        missing.append("G0-B union growth (g0b_union.json)")
    elif len(union.get("traces", {})) < pb["n_traces"]:
        missing.append(f"G0-B union: {len(union.get('traces', {}))} of {pb['n_traces']} traces present")
    if columns is None:
        missing.append("G0-B sampled columns (g0b_columns.json)")

    # ---- K0.7 --------------------------------------------------------------------------------
    k07 = {}
    if g0a is not None:
        for s, v in g0a.get("sets", {}).items():
            for pid, p in v.get("prompts", {}).items():
                if p.get("k07_rel_err") is not None:
                    k07[f"{s}/{pid}"] = (p["k07_rel_err"], k07_threshold(p.get("dtype"), prereg))
    if columns is not None and columns.get("k07_rel_err") is not None:
        k07["g0b_trace"] = (columns["k07_rel_err"], k07_threshold(columns.get("dtype"), prereg))
    if k07:
        fails = [k for k, (e, t) in k07.items() if not (_finite(e) and e <= t)]
        conds["k07"] = _cond({k: e for k, (e, _) in k07.items()}, {k: t for k, (_, t) in k07.items()},
                             not fails)
        derived["k07_failures"] = fails
    else:
        conds["k07"] = _cond(None, None, None)

    # ---- G0-A --------------------------------------------------------------------------------
    pre_c = pa["compressible"]
    nogo = prereg["verdicts"]["nogo"]
    k_short_plt = None
    C_short = None
    if g0a is not None and g0a.get("sets"):
        fr = fracs_by_set(g0a)
        comp = g0a_compressible(fr, prereg)
        conds["compressible"] = _cond(
            {s: f"{v['n_ok']}/{v['n']}" for s, v in comp["by_set"].items()},
            f"s95_rem_frac_T <= {pre_c['s95_rem_frac_T_at_most']} on >= {pre_c['prompts_required_at_least']} "
            f"prompts for >= {pre_c['sets_required_at_least']} set", comp["pass"])
        conds["not_nogo_s95"] = _cond(fr, f"NO-GO if every prompt of every set > {nogo['s95_rem_frac_T_above']}",
                                      not g0a_nogo_s95(fr, prereg))
        derived["k_eff_short"] = {s: k_eff_short(v) for s, v in g0a["sets"].items()}
        plt = plt_set(g0a["sets"], prereg)
        derived["plt_set"] = plt
        if plt is None:
            missing.append("G0-A: no PLT set (name ending with "
                           f"'{pa.get('plt_set_suffix', 'plt')}')")
        else:
            k_short_plt = derived["k_eff_short"][plt]
            cs = c_short(g0a["sets"][plt], prereg)
            derived["C"] = cs
            C_short = cs["C_short"]
    else:
        conds["compressible"] = _cond(None, None, None)
        conds["not_nogo_s95"] = _cond(None, None, None)

    # ---- G0-B union ----------------------------------------------------------------------------
    beta = union_beta(union) if union is not None else None
    L = union_L(union) if union is not None else None
    c_med = union.get("c_median") if union is not None else None
    derived.update({"beta": beta, "L": L, "c_median": c_med})
    conds["beta"] = _cond(beta, f">= {pb['beta_at_least']}",
                          None if beta is None else beta >= pb["beta_at_least"])

    # ---- G0-B columns --------------------------------------------------------------------------
    k_tr = columns.get("k_eff_trace") if columns is not None else None
    ratio = (k_tr / k_short_plt) if _finite(k_tr) and _finite(k_short_plt) and k_short_plt > 0 else None
    derived["k_eff_ratio"] = ratio
    conds["k_eff_ratio"] = _cond(ratio, f"<= {pb['k_eff_trace_over_k_eff_short_plt_at_most']}",
                                 None if ratio is None else ratio <= pb["k_eff_trace_over_k_eff_short_plt_at_most"])
    conds["not_nogo_k_eff"] = _cond(ratio, f"NO-GO if > {nogo['k_eff_trace_over_k_eff_short_plt_above']}",
                                    None if ratio is None else not ratio > nogo["k_eff_trace_over_k_eff_short_plt_above"])
    scr = screening_pass(columns.get("screen_recall"), prereg) if columns is not None else None
    derived["screening"] = scr
    conds["screening"] = _cond(None if scr is None else scr["best_recall"],
                               f">= {pb['screening']['energy_recall_at_least']} at some tau in {pb['screening']['taus']}",
                               None if scr is None else scr["pass"])

    # ---- S_proj --------------------------------------------------------------------------------
    sp = None
    if union is not None and columns is not None:
        H = (columns.get("hub_fraction") or 0.0) * columns.get("N", 0) if _finite(columns.get("hub_fraction")) else None
        terms = s_proj_terms(c_med, L, beta, C_short, k_tr, columns.get("T"),
                             (columns.get("timing") or {}).get("kappa"), H, prereg["s_proj"]["m_prime"])
        derived["s_proj_terms"] = terms
        sp = terms["s_proj"]
    conds["s_proj"] = _cond(sp, f">= {prereg['s_proj']['at_least']}",
                            None if sp is None else sp >= prereg["s_proj"]["at_least"])

    for name in GO_BIG_CONDITIONS:
        if conds[name]["pass"] is None:
            missing.append(f"condition '{name}' not evaluable")
    partial = bool(missing)

    # ---- decision table ------------------------------------------------------------------------
    failed = [n for n in GO_BIG_CONDITIONS if conds[n]["pass"] is False]
    if conds["k07"]["pass"] is False:
        verdict = "STOPPED"
    elif conds["not_nogo_s95"]["pass"] is False or conds["not_nogo_k_eff"]["pass"] is False:
        verdict = "NO-GO"
    elif not failed and not partial:
        verdict = "GO-BIG"
    elif conds["compressible"]["pass"] is True and failed and set(failed) <= STRUCTURE_ONLY_FAILURES:
        verdict = "GO-STRUCTURE"
    else:
        verdict = "UNDECIDED"
    derived["failed_conditions"] = failed
    return {"verdict": verdict, "partial": partial, "missing": missing, "conditions": conds, "s_proj": sp,
            "derived": derived}


def verdict_label(v: dict) -> str:
    return v["verdict"] + (" (PARTIAL)" if v["partial"] else "")
