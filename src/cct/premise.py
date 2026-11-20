"""Gate 0 premise verdict (GATE0_PREMISES.md Section 4, PREREG_GATE0.md, configs/prereg_gate0.yaml).

Pure functions over the stage result files written by scripts/premise_A.py to premise_F.py. Shared
contract of a stage file:
  {"run": {...}, "stage": "A", "premises": {"PR1": {"status", "values", "thresholds", "notes"}, ...},
   "stop": bool, "stop_reason": str | None, "gpu_seconds": float, "detail": {...}}

Statuses: "pass", "fail", "kill", "weak" (PR11 only), "pending" (PR11 only: the AUROC part passes but
Dawar's carrier labels are missing), "not_run". A premise whose stage file or entry is absent is
"missing". A kill is also a failure.

Resolutions of points that Section 4 leaves open (also in the module tests and the report):
- "PR4" means PR4 on Qwen3-1.7B (stage B). The Qwen3-4B result (stage B_qwen3_4b) is used only for
  SWITCH-MODEL and to decide whether a 1.7B PR4 kill is final.
- "PR4 kill" in the NO-GO rule means: kill on Qwen3-1.7B and Qwen3-4B did not pass (killed, failed or
  not run). If Qwen3-4B passes, SWITCH-MODEL applies instead.
- GO-METHOD ("everything passes except PR11") needs PR11 measured and not passed (fail, weak or
  kill). PR11 "pending" with everything else passing gives PENDING-LABELS (GO-BIG or GO-METHOD,
  pending Dawar's carrier labels). PR11 missing or not run gives no GO-METHOD.
- GO-STRUCTURE counts a kill of PR7, PR9 or PR10 as a failure.
- When no table row applies and some stage stopped by its stop rule, the verdict is STOPPED-EARLY
  with the stop reasons; otherwise UNDECIDED.
- GO-BIG needs every required premise present and passing, so missing stages never give GO-BIG.
"""
from __future__ import annotations

STAGES = ("A", "B", "B_qwen3_4b", "C", "D", "E", "F")
STAGE_FILES = {s: f"stage_{s}.json" for s in STAGES}
REQUIRED_STAGES = ("A", "B", "C", "D", "E", "F")
PREMISE_STAGE = {"PR1": "A", "PR2": "A", "PR3": "A", "PR4": "B", "PR5": "C", "PR6": "C",
                 "PR7": "D", "PR8": "D", "PR9": "D", "PR10": "E", "PR11": "F"}
PREMISES = tuple(PREMISE_STAGE)
PR4_4B = "PR4_qwen3_4b"
STATUSES = ("pass", "fail", "kill", "weak", "pending", "not_run")
FAILED = ("fail", "kill", "weak")

GO_BIG_REQUIRED = ("PR1", "PR2", "PR4", "PR5", "PR6", "PR7", "PR8", "PR9", "PR10", "PR11")
GO_METHOD_REQUIRED = tuple(p for p in GO_BIG_REQUIRED if p != "PR11")

LABELS = {
    "GO-BIG": "The thesis stands. Start the method and strengthening phase (PROJECT_V2.md Sections 5 and 8)",
    "GO-METHOD": "Paper: structure + whole-trace method; science narrowed to descriptive. Dawar decides scope",
    "GO-STRUCTURE": "Paper: structure of attribution matrices + uncapped full graphs + what the default cap "
                    "hides. Dawar decides venue",
    "SWITCH-MODEL": "Rerun Stages D to F on Qwen3-4B (about 3 more hours), then decide",
    "NO-GO": "Stop. One-page post-mortem on what attribution matrices actually look like",
    "PENDING-LABELS": "GO-BIG or GO-METHOD (pending Dawar's carrier labels)",
    "STOPPED-EARLY": "A stage stopped by its stop rule and no row of the decision table applies; Dawar decides",
    "UNDECIDED": "Not covered by the decision table: report written, Dawar decides",
}


def premise_entry(stage: dict | None, pr: str) -> dict | None:
    if not isinstance(stage, dict):
        return None
    prem = stage.get("premises")
    if not isinstance(prem, dict):
        return None
    e = prem.get(pr)
    return e if isinstance(e, dict) else None


def premise_status(stage: dict | None, pr: str) -> str:
    """Status of one premise in one stage file: one of STATUSES, "missing", or "unknown:<raw>"."""
    e = premise_entry(stage, pr)
    if e is None:
        return "missing"
    s = str(e.get("status", "")).strip().lower()
    return s if s in STATUSES else f"unknown:{e.get('status')}"


def collect_statuses(stages: dict) -> dict[str, str]:
    st = {pr: premise_status(stages.get(PREMISE_STAGE[pr]), pr) for pr in PREMISES}
    st[PR4_4B] = premise_status(stages.get("B_qwen3_4b"), "PR4")
    return st


def _stop_reasons(stages: dict) -> list[str]:
    out = []
    for s in STAGES:
        d = stages.get(s)
        if isinstance(d, dict) and d.get("stop"):
            out.append(f"stage {s}: {d.get('stop_reason') or 'stop rule triggered (no reason given)'}")
    return out


def _all_pass(st: dict, prs) -> bool:
    return all(st[p] == "pass" for p in prs)


def premise_verdict(stages: dict, prereg: dict) -> dict:
    """Apply GATE0_PREMISES.md Section 4 in the order of prereg["verdicts"]["order"].

    stages: {"A", "B", "B_qwen3_4b", "C", "D", "E", "F"} -> stage dict or None (absent keys are None).
    Returns {"verdict", "label", "premises": {PRk: status, "PR4_qwen3_4b": status}, "pending",
    "missing", "reasons", "candidates", "stop_reasons"}.
    """
    stages = {s: (stages.get(s) if isinstance(stages.get(s), dict) else None) for s in STAGES}
    st = collect_statuses(stages)
    missing = [f"stage {s} ({STAGE_FILES[s]})" for s in REQUIRED_STAGES if stages[s] is None]
    for pr in PREMISES:
        if st[pr] == "missing" and stages[PREMISE_STAGE[pr]] is not None:
            missing.append(f"{pr} (absent from {STAGE_FILES[PREMISE_STAGE[pr]]})")
    pr4_17b_failed = st["PR4"] in ("fail", "kill")
    if pr4_17b_failed and stages["B_qwen3_4b"] is None:
        missing.append(f"stage B_qwen3_4b ({STAGE_FILES['B_qwen3_4b']}), needed after PR4 {st['PR4']} on Qwen3-1.7B")
    pending = [pr for pr in PREMISES if st[pr] == "pending"]
    if st[PR4_4B] == "pending":
        pending.append(PR4_4B)
    unknown = [f"{pr} has unrecognised status '{s.split(':', 1)[1]}' (treated as not passing)"
               for pr, s in st.items() if s.startswith("unknown:")]
    stop_reasons = _stop_reasons(stages)
    pr4_kill_final = st["PR4"] == "kill" and st[PR4_4B] != "pass"
    # stop rules implied by statuses, in case a stage file did not set its stop flag
    implied = {"A": ["PR1", "PR2"], "B": ["PR4"] if pr4_kill_final and st[PR4_4B] == "kill" else [],
               "C": ["PR5", "PR6"], "D": ["PR7", "PR9"]}
    for s, prs in implied.items():
        kills = [p for p in prs if st[p] == "kill"]
        if kills and not (stages[s] or {}).get("stop"):
            stop_reasons.append(f"stage {s}: stop rule implied by {', '.join(k + ' kill' for k in kills)}")

    def nogo():
        r = [f"{p} kill" for p in ("PR1", "PR5", "PR6") if st[p] == "kill"]
        if st["PR10"] == "kill" and pr4_kill_final:
            r.append(f"PR10 kill and PR4 kill (Qwen3-1.7B kill; Qwen3-4B {st[PR4_4B]})")
        return r

    def switch_model():
        if pr4_17b_failed and st[PR4_4B] == "pass":
            return [f"PR4 {st['PR4']} on Qwen3-1.7B and pass on Qwen3-4B"]
        return []

    def go_big():
        if _all_pass(st, GO_BIG_REQUIRED):
            return [f"PR1, PR2 and PR4 to PR11 all pass (PR3 {st['PR3']}, informative)"]
        return []

    def go_method():
        if _all_pass(st, GO_METHOD_REQUIRED) and st["PR11"] in FAILED:
            return [f"everything passes except PR11 ({st['PR11']})"]
        return []

    def go_structure():
        if _all_pass(st, ("PR1", "PR5", "PR6")):
            bad = [f"{p} {st[p]}" for p in ("PR7", "PR9", "PR10") if st[p] in ("fail", "kill")]
            if bad:
                return [f"PR1, PR5, PR6 pass; {', '.join(bad)}"]
        return []

    rules = {"NO-GO": nogo, "SWITCH-MODEL": switch_model, "GO-BIG": go_big, "GO-METHOD": go_method,
             "GO-STRUCTURE": go_structure}
    order = list((prereg.get("verdicts") or {}).get("order") or
                 ["NO-GO", "SWITCH-MODEL", "GO-BIG", "GO-METHOD", "GO-STRUCTURE", "UNDECIDED"])
    verdict, reasons, candidates = None, [], []
    for name in order:
        if name in rules:
            r = rules[name]()
            if r:
                verdict, reasons = name, r
                break
        if name == "GO-METHOD" and verdict is None:
            # PENDING-LABELS sits where GO-BIG and GO-METHOD are decided: everything else passes and
            # only Dawar's carrier labels are missing.
            if _all_pass(st, GO_METHOD_REQUIRED) and st["PR11"] == "pending":
                verdict, candidates = "PENDING-LABELS", ["GO-BIG", "GO-METHOD"]
                reasons = ["PR1, PR2, PR4 to PR10 pass; PR11 AUROC part passes, carrier labels pending"]
                break
    if verdict is None:
        if stop_reasons:
            verdict, reasons = "STOPPED-EARLY", list(stop_reasons)
        else:
            verdict = "UNDECIDED"
            not_pass = [f"{p} {st[p]}" for p in PREMISES if p != "PR3" and st[p] != "pass"]
            reasons = ["no row of the decision table applies"] + (
                [f"not passing: {', '.join(not_pass)}"] if not_pass else [])
            if st["PR4"] == "fail" and st[PR4_4B] in ("missing", "not_run"):
                reasons.append("PR4 fail on Qwen3-1.7B without a Qwen3-4B result")
    reasons += unknown
    return {"verdict": verdict, "label": LABELS[verdict], "premises": st, "pending": pending,
            "missing": missing, "reasons": reasons, "candidates": candidates, "stop_reasons": stop_reasons}


def verdict_line(v: dict) -> str:
    """One-line verdict, e.g. 'GO-STRUCTURE (Paper: ...)'."""
    return f"{v['verdict']} ({v['label']})"
