# STATUS

## Now
P0 (laptop) code, tests and tiny run done. Waiting for: git identity and remote URL (to commit and
push), decisions on open questions Q1 to Q4 (`docs/decisions.md`), then the VM.

## Done (P0)
- Repository layout, packaging, requirements (VM and CPU).
- Literature notes and novelty gate: PASSED with one caveat (Google Scholar not queried directly;
  repeat before submission). `docs/related_work_notes.md`.
- circuit-tracer 0.5.0 source read; `docs/circuit_tracer_notes.md`. Backend written against it
  (not runnable on the laptop; first test is K0 on the VM).
- All modules, scripts, configs and prompt files; 31 CPU tests pass (T1 to T17 plus storage and
  decision-logic checks).
- Tiny end-to-end run: `results/tiny/KILL_REPORT.md` (TINY, numbers meaningless).

## PREREG commit
- Hash: (pending: first commit blocked until git user.name and user.email are set)

## Next (P1, VM)
See README "VM: first commands for P1". Stop after K0 and report.

## Blockers
- Git identity not configured on the laptop; no remote URL yet.
- Q1 (which circuit-tracer source), Q2 (PLT transcoder repository ID), Q4 (unverified rhyme prompt)
  should be settled before the PREREG commit freezes prompts and configs.

## Log
