# STATUS

## Now
P0 code, tests and tiny run done and pushed. Waiting for: decisions on open questions Q1 to Q4
(`docs/decisions.md`), then the VM.

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
- Draft PREREG commit: 55d84fc (not frozen: Q1 to Q4 open). The frozen hash is recorded here before the first real-model run.

## Next (P1, VM)
See README "VM: first commands for P1". Stop after K0 and report.

## Blockers
- None for git (repo-local identity thelast1coder with the GitHub no-reply address).
- Q1 (which circuit-tracer source), Q2 (PLT transcoder repository ID), Q4 (unverified rhyme prompt)
  should be settled before the PREREG commit freezes prompts and configs.

## Log
