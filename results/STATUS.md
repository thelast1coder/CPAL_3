# STATUS

## Now
Gate 0 premise tests (GATE0_PREMISES.md, PR1 to PR11) are coded and checked on the laptop. Waiting
for the GPU VM (H200 141 GB, or RTX PRO 6000 96 GB). Run: README "VM: Gate 0 premise tests".

## Done
- P0 (10 Nov): repository, literature notes, v1 code, CPU tests, tiny kill-test run.
- Gate 0 premise code (15 to 20 Nov):
  - Stages A to F: scripts/premise_A.py ... premise_F.py, shared logic in src/cct/premise_*.py,
    count-sketch decoder src/cct/sketch.py, closed-form A_0 src/cct/local_part.py, reachability
    src/cct/reach.py, verdict src/cct/premise.py, report scripts/premise_report.py, queue
    scripts/run_premises.sh (parallel lanes, stop rules, SWITCH-MODEL path).
  - Speed: pruning and graph scores on the GPU (CCT_PRUNE_DEVICE), vectorised mixed-pass injection.
  - Checks on 20 Nov: 133 CPU tests pass (127 fast, 6 slow end-to-end, including tiny runs of every
    stage). Full tiny rehearsal through the queue: stop rules fire as designed (tiny matrices are
    dense, so the tiny verdict is NO-GO; numbers meaningless).
- The real-model paths are written against the circuit-tracer 0.5.0 source but have not run on a real
  model yet; the first minutes of Stage A test them.

## PREREG commits
- Gate 0 premise tests: f53f188, APPROVED by Dawar on 15 Nov 2026 (thresholds fixed).
- Superseded: Gate 0 draft 60f3720 (PROJECT_V2 Section 7). Kill test draft 55d84fc (v1; after Gate 0).

## Next (VM)
Expected about 4 to 6 hours on an H200 (cap 6 H100-hours), less if a stop rule fires early.
Then: stop at results/gate0/PREMISE_REPORT.md; Dawar labels the carrier features (Stage F) and decides.

## Blockers
- None on the code side. Repository visibility: the spec asks for a private repository.

## Log
- 20 Nov: premise code checked (133 tests, tiny rehearsal); pushed.
