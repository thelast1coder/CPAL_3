# STATUS

## Now
Gate 0 (PROJECT_V2.md Section 7) code, CPU tests and the tiny rehearsal are done. Waiting for:
Dawar's confirmation of the Gate 0 thresholds (PREREG_GATE0.md, draft), then the GPU VM
(RTX PRO 6000, 96 GB).

## Done
- P0 (10 Nov): repository, literature notes (novelty gate passed, Scholar caveat), circuit-tracer
  0.5.0 notes, all v1 code, CPU tests, tiny kill-test run.
- Gate 0 code (15 Nov):
  - `cct/local_part.py`: closed-form A_0; equals the attention-detached backward pass to about 1e-16 on
    the tiny model (PLT, CLT, skip transcoders).
  - `cct/reach.py`: attention-rollout reachability and candidate sets.
  - Backend: attention detaching (K0.7), read-map detection, skip map, attention patterns.
  - `run_cct` subtracts A_0 before recovery.
  - Scripts: `gate0_short.py` (G0-A), `gen_traces.py`, `gate0_union.py` (G0-B union growth),
    `gate0_columns.py` (G0-B columns and screening), `gate0_report.py`, `run_gate0.sh` (job queue).
  - Tiny rehearsal through the queue: `results/tiny/gate0/GATE0_REPORT.md` (TINY, numbers meaningless).
- Settled: Gemma PLT repository `mntss/gemma-scope-transcoders` (PROJECT_V2); Qwen3 PLTs
  `mwhanna/qwen3-1.7b-transcoders-lowl0` (circuit-tracer README); dependency pins (decisions I26).

## PREREG commits
- Kill test draft: 55d84fc (v1; runs after Gate 0 on the GO-BIG path).
- Gate 0 draft: 60f3720 (thresholds awaiting confirmation); the confirmed hash is recorded here before the first
  real-model run.

## Next (VM)
README "VM: Gate 0". Expected: about 2 to 2.5 hours of wall-clock time on the VM including setup,
with a 4-hour ceiling (first real run of the backend). Stop at the report.

## Blockers
- Q5: Dawar confirms the Gate 0 thresholds. Q6: compute cap and whether the GPU is exclusive.
- Repository visibility: the spec asks for a private repository.

## Log
