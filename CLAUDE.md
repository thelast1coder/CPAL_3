# CLAUDE.md (CCT_CPAL: Compressed Circuit Tracing)

Read these before doing anything, in this order:
1. `PROJECT_V2.md`: the current project (big-paper scope, 15 Nov). It supersedes `PROJECT.md` (v1)
   wherever they disagree; v1 is still the reference for anything V2 does not change.
2. `GATE0_PREMISES.md` (replaces PROJECT_V2 Section 7) and `PREREG_GATE0.md` (approved thresholds,
   commit f53f188): the first GPU job, premise tests PR1 to PR11.
3. `EXPERIMENTS.md`, `PHASE1_KILL_TEST.md`, `PREREG_KILL_TEST.md`: v1 phases; the short-prompt kill
   test (K0 to K3) runs after Gate 0 on the GO-BIG path (PROJECT_V2 Section 8).

## Current phase: Gate 0 code ready, waiting for the GPU VM
Gate 0 code, CPU tests and the tiny rehearsal are done on the laptop. Do not download any model,
transcoder or dataset on the laptop. Gate 0 runs with `bash scripts/run_premises.sh` on the VM; stop
at results/gate0/PREMISE_REPORT.md and wait for Dawar's decision. GPU cap: 6 H100-hours.

## Phase status
| Phase | What | Where | Status |
|---|---|---|---|
| P0 | Repo, literature notes, all v1 code, CPU tests, tiny run | laptop | done |
| Gate 0 code | premise tests PR1 to PR11 (Stages A to F), report, queue | laptop | in progress; prereg approved (f53f188) |
| Gate 0 run | G0-A and G0-B, report | RTX PRO 6000 VM | waiting for the VM |
| After Gate 0 | K0 to K3, E1 to E5 (PROJECT_V2 Section 8) | VM | only after GO-BIG and Dawar's confirmation |

## Rules (always)
- Never invent numbers, citations, model IDs, transcoder IDs or API details. Anything not checked
  stays marked (verify) until it is checked.
- Every reported number comes from a script in this repo with a fixed seed. Each result file stores
  config, seed, git hash, package versions, GPU name and wall time.
- The exact graph from the pinned circuit-tracer version is the ground truth. Same model,
  transcoders, dtype, prompts and pruning settings for every method.
- Prompts are fixed and committed before any real-model run. Validation prompts (for tuning) never
  overlap test prompts.
- Thresholds in `PREREG_KILL_TEST.md` are fixed once committed. Record the commit hash in
  `results/STATUS.md` before the first real-model run. Deviations go in the deviations log with a
  reason, and are reported.
- Count every cost: mixed passes, forward-mode passes, recovery time, certification. Report speed-ups
  in passes and in end-to-end wall-clock time, against the exact default and the truncation baseline.
- Stop and report at every gate (K0, the kill-test decision, and the gates in EXPERIMENTS.md). Do not
  start the next phase without my go-ahead.
- Ask me before changing a claim, model, threshold or prompt set, or before exceeding a phase's
  compute budget by more than 25%. Implementation details are yours; log them in `docs/decisions.md`.
- Check `nvidia-smi` before every job, cap memory per process with `cct.utils.gpu.cap_memory`, queue
  jobs, and never kill processes you did not start. The Gate 0 VM (RTX PRO 6000, 96 GB) is assumed to
  be for this project only; if it is shared, lower the caps (`CCT_MEMORY_CAP_GB`) first.
- On the VM, all downloads go under `$HF_HOME`. Never commit tokens, keys, weights or large artefacts.
  Use a fresh Hugging Face token.
- Keep `results/STATUS.md` current: what ran, what is running, what is next, blockers.
- Reports: British spelling, plain language, no em-dashes, no emoji. Every report starts with a
  three-line plain summary, then the numbers, then what they mean.
