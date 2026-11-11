# CLAUDE.md (CCT_CPAL: Compressed Circuit Tracing)

Read these before doing anything, in this order:
1. `PROJECT.md`: the whole project (problem, maths, claims, models, baselines, rules, decisions, timeline).
2. `EXPERIMENTS.md`: every phase from P0 to P7, with experiments, gates and compute budgets.
3. `PHASE0_SETUP.md`: what to do now.
4. `PHASE1_KILL_TEST.md` and `PREREG_KILL_TEST.md`: the go/no-go experiment that runs first on the GPU.

## Current phase: P0 (laptop, no GPU)
Set up the repository, write the literature notes (with the novelty gate), write all code and CPU
tests against a tiny pure-PyTorch backend, run the tiny end-to-end pipeline, and commit
`PREREG_KILL_TEST.md`. Do not download any model, transcoder or dataset on the laptop (about 0.5 GB
free disk). You may download the circuit-tracer wheel without dependencies to read its source. When
P0 is complete, stop and tell me. P1 starts when I give you the GPU VM.

## Phase status
| Phase | What | Where | Status |
|---|---|---|---|
| P0 | Repo, literature notes, all code, CPU tests, tiny end-to-end run, PREREG committed | laptop | NEXT |
| P1 | Kill test (K0 sanity, then K1, K2, K3) and report | H100 VM | waiting for the VM |
| P2 to P7 | Paper experiments | H100 VM | only after GO and my confirmation |

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
- The GPU is shared with Mirror_CPAL, MoE_CPAL and a CVPR project. Check `nvidia-smi` before every
  job, cap memory per process with `cct.utils.gpu.cap_memory`, queue jobs, and never kill processes
  you did not start.
- On the VM, all downloads go under `$HF_HOME`. Never commit tokens, keys, weights or large artefacts.
  Use a fresh Hugging Face token.
- Keep `results/STATUS.md` current: what ran, what is running, what is next, blockers.
- Reports: British spelling, plain language, no em-dashes, no emoji. Every report starts with a
  three-line plain summary, then the numbers, then what they mean.
