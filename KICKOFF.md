# KICKOFF.md: what to paste into Claude Code

Put all files from this zip in an empty folder called `CCT_CPAL` on the laptop, open Claude Code in
that folder, and paste the prompt below.

## Prompt for P0 (laptop, now)
```
Read CLAUDE.md, then PROJECT.md, EXPERIMENTS.md, PHASE0_SETUP.md, PHASE1_KILL_TEST.md and
PREREG_KILL_TEST.md in full before writing anything.

We are in Phase P0 on my laptop: no GPU and only about 0.5 GB of free disk. Do not download any
model, transcoder or dataset here. You may download the circuit-tracer wheel without dependencies only
to read its source. Do all of P0 as PHASE0_SETUP.md describes:
1. Set up the repository and the small environment (reuse the system PyTorch).
2. Write docs/related_work_notes.md and apply the novelty gate. If the gate trips, stop and tell me.
3. Read circuit-tracer's attribution code and write docs/circuit_tracer_notes.md.
4. Write every module, script and config in Section 2 (the circuit-tracer backend against the pinned
   source), the prompt files, and every test in Section 4.
5. Run the tests on CPU and the tiny end-to-end run (scripts/run_kill_test.sh --tiny).
6. Commit PREREG_KILL_TEST.md, configs/prereg_kill_test.yaml and the prompt files in their own commit
   and record the hash.
Ask me for the private GitHub repo URL when you need it. Use a task list. When P0 is complete, stop
and report: what is ready, test results, the novelty-gate outcome, what circuit-tracer's source means
for the plan (seed injection, forward mode), anything marked (verify), and the first VM commands.
```

## Prompt for P1 (when the VM is ready)
```
The H100 VM is ready: <how to reach it, the path for HF_HOME>. The GPU is shared, so check
nvidia-smi first and agree the order with the MoE_CPAL kill test. Start Phase P1 exactly as
PHASE1_KILL_TEST.md describes, with the thresholds in PREREG_KILL_TEST.md. Run K0 first and stop if
any K0 check fails. Keep results/STATUS.md current. When the kill test finishes, write
results/KILL_REPORT.md, apply the decision table, and stop and report to me. Do not start P2 without
my go-ahead.
```
