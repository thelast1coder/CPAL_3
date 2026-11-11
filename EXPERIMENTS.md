# EXPERIMENTS.md: all phases, experiments, gates and budgets

Read PROJECT.md first. IDs here (X, S, G, Z, A, R) are used in code, results files and the paper.
GPU-hour figures are planning estimates for one H100. Log the actual wall time of every job.

## Dependency graph
```
P0 (laptop: repo, notes, code, CPU tests on a tiny backend)
 |
P1 kill test (K0 sanity -> exact graphs -> K1, K2 offline -> K3 real) --> STOP, report, wait for Dawar
 |  GO
 +--> P2 core results (X1 to X4) -------------------------+
 +--> P3 scaling (S1 to S4)                               |
 +--> P4 application: group-testing ablations (G1, G2)    +--> P7 seeds, figures, FINAL_RESULTS.md
 +--> P5 theory and certificate checks (Z1 to Z4, offline)|
 +--> P6 ablations (A1 to A10, mostly offline)          --+
```
Offline work (sensing simulations on stored exact matrices, recovery, metrics) runs on CPU or as small
GPU jobs while exact-graph jobs run. Simulation is legitimate because mixed passes are exactly linear
(K0.2 proves it); every headline number is then confirmed with real mixed passes.

## P0: setup and code (laptop, no GPU), 11 to 12 Nov
Full instructions: PHASE0_SETUP.md. Exit criteria: CPU tests pass on the tiny backend, the tiny
end-to-end run writes a TINY kill report, literature notes done with the novelty gate passed, the
circuit-tracer backend written against the pinned source, PREREG committed, README has VM commands.

## P1: kill test (VM), 13 to 15 Nov, about 4 to 6 GPU-hours
Full instructions: PHASE1_KILL_TEST.md; thresholds: PREREG_KILL_TEST.md.
Gate G1: the decision table in the prereg. Stop and report whatever the outcome.

## P2: core results, 16 to 22 Nov, about 12 to 18 GPU-hours
**X1, main table.** Models and transcoders: Gemma-2-2B (PLT, CLT-426k, CLT-2.5M if memory allows),
Llama-3.2-1B (PLT, CLT), Qwen3-4B (PLT). 100 test prompts (the 32 kill-test prompts plus 68 new
ones in the same 8 categories, committed before running). Methods: B0, B1, B2, B3, B5 (node metrics
only), B6, CCT-fast, CCT-certified. Report, per model: cost at fidelity (passes and wall-clock),
speed-up against B0 and against B1, all fidelity metrics.
**X2, Pareto curves.** Fidelity (influence-weighted node recall, Jaccard) against cost for CCT, B1,
B2, B3 on Gemma-2-2B PLT and CLT and Llama-3.2-1B CLT.
**X3, canonical circuits side by side.** The two-hop "capital of the state containing Dallas", the
"Michael Jordan plays the sport of", the multilingual antonym, the addition and the acronym prompts
(exact strings as in PHASE1 Section 4): exact graph against CCT graph at the headline budget,
rendered the same way, with the top paths listed.
**X4, real-pass confirmation.** Every headline number of X1 rerun with real mixed passes (not
simulation) on 20 prompts per model.
Gate G2 (stop and report): on at least two model and transcoder pairs, median speed-up against B0 of
at least 10x at fidelity, and against B1 of at least 3x, confirmed by X4.
- Yes: full paper as planned.
- Only in regime F (full graphs): headline moves to full graphs and long prompts (PD2).
- Neither: discuss; the structural finding (C1) and certificate (C4) may still make a smaller paper.

## P3: scaling (C3), 18 to 25 Nov, about 10 to 15 GPU-hours
**S1, prompt length.** Gemma-2-2B CLT-426k, prompt lengths 16, 32, 64, 128, 256, 512 tokens (8 prompts
each, from the long-prompt set). Exact full graphs (B0F) up to the length that fits in the budget;
CCT at every length. Plot passes and wall-clock against T (number of targets): exact grows linearly,
CCT should grow like k log T.
**S2, model size.** Qwen3 0.6B, 1.7B, 4B, 8B, 14B with PLTs (as memory allows), 16 prompts each,
regime D and F.
**S3, full graphs.** Regime F on 32 prompts for Gemma-2-2B (PLT and CLT): what the cap drops, and CCT's
cost relative to exact full enumeration.
**S4, workload demo.** Build 1,000 graphs (for example, every prompt in a public prompt set, or one
graph per reasoning step on 50 chain-of-thought examples) with CCT, report total GPU time against the
exact estimate (from measured per-graph time on a 50-graph sample).
Gate G3: S1 shows CCT's cost growing clearly slower than T (fitted exponent at most 0.5 on log-log),
with fidelity held.

## P4: application, group-testing ablations (C5), 22 to 26 Nov, about 4 to 6 GPU-hours
**G1.** For 20 prompts, find the 10 features whose single ablation (in the real model, transcoder
features clamped to zero) most changes the top logit, among the N active features: exhaustive single
ablations (N runs; ground truth on a subset where affordable) against random group ablations decoded
with sparse regression and with a group-testing decoder at C k log N runs. Report recall of the true
top 10 against the number of runs.
**G2.** Use G1 to validate CCT graphs: are the top influence nodes in the CCT graph the ones whose
ablation matters? Compare with the same check for exact graphs.

## P5: theory and certificate checks (C1, C4), 22 to 27 Nov, mostly CPU
**Z1, structure.** s95 of columns and rows by node type (embedding, error, feature), layer and
position; hub statistics; which side is sparser and why (for example, the direct residual path).
**Z2, recovery against tail.** Per column, recovery error against the tail sigma_k(a)_1 / sqrt(k);
check the predicted scaling and the constant.
**Z3, certificate tightness.** A posteriori estimate against the true error; the Neumann influence
bound and the data-dependent interval against the true influence error; how often refinement is
triggered.
**Z4, depth.** Measured nilpotency index of A_bar against the layer count, per model.

## P6: ablations, 22 to 28 Nov, about 6 to 8 GPU-hours (mostly offline)
| ID | Ablation | Values |
|---|---|---|
| A1 | Recovery algorithm | masked OMP, CoSaMP, FISTA-LASSO, IHT |
| A2 | Sensing matrix | Rademacher, Gaussian, sparse +-1 (d nonzeros per column), position-blocked |
| A3 | Causal mask | on, off |
| A4 | Energy screening | kept energy 99%, 99.5%, 99.9%, none |
| A5 | Hubs | rule thresholds, budget H = 0, 32, 128, 512; class rules (embeddings, errors) on or off |
| A6 | Certification probes | m' = 8, 16, 32 |
| A7 | Precision | fp32, bf16 |
| A8 | Sensing side | target side (default), source side (dual), both |
| A9 | Plain Jacobian sensing (B6) against full CCT | component-by-component additions |
| A10 | Batch size and wall-clock | 64, 128, 256, 512 rows per batch |

## P7: robustness, figures, final results, 27 to 29 Nov, about 4 to 6 GPU-hours
**R1.** Three sensing seeds for every headline number; report median and range.
**R2.** All figures and tables generated by `scripts/make_paper_assets.py` from results files only.
**R3.** `results/FINAL_RESULTS.md`: every number in the paper with its source file, seed and git hash.

## Paper assets plan
- Figure 1: compressibility (column energy curves, s95 histograms, hubs), Gemma-2-2B PLT and CLT.
- Figure 2: fidelity against cost, CCT against truncation, random rows and low-rank (X2).
- Figure 3: scaling with the number of targets and prompt length (S1), log-log.
- Table 1: main results (X1). Table 2: scaling with model size (S2).
- Figure 4: canonical circuits, exact against CCT (X3).
- Table 3: certification (Z3) and group testing (G1).
- Appendix: ablations (A1 to A10), seeds (R1), real-pass confirmation (X4).

## Compute summary (planning)
| Phase | GPU-hours |
|---|---|
| P1 | 4 to 6 |
| P2 | 12 to 18 |
| P3 | 10 to 15 |
| P4 | 4 to 6 |
| P5 | under 1 (mostly CPU) |
| P6 | 6 to 8 |
| P7 | 4 to 6 |
| Total | about 40 to 60 |
