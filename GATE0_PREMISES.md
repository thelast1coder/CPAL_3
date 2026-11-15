# GATE0_PREMISES.md: test every assumption the CCT thesis rests on, before building the method

Version 1, 15 November 2026. This file replaces Section 7 of PROJECT_V2.md (Gate 0). Where the two
disagree, this file wins. Section 19 of PROJECT_V2.md stays valid for the code layout.

## 0. Purpose and rules

The thesis (PROJECT_V2.md, Section 2.4) depends on several things being true about real models. This
gate measures each one directly, cheapest and most fundamental first, and stops early when one fails.
It is the lesson from the MoE post-mortem: that idea rested on one second-hand number, and even its
oracle could not reach the bar.

Rules:
- **Build no method.** Only measurement code, plus the minimum needed to simulate recovery offline.
  No GPU decoder optimisation, no export, no circuit-tracer integration beyond what each test needs.
- **Commit the thresholds first.** Copy Section 3 into PREREG_GATE0.md and commit it before any
  real-model run. Dawar approves by that commit. After that, thresholds never change.
- **Stop at every stop rule** (Section 4), write the report, and wait for Dawar.
- **Gate 0 GPU cap: 6 H100-hours.** Log real times per stage.
- **Every number** is stored in results/gate0.json with config, seed, git hash, package versions,
  GPU name and wall time.
- **Thresholds are proposals until committed.** Each marked number is a judgement call, not a
  standard.

---

## 1. The assumptions, in plain words

The thesis needs all of these. Each one has its own test in Section 3.

| ID | Assumption | If false |
|---|---|---|
| PR1 | A backward pass seeded with a signed mix of targets returns exactly that mix of their rows, in the dtype we use, through circuit-tracer's frozen model, and costs about the same as an ordinary pass | The method cannot work. Stop |
| PR2 | One frozen forward pass over a whole trace defines one table, and every token's graph is a slice of it | Sharing across tokens is false. The whole-trace thesis dies |
| PR3 | The same-position residual part A_0 has an exact closed form | Not fatal: drop the "local part" from the thesis |
| PR4 | Transcoders work on reasoning text: few error nodes, graphs agree with real-model ablations, pruned graphs stay readable | Graphs of traces are mostly error nodes. The science pillar is empty. Change model first |
| PR5 | After removing A_0, each node's out-edges sit on a small share of the targets it can reach (short prompts, on the transcoder type the traces will use) | No sparsity, so nothing to sense. Stop |
| PR6 | Sparse recovery from mixed passes reproduces the pruned graph, not just the edges | Sparsity without fidelity is useless. Stop or rethink |
| PR7 | Sparsity survives long traces, and hubs (dense columns) are rare | Cost grows with length, so the whole-trace win disappears |
| PR8 | Attention reachability can skip irrelevant targets safely and also cuts the candidate set a lot | Recovery cannot run at trace scale |
| PR9 | Per-column recovery at trace length reaches the error level that PR6 showed is good enough | No trustworthy graphs at length |
| PR10 | Row-based methods (truncation) must compute many distinct rows to explain many tokens, so shared sensing wins by an order of magnitude, even against an oracle that knows which rows matter | No speed-up worth a paper. Fall back to the structure-only paper |
| PR11 | Feature-level step dependencies predict real-model step interventions at least as well as sentence-level methods (Thought Anchors' attention suppression), and the carrier features are nameable | The science pillar is weak. Narrow it to descriptive |

---

## 2. Shared setup

- **Short-prompt models and transcoders:**
  - Gemma-2-2B with PLT `mntss/gemma-scope-transcoders`;
  - Gemma-2-2B with CLT `mntss/clt-gemma-2-2b-426k`;
  - Qwen3-1.7B with circuit-tracer's Qwen3 PLT (verify the ID, and the checkpoint the transcoder was
    trained on).
- **Short prompts:**
  - Gemma: the 3 canonical prompts ("Fact: the capital of the state containing Dallas is", "Fact:
    Michael Jordan plays the sport of", "calc: 36+59="), with exact strings copied from the sources.
  - Qwen3-1.7B: 3 short prompts of the same kinds (one two-hop fact, one single-hop fact, one
    addition).
- **Traces:** 3 Qwen3-1.7B thinking-mode traces, one each of arithmetic word problem, multi-hop
  factual question and logic puzzle.
  - 200 to 300 generated tokens each.
  - Fixed seed. Commit the text and the generation settings.
  - Split each trace into sentences with the same rule as Thought Anchors (verify their
    segmentation code).
- **Sampled outputs (shared by PR4, PR10 and PR11):**
  - Per trace, take the first two tokens of 8 sentences spread across the second half of the trace,
    plus the final answer token.
  - That gives 17 outputs per trace, 51 in total.
  - For each, build an exact reference graph with circuit-tracer on the prefix:
    - budget 16,384 rows, with convergence checked against 8,192 (node Jaccard ≥ 0.95, otherwise
      raise the budget and log it);
    - save the visiting order and every computed row.
- **Fidelity of a graph against its reference:** influence-weighted node recall ≥ 0.95 and node
  Jaccard ≥ 0.85, with the same pruning settings (circuit-tracer defaults).
- **dtype:** fp32 where it fits, otherwise bf16. Decide per model in Stage A, record it, never mix.

---

## 3. The tests, thresholds and verdicts (copy into PREREG_GATE0.md)

### Stage A: foundations (≤ 0.3 H100-hours)

**PR1: exact linearity and cost of mixed passes**
- Measure:
  - Gemma PLT, 2 short prompts: 5 random-sign mixes over all targets. Compare the returned rows
    with Φ times the exact rows.
  - One 128-token trace prefix on Qwen3-1.7B: mix over 256 random targets, so the exact answer is
    computable, and compare the same way.
  - Timing: one mixed pass against one single-target pass, at lengths 128 and 300.
- Pass:
  - relative error ≤ 1e-5 in fp32, or ≤ 2e-2 in bf16;
  - time ratio ≤ 1.2.
- Kill: fails in both dtypes after the injection code is checked. **Stop: the method is impossible.**

**PR2: one table serves every token**
- Measure: for 3 positions in one trace, compare exact rows of 50 random targets computed on the
  prefix with the same rows from the full-trace frozen pass.
- Pass: relative error ≤ 1e-4 in fp32 (bf16 tolerance as PR1).
- Kill: a mismatch that dtype does not explain, for example a chat template or a special token that
  changes between prefix and full trace. If the cause cannot be fixed, **stop whole-trace work.**

**PR3: closed-form local part**
- Measure:
  - the closed-form A_0 against a frozen backward pass with every attention output zeroed, on 2 short
    prompts per model;
  - the energy share of A_0;
  - the reduction in median column s95 after subtracting A_0.
- Pass: relative error ≤ 1e-4.
- Not a kill. If the formula fails, fix it or drop A_0 subtraction and record that.
- Informative: if A_0 holds less than 10% of edge energy and subtracting it cuts s95 by less than
  1.5x, drop "local part" from the thesis.

**Stop rule A:** any PR1 or PR2 kill means stop, report and wait.

### Stage B: are reasoning-trace graphs meaningful at all? (≤ 0.5 H100-hours)

**PR4: transcoder quality on thinking text**
- Measure, on Qwen3-1.7B, for the 51 trace outputs and for the 3 short prompts:
  1. Completeness and replacement scores (circuit-tracer's definitions).
  2. Pruned graph size: nodes and edges at default thresholds.
  3. Faithfulness:
     - for 8 trace outputs and the 3 short prompts, ablate each of the top 10 influence feature nodes
       one at a time in the real model (circuit-tracer's intervention utilities);
     - Spearman correlation between graph-predicted and actual change in the output logit.
- Pass (proposals):
  - median completeness on traces ≥ 0.70, and no more than 0.10 below short prompts;
  - median trace faithfulness Spearman ≥ 0.5, and no more than 0.15 below short prompts.
- Informative: median pruned graph on traces above 2,000 nodes means graphs are too big to read.
  Flag it for the science plan.
- Kill: median completeness < 0.50, or faithfulness Spearman < 0.2.
  - **Then switch model before anything else:** try Qwen3-4B PLT on the same traces regenerated,
    within 0.5 hours.
  - If it still fails, the whole-trace science pillar is not viable on available transcoders.

**Stop rule B:** a PR4 kill on both models means stop and report. The structure-only path remains
open, but whole-trace science does not.

### Stage C: structure and recovery on short prompts, with full exact matrices (≤ 1 H100-hour)

For each of the 3 sets (Gemma PLT, Gemma CLT-426k, Qwen3-1.7B PLT) and each of their 3 short
prompts, compute the full exact matrix with every active feature as a target. Store it sparse or in
fp16, with a checksum.

**PR5: compressibility**
- Measure, for A and for the remainder A − A_0:
  - per column: s95 by squared values AND by absolute values, as a fraction of that source's causal
    candidate set (later layers, same or later position), not of T;
  - the energy-weighted median over columns;
  - hub share by node kind: features, error nodes, embeddings, first-token nodes.
- Pass: on Qwen3-1.7B PLT, remainder median s95 (absolute-value version) ≤ 2% of the causal set on
  ≥ 2 of 3 prompts.
- Kill: > 10% on every prompt for all three sets. **Stop.**

**PR6: recovery gives the right pruned graph**
- Measure:
  - simulate Y = Φ (A − A_0) with two decoders:
    - D1: Rademacher signs plus masked OMP;
    - D2: count-sketch, with 3 hash seeds;
  - m ∈ {256, 512, 1024, 2048};
  - add A_0 back, prune, and score fidelity against the exact pruned graph;
  - time the recovery on the GPU.
- Calibration: record per-column relative error at each m. Find ε*, the largest energy-weighted
  per-column error at which fidelity still holds on ≥ 2 of 3 prompts.
- Pass: on Qwen3-1.7B PLT, some m ≤ T/10 meets fidelity on ≥ 2 of 3 prompts, with D2 (the decoder
  the traces need). D1 is reported too.
- Kill: no m ≤ T/4 meets fidelity on any Qwen3 prompt with either decoder. **Stop, or write a
  post-mortem on why sparse columns still give wrong graphs.**

**Stop rule C:** PR5 or PR6 kill means stop.

### Stage D: structure and recovery at trace length, from sampled exact columns (≤ 1 H100-hour)

Per trace, take 300 sources stratified by node kind and position (first token, early, middle, late).
Compute their exact columns over every target in the trace with forward-mode passes
(torch.func.jvp, or the double-backward trick; record which).

**PR7: sparsity survives length; hubs are rare**
- Measure:
  - absolute k_eff (energy-weighted median s95 of the remainder, absolute-value version) on traces,
    compared with Qwen3-1.7B short prompts;
  - the hub rate by node kind.
  - A hub is a column whose simulated D2 recovery error stays above ε* at the largest m tested in
    PR9.
- Pass: k_eff(trace) ≤ 3 × k_eff(short), same model.
- Kill: > 10x. **Stop whole-trace work.**

**PR8: screening is safe and selective**
- Measure: for τ ∈ {1e-2, 1e-3, 1e-4}:
  - the share of each column's absolute-value mass inside its reachability candidate set Ω_s;
  - the median |Ω_s| as a share of the causal set.
- Pass: at some τ, mass recall ≥ 0.99 AND median |Ω_s| ≤ 10% of the causal set.
- Not a kill on its own. It feeds the recovery cost in PR10.

**PR9: per-column recovery at trace length**
- Measure:
  - for each sampled column, simulate count-sketch measurements exactly: hash the trace's full
    target set and apply Φ to the exact column. By PR1, this is exactly what real mixed passes would
    return for that column;
  - decode over Ω_s at m ∈ {512, 1024, 2048, 4096, 8192}, with 3 seeds;
  - measure per-column relative error and decode time per column.
- Output: m_trace, the smallest m at which ≥ 90% of non-hub column energy has error ≤ ε*.
- Pass: m_trace exists within the grid.
- Kill: none at 8,192. **Stop whole-trace work.**

### Stage E: does shared sensing beat row methods? (≤ 1.5 H100-hours; reuses the reference graphs)

**PR10: reuse and speed-up, against realistic and oracle row methods**
- Measure, per trace, for n ∈ {1, 2, 4, 8, 16, 32, 51}:
  - **realistic union |U_trunc(n)|:**
    - for each output, find the smallest influence-ordered prefix of its reference visiting order
      that meets fidelity;
    - union the (layer, position, feature) rows across the first n outputs.
  - **oracle union |U_oracle(n)|:**
    - the union of the exact kept node sets, plus logit rows;
    - this lower-bounds any method that computes exact rows, however clever its selection;
    - note that CCT with exact refinement of kept nodes costs at least this much, so the headline
      must be the version without refinement.
  - **cost weights:** weight each truncation row by the measured pass time at its prefix length,
    measured at 64, 128, 256 and full length. Truncation is cheaper per pass on short prefixes, and
    it gets that credit.
  - **Cost_CCT(L), for all L tokens of the trace,** in row-equivalents at full length:
    - m_trace (PR9);
    - plus κ × the projected hub count (PR7 hub rates × the number of sources of each kind);
    - plus 16 check probes;
    - plus recovery time (decode time per column × the number of columns kept after energy
      screening, divided by the row time).
- Speed-ups:
  - local slope = the slope of log |U_trunc(n)| between n = 16 and n = 32, and between 32 and 51;
    use the smaller;
  - S_meas = |U_trunc(51)| (cost-weighted) / Cost_CCT(L). This is a measured comparison: 51 explained
    outputs for truncation against all L for CCT, so it is conservative for CCT;
  - S_proj = |U_trunc| extrapolated to L with the local slope, divided by Cost_CCT(L);
  - S_oracle_proj = the same with |U_oracle|.
- Pass, all of:
  - local slope ≥ 0.6 on ≥ 2 of 3 traces;
  - S_meas ≥ 3;
  - S_proj ≥ 20 (twice the paper's 10x target, because projections leave out overheads);
  - S_oracle_proj ≥ 5.
- Kill: S_proj < 3 on every trace. **Whole-trace headline is dead.**

### Stage F: the science premise, tested with exact graphs and no CCT (≤ 1.5 H100-hours)

**PR11: feature-level step dependencies predict interventions**
- Measure, on the 3 traces, using the reference graphs already built:
  1. **Feature-level dependency.** D[j, i] = the influence on the sampled tokens of sentence j from
     nodes at sentence i's positions, normalised per j.
  2. **Ground truth, in the real model.**
     - Replace sentence i with neutral filler of equal token length.
     - Teacher-force the rest of the trace unchanged.
     - Measure the change in log-probability of sentence j's sampled tokens.
     - Repeat with a second variant: delete sentence i.
     - Use all pairs i < j.
  3. **Baselines, predicting the same ground truth:**
     - S4: Thought Anchors' attention suppression (suppress attention to sentence i and measure the
       effect on sentence j);
     - S1: attention rollout;
     - S2: gradient × input summed over sentence i;
     - S5: FlashTrace token attribution, if its public code runs in time; otherwise log it as
       skipped.
  4. **Metrics:**
     - Spearman correlation per trace between predicted and actual effect;
     - AUROC for detecting the top-quartile actual effects.
  5. **"What is carried".** For the top 5 dependencies per trace, list the 5 carrier features (the
     nodes at sentence i holding most of the path influence into j), with their top-activating
     tokens on a reference corpus. Dawar judges whether each set is nameable.
- Pass (pilot thresholds; small sample):
  - feature-level AUROC ≥ S4 AUROC − 0.02, and ≥ 0.65 absolute;
  - at least 9 of the 15 top dependencies have nameable carriers.
- Weak: feature-level AUROC < S4 − 0.10, or fewer than 5 nameable carriers. Then narrow the science
  pillar to descriptive "what is carried", or drop it.

---

## 4. Overall decision (results/PREMISE_REPORT.md, applied automatically from the YAML)

| Verdict | Condition | What it means |
|---|---|---|
| **GO-BIG** | PR1, PR2, PR4 to PR11 all pass (PR3 informative) | The thesis stands. Start the method and strengthening phase (PROJECT_V2.md Sections 5 and 8) |
| **GO-METHOD** | Everything passes except PR11 | Paper: structure + whole-trace method; science narrowed to descriptive. Dawar decides scope |
| **GO-STRUCTURE** | PR1, PR5, PR6 pass; PR7, PR9 or PR10 fail | Paper: structure of attribution matrices + uncapped full graphs + what the default cap hides. Dawar decides venue |
| **SWITCH-MODEL** | PR4 fails on Qwen3-1.7B but passes on Qwen3-4B | Rerun Stages D to F on Qwen3-4B (about 3 more hours), then decide |
| **NO-GO** | PR1 kill, or PR5 kill, or PR6 kill, or (PR10 kill and PR4 kill) | Stop. One-page post-mortem on what attribution matrices actually look like |

Anything not covered: write the report and stop; Dawar decides.

---

## 5. Report format (results/PREMISE_REPORT.md)

1. Three-line plain summary and the verdict.
2. One table: premise, what was measured, threshold, value, pass or fail.
3. Plots:
   - s95 histograms (short prompts and traces);
   - fidelity against m (D1 and D2);
   - union growth, realistic and oracle, log-log, with local slopes;
   - feature-level against baseline AUROC.
4. Cost table: the inputs to Cost_CCT, the truncation cost weights, S_meas, S_proj, S_oracle_proj.
5. The 15 top dependencies with carrier features, for Dawar to label.
6. Deviations from the prereg, if any, and why.
7. What surprised us, in plain words.

---

## 6. Tasks for Claude Code (delta on top of the code already written)

1. Copy Section 3 thresholds into PREREG_GATE0.md and `configs/prereg_gate0.yaml`. Commit. Stop for
   Dawar's approval if not already given.
2. Add the Qwen3-1.7B short prompts and the 3 trace generations, then commit the text.
3. `scripts/premise_A.py`: PR1 (including the timing ratio), PR2, PR3.
4. `scripts/premise_B.py`: PR4 (completeness, replacement, graph size, ablation faithfulness). Also
   build and save the 51 reference graphs here, because Stages E and F reuse them.
5. `scripts/premise_C.py`: full exact matrices, PR5, PR6 with D1 and D2 simulation, and the ε*
   calibration.
6. `scripts/premise_D.py`: forward-mode columns, PR7, PR8, PR9.
7. `scripts/premise_E.py`: realistic and oracle unions, per-length pass timing, Cost_CCT, S_meas,
   S_proj, S_oracle_proj.
8. `scripts/premise_F.py`: dependency matrix, sentence interventions (two variants), baselines S4,
   S1, S2 (and S5 if feasible), metrics, carrier-feature lists.
9. `scripts/premise_report.py`: report and gate0.json, verdict applied from the YAML.
10. Each script checks its stage's stop rule and exits with a clear message if it triggers.

CPU tests before any GPU run:
- count-sketch simulation on synthetic sparse columns with a known tail;
- union and oracle computation on toy graphs;
- dependency aggregation on a toy DAG;
- the verdict logic on fabricated result files, covering every branch of Section 4.
