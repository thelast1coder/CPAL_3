# PREREG_KILL_TEST.md: pre-registration of the kill test

Status: DRAFT until committed. Commit this file, `configs/prereg_kill_test.yaml` and the prompt files
in their own commit before the first real-model run, and write the hash into `results/STATUS.md`.
After that, nothing in Sections 1 to 6 changes. Anything done differently goes in Section 7.

## 1. Hypotheses
- **H-K1 (compressible):** in attribution matrices from circuit-tracer on Gemma-2-2B, most of each
  source's out-edge energy lies on a small fraction of targets (energy-weighted median column s95 at
  most 1% of the targets on most prompts).
- **H-K2 (cheap and better than truncation):** CCT reaches the exact pruned graph (influence-weighted
  node recall >= 0.95 and node Jaccard >= 0.85) on at least 75% of prompts at a tenth or less of the
  exact cost, and at a third or less of the cost truncation needs for the same rate.
- **H-K3 (holds for real):** with real mixed passes, fidelity holds and end-to-end wall-clock time is at
  least 5x lower than the exact default.

## 2. Setup
Gemma-2-2B with two transcoder sets: PLT (`mntss/gemma-scope-transcoders`) and CLT
(`mntss/clt-gemma-2-2b-426k`) (IDs verify). Pinned circuit-tracer version with its defaults (recorded
at K0). dtype per set by decision D10, fixed before K0. Prompts: 32 test, 4 validation, 8 long, as
committed. Sensing seeds 0, 1, 2. Certification probes m' = 16.

## 3. Measurements
- s95 of a vector: the number of its largest entries holding 95% of its squared norm. Columns are
  restricted to their causal masks.
- Fidelity for one prompt and seed: influence-weighted node recall and node Jaccard of the pruned
  feature-node sets, against the exact pruned graph (same pruning code and thresholds). A prompt meets
  fidelity if both thresholds hold for at least 2 of 3 seeds.
- Cost in row-equivalents, with per-row time and kappa measured at K0.5: mixed rows + certification
  rows + kappa x forward-mode passes + recovery seconds / row seconds (+ refinement rows for
  CCT-certified). B1 cost = number of rows. Exact default cost T_D = rows in the exact default run.
- Hyperparameters other than the budget are chosen on the 4 validation prompts only (PHASE1 Section 7,
  Step 1).
- c75 of a method: the smallest budget on its grid at which at least 75% of the 32 test prompts meet
  fidelity; its cost is the median cost over prompts at that budget.
- S_exact = median over prompts of T_D divided by CCT-fast's c75 cost. S_B1 = B1's c75 cost divided by
  CCT-fast's c75 cost.
- Wall-clock speed-up (K3) = median over the 8 K3 prompts of exact default wall time / CCT wall time,
  each the median of 3 repeats in the same session.

## 4. Thresholds (machine-readable copy; must equal configs/prereg_kill_test.yaml, test T16)
```yaml
prereg_version: 1
model: gemma2_2b
transcoder_sets: [plt, clt426k]
prompts: {test: 32, validation: 4, long: 8}
seeds: [0, 1, 2]
certification_probes: 16
fidelity:
  influence_weighted_node_recall_at_least: 0.95
  node_jaccard_at_least: 0.85
  seeds_required: 2
k1:
  statistic: share_of_test_prompts_by_energy_weighted_median_column_s95
  go: {s95_fraction_of_T_at_most: 0.01, prompt_share_at_least: 0.70}
  nogo: {s95_fraction_of_T_above: 0.10, prompt_share_at_least: 0.70}
k2:
  primary_variant: cct_fast
  cct_budgets_m: [64, 128, 256, 512, 1024]
  b1_budgets_rows: [64, 128, 256, 512, 1024, 2048, 4096, T_D]
  c75_prompt_share: 0.75
  go: {s_exact_at_least: 10.0, s_b1_at_least: 3.0}
  amber: {s_exact_at_least: 4.0, s_b1_at_least: 1.5}
  nogo: "otherwise, or CCT never reaches the 75% fidelity rate on its grid"
  upgrade_to_go_f: "if K2 is AMBER on both sets in regime D but regime F gives s_exact >= 20 and s_b1 >= 3 on at least one set"
k3:
  prompts: 8
  repeats: 3
  go: {prompts_meeting_fidelity_at_least: 6, wallclock_speedup_at_least: 5.0, real_vs_simulated_rel_frobenius_at_most: {fp32: 0.01, bf16: 0.05}}
  amber: {prompts_meeting_fidelity_at_least: 5, wallclock_speedup_at_least: 2.5}
  nogo: "otherwise"
decision:
  go: "K2 GO or GO-F on at least one set and not NO-GO on the other; K3 GO on that set; K1 not NO-GO on that set"
  nogo: "K1 NO-GO on both sets, or K2 NO-GO on both sets, or K3 NO-GO"
  amber: "anything else"
```

## 5. Secondary analyses (reported, not used for the decision)
Row s95 and hub statistics; CCT-certified results; B2 and B3 curves; per-prompt cheapest budgets;
regime F results when not used for the upgrade; Spearman of node influence; replacement and
completeness score differences; edge errors; recovery time share of the total cost.

## 6. What may not happen
No hyperparameter tuning on test prompts (only the budget grid is evaluated on them, symmetrically for
every method). No change of prompts, thresholds, grids, transcoder sets or circuit-tracer defaults
after seeing CCT results. No dropping of prompts after seeing results, except a regime-F graph that
exceeds the 30-minute exact budget, which is logged before any CCT run on it. The only
pre-registered adaptations are the GO-F upgrade path and the AMBER fix window, which needs Dawar's
approval and is reported as exploratory.

## 7. Deviations log
| Date | What changed | Why | Effect on the decision |
|---|---|---|---|
| | | | |
