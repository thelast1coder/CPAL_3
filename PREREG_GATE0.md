# PREREG_GATE0.md: preregistration of Gate 0 (premise tests)

**Status: APPROVED by Dawar on 15 November 2026. Thresholds are fixed from this commit.** The commit
hash is recorded in results/STATUS.md before the first real-model run. Any later change goes in the
deviations log below, with its reason, and is reported.

This preregistration replaces the earlier Gate 0 draft (commit 60f3720, PROJECT_V2 Section 7). The
tests, measurements and verdict table are those of GATE0_PREMISES.md Sections 2, 3 and 4, committed
in the same commit; this file adds the operational definitions needed to compute them and the
machine-readable thresholds.

## 1. What is measured
GATE0_PREMISES.md Section 3, premises PR1 to PR11 in Stages A to F, on:
- Gemma-2-2B with `mntss/gemma-scope-transcoders` (PLT) and `mntss/clt-gemma-2-2b-426k` (CLT);
- Qwen3-1.7B (thinking mode) with `mwhanna/qwen3-1.7b-transcoders-lowl0` (PLT; config model_name
  Qwen/Qwen3-1.7B); for SWITCH-MODEL, Qwen3-4B with `mwhanna/qwen3-4b-transcoders`.
- Prompts: configs/prompts/kill_test.json ids kill_test_00, kill_test_04 and kill_test_12 ("calc: 36+59=")
  for Gemma; configs/prompts/gate0_qwen_short.json for Qwen3; 3 traces generated once by
  scripts/gen_traces.py from configs/prompts/gate0_traces.json (one per category: arithmetic word
  problem, multi-hop factual question, logic puzzle) and committed before any analysis.

## 2. Operational definitions (fixed before data)
- **s95 (squared):** smallest number of entries of a column holding 95% of its squared norm.
  **s95 (absolute):** smallest number holding 95% of its absolute (L1) mass. Columns are over
  feature targets (logit rows excluded) unless stated.
- **Causal candidate set of a source:** feature targets at a later read layer (at least one write
  layer below the read layer) and the same or a later position. Fractions in PR5, PR7 and PR8 divide
  by its size, per column.
- **Energy-weighted median:** median over columns weighted by the column's squared norm.
- **k_eff:** energy-weighted median of the remainder's absolute-value s95 (entries, not fraction).
- **Remainder:** A - A_0, with A_0 the closed-form local part (cct/local_part.py). The PR3 check uses
  the frozen backward pass with every attention output stop-gradiented (forward values unchanged).
- **Per-column relative error:** ||a_hat_s - a_s|| / ||a_s|| on the remainder column. **epsilon***:
  over the PR6 runs on the Qwen3 prompts, the largest energy-weighted per-column error among runs
  whose pruned graph meets fidelity, taken at the smallest m where fidelity holds on at least 2 of 3
  prompts (the minimum over decoders if both qualify).
- **Hub (PR7):** a sampled column whose simulated D2 error is above epsilon* at m = 8192.
- **Mass recall (PR8):** absolute mass of the column inside Omega_s divided by its total absolute
  mass, energy-weighted over columns.
- **Count-sketch (D2):** R = 5 repetitions of B = floor(m / 5) buckets; each target hashed to a
  bucket with a random sign in each repetition; estimate = median over repetitions of the signed
  bucket value; per column, the B / 4 largest estimates inside Omega_s (PR9) or the causal set (PR6)
  are kept, the rest set to zero. "3 hash seeds" means 3 independent sketches (seeds 0, 1, 2); fidelity
  "on 2 of 3 seeds" as for D1. (R = 3 was rejected on synthetic data before any real run: with
  thousands of targets, a median of 3 lets collisions with heavy entries create false entries.)
- **Truncation union, oracle union and costs (PR10):** as GATE0_PREMISES.md Section 3, Stage E; row
  costs weighted by measured pass time at prefix lengths 64, 128, 256 and full, interpolated
  linearly in length.
- **Faithfulness (PR4):** per output, Spearman correlation over its top 10 influence features between
  the graph-predicted logit change (direct-path attribution of the feature to the output logit times
  minus its activation) and the change measured by setting the feature to zero with circuit-tracer's
  feature_intervention (attention frozen as its default).
- **Sentences (PR11):** Thought Anchors' segmentation rule (verified against their code before the
  trace analysis; the rule used is recorded in the report).

## 3. Thresholds (machine-readable, identical to configs/prereg_gate0.yaml)
```yaml
prereg_version: 2
gate: gate0_premises
source: "GATE0_PREMISES.md Sections 3 and 4 (15 Nov 2026)"
status: "APPROVED by Dawar on 15 Nov 2026 (thresholds as proposed in GATE0_PREMISES.md); fixed from this commit"
gpu_cap_h100_hours: 6
dtype_rule: "fp32 where it fits, otherwise bf16; decided per model in Stage A, recorded, never mixed"
fidelity:
  influence_weighted_node_recall_at_least: 0.95
  node_jaccard_at_least: 0.85
  seeds_required: 2
pruning: {node_threshold: 0.8, edge_threshold: 0.98}
reference_graphs:
  outputs_per_trace: 17
  sentences_per_trace: 8
  tokens_per_sentence: 2
  budget: 16384
  convergence_budget: 8192
  convergence_jaccard_at_least: 0.95
stage_a:
  pr1:
    rel_err_fp32_at_most: 1.0e-5
    rel_err_bf16_at_most: 2.0e-2
    time_ratio_at_most: 1.2
    n_mixes: 5
    trace_prefix_tokens: 128
    trace_mix_targets: 256
    timing_lengths: [128, 300]
  pr2:
    n_positions: 3
    n_targets: 50
    rel_err_fp32_at_most: 1.0e-4
    rel_err_bf16_at_most: 2.0e-2
  pr3:
    rel_err_at_most: 1.0e-4
    informative_drop_local_part_if: {a0_energy_share_below: 0.10, s95_reduction_factor_below: 1.5}
  stop_rule: "any PR1 or PR2 kill: stop, report and wait"
stage_b:
  pr4:
    n_faithfulness_trace_outputs: 8
    top_k_ablated_features: 10
    pass:
      median_completeness_traces_at_least: 0.70
      completeness_gap_vs_short_at_most: 0.10
      median_faithfulness_spearman_traces_at_least: 0.5
      faithfulness_gap_vs_short_at_most: 0.15
    informative_median_pruned_nodes_above: 2000
    kill:
      median_completeness_below: 0.50
      faithfulness_spearman_below: 0.2
    switch_model: {model_set: qwen3_4b_plt, time_hours_at_most: 0.5}
  stop_rule: "PR4 kill on both models: stop and report"
stage_c:
  pr5:
    pass: {remainder_abs_s95_frac_causal_at_most: 0.02, prompts_at_least: 2, set: qwen3_1p7b_plt}
    kill: {remainder_abs_s95_frac_causal_above: 0.10, on: "every prompt of all three sets"}
  pr6:
    m_grid: [256, 512, 1024, 2048]
    d2_hash_seeds: 3
    pass: {m_at_most_T_over: 10, prompts_at_least: 2, decoder: D2, set: qwen3_1p7b_plt}
    kill: {m_at_most_T_over: 4, on: "no Qwen3 prompt with either decoder"}
  stop_rule: "PR5 or PR6 kill: stop"
stage_d:
  n_sources_per_trace: 300
  pr7:
    pass: {k_eff_trace_over_short_at_most: 3.0}
    kill: {k_eff_trace_over_short_above: 10.0}
  pr8:
    taus: [1.0e-2, 1.0e-3, 1.0e-4]
    pass: {mass_recall_at_least: 0.99, median_omega_frac_causal_at_most: 0.10}
    kill: null
  pr9:
    m_grid: [512, 1024, 2048, 4096, 8192]
    seeds: 3
    non_hub_energy_share_within_eps_star_at_least: 0.90
    kill: "no m_trace within the grid"
stage_e:
  pr10:
    n_grid: [1, 2, 4, 8, 16, 32, 51]
    timing_lengths: [64, 128, 256, full]
    check_probes: 16
    pass:
      local_slope_at_least: 0.6
      traces_at_least: 2
      s_meas_at_least: 3.0
      s_proj_at_least: 20.0
      s_oracle_proj_at_least: 5.0
    kill: {s_proj_below: 3.0, on: "every trace"}
stage_f:
  pr11:
    top_dependencies_per_trace: 5
    carriers_per_dependency: 5
    top_quartile_auroc: true
    pass: {auroc_minus_s4_at_least: -0.02, auroc_at_least: 0.65, nameable_at_least: 9}
    weak: {auroc_minus_s4_below: -0.10, nameable_below: 5}
verdicts:
  order: [NO-GO, SWITCH-MODEL, GO-BIG, GO-METHOD, GO-STRUCTURE, UNDECIDED]
  NO-GO: "PR1 kill, or PR5 kill, or PR6 kill, or (PR10 kill and PR4 kill)"
  SWITCH-MODEL: "PR4 fails on Qwen3-1.7B but passes on Qwen3-4B"
  GO-BIG: "PR1, PR2, PR4 to PR11 all pass (PR3 informative)"
  GO-METHOD: "everything passes except PR11"
  GO-STRUCTURE: "PR1, PR5, PR6 pass; PR7, PR9 or PR10 fail"
  UNDECIDED: "anything not covered: write the report and stop; Dawar decides"
```

## 4. Deviations log
| Date | What | Why |
|---|---|---|
