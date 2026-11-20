TINY: numbers meaningless. Gate 0 verdict: NO-GO. Stop. One-page post-mortem on what attribution matrices actually look like.
Passed: PR2, PR3. Failed, killed or weak: PR1 (fail), PR4 (kill), PR5 (kill), PR6 (fail). Pending, not run or missing: PR7 (missing), PR8 (missing), PR9 (missing), PR10 (missing), PR11 (missing).
Why: PR5 kill. GPU time used: 0.02 of 6 H100-hours.

# Gate 0 premise report (TINY: numbers meaningless)

TINY: this run uses the tiny random model and synthetic inputs. It tests the code path only; no number here says anything about real models.

## Verdict

- Verdict: **NO-GO** (Stop. One-page post-mortem on what attribution matrices actually look like).
- Applied from configs/prereg_gate0.yaml, order: NO-GO, SWITCH-MODEL, GO-BIG, GO-METHOD, GO-STRUCTURE, UNDECIDED.
- Reason: PR5 kill
- Missing: stage D (stage_D.json); stage E (stage_E.json); stage F (stage_F.json). No GO-BIG is issued while anything is missing.
- Stop rules triggered: stage B: PR4 kill on Qwen3-1.7B: switch model before anything else (premise_B.py --model-set qwen3_4b_plt); stop rule B applies if Qwen3-4B also fails; stage B_qwen3_4b: Stop rule B: PR4 kill on Qwen3-1.7B and Qwen3-4B; whole-trace science is not viable on available transcoders (structure-only path remains open); stage C: PR5 kill: remainder s95 above the kill level on every prompt of all three sets.

## Premises

| Premise | What was measured | Threshold | Value | Result |
|---|---|---|---|---|
| PR1 | mixed-pass rows against Phi times exact rows (rel. error); mixed/single pass time ratio | rel_err_at_most_gemma: 1e-05; rel_err_at_most_qwen: 1e-05; time_ratio_at_most: 1.2 | gemma_plt_mix_rel_err: short_tiny_test_00: 6.04e-16, 7.64e-16, 2.67e-16, 4.17e-16, 8.17e-16; short_tiny_test_01: 6.95e-16, 7.32e-16, 6.41e-16, 5.43e-16, 3.78e-16; gemma_plt_mix_rel_err_max: 8.17e-16; qwen_trace_mix_re... (note: rows exact, but time ratio 1.375 > 1.2) | fail |
| PR2 | prefix rows against full-trace frozen-pass rows, 3 positions x 50 targets | rel_err_at_most: 0.0001 | rel_err: 26: 1.5e-16; 44: 2.18e-16; 62: 1.96e-16; rel_err_max: 2.18e-16; only_full_rel_mass_max: 0; targets_missing_in_full: 0 (note: prefix rows equal the full-trace rows of the same nodes within tolerance) | pass |
| PR3 | closed-form A_0 against frozen backward with attention stop-gradiented (informative) | rel_err_at_most_by_set: tiny_plt: 0.0001; tiny_clt: 0.0001; tiny_qwen_plt: 0.0001; informative_drop_local_part_if: a0_energy_share_below: 0.1; s95_reduction_factor_below: 1.5 | by_set: tiny_plt: tiny_test_00: rel_err: 2.44e-16; a0_energy_share: 0.556; s95_sq_reduction: 0.468; s95_abs_reduction: 0.628; tiny_test_01: rel_err: 2.47e-16; a0_energy_share: 0.536; s95_sq_reduction: 0.551; s95_abs_r... (note: closed-form A_0 equals the attention-detached rows within tolerance) | pass |
| PR4 | Qwen3-1.7B trace completeness and ablation faithfulness against short prompts | n_faithfulness_trace_outputs: 8; top_k_ablated_features: 10; pass: median_completeness_traces_at_least: 0.7; completeness_gap_vs_short_at_most: 0.1; median_faithfulness_spearman_traces_at_least: 0.5; faithfulness_gap_... | median_completeness_traces: 0.809; median_completeness_short: 0.837; median_replacement_traces: 0.578; median_replacement_short: 0.621; median_faithfulness_spearman_traces: 0.182; median_faithfulness_spearman_short: 0... | kill |
| PR4_qwen3_4b | the same on Qwen3-4B (SWITCH-MODEL path) | n_faithfulness_trace_outputs: 8; top_k_ablated_features: 10; pass: median_completeness_traces_at_least: 0.7; completeness_gap_vs_short_at_most: 0.1; median_faithfulness_spearman_traces_at_least: 0.5; faithfulness_gap_... | median_completeness_traces: 0.809; median_completeness_short: 0.837; median_replacement_traces: 0.578; median_replacement_short: 0.621; median_faithfulness_spearman_traces: 0.182; median_faithfulness_spearman_short: 0... | kill |
| PR5 | remainder abs. s95 as share of the causal set, Qwen3-1.7B PLT short prompts | pass: remainder_abs_s95_frac_causal_at_most: 0.02; prompts_at_least: 2; set: qwen3_1p7b_plt; kill: remainder_abs_s95_frac_causal_above: 0.1; true: every prompt of all three sets | remainder_abs_s95_frac_causal: tiny_plt: tiny_test_00: 0.738; tiny_test_01: 0.707; tiny_test_02: 0.71; tiny_clt: tiny_test_00: 0.705; tiny_test_01: 0.69; tiny_test_02: 0.698; tiny_qwen_plt: tiny_test_03: 0.686; tiny_t... (note: remainder s95 above the kill level on every prompt of all three sets) | kill |
| PR6 | pruned-graph fidelity from simulated D2 recovery, smallest m | pass: m_at_most_T_over: 10; prompts_at_least: 2; decoder: D2; set: qwen3_1p7b_plt; kill: m_at_most_T_over: 4; true: no Qwen3 prompt with either decoder; fidelity: influence_weighted_node_recall_at_least: 0.95; node_ja... | d2_prompts_meeting_within_T_over_10_by_m: 16: ; 32: ; 64: ; 128: ; passing_m: ; any_meet_within_T_over_4: true; T_features: tiny_test_03: 544; tiny_test_04: 390; tiny_test_05: 355; fidelity_tables: tiny_plt: tiny_test... (note: fidelity reached only above T/10, or with D1 only, or on fewer than 2 Qwen prompts) | fail |
| PR7 | k_eff(trace) / k_eff(short), hub rate | ratio <= 3; kill > 10 | not available | missing |
| PR8 | reachability mass recall and median size of Omega_s as share of the causal set | some tau: recall >= 0.99 and median size of Omega_s <= 0.1 of causal set | not available | missing |
| PR9 | m_trace: smallest m with 90% of non-hub energy within epsilon* | m_trace in [512, 1024, 2048, 4096, 8192] | not available | missing |
| PR10 | union growth slope, S_meas, S_proj, S_oracle_proj | slope >= 0.6 on >= 2 traces; S_meas >= 3; S_proj >= 20; S_oracle_proj >= 5; kill: S_proj < 3 everywhere | not available | missing |
| PR11 | feature-level AUROC against S4; nameable carriers | AUROC - S4 >= -0.02 and >= 0.65; >= 9 of 15 nameable; weak: AUROC - S4 < -0.1 or < 5 nameable | not available | missing |

## Plots

- s95 histograms (remainder, absolute version): skipped: matplotlib is not installed here (it is on the VM).
- Fidelity against m (D1 and D2): skipped: matplotlib is not installed here (it is on the VM).
- Union growth, realistic and oracle (log-log): skipped: data not available in the stage files.
- Feature-level against baseline AUROC: skipped: data not available in the stage files.

## Cost table (PR10)

| Quantity | Value |
|---|---|
| m_trace (PR9) | not available |
| kappa (hub cost in row-equivalents) | not available |
| projected hub count | not available |
| check probes | not available |
| recovery cost (row-equivalents) | not available |
| decode time per column (s) | not available |
| columns kept after energy screening | not available |
| row time at full length (s) | not available |
| Cost_CCT(L) (row-equivalents) | not available |
| L (trace tokens) | not available |
| truncation cost weights by prefix length | not available |
| local slope | not available |
| U_trunc(51) size (cost-weighted) | not available |
| S_meas | not available |
| S_proj | not available |
| S_oracle_proj | not available |
| epsilon* (PR6 calibration) | 0.0112 |

## Top 15 dependencies and carrier features (for Dawar to label)

Top dependencies and carriers: not available (no stage F list and no carriers_for_labelling.md).

## Deviations from the prereg

None recorded in the PREREG_GATE0.md deviations log.

## GPU time per stage

| Stage | GPU hours | Stage budget (H100-hours) |
|---|---|---|
| A | 0.00218 | 0.3 |
| B | 0.00541 | 0.5 |
| B_qwen3_4b | 0.00296 | 0.5 |
| C | 0.00849 | 1 |
| D | not available | 1 |
| E | not available | 1.5 |
| F | not available | 1.5 |
| total | 0.019 | cap 6 |

## Provenance

| Stage | git hash | dirty | GPU | wall seconds | UTC |
|---|---|---|---|---|---|
| A | f53f188a52f1 | true | cpu (AMD64) | 9.3 | 2026-11-20T12:47:12+00:00 |
| B | f53f188a52f1 | true | cpu (AMD64) | 19.7 | 2026-11-20T12:47:37+00:00 |
| B_qwen3_4b | f53f188a52f1 | true | cpu (AMD64) | 14.1 | 2026-11-20T12:48:08+00:00 |
| C | f53f188a52f1 | true | cpu (AMD64) | 32.2 | 2026-11-20T12:47:49+00:00 |

## What surprised us

(To be filled in by hand after reading the results.)

Stop: waiting for Dawar's decision.
