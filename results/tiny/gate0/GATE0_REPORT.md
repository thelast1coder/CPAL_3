# Gate 0 report (TINY: numbers meaningless)

Verdict: **NO-GO**.
Short prompts (G0-A): remainder compressible on tiny_clt: 0/3; tiny_plt: 0/3 prompts per set; k_eff_short(PLT) = 65, C_short = 0.279.
Whole traces (G0-B): beta = 0.61, k_eff ratio = 7.18, best screening recall = 1, S_proj(L) = 10.8.

TINY: this run uses a tiny random transformer and synthetic inputs. It tests the code and the decision table only; no number below says anything about real attribution graphs.

## Setup

- git: 7ecc83534940a9584abc8d56b440c36e96076fa4 (dirty: True)
- prereg: configs/prereg_gate0.yaml, version 1 (DRAFT: thresholds proposed in PROJECT_V2.md Section 7, awaiting Dawar's confirmation)

## G0-A: short prompts

### tiny_clt

Row time 0.00151 s; kappa 1.98; k_eff_short 62.

| prompt | T | N | K0.7 error | dtype | A_0 energy share | s95 A | s95 remainder | s95 rem / T | c_trunc | m_min |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny_test_00 | 160 | 190 | 2.46e-16 | float64 | 0.579 | 32 | 68 | 0.425 | 160 | 16 |
| tiny_test_01 | 160 | 200 | 2.08e-16 | float64 | 0.587 | 38 | 61 | 0.381 | 160 | 16 |
| tiny_test_02 | 160 | 190 | 1.97e-16 | float64 | 0.6 | 38 | 62 | 0.388 | 160 | 16 |

| prompt | hub share by kind | fidelity by m | recovery seconds by m |
|---|---|---|---|
| tiny_test_00 | feature: 0.0335; error: 0.128; token: 0.836 | 16: n_meet: 3; meets: True; median_cost: 305; 32: n_meet: 3; meets: True; median_cost: 309; 64: n_meet: 3; meets: True; median_cost: 368; 128: n_meet: 3; meets: True; median_cost: 478 | 16: 0.0526; 32: 0.133; 64: 0.18; 128: 0.225 |
| tiny_test_01 | feature: 0.0325; error: 0.191; token: 0.777 | 16: n_meet: 3; meets: True; median_cost: 274; 32: n_meet: 3; meets: True; median_cost: 321; 64: n_meet: 3; meets: True; median_cost: 334; 128: n_meet: 3; meets: True; median_cost: 413 | 16: 0.022; 32: 0.127; 64: 0.129; 128: 0.149 |
| tiny_test_02 | feature: 0.0149; error: 0.109; token: 0.872 | 16: n_meet: 3; meets: True; median_cost: 241; 32: n_meet: 3; meets: True; median_cost: 237; 64: n_meet: 3; meets: True; median_cost: 276; 128: n_meet: 3; meets: True; median_cost: 355 | 16: 0.0206; 32: 0.0386; 64: 0.126; 128: 0.15 |

### tiny_plt (PLT)

Row time 0.003 s; kappa 2.38; k_eff_short 65.

| prompt | T | N | K0.7 error | dtype | A_0 energy share | s95 A | s95 remainder | s95 rem / T | c_trunc | m_min |
|---|---|---|---|---|---|---|---|---|---|---|
| tiny_test_00 | 160 | 190 | 2.62e-16 | float64 | 0.624 | 26 | 73 | 0.456 | 160 | 16 |
| tiny_test_01 | 160 | 200 | 2.37e-16 | float64 | 0.646 | 40 | 60 | 0.375 | 160 | 16 |
| tiny_test_02 | 160 | 190 | 2.24e-16 | float64 | 0.556 | 38 | 65 | 0.406 | 160 | 64 |

| prompt | hub share by kind | fidelity by m | recovery seconds by m |
|---|---|---|---|
| tiny_test_00 | feature: 0.0315; error: 0.0882; token: 0.879 | 16: n_meet: 3; meets: True; median_cost: 251; 32: n_meet: 3; meets: True; median_cost: 267; 64: n_meet: 3; meets: True; median_cost: 335; 128: n_meet: 3; meets: True; median_cost: 440 | 16: 0.0503; 32: 0.0903; 64: 0.222; 128: 0.302 |
| tiny_test_01 | feature: 0.0365; error: 0.0961; token: 0.867 | 16: n_meet: 3; meets: True; median_cost: 300; 32: n_meet: 3; meets: True; median_cost: 328; 64: n_meet: 3; meets: True; median_cost: 408; 128: n_meet: 3; meets: True; median_cost: 544 | 16: 0.0772; 32: 0.171; 64: 0.29; 128: 0.536 |
| tiny_test_02 | feature: 0.0204; error: 0.0813; token: 0.898 | 16: n_meet: 0; meets: False; median_cost: 268; 32: n_meet: 1; meets: False; median_cost: 298; 64: n_meet: 2; meets: True; median_cost: 353; 128: n_meet: 2; meets: True; median_cost: 447 | 16: 0.0302; 32: 0.0895; 64: 0.311; 128: 0.371 |

Sensing constant on tiny_plt: C by prompt tiny_test_00: 0.279; tiny_test_01: 0.272; tiny_test_02: 1.09; C_short = 0.279.

## G0-B: union growth

| n | U(n) tiny_trace_00 | U(n) tiny_trace_01 |
|---|---|---|
| 1 | 266 | 202 |
| 2 | 396 | 378 |
| 4 | 595 | 567 |
| 8 | 843 | 816 |

| trace | L | generated tokens | beta | outputs with c |
|---|---|---|---|---|
| tiny_trace_00 | 28 | 20 | 0.558 | 8 |
| tiny_trace_01 | 28 | 20 | 0.663 | 8 |

beta (mean over traces) = 0.61; c (median c_i) = 256; outputs without c = 0; L (median generated tokens) = 20.

## G0-B: sampled exact columns

| T_trace | N_trace | sampled | K0.7 error | dtype | s95 A | s95 remainder | k_eff_trace | hub fraction | row seconds | kappa |
|---|---|---|---|---|---|---|---|---|---|---|
| 1275 | 1405 | 80 | 2.43e-16 | float64 | 211 | 467 | 467 | 0 | 0.00352 | 1.37 |

| tau | energy recall | mean Omega_s size / T |
|---|---|---|
| 0.01 | 0.9997 | 0.327 |
| 0.001 | 1 | 0.343 |
| 0.0001 | 1 | 0.343 |

Screening: passes at tau = 0.01, |Omega_s| / T = 0.327.

## Projected speed-up

S_proj(L) = c L^beta / (C_short k_eff_trace ln(T_trace / k_eff_trace) + kappa H_est + m')

| term | value |
|---|---|
| c | 256 |
| L | 20 |
| beta | 0.61 |
| C_short | 0.279 |
| k_eff_trace | 467 |
| T_trace | 1275 |
| kappa | 1.37 |
| H_est | 0 |
| m_prime | 16 |
| trunc_cost | 1.59e+03 |
| sensing_cost | 131 |
| hub_cost | 0 |
| s_proj | 10.8 |

## What the numbers mean

Nothing about real models: this is a tiny run. It shows that the report reads every input, applies the decision table and writes its outputs.

- k07: the closed-form local part matches the attention-detached backward pass.
- compressible: the remainder is not compressible enough on short prompts.
- beta: union-truncation cost grows fast enough with the number of explained outputs.
- k_eff_ratio: the remainder is much less sparse at trace length than on short prompts.
- screening: attention reachability keeps almost all column energy at some threshold.
- s_proj: the cost law projects at least a 10x saving over union-truncation at trace length.

## Verdict

| condition | value | threshold | pass |
|---|---|---|---|
| k07 | max 2.62e-16 over 7 checks | 0.0001 (by dtype) | True |
| compressible | tiny_clt: 0/3; tiny_plt: 0/3 | s95_rem_frac_T <= 0.02 on >= 2 prompts for >= 1 set | False |
| not_nogo_s95 | tiny_clt: tiny_test_00: 0.425; tiny_test_01: 0.381; tiny_test_02: 0.388; tiny_plt: tiny_test_00: 0.456; tiny_test_01: 0.375; tiny_test_02: 0.406 | NO-GO if every prompt of every set > 0.1 | False |
| beta | 0.61 | >= 0.6 | True |
| k_eff_ratio | 7.18 | <= 3.0 | False |
| not_nogo_k_eff | 7.18 | NO-GO if > 10.0 | True |
| screening | 1 | >= 0.99 at some tau in [0.01, 0.001, 0.0001] | True |
| s_proj | 10.8 | >= 10.0 | True |

Verdict: **NO-GO**.

Stop: waiting for Dawar's decision (PD-G0).
