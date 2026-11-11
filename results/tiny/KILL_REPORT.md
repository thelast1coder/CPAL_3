# TINY RUN, NUMBERS ARE MEANINGLESS

This report exercises the whole kill-test pipeline on a tiny random transformer. It tests the
code and the decision logic only. No number below says anything about real attribution graphs.

# Kill test report

## Summary

Verdict: **NO-GO**.
K1 asks whether attribution matrices are compressible, K2 whether a few mixed passes rebuild the graph more cheaply than exact enumeration and truncation, K3 whether this holds with real passes and clocks.
Per-set verdicts are in the decision table below; thresholds are those of PREREG_KILL_TEST.md.

## Setup

- git: HEAD (dirty: True)
- prereg version: 1; sets: tiny_plt, tiny_clt; seeds: [0, 1, 2]
- packages: {'python': '3.12.1', 'torch': '2.15.0+cpu', 'numpy': '2.5.4', 'scipy': '1.18.1', 'circuit_tracer': None, 'transformer_lens': None, 'nnsight': None, 'transformers': None, 'huggingface_hub': None, 'yaml': '6.0.3'}
- device: cpu (AMD64)

## K0 sanity

| set | check | pass | value |
|---|---|---|---|
| tiny_plt | K0.1 | True | 2.67e-15 |
| tiny_plt | K0.2 | True | 6.72e-16 |
| tiny_plt | K0.3 | True | 3.63e-16 |
| tiny_plt | K0.4 | True | True |
| tiny_plt | K0.5 | True | row_seconds=0.000193, kappa=3.73, batch=64, recovery_columns_per_second=1.2e+03 |
| tiny_plt | K0.6 | True | True |
| tiny_clt | K0.1 | True | 3.32e-15 |
| tiny_clt | K0.2 | True | 6.87e-16 |
| tiny_clt | K0.3 | True | 5.97e-16 |
| tiny_clt | K0.4 | True | True |
| tiny_clt | K0.5 | True | row_seconds=0.000263, kappa=1.89, batch=64, recovery_columns_per_second=1.55e+03 |
| tiny_clt | K0.6 | True | True |

## K1 structure

| set | median col s95 / T | median row s95 / N | median hub energy share | share GO | share NO-GO | verdict |
|---|---|---|---|---|---|---|
| tiny_plt | 0.194 | 0.0211 | 0.975 | 0 | 0.875 | NO-GO |
| tiny_clt | 0.213 | 0.0316 | 0.953 | 0 | 1 | NO-GO |

## K2 simulation

### tiny_plt

Chosen on validation: `{'setting': {'recovery': {'algo': 'omp', 'k_max': 8}, 'screening': {'q': 0.995}, 'hubs': {'rho_hub': 0.3, 'budget': 32}}, 'index': 0, 'm': None, 'cost': None, 'fallback': True}`

| method | c75 budget | c75 cost | share at c75 |
|---|---|---|---|
| cct_fast | never | n/a | n/a |
| b1 | T_D | 160 | 1 |
| cct_certified | 16 | 433 | 1 |
| b2 | never | n/a | n/a |
| b3 | never | n/a | n/a |

Median T_D = 160; S_exact = n/a; S_B1 = n/a; verdict NO-GO.

| method | budget | share meeting fidelity | median cost |
|---|---|---|---|
| cct_fast | 16 | 0.125 | 235 |
| cct_fast | 32 | 0 | 254 |
| cct_fast | 64 | 0 | 298 |
| cct_fast | 128 | 0 | 386 |
| b1 | 16 | 0 | 16 |
| b1 | 32 | 0 | 32 |
| b1 | 64 | 0 | 64 |
| b1 | 128 | 0.125 | 128 |
| b1 | T_D | 1 | 160 |
| cct_certified | 16 | 1 | 433 |
| cct_certified | 32 | 1 | 425 |
| cct_certified | 64 | 1 | 457 |
| cct_certified | 128 | 1 | 496 |
| b2 | 16 | 0 | 16 |
| b2 | 32 | 0 | 32 |
| b2 | 64 | 0 | 64 |
| b2 | 128 | 0 | 128 |
| b3 | 16 | 0.375 | 47.8 |
| b3 | 32 | 0.125 | 85.6 |
| b3 | 64 | 0.125 | 161 |
| b3 | 128 | 0.25 | 312 |

Regime F (secondary): S_exact = n/a, S_B1 = n/a, NO-GO.

### tiny_clt

Chosen on validation: `{'setting': {'recovery': {'algo': 'omp', 'k_max': 16}, 'screening': {'q': 0.995}, 'hubs': {'rho_hub': 0.3, 'budget': 128}}, 'index': 3, 'm': 16, 'cost': 317.6133217705258}`

| method | c75 budget | c75 cost | share at c75 |
|---|---|---|---|
| cct_fast | 16 | 322 | 1 |
| b1 | T_D | 160 | 1 |
| cct_certified | 16 | 491 | 1 |
| b2 | never | n/a | n/a |
| b3 | never | n/a | n/a |

Median T_D = 160; S_exact = 0.496; S_B1 = 0.496; verdict NO-GO.

| method | budget | share meeting fidelity | median cost |
|---|---|---|---|
| cct_fast | 16 | 1 | 322 |
| cct_fast | 32 | 1 | 374 |
| cct_fast | 64 | 1 | 410 |
| cct_fast | 128 | 1 | 470 |
| b1 | 16 | 0 | 16 |
| b1 | 32 | 0 | 32 |
| b1 | 64 | 0 | 64 |
| b1 | 128 | 0.125 | 128 |
| b1 | T_D | 1 | 160 |
| cct_certified | 16 | 1 | 491 |
| cct_certified | 32 | 1 | 522 |
| cct_certified | 64 | 1 | 609 |
| cct_certified | 128 | 1 | 661 |
| b2 | 16 | 0 | 16 |
| b2 | 32 | 0 | 32 |
| b2 | 64 | 0 | 64 |
| b2 | 128 | 0 | 128 |
| b3 | 16 | 0.625 | 33.1 |
| b3 | 32 | 0.375 | 56.2 |
| b3 | 64 | 0.125 | 102 |
| b3 | 128 | 0 | 195 |

Regime F (secondary): S_exact = 1.17, S_B1 = 1.17, NO-GO.

## K3 real passes

| set | regime | m | prompts meeting | median speed-up | max rel. Frobenius real vs sim | verdict |
|---|---|---|---|---|---|---|
| tiny_plt | D | 128 | 0 | 1.19 | 2.66e-08 | NO-GO |
| tiny_clt | D | 16 | 4 | 1.41 | 2.67e-08 | NO-GO |

## Decision table

| set | K1 | K2 | K3 |
|---|---|---|---|
| tiny_plt | NO-GO | NO-GO | NO-GO |
| tiny_clt | NO-GO | NO-GO | NO-GO |

Overall: **NO-GO**.

## What the numbers mean

Nothing: this is a tiny random model. The run shows that every stage executes and the decision table is applied.
