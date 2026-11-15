# PREREG_GATE0.md: pre-registration of Gate 0

Status: DRAFT: thresholds proposed in PROJECT_V2.md Section 7, awaiting Dawar's confirmation.

Commit this file and `configs/prereg_gate0.yaml` in their own commit before the first real-model run
of Gate 0, and write the hash into `results/STATUS.md`. After that, nothing in Sections 1 to 6 changes.
Anything done differently goes in Section 7. Settings that are not decision thresholds (budgets, grids,
batch sizes) live in `configs/experiments/gate0.yaml`.

## 1. Purpose
Gate 0 tests the premise of the whole-trace paper (PROJECT_V2.md Sections 2 to 7) before any further
GPU time is spent. It asks four questions:
- **G0-A, short prompts:** once the closed-form local part A_0 is removed, is the remainder A - A_0
  compressible (few large entries per column), and how many mixed passes does sensing need in practice
  (the constant C)?
- **G0-B, union growth:** when circuit-tracer explains n outputs of one reasoning trace, does the
  union of rows that truncation needs grow like n^beta with beta large enough for sensing to win?
- **G0-B, columns at trace length:** is the remainder still sparse at trace length (k_eff_trace close
  to k_eff_short), and does attention reachability screening keep the column energy?
- **Projection:** does the measured cost law project a speed-up of at least 10x over union-truncation
  at trace length?

The answer is one of GO-BIG, GO-STRUCTURE, NO-GO or UNDECIDED (Section 4). Dawar takes decision PD-G0
from the report. Nothing after Gate 0 starts without that decision.

## 2. Models, transcoders, prompts
**G0-A (short prompts).**
- Model: Gemma-2-2B (`google/gemma-2-2b`, revision recorded from the download).
- Transcoder sets: `gemma2_2b_plt` (`mntss/gemma-scope-transcoders`, as PROJECT_V2 Section 11; Gemma
  Scope lowest-L0 transcoders, reading `ln2.hook_normalized`) and `gemma2_2b_clt426k`
  (`mntss/clt-gemma-2-2b-426k`, reading `hook_resid_mid`). Both repositories and their config files
  were checked on Hugging Face on 15 November 2026; revisions are recorded at download.
  The PLT set is the one whose name ends with `plt`.
- Prompts: `kill_test_00`, `kill_test_04`, `kill_test_09` from `configs/prompts/kill_test.json`
  (the three verified canonical prompts: two-hop "Dallas", single-hop "Michael Jordan", antonym
  "small").
- Exact graphs: circuit-tracer defaults, regime D (PHASE1_KILL_TEST.md Section 5), with the full row
  matrix saved. dtype per set as decided at K0 (D10).

**G0-B (whole traces).**
- Model: Qwen3-1.7B (`Qwen/Qwen3-1.7B`) with thinking mode on.
- Transcoders: `mwhanna/qwen3-1.7b-transcoders-lowl0` (PLT).
- Traces: 2 traces, generated once by the rule in `scripts/gen_traces.py` from the problems in
  `configs/prompts/gate0_traces.json` (greedy decoding, settings in `configs/experiments/gate0.yaml`,
  section `g0b.generation`; traces of 200 to 300 generated tokens, with the fallback range 150 to 400
  only if fewer than 2 fall in the main range). The trace text and settings are committed before any
  attribution run on them.

## 3. Measurements and exact definitions
**Column statistics.**
- s95 of a column: the smallest number of its entries holding 95% of its squared norm
  (`cct.killtest.s95`).
- Column statistics are taken over feature targets; logit rows are excluded.
- Energy-weighted median: the median of a per-column statistic with each column weighted by its
  squared norm (`cct.killtest.weighted_median`).
- Remainder: A - A_0, where A_0 is the closed-form local part (PROJECT_V2.md T2).
- T: the number of targets in the exact regime-D matrix of the prompt.
- s95_rem_frac_T: the energy-weighted median remainder s95 divided by T.
- k_eff_short for a transcoder set: the median over the 3 prompts of the energy-weighted median
  remainder s95.
- k_eff_trace: the energy-weighted median remainder s95 over the sampled exact columns of the trace,
  taken over all feature targets of the full trace. The restriction to the screening set Omega_s is
  reported, not used.
- Also reported (not used for the decision): the energy share of A_0, the energy-weighted median s95 of
  A, the remainder s95 as a fraction of |Omega_s|, and hub energy share by node kind.

**K0.7 sanity (closed-form local part).** The rows of the closed-form A_0 must equal the rows of the
attention-detached backward pass, with relative Frobenius error at most 1e-4 in float32 or at most
2e-2 in bfloat16. Checked on each G0-A prompt and set, and on the trace used for the sampled columns.
Any failure stops Gate 0: the failure is reported and no verdict is issued (verdict STOPPED).

**Fidelity.** Influence-weighted node recall at least 0.95 and node Jaccard at least 0.85 against the
reference pruned graph (node threshold 0.8, edge threshold 0.98). The same definition is used for
truncation and for sensing.

**Truncation budget (G0-A).** c_trunc: the smallest budget on the truncation grid of
`configs/experiments/gate0.yaml` that meets fidelity.

**Practical sensing constant C (G0-A).**
- For one prompt: m_min is the smallest m in {128, 256, 512, 1024} at which sensing of A - A_0 (with
  A_0 added back before pruning) meets fidelity on at least 2 of the 3 sensing seeds.
- C for that prompt = m_min / (k_eff ln(T / k_eff)), with k_eff the prompt's energy-weighted median
  remainder s95 and T its number of targets.
- If no m on the grid works for a prompt, its C is taken as 1024 / (k_eff ln(T / k_eff)) and the
  prompt is flagged in the report.
- C_short: the median over the 3 prompts of the PLT set.

**Union growth (G0-B).**
- Explained outputs n in {1, 2, 4, 8, 16, 32}, nested and evenly spaced, always including the final
  answer token.
- For each output, circuit-tracer runs on the prefix ending there. Its reference graph uses a budget of
  8192 rows; convergence is checked against 4096 rows (node Jaccard at least 0.95), and the budget is
  raised to 16384 when the check fails.
- c_i: the smallest truncation budget meeting fidelity against output i's reference graph.
- |U(n)|: the number of distinct (layer, position, feature) rows across the truncation prefixes of the
  first n outputs.
- beta per trace: the ordinary least squares slope of log |U(n)| on log n over the completed nested
  levels. beta: the mean over the 2 traces.
- c: the median of c_i over all explained outputs.
- The run stops at the last complete level within 60 minutes; incomplete levels are not used.

**Sampled exact columns (G0-B).** 300 sources, stratified by node kind (feature, error, embedding) and
position (first token, early, middle, late), computed exactly with forward-mode passes. k_eff_trace as
above. f_hub: the share of sampled columns whose remainder s95 is above 10 x k_eff_trace.

**Screening recall (G0-B).** For each tau in {1e-2, 1e-3, 1e-4}, Omega_s is the set of targets whose
attention rollout reachability from source s is at least tau. Recall at tau: over the sampled columns,
the sum of squared column energy inside Omega_s divided by the total squared energy (energy-weighted).
Screening passes if recall is at least 0.99 at some tau; the mean |Omega_s| / T is reported at the
largest passing tau.

**Projected speed-up (G0-B).**

    S_proj(L) = c L^beta / (C_short k_eff_trace ln(T_trace / k_eff_trace) + kappa H_est + m')

- L: the median number of generated tokens of the 2 traces.
- T_trace: the number of targets of the full trace.
- m' = 16 (fresh error-estimate probes).
- kappa: forward-mode column cost per backward row, measured on the trace.
- H_est = f_hub x N_trace, with N_trace the number of sources of the full trace.

## 4. Decision table
| Verdict | Condition |
|---|---|
| STOPPED | K0.7 fails anywhere. No verdict; report only. |
| GO-BIG | All of: (a) compressible: s95_rem_frac_T at most 0.02 on at least 2 of 3 prompts, for at least one transcoder set; (b) beta at least 0.6; (c) k_eff_trace at most 3 x k_eff_short(PLT); (d) screening recall at least 0.99 at some tau; (e) S_proj(L) at least 10. Never issued on partial data. |
| NO-GO | s95_rem_frac_T above 0.10 on every prompt of both sets, or k_eff_trace above 10 x k_eff_short(PLT). |
| GO-STRUCTURE | Not NO-GO, (a) holds, and the GO-BIG condition fails only because beta is below 0.6 or S_proj is below 10 (or both). |
| UNDECIDED | Any other combination. The failed conditions are listed and Dawar decides. |

NO-GO is checked before GO-BIG and GO-STRUCTURE. If a G0-B part did not run (or a G0-A set or prompt is
missing), the verdict is computed on what exists and labelled PARTIAL with the missing items listed.
On partial data, NO-GO and GO-STRUCTURE may be issued from the conditions that were evaluated; GO-BIG is
never issued, and a case that would otherwise be GO-BIG is UNDECIDED (PARTIAL).

The rule is implemented in `cct.gate0.gate0_verdict` and applied automatically by
`scripts/gate0_report.py`.

## 5. What is reported
`results/gate0/GATE0_REPORT.md` and `results/gate0/gate0.json`, starting with a three-line plain
summary, then:
- per set and prompt: T, N, K0.7 error and dtype, energy share of A_0, s95 of A and of the remainder,
  s95_rem_frac_T, hub share by kind, c_trunc, m_min, fidelity and recovery time by m;
- k_eff_short and C_short, with flagged prompts;
- the union growth table |U(n)| per trace, beta per trace and mean, c, and the number of outputs
  without a c;
- the columns summary: T_trace, N_trace, K0.7, k_eff_trace, f_hub, screening recall and |Omega_s| / T
  at each tau;
- S_proj(L) with each term of the formula;
- what the numbers mean, the verdict with every condition (value, threshold, pass), and the line
  "Stop: waiting for Dawar's decision (PD-G0)".
Costs are counted in full: mixed passes, forward-mode passes (through kappa), recovery time and the
m' probes. Every result file stores config, seed, git hash, package versions, GPU name and wall time.

## 6. Thresholds (machine-readable copy; must equal configs/prereg_gate0.yaml, tests/test_gate0_decision.py)
```yaml
prereg_version: 1
gate: gate0
status: "DRAFT: thresholds proposed in PROJECT_V2.md Section 7, awaiting Dawar's confirmation"
k07:
  fp32_rel_err_at_most: 1.0e-4
  bf16_rel_err_at_most: 2.0e-2
  on_failure: "stop Gate 0: report the failure, issue no verdict"
fidelity:
  influence_weighted_node_recall_at_least: 0.95
  node_jaccard_at_least: 0.85
  seeds_required: 2
pruning: {node_threshold: 0.8, edge_threshold: 0.98}
g0a:
  model: gemma2_2b
  transcoder_sets: [gemma2_2b_plt, gemma2_2b_clt426k]
  plt_set_suffix: plt
  prompt_ids: [kill_test_00, kill_test_04, kill_test_09]
  regime: D
  sensing_m: [128, 256, 512, 1024]
  sensing_seeds: [0, 1, 2]
  compressible: {s95_rem_frac_T_at_most: 0.02, prompts_required_at_least: 2, sets_required_at_least: 1}
g0b:
  model_set: qwen3_1p7b_plt
  n_traces: 2
  union_levels: [1, 2, 4, 8, 16, 32]
  reference_budget: 8192
  convergence_budget: 4096
  convergence_jaccard_at_least: 0.95
  raised_budget: 16384
  beta_at_least: 0.6
  n_sampled_columns: 300
  k_eff_trace_over_k_eff_short_plt_at_most: 3.0
  hub_s95_over_k_eff_trace_above: 10.0
  screening: {taus: [1.0e-2, 1.0e-3, 1.0e-4], energy_recall_at_least: 0.99}
s_proj:
  m_prime: 16
  at_least: 10.0
  m_min_fallback: 1024
verdicts:
  nogo: {s95_rem_frac_T_above: 0.10, k_eff_trace_over_k_eff_short_plt_above: 10.0}
  go_big: "compressible AND beta AND k_eff_ratio AND screening AND s_proj, on complete data only"
  go_structure: "not NO-GO, compressible holds, and the GO-BIG condition fails only through beta or s_proj"
  nogo_rule: "s95_rem_frac_T above 0.10 on every prompt of both sets, OR k_eff ratio above 10"
  undecided: "any other combination; the failed conditions are listed for Dawar"
  partial: "a G0-B part did not run: verdict on what exists, labelled PARTIAL, never GO-BIG"
  k07_failure: "STOPPED: no verdict"
```

## 7. Deviations log
| Date | What changed | Why | Effect on the decision |
|---|---|---|---|
| | | | |
