# PHASE1_KILL_TEST.md: the go/no-go experiment (VM, about 4 to 6 GPU-hours)

Thresholds and decision rules live in PREREG_KILL_TEST.md and `configs/prereg_kill_test.yaml`. If
this file and the prereg disagree, the prereg wins.

## 0. What the kill test decides, in plain words
The paper stands on three things being true. The kill test checks each one, in order:
- **K1, is the graph compressible?** In real attribution matrices, does most of each source's
  out-edge energy sit on a small fraction of the targets? If the columns are dense, sensing cannot
  work and the project stops.
- **K2, can a few mixed passes rebuild the graph, and beat the obvious shortcut?** Simulated exactly
  from the stored exact matrices (allowed because mixed passes are linear, which K0 proves), does CCT
  reach the same pruned graph at a tenth of the cost or less, and at a third of the cost of simply
  computing fewer rows (truncation)?
- **K3, does it hold with real passes and real clocks?** With real mixed passes on the GPU, is the
  graph still right and is the end-to-end time at least 5x lower?

---

## 1. VM setup
```bash
git clone <private repo URL> CCT_CPAL && cd CCT_CPAL
export HF_HOME=/path/on/vm/big_disk/hf        # all downloads go here
df -h "$HF_HOME"                               # need about 150 GB free (matrices are stored, Section 5)
nvidia-smi                                     # see what the other three projects are using
bash scripts/vm_setup.sh                       # venv, installs, versions, pytest -q
hf auth login                                  # fresh token (verify the command); Gemma licence accepted
```
All CPU tests must pass on the VM first. Write the PREREG commit hash and free GPU memory into
`results/STATUS.md`.

## 2. Downloads and the dtype decision (D10)
```bash
python scripts/download_models.py --set gemma2_2b_plt
python scripts/download_models.py --set gemma2_2b_clt426k
```
Record repository revisions and sizes. **D10:** for each transcoder set, use fp32 if model plus
transcoders plus working memory fit under the memory cap agreed with the other projects; otherwise
bf16 (or circuit-tracer's offload option). Decide before K0, record in STATUS.md, never mix within a
set.

## 3. K0: sanity on the real setup (all must pass; otherwise fix and rerun)
| Check | Pass condition |
|---|---|
| K0.1 Backend exact rows | our `exact_rows` equals circuit-tracer's adjacency rows on 3 prompts: max relative error <= 1e-5 (fp32) or <= 1e-2 (bf16) |
| K0.2 Linearity of mixed passes | for 5 Rademacher seeds over all targets on 3 prompts, `rows(seed)` equals `seed @ A_exact`: relative error <= 1e-5 (fp32) or <= 2e-2 (bf16) |
| K0.3 Forward-mode columns | for 20 sources (including embeddings and error nodes), `cols(one-hot)` equals the exact column within the same tolerances; record whether true forward mode or the double-backward fallback is used |
| K0.4 Pruning | our pruning of the exact matrix gives exactly circuit-tracer's pruned node and edge sets |
| K0.5 Timing | per-row time at batch 256, forward-mode cost ratio kappa, recovery throughput, exact default graph wall time and peak memory: recorded in the model config (3 repeats, median) |
| K0.6 Truncation equivalence | a real run with `max_feature_nodes = 1024` gives the same graph as the first 1,024 rows of the full run's visiting order; if not, B1 uses real runs at each budget and this is logged |

## 4. Prompts (write and commit in P0; exact strings checked against the sources, verify)
- **Test set, 32 prompts, 8 categories x 4:** two-hop factual recall (including "Fact: the capital
  of the state containing Dallas is"); single-hop factual recall (including "Fact: Michael Jordan
  plays the sport of"); antonyms, English and multilingual (including the French
  'Le contraire de "petit" est "'); two-digit addition (including "calc: 36+59="); acronyms
  (including "The National Digital Analytics Group (N"); rhyme and planning couplets; subject-verb
  agreement; short code completion. Lengths 6 to 30 tokens.
- **Validation set, 4 prompts** from other kinds (for example translation, induction copying, a date,
  a unit conversion). Used only for choosing hyperparameters.
- **Long set, 8 prompts** of 128 to 256 tokens (for example, a WikiText-2 test paragraph followed by a
  factual question about it). Used in regime F and in P3.
The prompt files are committed before any real run. The exact strings of the canonical prompts are
copied from the Anthropic papers or circuit-tracer demos, not typed from memory.

## 5. Exact graphs (`scripts/run_exact_graphs.py`)
For each transcoder set (PLT, CLT-426k):
- **Regime D:** all 32 test and 4 validation prompts with circuit-tracer defaults (record the
  defaults actually used: max_n_logits, desired_logit_prob, max_feature_nodes, batch size, node and
  edge thresholds). Save the full matrix of computed rows (T_D x N), the visiting order, metadata,
  logit weights, the pruned graph, wall time, peak memory.
- **Regime F:** the first prompt of each category (8 prompts) with all active features as targets
  (T_F x N). Save the same items. If one regime-F graph exceeds the time budget (30 minutes), skip it
  and log why.
Storage: fp16 on disk with an fp32 checksum, about 60 GB in total (estimate; check as you go).

## 6. K1: structure (`scripts/analyse_structure.py`, offline)
Per prompt, on the regime-D matrix:
- s95 of every column (out-edges, over the targets in its causal mask) and of every row.
- Energy-weighted median s95 over columns (weights ||a_s||^2), as a fraction of T_D. The same for rows.
- Hub energy share: fraction of total energy in columns with s95 > T_D / 20; hub counts by node kind.
- Plots: `plots/k1_column_energy_curves_<set>.png`, `plots/k1_s95_hist_<set>.png`.
**K1 statistic:** the share of test prompts whose energy-weighted median column s95 is at most
T_D / 100 (GO) or above T_D / 10 (NO-GO). Thresholds in the prereg.

## 7. K2: sensing simulation (`scripts/simulate_sensing.py`, offline on the GPU)
Simulation uses `DenseOperator` on the stored exact matrices; recovery runs on the GPU with the same
code as the real pipeline, and its time is measured.
**Step 1, fix hyperparameters on the 4 validation prompts only:** recovery algorithm (OMP, CoSaMP,
FISTA), k_max, screening fraction q, hub threshold rho_hub, hub budget H, certification probes
m' = 16 (fixed). Rule: the setting that meets the fidelity criteria on all 4 validation prompts at the
smallest median cost, searched over a small grid written in `configs/experiments/kill_test.yaml` before
running. Record the choice.
**Step 2, cost-fidelity curves on the 32 test prompts:**
- CCT-fast (primary) and CCT-certified (secondary) at m in {64, 128, 256, 512, 1024}, sensing seeds
  0, 1, 2.
- B1 truncation at c in {64, 128, 256, 512, 1024, 2048, 4096, T_D} rows (prefixes of the exact
  visiting order).
- B2 random rows and B3 low-rank at the same budgets as CCT (secondary).
- Cost per graph, in row-equivalents: mixed rows + certification rows + kappa x forward-mode passes
  + recovery seconds / row seconds (+ refinement rows for CCT-certified). B1's cost is its row count.
- Fidelity met for a prompt: influence-weighted node recall >= 0.95 **and** node Jaccard >= 0.85
  against the exact pruned graph, for at least 2 of the 3 seeds.
**K2 statistics, per transcoder set:**
- For each method, c75 = the smallest budget on its grid at which at least 75% of the 32 test prompts
  meet fidelity; its cost is the median cost over prompts at that budget.
- S_exact = median over prompts of T_D (rows of the exact default run) divided by CCT's c75 cost.
- S_B1 = B1's c75 cost divided by CCT's c75 cost.
- If CCT never reaches 75% on its grid, K2 is NO-GO for that set.
**Secondary, regime F (8 prompts):** the same statistics against the exact full graphs, with B1 as
prefixes of the full run's visiting order (the default cap is one point on that curve).

## 8. K3: real mixed passes and real clocks (`scripts/run_real_cct.py`)
For each transcoder set, on 8 test prompts (the first of each category): run the full CCT pipeline
with real mixed passes through the circuit-tracer backend at CCT's c75 budget and the hyperparameters
from Step 1, and rerun the exact default graph in the same session (3 repeats each, median wall time).
**K3 statistics:** number of prompts meeting fidelity; relative Frobenius difference between the real
and the simulated recovered matrices (same seed); median end-to-end wall-clock speed-up (exact
default time / CCT time, everything included).
If K2 ended as GO-F (prereg upgrade path), K3 uses the regime-F prompts against exact full graphs.

## 9. Decision (applied automatically by `scripts/make_kill_report.py` from the prereg YAML)
Per transcoder set, each check is GO, AMBER or NO-GO (thresholds in the prereg). Overall:
- **GO:** K2 GO (or GO-F) on at least one set and not NO-GO on the other; K3 GO on that set; K1 not
  NO-GO on that set.
- **NO-GO:** K1 NO-GO on both sets, or K2 NO-GO on both sets, or K3 NO-GO.
- **AMBER:** anything else.

## 10. What happens after each outcome
- **GO:** stop and report. After Dawar confirms (and decides MoE against CCT, PD1), start P2.
- **AMBER:** stop and report with a diagnosis. Proposed 48-hour fix window, only after Dawar agrees:
  (a) if columns are dense but rows are sparse, switch to dual sensing (forward-mode mixtures,
  recover rows) and rerun K2; (b) if recovery time eats the saving, use sparse sensing matrices and
  stronger screening; (c) if truncation is the problem in regime D, move the headline to regime F and
  long prompts. Then decide again.
- **NO-GO:** stop. One-page post-mortem: what the structure of attribution matrices actually is. If
  K1 is GO but K2 is NO-GO only because of truncation, record that "pruned graphs need only the
  most influential rows", which is itself a useful finding for the community.

## 11. Report (`results/KILL_REPORT.md`, generated, then edited for clarity)
1. Three-line plain summary and the verdict.
2. Setup: model, transcoders, dtype (D10), revisions, circuit-tracer version and defaults, prompt
   files, prereg commit hash, deviations.
3. K0 table, including kappa and per-row time.
4. K1: column and row s95 statistics, hub share, plots.
5. K2: the hyperparameters chosen on validation; cost-fidelity curves for CCT, B1, B2, B3
   (`plots/k2_curves_<set>.png`); c75 table; S_exact and S_B1; regime F results.
6. K3: real against simulated, fidelity count, wall-clock table.
7. Decision table with every cell, and the verdict.
8. What the numbers mean, in plain words; surprises; next steps.
Also write `results/kill_test.json` with every number and its run record.

## 12. Schedule and parallel work
| Step | Estimate |
|---|---|
| Downloads | 0.2 to 0.5 h (check the CLT-426k size first) |
| K0 | 0.5 h |
| Exact graphs, regime D (72 graphs) | 1 to 2 h |
| Exact graphs, regime F (16 graphs) | 1 to 2 h |
| K1 and K2 (GPU recovery simulations) | 1 to 2 h |
| K3 | 0.5 h |
Run K1 and K2 for the PLT set while the CLT exact graphs are computing. This project's GPU memory use
is small (Gemma-2-2B), so it can share the GPU with MoE_CPAL's kill test; agree the order with the
other jobs first. These are estimates; log the real times.
