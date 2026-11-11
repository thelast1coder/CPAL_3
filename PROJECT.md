# PROJECT.md: Compressed Circuit Tracing (CPAL 2027, idea 3)

Version 1, 10 November 2026. This file is the single source of truth for the project. If another
file disagrees with it, this file wins until Dawar changes it.

## 0. At a glance
- **Venue:** CPAL 2027 Proceedings Track (PMLR, 9 pages). Abstract registration 23 November 2026;
  full paper 5 December 2026 AoE. Fallback: Recent Spotlight Track, 18 January 2027.
- **Place in the lab's plan:** paper 1 is Mirror Surgery; Projection Routing for MoE (MoE_CPAL) is
  the other candidate for the second paper. This project may run in parallel with MoE_CPAL on the
  same VM; the kill tests decide which one (or both) goes to the paper stage.
- **Contribution type:** a structural finding (attribution graphs are compressible), a method that
  exploits it (compressed sensing of the attribution matrix), a certificate, and scaling results.
- **Compute budget:** about 40 to 60 H100-hours in total. Main models are small (Gemma-2-2B,
  Llama-3.2-1B, Qwen3 up to 14B), which suits a shared GPU.
- **First GPU job:** the kill test (PHASE1_KILL_TEST.md, about 4 to 6 GPU-hours), which decides GO,
  AMBER or NO-GO.

## 1. Working title and thesis
**Compressed Circuit Tracing: Attribution Graphs from a Few Mixed Backward Passes** (working title).

Thesis. Circuit tracing (Ameisen et al., 2025) builds an attribution graph by running one backward
pass per target node, thousands per prompt. The resulting matrix of edge weights is compressible:
most features send almost all of their influence to a few targets. Because the local replacement
model is exactly linear, one backward pass seeded with a random signed mixture of many targets
returns the same mixture of their rows. A few hundred such mixed passes, followed by sparse recovery,
reproduce the graph. A small set of dense "hub" columns is computed exactly with forward-mode passes,
and a cheap a posteriori check certifies the result, using the fact that the graph is a DAG (so the
influence series is finite).

## 2. The problem (open statement)
Attribution graphs are the main tool for reading circuits in LLMs (Anthropic's circuit tracing and
"On the Biology of a Large Language Model", 2025; the open `circuit-tracer` library with Neuronpedia).
They are expensive: the open implementation computes one backward pass per target node, caps the
graph at 7,500 feature nodes by default (`max_feature_nodes`, verify current default), and is used
on prompts of tens of tokens. Graph cost grows linearly with the number of targets, and the number of
active features grows with prompt length and model size. So long prompts, large models and large
graph datasets are out of reach, and the cap silently drops part of the graph.

Open question: **does building an attribution graph really need one backward pass per target, or is
the graph's structure simple enough to be measured with far fewer passes?** Nobody has measured the
compressibility of attribution matrices, and nobody has computed them with sensing instead of
row-by-row enumeration (confirm in the P0 literature check).

## 3. Background a reader needs (plain)
**3.1 Transcoders and the replacement model.** A transcoder replaces an MLP with a wide, sparse layer
of interpretable features: an encoder reads the residual stream, a JumpReLU keeps a few features
active, a decoder writes back. Per-layer transcoders (PLT) write to their own layer; cross-layer
transcoders (CLT) write to every later layer.

**3.2 The local replacement model is linear.** For one prompt, freeze the attention patterns, the
layer-norm denominators and the MLP outputs (stop-gradient) at their values on that prompt, and add
an error node per layer and position for what the transcoder misses. With these frozen, the map from
"what each node writes" to "each feature's pre-activation" is linear. So every edge weight
A[t, s] (from source s to target t) is a fixed number, and the whole graph is one matrix A.

**3.3 How graphs are computed today.** Row t of A (all edges into target t) costs one backward pass:
inject the target's encoder vector at its layer and position, backpropagate through the frozen model,
and read the gradient against every source's decoder vector. The open implementation does this in
batches (default 256 rows per batch, verify), first for the top logits, then for features in order
of estimated influence, until the cap. Then it prunes: it computes indirect influence on the logits
and keeps nodes that carry 80% of the influence and edges that carry 98% (defaults, verify).

**3.4 Compressed sensing.** If a vector of length T has only k large entries, about C k log(T / k)
random linear measurements are enough to recover it (Candes, Romberg and Tao 2006; Donoho 2006),
using orthogonal matching pursuit (OMP), CoSaMP or LASSO. Recovery error is controlled by how much
energy lies outside the top k entries.

**3.5 The idea in one line.** Each mixed backward pass is one random linear measurement of every
column of A at once. If columns are compressible, m much smaller than T mixed passes recover all of A.

## 4. Notation
- T: number of target nodes (top logits plus the features whose incoming edges we want).
- N: number of source nodes (token embeddings, active features, error nodes).
- A in R^(T x N): the attribution matrix, A[t, s] = edge from source s to target t.
- a_s in R^T: column s of A (all out-edges of source s). r_t in R^N: row t (all in-edges of target t).
- e_t: target t's read vector (encoder vector for a feature, logit direction for a logit), placed at
  its layer and position. w_s: source s's write vector (activation times decoder vector for a
  feature, the error vector, the token embedding), placed at its layer(s) and position.
- Phi in R^(m x T): sensing matrix (entries +1 or -1, scaled by 1 / sqrt(m)). Y = Phi A in R^(m x N).
- Omega_s: the set of targets that source s can reach (causal mask: later layer, same or later
  position; check the exact rule for CLT write layers and for logits).
- sigma_k(a)_1: the sum of |entries| of a outside its k largest (the "tail").
- s95(a): the number of largest entries holding 95% of the vector's squared norm.

## 5. Mathematics (load-bearing; every formula has a CPU test in tests/)
**5.1 Edges.** For a feature source s at position p_s and a target t at layer l_t, position p_t:

    A[t, s] = sum over write layers l of s:  w_(s,l)^T  J_((l, p_s) -> (l_t, p_t))  e_t,

where J is the Jacobian of the frozen (stop-gradient) model between residual-stream locations. Error
and embedding sources use their own write vectors. This matches Ameisen et al. 2025 (A = a_s w_(s->t)
with the virtual weight w built from decoder, frozen Jacobian and encoder); check the exact
definition used by the pinned circuit-tracer version and match it.

**5.2 Rows, columns and linearity.**
- Row t: one backward pass (vector-Jacobian product, VJP) seeded with e_t, read against every w_s.
- Column s: one forward-mode pass (Jacobian-vector product, JVP) with tangent w_s, read against
  every e_t.
- Because the frozen model is linear, a backward pass seeded with the mixture sum_t phi_t e_t (each
  e_t at its own location) returns sum_t phi_t r_t = phi^T A. So m mixed backward passes give
  Y = Phi A exactly (up to floating point). Its cost does not depend on N.

**5.3 Compressed sensing of the columns.** Column s is measured by y_s = Phi a_s, with a_s supported
on Omega_s. Recover a^_s by masked OMP (up to k_max atoms) or masked LASSO/CoSaMP, batched over all
columns on the GPU. Standard guarantee (restricted isometry, which a Rademacher Phi satisfies with
high probability when m >= C k log(T / k)):

    ||a^_s - a_s||_2 <= C0 sigma_k(a_s)_1 / sqrt(k)   (+ a noise term).

Exact enumeration costs T backward passes. Sensing costs m passes plus recovery compute.

**5.4 Energy screening (free).** With normalised Rademacher Phi, E ||y_s||^2 = ||a_s||^2, so the
measurements already estimate every column's total out-edge energy. Recover only the columns that
together hold 99.5% of the estimated energy (threshold tuned on validation prompts); set the rest to
zero. Most active features have negligible out-edges, so this cuts recovery compute sharply.

**5.5 Hubs.** Some sources may have dense columns (for example token embeddings, error nodes, or
features at the first position), which sparse recovery cannot fit. Flag a column as a hub when its
relative residual ||y_s - Phi a^_s|| / ||y_s|| exceeds rho_hub or its recovered support reaches k_max.
Compute hub columns exactly with forward-mode passes (budget H, batched). Hubs only affect their own
column, so they do not spoil the others.

**5.6 Influence, pruning and the finite Neumann series.** Pruning uses A_bar, the matrix of |A| with
each target's incoming edges normalised to sum to 1, and the indirect influence
B = A_bar + A_bar^2 + ... = (I - A_bar)^(-1) - I (Ameisen et al. 2025). Every edge goes to a strictly
later layer (check for CLTs and logits), so the graph is a DAG, A_bar is nilpotent, and the series
stops after D terms, with D at most the number of layers plus one. Node influence on the logits is
w^T B, where w holds the logit probabilities.
**Perturbation bound.** If the normalised matrix is off by E, with eps = ||E||_inf (largest absolute
row sum), then

    || w^T (B' - B) ||_1  <=  sum_(k = 1..D) ( (1 + eps)^k - 1 )   (about eps D (D + 1) / 2 for small eps).

Checked numerically on 300 random DAGs (it always held; the worst case used 24% of the bound). A
sharper, data-dependent version propagates per-row error bounds through the DAG in layer order;
because the DAG is finite, this gives a per-node influence interval exactly.

**5.7 Certification.**
- **A posteriori check (cheap).** Run m' extra mixed backward passes with a fresh, independent sensing
  matrix Phi'. Then Z = Phi' A - Phi' A^ = Phi'(A - A^) measures the recovery error of every column;
  ||z_s|| estimates ||a_s - a^_s||, and standard randomised-error results give high-probability bounds
  (Halko, Martinsson and Tropp 2011, Section 4.3; check the constant for Rademacher probes, verify).
  Feed these into 5.6 for influence intervals.
- **Exact refinement (optional).** Recompute exact rows for nodes kept after pruning and for nodes
  whose influence interval straddles the pruning threshold. CCT-fast = check only; CCT-certified =
  check plus refinement. The refinement costs one pass per kept node, which caps the saving at T
  divided by the kept count; report both variants.

**5.8 Dual sensing (ablation).** Mix sources instead: m forward-mode passes give A Psi^T, and rows are
recovered (sparse over sources). Use whichever side of A is more compressible; K1 measures both.

**5.9 Group testing for interventions (application).** To find the k features whose ablation changes
a logit most among N candidates in the real (nonlinear) model, ablate random groups and decode with
sparse regression or a group-testing decoder: about C k log(N) intervention runs instead of N.

## 6. Claims (each tied to experiments in EXPERIMENTS.md)
- **C1, the structure:** attribution matrices are compressible. Most of a feature's out-edge energy
  sits on a few targets (s95 much smaller than T), and dense hub columns are few (K1, Z1).
- **C2, the method:** CCT reproduces circuit-tracing graphs (pruned node sets, influence rankings,
  replacement and completeness scores) with at least 10x fewer backward passes at default settings,
  and beats budget-matched truncation (hypothesis; K2, X1).
- **C3, scaling:** cost grows like k log T instead of T. For full graphs (no cap) and long prompts
  the saving approaches 100x, making graphs feasible for long prompts and larger models (hypothesis;
  P3).
- **C4, certificates:** a cheap a posteriori check with the finite-depth influence bound tells the
  user how far the graph can be from the exact one; exact refinement of kept nodes is available
  (P5).
- **C5, application:** group-testing ablations validate graph features in the real model with
  O(k log N) interventions (P4).

**Headline targets (hypotheses; never report them as results):**
- Default circuit-tracer setting (cap 7,500): the same pruned graph (influence-weighted node recall at
  least 0.95) at one tenth of the backward passes or less.
- Full graphs on longer prompts: about 1% of the passes exact enumeration would need.
- Headline sentence if it works: "Anthropic-style attribution graphs at a tenth of the compute; full
  graphs at about one percent."

## 7. The method in one paragraph (what we ship)
Run the forward pass and transcoders as circuit-tracer does. Draw a Rademacher matrix Phi (m rows).
Run m mixed backward passes, each seeded with a signed mixture of all target read vectors, to get
Y = Phi A. Screen columns by their measured energy, recover the kept columns with masked sparse
recovery on the GPU, flag hubs and compute them exactly with forward-mode passes. Run m' fresh probes
to certify, optionally refine kept rows exactly, then prune and export the graph in circuit-tracer's
format so the same viewers and analyses work. No training; no change to the model or transcoders.

## 8. Models and transcoders (IDs from the circuit-tracer documentation, March 2026; verify each)
| Role | Model | Transcoders | Notes |
|---|---|---|---|
| Kill test, main | google/gemma-2-2b | PLT `mntss/gemma-scope-transcoders`; CLT `mntss/clt-gemma-2-2b-426k` | the standard circuit-tracer setting |
| Main, larger CLT | google/gemma-2-2b | CLT `mntss/clt-gemma-2-2b-2.5M` | memory heavy; check size |
| Generality | meta-llama/Llama-3.2-1B | PLT `mntss/transcoder-Llama-3.2-1B`; CLT `mntss/clt-llama-3.2-1b-524k` | gated licence |
| Scaling | Qwen3 0.6B to 14B | circuit-tracer's Qwen3 PLT repositories (find exact IDs) | model-size curve |
| Scale demo (optional) | openai/gpt-oss-20b | CLT `mntss/clt-131k` | only if memory allows |
| Optional | Gemma-3 | `mwhanna/gemma-scope-2-transcoders-circuit-tracer` collection | needs the NNSight backend |
Record in `configs/models/*.yaml` the exact repository IDs, revisions, widths, layer counts and
d_model from the downloaded files, never from memory.

## 9. Metrics
**Fidelity** (against the exact graph with the same targets, same transcoders, same pruning):
- Influence-weighted node recall: exact influence mass of exact-kept nodes that CCT also keeps,
  divided by the exact influence mass of all exact-kept nodes (primary).
- Node Jaccard of the pruned feature-node sets (primary).
- Replacement score and completeness score differences (definitions in Ameisen et al. 2025; use the
  pinned circuit-tracer implementation if it has one).
- Relative Frobenius error on the exact pruned edge set; top-20 and top-50 node overlap; Spearman
  correlation of node influence.
- Case-study check: the canonical circuits (Section 4 of PHASE1) look the same to a reader.
**Structure:** s95 of columns and rows (energy-weighted median), tail ratios, hub count and energy
share.
**Cost:** backward-row equivalents (unit = one row at circuit-tracer's batch size), forward-mode
passes (converted with the measured cost ratio kappa), recovery time (converted with measured time),
certification passes; end-to-end wall-clock time and peak GPU memory.
**Statistics:** medians and interquartile ranges over prompts; 3 sensing seeds per configuration;
bootstrap intervals over prompts for medians.

**Honesty rule.** Speed-ups are always reported both in passes and in end-to-end wall-clock time,
including recovery and certification, against both the exact default and the truncation baseline.

## 10. Baselines
| ID | Method | Notes |
|---|---|---|
| B0 | Exact circuit-tracer, default cap | reference graph (regime D) |
| B0F | Exact, all active features as targets | reference for full graphs (regime F) |
| B1 | Truncation: circuit-tracer with a smaller `max_feature_nodes` (influence-ordered prefix) | **the key budget-matched baseline** |
| B2 | Random rows at equal budget | weak baseline |
| B3 | Low-rank sketch with the same m (randomised range finder) | sparsity prior against low-rank prior |
| B4 | Structural colouring (Curtis-Powell-Reid style) using only the causal zero pattern | shows structure alone is not enough |
| B5 | One-pass total-effect node ranking (gradient times activation to the logit, features unfrozen) | node ranking only, no edges |
| B6 | Plain Jacobian sensing (no masks, no screening, no hubs, no certification) | technique prior art (Choromanski and Sindhwani 2017) |

## 11. Prior work and our delta (complete and verify in P0, `docs/related_work_notes.md`)
| Work | Fact (verify each) | Our delta |
|---|---|---|
| Ameisen et al. 2025, Circuit Tracing (transformer-circuits.pub) | Defines CLTs, the local replacement model, attribution graphs, influence pruning, replacement and completeness scores | Same graphs, measured with far fewer passes |
| Lindsey et al. 2025, On the Biology of a Large Language Model | Case studies built on attribution graphs | We reproduce canonical cases at lower cost |
| circuit-tracer library (v0.5.0, March 2026, PyPI) and Neuronpedia | Open implementation; defaults max_feature_nodes 7,500, batch 256, node threshold 0.8, edge threshold 0.98 | Drop-in faster attribution |
| Choromanski and Sindhwani, NeurIPS 2017, On Blackbox Backpropagation and Jacobian Sensing | Recovers Jacobians from few random queries with compressed sensing and graph colouring | Technique is known. New: applied to attribution graphs, measured compressibility, causal masks, energy screening, hubs, DAG-based certification, scale |
| Curtis, Powell and Reid 1974; Gebremedhin et al. (sparse Jacobian colouring) | Known sparsity pattern lets columns share passes | Baseline B4 |
| Dunefsky et al. 2024 (transcoders); Marks et al. 2024 (sparse feature circuits) | Feature circuits with transcoders and attribution patching | Different object or cost profile; cite |
| Syed et al. 2023 (edge attribution patching) | Scores every edge's effect on one metric with one backward pass | Not the edge matrix between nodes |
| CLT-Forge, arXiv 2603.21014 | Scalable CLT training; uses circuit-tracer for graphs; no reduction in passes | Orthogonal |
| HAGD, arXiv 2601.12879 | Faster search over graphs; attribution still computed in full | Orthogonal |
| Li, arXiv 2611.00606 | Attribution-graph accuracy for TopK SAEs at the margin; same cost | Orthogonal; cite |
| Halko, Martinsson and Tropp 2011 | A posteriori randomised error estimates | Used for certification |

**Delta in one line:** circuit tracing enumerates the attribution matrix one row at a time; we show
the matrix is compressible and measure it with a few mixed passes, with an exact path for dense hubs
and a certificate, which turns a cost linear in the number of targets into one logarithmic in it.

**Honest novelty note:** compressed sensing of Jacobians is not new (2017). The new parts are the
measurement that attribution matrices are compressible, the attribution-specific machinery (masks,
screening, hubs, DAG certificate) and the scale it unlocks. The P0 novelty gate checks that nobody
has already applied sensing to attribution graphs.

## 12. Rules and constraints
- **No training** of models or transcoders. Same model, transcoders, dtype, prompts and pruning
  settings for every method.
- **Ground truth** is always the exact graph computed by the pinned circuit-tracer version (or our
  exact backend after K0 proves it identical).
- **Prompts** are fixed in `configs/prompts/` and committed before any real run; validation prompts
  (for tuning) never overlap test prompts.
- **Pre-registration:** PREREG_KILL_TEST.md is committed before the first real-model run; later
  gated claims get their own PREREG file.
- **Cost accounting** counts every pass and all recovery and certification time (Section 9).
- **Reproducibility:** fixed seeds (sensing seeds 0, 1, 2); every result file stores config, seed, git
  hash, package versions, GPU name and wall time.
- **No invented facts:** numbers, citations, IDs and API details are checked or marked (verify).
- **Laptop:** about 0.5 GB free and no GPU: no model, transcoder or dataset downloads; reuse the system
  PyTorch; tests use a tiny pure-PyTorch backend. Downloading the circuit-tracer wheel without
  dependencies, only to read its source, is allowed.
- **GPU:** one H100 shared with Mirror_CPAL, MoE_CPAL and a CVPR project. Check `nvidia-smi` before
  each job, cap memory per process, queue jobs, never kill processes this project did not start.
- **Secrets:** fresh Hugging Face token on the VM (the earlier one must be revoked); never commit
  tokens, keys, weights or large artefacts. Gemma and Llama licences must be accepted on the account.
- **Writing:** British spelling, plain language, no em-dashes, no emoji.

## 13. Decision register
**Taken**
| ID | Decision | Reason |
|---|---|---|
| D1 | CCT is CPAL idea 3; it may run in parallel with MoE_CPAL; the kill tests decide | Agreed 10 Nov |
| D2 | Build on circuit-tracer (pinned version) as the reference and the backend | It defines the graphs people use; drop-in result |
| D3 | Kill-test model: Gemma-2-2B with PLT and CLT-426k | Standard setting, small, both transcoder types |
| D4 | Primary regime D (same targets as the exact default run); regime F (all active features as targets) secondary in the kill test | Like-for-like first; scaling second |
| D5 | Sense on the target side (mixed backward passes, recover columns); forward dual sensing is an ablation | Matches circuit-tracer's machinery |
| D6 | Rademacher sensing matrix; sparse and structured matrices are ablations | Standard guarantees |
| D7 | Cost unit: one backward row at circuit-tracer's batch size; other costs converted with measured times | Honest, hardware-grounded |
| D8 | Fidelity criteria as in the prereg (influence-weighted node recall >= 0.95 and node Jaccard >= 0.85) | Matches what users read from graphs |
| D9 | Prompts: 32 test prompts (8 categories x 4), 4 validation prompts, 8 long prompts | Small, varied, fixed |
| D10 | dtype decided per transcoder set at K0 (fp32 where it fits under the memory cap, otherwise bf16), recorded, never mixed | Shared GPU |
| D11 | Kill-test thresholds as in PREREG_KILL_TEST.md | Fixed before data |

**Pending (who decides, when)**
| ID | Decision | Who, when |
|---|---|---|
| PD1 | GO, AMBER or NO-GO path; and MoE versus CCT versus both | Dawar, after both kill-test reports |
| PD2 | Headline regime (default-matched or full graphs) | After P2 and P3 |
| PD3 | Title and abstract for registration | Dawar, 22 Nov |
| PD4 | Whether CCT-certified or CCT-fast is the main variant | After P2 |
| PD5 | Which large models fit (Qwen3-14B, gpt-oss-20b) on the shared GPU | Start of P3 |
| PD6 | Whether group testing goes in the main paper | After P4 |
| PD7 | If a paper applying sensing to attribution graphs appears, stop and discuss | Any time; recheck final week |

## 14. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Truncation (B1) is already enough, because pruned graphs keep only a few hundred nodes and circuit-tracer visits the most influential first | K2 requires CCT to beat B1 at equal cost by at least 3x; regime F tests where truncation loses structure |
| Columns are not compressible (a dense noise floor through the residual path) | K1 measures it directly; hubs and screening; dual sensing |
| Recovery compute eats the saving | Energy screening, masks, batched GPU recovery; cost counted in wall-clock (K3) |
| circuit-tracer injects one location per batch row | Extend the injection to sums over locations (bookkeeping only); K0.2 tests linearity |
| Forward-mode passes do not work through the hooks | Emulate JVPs with the double-backward trick, or give hubs a larger recovery budget |
| bf16 numerics break linearity | K0.2 in the chosen dtype; fp32 where it fits |
| Novelty (Jacobian sensing exists) | State it plainly; P0 gate for prior application to attribution graphs |
| Memory for large CLTs on a shared GPU | circuit-tracer offload options; schedule windows |
| Time | Gates with dates; stop rules |

## 15. Timeline (today is 10 Nov 2026)
| Dates | Work |
|---|---|
| 11 to 12 Nov | P0 on the laptop: repo, literature notes and novelty gate, all code, CPU tests, tiny end-to-end run, PREREG committed |
| 13 to 15 Nov | P1 on the VM (alongside MoE_CPAL's kill test): K0, K1, K2, K3, report |
| 15 Nov | Decision (PD1) |
| 16 to 22 Nov | P2 core results, P3 scaling |
| 22 Nov | Freeze title and abstract (PD3); 23 Nov registration closes |
| 22 to 28 Nov | P4 application, P5 theory checks, P6 ablations |
| 27 to 29 Nov | P7 seeds, figures, FINAL_RESULTS.md |
| 24 Nov to 4 Dec | Writing; 5 Dec submit |

## 16. Glossary (plain)
- **Feature:** one unit of a transcoder, meant to stand for one concept.
- **Attribution graph:** nodes are features, tokens, error terms and output logits; an edge says how
  much one node directly pushes another on this prompt.
- **Row / column of A:** all edges into one target / all edges out of one source.
- **Backward pass (VJP) / forward-mode pass (JVP):** one gives a row, the other a column.
- **Mixed backward pass:** a backward pass seeded with a random signed sum of many targets; it returns
  the same signed sum of their rows.
- **Compressible:** most of the vector's size sits in a few entries.
- **Hub:** a source whose out-edges are spread over many targets.
- **Pruning:** keeping only nodes and edges that carry most of the influence on the output.
- **Certificate:** a computed bound on how far the recovered graph can be from the exact one.
