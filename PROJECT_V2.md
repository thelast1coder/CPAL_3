# PROJECT_V2.md: Compressed Circuit Tracing, big-paper scope (CPAL 2027)

Version 2, 15 November 2026. This file supersedes PROJECT.md (v1, 10 November) wherever the two
disagree. PHASE1_KILL_TEST.md stays valid for the short-prompt kill test (K0 to K3), except where
Section 8 here changes it. BRAINSTORM.md is background only.

Read Sections 0 to 2 before writing any code. Read Section 19 for the ordered task list.

---

## 0. What changed from v1

| Area | v1 | v2 | Why |
|---|---|---|---|
| Headline | Same graph at 10x fewer passes, one short prompt, default cap | Attribution for every token of a reasoning trace at roughly the cost of one graph | In the single-graph setting, truncation and sensing cost about the same from first principles (Section 2.3) |
| Key baseline | Truncation per graph (B1) | Union-truncation across all explained tokens, with rows cached and reused (B1U) | A row is the same whichever output it serves, so truncation must get credit for reuse |
| Structure claim | Columns are compressible | A = closed-form local part + sparse, attention-routed remainder; a measured atlas plus an explanation of why | Turns an observation into an explanatory finding, and removes the dense floor at zero cost |
| Method | Rademacher signs + OMP, hubs, global certificate | Exact local part (zero passes) + sketched remainder (count-sketch for large T, Rademacher + OMP for small T) + reachability screening + hubs + fresh-probe error estimate | Must scale to T, N of 1e5 to 1e6 nodes; the global bound is too loose to lead with |
| Science | Optional | Required: a mechanistic dependency map of reasoning traces, validated by interventions in the real model | A speed-up alone is incremental; a result about reasoning is what gets cited |
| First GPU job | Short-prompt kill test | Gate 0: short-prompt structure + whole-trace union growth + sampled exact columns at trace length | Test the flagship's premise first, not last |
| Demoted or dropped | | Regime D as headline (kept as an honest comparison); global certificate (Tier B); compressive principal component pursuit (only if hubs remain large after the local part is removed); group testing, dual sensing, dataset-scale motifs (follow-up) | Scope and time |

---

## 1. Why this can be a big paper, and the honest limits

A paper gets cited by the people who own a method when it does three things: it removes a limitation
they or their users name, it explains something about their object that they did not explain, and it
ships inside the tools they use. v2 is designed around all three.

1. **It removes a named barrier.** Researchers now apply attribution graphs to chain-of-thought
   reasoning, and they say cost is what stops them (Section 2.1). CCT-v2 targets graphs for every
   token of a trace.
2. **It explains the object.** Attribution matrices split into a part that is exact and free (the
   residual-stream path) and a remainder whose spread follows attention. That is a statement about why
   circuit-tracing graphs look the way they do, not only how to compute them faster.
3. **It ships as a drop-in.** It produces the same graph format as circuit-tracer, so Neuronpedia
   viewers and existing analyses work unchanged.
4. **It produces a result about reasoning that needs full coverage:** which steps of a trace the
   model actually uses, which steps are decorative, and where the answer is decided. These are
   validated by interventions in the real model.

Honest limits, stated once:
- No design guarantees citations. This plan targets the conditions that earn them.
- The big version rests on one measured number, the **reuse exponent β** (Section 2.3). If β is
  small, union-truncation already handles whole traces, and the paper shrinks to the structure
  finding. Gate 0 measures β first.
- There are 20 days to submission. The plan is tiered (Section 13), and Tier A alone must make a
  complete paper.

---

## 2. Problem, question, thesis

### 2.1 The open problem, with evidence (verify each quote before citing)
- Attribution graphs (Ameisen et al. 2025; Lindsey et al. 2025) cost one backward pass per target
  node. circuit-tracer's defaults (checked on PyPI, November 2026): batch_size 256,
  max_feature_nodes 7,500, desired_logit_prob 0.95.
- **CRV** (Zhao et al., "Verifying Chain-of-Thought Reasoning via Its Computational Graph", ICLR 2026,
  arXiv 2510.09312, Meta FAIR) builds one attribution graph per reasoning step on Llama-3.1-8B-Instruct
  with per-layer transcoders and circuit-tracer. It calls its approach too computationally intensive
  to be a practical, drop-in verifier.
- **CIE** (arXiv 2605.25603, "Detecting Unfaithful Chain-of-Thought via Circuit-Guided
  Internal-External Discrepancy") reuses CRV's transcoders. It selects a subset of tokens to trace
  because full-trace construction is computationally expensive, cutting traced tokens by about 62 to
  69% relative to CRV.
- **DifFRACT** (arXiv 2606.15796, diffusion models) states that a backward pass per feature vertex is
  infeasible and uses budgeted greedy expansion, which is truncation.
- **GraphGhost** (arXiv 2510.08613) traces token-level graphs on Qwen3-0.6B reasoning.

So the field wants graphs over whole reasoning traces, and every group either traces a few tokens or
pays heavily for many.

### 2.2 The question
What does it really cost to know a model's full attribution structure over a long generation? Is the
cost inherently linear in the number of nodes, or does the structure of attribution matrices allow
far less?

### 2.3 First principles: the arena where sensing can win
The two methods exploit different sparsity:
- **Truncation** exploits "only a few targets matter for this output". Its cost is c, the number of
  rows needed to reproduce one output's pruned graph.
- **Sensing** exploits "each source pushes only a few targets". Its cost is m ≈ C k log(T/k), plus
  recovery.

**For one graph, both land in the hundreds to low thousands.** For example, with k = 50,
T = 7,500 and C between 2 and 4 (illustrative values, not measurements), m is about 500 to 1,000.
So a 10x win on a single graph is unlikely. This is not the arena.

**For n explained outputs, the costs diverge:**
- Truncation pays |U(n)|, the size of the union of the rows each output needs. Rows are reused, so
  this is not n × c.
- Sensing pays its cost once, for all n outputs, because every mixed pass serves every row.

If |U(n)| ≈ c n^β, the speed-up is S(n) ≈ c n^β / Cost_CCT. Illustration with 300 explained tokens,
c = 500 and Cost_CCT = 3,000 row-equivalents:
- β = 1 gives |U| = 150,000 rows, so S = 50x;
- β = 0.5 gives |U| ≈ 8,700 rows, so S ≈ 3x.

**β decides the paper.** β is also a scientific quantity in its own right: how much internal
machinery a model reuses across its own reasoning.

### 2.4 Thesis
Attribution matrices split into an exact, closed-form local part and a sparse, attention-routed
remainder. Because of that, the attribution structure of an entire generation can be measured at
roughly the cost of one graph, which gives graphs for every token of a reasoning trace. Those graphs
reveal which reasoning steps the model actually uses, and real-model interventions confirm it.

---

## 3. Notation added to v1 Section 4
- L: number of tokens in the trace (prompt plus generated). n: number of explained output tokens.
- U(n): union of the rows that union-truncation needs to reproduce the pruned graphs of n outputs.
  β: exponent fitted from |U(n)| ≈ c n^β.
- A_0: the residual-path part of A (closed form). A_attn = A − A_0: the paths through at least one
  attention head.
- r_t: target t's effective read vector. For a feature, its encoder vector times the frozen
  normalisation gain and scale at its layer and position. For a logit, the unembedding row times the
  frozen final normalisation.
- w_(s,l): source s's write vector into layer l (activation times decoder, error vector, or token
  embedding).
- ρ(q ← p): attention reachability from position p to position q between two layers.
  Ω_s: candidate targets of source s after reachability screening.
- k_eff: energy-weighted median s95 of the columns of A_attn, restricted to Ω_s.

---

## 4. Theory (load-bearing only; every result gets a CPU test or a numerical check)

**T1. Linearity of mixed passes (from v1).**
- A backward pass seeded with Σ_t φ_t r_t (each read vector placed at its own location) returns
  Σ_t φ_t row_t.
- So m mixed passes give Y = Φ A exactly, up to floating point. The cost does not depend on N.

**T2. Path decomposition and the closed-form local part (new; derive against circuit-tracer's code).**
- In the local replacement model, the MLPs are stop-gradiented, and their effects are carried by
  nodes. So between a source's write point and a target's read point, the only live paths are the
  residual stream (identity) and attention heads with frozen patterns.
- Expand the Jacobian over paths, in the style of Elhage et al. 2021: J = J_0 + J_1 + J_2 + ...
  - J_0 is the residual-only path. It exists only when source and target sit at the same position.
  - J_r contains the paths through exactly r attention heads.
- So A = A_0 + A_attn, with

      A_0[t, s] = 1[pos(s) = pos(t)] × Σ over write layers l of s that come before t's read point:  r_t^T w_(s,l)

- A_0 is one dense matrix product per position (targets at that position × sources at that
  position). It costs nothing like a backward pass, and it is exact.
- Match circuit-tracer's conventions exactly:
  - CLT write layers;
  - where Gemma-2's pre-MLP and post-attention norms sit;
  - which input the transcoder reads (normalised or raw);
  - logit targets.
- Numerical check (K0.7): run the frozen backward pass with every attention output zeroed. It must
  equal the closed-form A_0 (relative error ≤ 1e-4 in fp32). A − A_0 must equal the attention-only
  part.
- Prediction (H-local): A_0 carries a large share of the edge energy and most of the dense,
  small-interference edges. These are dense because r_t^T w_s is rarely exactly zero.

**T3. Why the remainder should be sparse (hypothesis H-struct; explanatory claim C2).**
- For a path through one head h at layer l, the block between positions q and p is
  α^(l,h)[q, p] × (r_t^T M^(l,h) w_s), where M^(l,h) is the frozen OV map with its normalisations.
- Multi-head paths multiply attention weights along the path.
- So a source's out-edges across positions are bounded by attention reachability ρ(q ← p):
  - concentrated attention means few reachable positions, which means sparse columns;
  - positions that act as attention sinks (for example, the first token) predict the hubs.
- Testable prediction: per-source column spread correlates with the entropy of reachability from its
  position.

**T4. Recovery guarantees (two decoders).**

*D1, for small T (short prompts):*
- Rademacher Φ (entries ±1/√m), masked to Ω_s.
- OMP or CoSaMP per column.
- Guarantee: ||â − a||_2 ≤ C0 σ_k(a)_1 / √k when m ≥ C k log(T/k) (Candès, Romberg and Tao 2006;
  Needell and Tropp 2009).

*D2, for large T (whole traces): count-sketch (Charikar, Chen and Farach-Colton 2002).*
- Setup:
  - R repetitions × B buckets;
  - a hash h_r(t) ∈ [B] and a sign σ_r(t) ∈ {±1} for each target.
- Measurement:
  - mixed pass (r, b) is seeded with Σ over {t : h_r(t) = b} of σ_r(t) r_t;
  - m = R × B passes.
- Estimate: â[t, s] = median over r of σ_r(t) Y[(r, h_r(t)), s].
- Guarantee (verify constants): with B ≥ c k and R = O(log(#estimated entries / δ)), every estimate
  is within ||a_tail(k)||_2 / √k of the truth with probability ≥ 1 − δ.
- Decode cost: R lookups per candidate pair (s, t ∈ Ω_s). There is no dense T × N product, which
  is what makes 1e5 to 1e6 nodes feasible.

*Both decoders:*
- **Subtract the known local part first:** Y_rem = Y − Φ A_0. This needs no passes.
- **Energy screening:** under normalised Rademacher signs, E||y_s||² = ||a_s||². For count-sketch,
  the sum of squares within one repetition is unbiased for ||a_s||². Sources below the energy
  threshold are skipped.

**T5. Cost law and order-optimality.**
- Cost_CCT = m + m' + κ H + recovery_time / row_time, in row-equivalents.
  - It does not depend on n.
  - κ is the measured cost of one forward-mode pass relative to one backward row.
- Cost_trunc(n) = |U(n)|. Cost_exact_full = T.
- Crossover: n* = (Cost_CCT / c)^(1/β).
- Lower bound: any non-adaptive linear measurement scheme needs Ω(k log(T/k)) measurements for stable
  recovery of one k-sparse column (Ba, Indyk, Price and Woodruff 2010; verify).
  - CCT's measurements serve all columns at once, so it is order-optimal in that class.
  - Truncation cannot go below |U(n)|, because it computes every row it uses.
- This is the predictive core of the paper. The fitted law must match measured costs in E3.

**T6. Error estimation (Tier A) and influence intervals (Tier B).**
- Tier A: m' fresh probes with an independent Φ' give Z = Φ'(A − Â), so ||z_s|| estimates
  ||a_s − â_s|| (Halko, Martinsson and Tropp 2011, Section 4.3; verify the constant for Rademacher
  probes).
- Tier B: per-node influence intervals.
  - Propagate column-wise ℓ2 error estimates through the DAG in layer order, using the Cauchy-Schwarz
    form ||w^T E||_1 ≤ ||w||_2 Σ_s ||E_(:,s)||_2.
  - Account for the row normalisation in A_bar.
  - If no useful bound exists by 24 November, drop the certificate claim. Report the empirical error
    estimates and exact-row refinement instead.
- The global bound ε D(D+1)/2 from v1 is a factor of about 378 at 26 layers (verify the layer count).
  Report it only as a remark.

**T7. Whole-trace exactness.**
- Causal attention, per-position normalisation and per-position transcoders mean that freezing on the
  full trace gives, for every position q, the same frozen values as freezing on the prefix ending at
  q.
- So every output's graph is the restriction of one table A to positions ≤ q.
- Numerical check (K0.8): for 3 positions, the circuit-tracer graph of the prefix equals the slice of
  the full-trace table.

---

## 5. The method we ship (CCT-v2)

1. Run the forward pass over the full trace. Compute transcoder activations, then freeze and record
   attention patterns and normalisation scales (circuit-tracer machinery).
2. Define targets and sources.
   - Targets: every active feature at every position, plus one logit target per generated position
     (the token actually generated; optionally the top-k).
   - Sources: embeddings, active features, error nodes.
3. Compute A_0 exactly (T2).
4. Compute reachability ρ by attention rollout across layers, weighted by OV norms (Abnar and
   Zuidema 2020; verify). This uses L × L matrices per layer and is cheap. Set the candidate sets
   Ω_s at threshold τ.
5. Draw the sketch: D1 for short prompts, D2 for whole traces. Run m mixed backward passes and form
   Y_rem = Y − Φ A_0.
6. Screen sources by measured energy.
7. Decode the remainder columns over Ω_s.
8. Hubs: flag a column if it has a large decode residual or carries heavy mass in many buckets.
   Compute flagged columns exactly with forward-mode passes (torch.func.jvp, or the double-backward
   trick), up to a hub budget H.
9. Error estimate: run m' fresh probes (T6).
10. Optional refinement: compute exact rows for nodes kept in any pruned graph (the CCT-certified
    variant). Report it separately from CCT-fast.
11. Graphs on demand: for any root (the logit at any position, or any feature: "why did this feature
    fire here?"), restrict Â to positions ≤ the root's position, prune with circuit-tracer's rule,
    and export in circuit-tracer's JSON format.

Engineering:
- extend circuit-tracer's injection to multi-location seeds (K0.2 linearity check);
- store Y as m × N in fp16, chunked over N;
- store Â sparse, in COO format by column;
- run per-output influence with sparse triangular operations in layer order.

---

## 6. Claims, the evidence each needs, and the condition that kills each

| ID | Claim | Evidence | Kills it |
|---|---|---|---|
| C1 | Attribution matrices split into an exact local part A_0 and a compressible remainder; measured across ≥ 3 models, both transcoder types, lengths 16 to 512 | K0.7, G0-A, E1 | Remainder s95 > 10% of the candidate set on most prompts |
| C2 | The remainder's spread tracks attention reachability; hubs sit at attention sinks | E1 | No correlation: keep C1 as descriptive, drop "explain" |
| C3 | Whole-trace cost of CCT is roughly flat in the number of explained tokens; union-truncation grows like n^β; the cost law predicts the crossover | G0-B, E2, E3 | β < 0.6 or the measured crossover is beyond trace length |
| C4 | CCT gives per-token graphs that meet fidelity for ≥ 90% of sampled tokens at ≥ 10x lower cost per explained token than union-truncation | E2 | Below 3x: move to GO-STRUCTURE |
| C5 | Dependency maps of reasoning traces identify load-bearing and decorative steps, and real-model interventions confirm the split better than cheap baselines | E5 | Interventions do not separate the two groups. Report it honestly as a finding about graph validity on reasoning |
| C6 | Drop-in: same graphs, same format, released code | Export tests, Neuronpedia render | |
| C7 (Tier B) | Influence intervals and adaptive stopping | E8 | No useful bound: drop the claim |

**Headline targets are hypotheses. Never report them as results.**
- ≥ 10x fewer row-equivalents per explained token than union-truncation on traces of 200 to 400
  tokens.
- ≥ 100x fewer than exact full enumeration.
- Regime D (single short prompt, default cap) is reported honestly, whatever it shows.

---

## 7. Gate 0: the first GPU session (≤ 3 H100-hours; preregister first)

Write PREREG_GATE0.md with the thresholds below and commit it before any real-model run. The
thresholds are proposals; Dawar confirms them in the prereg.

**G0-A. Short prompts** (Gemma-2-2B; PLT and CLT-426k; the 3 canonical prompts; about 1 hour).
1. Build exact graphs and save the full row matrix, as in PHASE1 Section 5, regime D.
2. Compute the closed-form A_0 and check it against the attention-zeroed backward pass (T2).
3. Measure:
   - the energy share of A_0;
   - the energy-weighted median s95 of columns, for A and for A − A_0, as fractions of T and of
     |Ω_s|;
   - hub energy share by node kind.
4. Truncation curve: find the smallest c that meets fidelity (influence-weighted node recall
   ≥ 0.95 and node Jaccard ≥ 0.85).
5. Simulate D1 on A − A_0 at m ∈ {128, 256, 512, 1024}: find the smallest m that meets fidelity
   once A_0 is added back, and time the recovery. This gives the practical constant C and
   Cost_CCT(short).

**G0-B. One whole trace setting** (Qwen3-1.7B with thinking, PLT; 2 traces of 200 to 300 generated
tokens; about 1.5 hours).
1. Generate the traces with a fixed seed. Commit the text and the generation settings.
2. Union growth.
   - Explained outputs: n ∈ {1, 2, 4, 8, 16, 32} positions, evenly spaced, always including the
     final answer token.
   - For each output, run circuit-tracer on the prefix ending there. Its reference graph uses a
     budget of 8,192 rows; check convergence against 4,096 (node Jaccard ≥ 0.95, otherwise raise the
     budget).
   - Record the smallest influence-ordered prefix that reaches fidelity.
   - |U(n)| is the number of distinct (layer, position, feature) rows across those prefixes. Fit β.
   - Stop at the largest n finished within 60 minutes.
3. Sampled exact columns.
   - Take 300 sources, stratified by node kind (feature, error, embedding) and by position (first
     token, early, middle, late).
   - Compute their exact columns with forward-mode passes.
   - Measure s95 for A and for A − A_0, and estimate k_eff at trace length.
4. Screening recall. For τ ∈ {1e-2, 1e-3, 1e-4}, measure the share of each sampled column's energy
   that falls inside Ω_s.
5. Projected speed-up: S_proj(L) = c L^β / (C k_eff ln(T / k_eff) + H_est + m'), with C taken from
   G0-A.

**Decision**, reported in results/GATE0_REPORT.md. Then stop and wait for Dawar.

| Verdict | Condition |
|---|---|
| GO-BIG | G0-A: remainder s95 ≤ 2% of T on ≥ 2 of 3 prompts, for at least one transcoder set. G0-B: β ≥ 0.6, k_eff(trace) ≤ 3 × k_eff(short), screening recall ≥ 0.99 at some τ, and S_proj(L) ≥ 10. |
| GO-STRUCTURE | Compressibility holds, but β < 0.6 or S_proj < 10. The paper becomes: the structure finding + uncapped full graphs + what the default cap hides. Dawar decides whether CPAL still gets it. |
| NO-GO | Remainder s95 > 10% of T on all prompts and both sets, or k_eff(trace) > 10 × k_eff(short). Write a one-page post-mortem on what attribution matrices actually look like. |

---

## 8. Experiments after Gate 0 (GO-BIG path)

**K. Short-prompt kill test** (PHASE1 K0 to K3, with these changes):
- add K0.7 (closed-form A_0), K0.8 (whole-trace slice exactness, T7) and K0.9 (count-sketch decoder
  unit tests on synthetic sparse matrices with a known tail);
- report regime D results honestly, with no headline.

**E1. Structure atlas (C1, C2).**
- Models and transcoders: Gemma-2-2B (PLT, CLT-426k), Llama-3.2-1B (CLT), Qwen3-0.6B, 1.7B and 4B
  (PLT).
- Lengths 16, 64, 256 and 512, with 8 prompts each. Long sequences use sampled exact columns only.
- Report:
  - A_0 energy share;
  - remainder k_eff against length and model size;
  - hub share by node kind and position;
  - compressibility layer by layer;
  - correlation between column spread and reachability entropy (C2).

**E2. Whole-trace cost and fidelity (C3, C4).**
- Qwen3-1.7B and 4B, 24 traces each (8 arithmetic word problems, 8 multi-hop factual questions,
  8 logic puzzles), 150 to 400 tokens.
- Methods: CCT-v2 (D2), B1U, B6, B7, B8, and sampled exact references (Section 10).
- Metrics:
  - cost and wall-clock per explained token;
  - fidelity on 16 sampled outputs per trace;
  - edge error on sampled exact rows and columns;
  - peak memory;
  - coverage.

**E3. Scaling (C3).**
- Cost per explained token against trace length (64 to 512) and model size (0.6B to 4B, plus 8B if
  it fits).
- Fit the exponents, compare them with the T5 cost law, and report the crossover n*.

**E4. Ablations.**
- without A_0 subtraction;
- without screening;
- D1 against D2;
- m sweep;
- hub budget;
- 3 sketch seeds;
- dtype.

**E5. Reasoning science (C5, Tier A).**

*The maps:*
- Split each trace into steps (sentences or lines).
- Step dependency D[j, i] is the influence on step j's token logits from nodes at step i's
  positions, computed from Â with the finite DAG series.
- Answer dependency is the same quantity, for the final answer token.
- Classify steps as load-bearing (downstream dependency above a threshold set on validation traces)
  or decorative.

*Validation, in the real model (not the replacement):*
- Replace a step with neutral filler of equal token length, or delete it.
- Teacher-force the rest of the trace and measure the change in the answer's log-probability.
- Prediction: perturbing load-bearing steps changes the answer far more than perturbing decorative
  ones. Report effect sizes, and AUROC of graph dependency as a predictor of intervention effect.
- Add a mistake-insertion perturbation in the style of Lanham et al. 2023 (verify).

*Cheap baselines the maps must beat:*
- S1: attention rollout dependency;
- S2: gradient × input saliency per step;
- S3: selected-token graphs in the CRV or CIE style.

If S1 or S2 predict interventions as well as CCT's maps, say so plainly. That is a finding too.

**E6. CRV at scale (Tier B).**
- Use Llama-3.1-8B-Instruct with CRV's released transcoders and step-labelled data (verify
  availability, licence and fit on the GPU).
- Build graphs for every step at CCT's cost.
- Report the cost against per-step construction, and check whether CRV's structural features keep
  their predictive power (AUROC within 0.02).

**E7. Answer commitment (Tier B).**
- Find the earliest position where the final answer's features already carry influence (the
  mechanistic decision point), and compare it with where the text states the answer.
- Validate with early-answering interventions.

**E8. Adaptive stopping and influence intervals (Tier B, C7).**

---

## 9. Baselines

| ID | Method | Role |
|---|---|---|
| B0 | Exact circuit-tracer at the default cap | Regime D reference |
| B0F | Exact, uncapped (short prompts only) | Full-graph reference |
| B1 | Truncation, single graph | Regime D comparison |
| **B1U** | **Union-truncation: influence-ordered rows per output, with rows cached and reused across outputs** | **The key baseline for the headline** |
| B2 | Random rows at equal budget | Weak baseline |
| B3 | Low-rank sketch at equal budget | Sparsity prior against low-rank prior |
| B4 | Colouring on the causal zero pattern | Shows structure alone is not enough |
| B5 | One-pass total-effect node ranking | Nodes only, no edges |
| B6 | Plain Jacobian sensing (no A_0, no screening, no hubs) | Prior technique (Choromanski and Sindhwani 2017) |
| B7 | A_0 only (zero passes) | How far the free part goes |
| B8 | Selected-token tracing, CRV or CIE style | Coverage against cost |

---

## 10. Metrics and the ground-truth protocol for whole traces

- **Reference graphs for sampled outputs:** circuit-tracer on the prefix, with a budget of 16,384
  rows. Check convergence against 8,192 (node Jaccard ≥ 0.95); raise the budget if needed. Report
  that references at long lengths are themselves budgeted.
- **Edge error:**
  - exact rows for 2,000 random targets, stratified by layer and position;
  - exact columns for 500 random sources;
  - report relative Frobenius error on the pruned edge set.
- **Fidelity per sampled output:** influence-weighted node recall ≥ 0.95 and node Jaccard ≥ 0.85
  (as in v1); also top-20 overlap and Spearman correlation of influence.
- **Cost:**
  - unit: row-equivalents (one backward row at batch 256, at the same sequence length);
  - forward-mode passes converted with the measured κ;
  - recovery time converted with the measured row time;
  - also end-to-end wall-clock and peak memory;
  - all divided by the number of explained outputs.
- **Coverage:** the share of generated tokens that have a graph.
- **Replacement and completeness scores per trace:** if error nodes dominate on thinking text, say so,
  because it limits what any graph can show.
- **Statistics:** medians with interquartile ranges over traces; 3 sketch seeds; bootstrap intervals.

---

## 11. Models, transcoders, data (record exact IDs and revisions from the downloads, never from memory)

| Role | Model | Transcoders | Notes |
|---|---|---|---|
| Gate 0, structure | google/gemma-2-2b | PLT `mntss/gemma-scope-transcoders`; CLT `mntss/clt-gemma-2-2b-426k` | Standard setting |
| Structure | meta-llama/Llama-3.2-1B | CLT `mntss/clt-llama-3.2-1b-524k` | Gated licence |
| Whole traces (main) | Qwen3-1.7B, Qwen3-4B, thinking mode | circuit-tracer's Qwen3 PLTs | Verify which sizes exist and which checkpoint they were trained on |
| Scaling | Qwen3-0.6B, Qwen3-8B | Same family | 8B only if memory allows |
| Tier B | Llama-3.1-8B-Instruct | CRV transcoders | Verify release, licence, size |
| Follow-up only | openai/gpt-oss-20b | CLT `mntss/clt-131k` | Not in this paper unless everything else is done |

Data:
- Commit prompt files before any run.
- Generate traces once with fixed seeds, then commit the text and the generation settings (thinking
  mode, temperature, maximum tokens).
- Validation traces (used for thresholds) never overlap test traces.

---

## 12. Compute budget (pending decision PD-C)

| Block | H100-hours |
|---|---|
| Gate 0 | ≤ 3 |
| K tests | ≤ 4 |
| E1 | ≤ 6 |
| E2 | ≤ 10 |
| E3 | ≤ 5 |
| E4 | ≤ 3 |
| E5 | ≤ 5 |
| Tier B | ≤ 8 |
| **Total** | **≤ 44** |

v1 said 40 to 60 hours and the 15 November briefing said 15 to 20. If the cap is 20 hours:
- E1 shrinks to 3 models × 2 lengths;
- E2 shrinks to 12 traces on Qwen3-1.7B only;
- E3 drops the model-size axis;
- Tier B is dropped.

---

## 13. Timeline and tiers (today is Sunday 15 November 2026)

| Dates | Work |
|---|---|
| 15 to 16 Nov | Gate 0 code (A_0, union growth, forward-mode columns, reachability) and CPU tests; commit PREREG_GATE0; run G0-A and G0-B |
| 17 Nov | Gate 0 report; Dawar decides GO-BIG, GO-STRUCTURE or NO-GO |
| 17 to 20 Nov | Build CCT-v2 (injection, D2, screening, hubs, export); K0 to K3; first end-to-end run on 2 traces |
| 21 Nov | E2 pilot on 4 traces; fixes |
| 22 Nov | Freeze title and abstract (two pre-written variants, BIG and STRUCTURE). Registration 23 Nov: verify the date on OpenReview, and whether the abstract can be edited until 5 Dec |
| 22 to 28 Nov | E1, E2, E3, E5 (Tier A) |
| 26 Nov | Go or no-go for Tier B, based on slack |
| 29 Nov | Results freeze; FINAL_RESULTS.md |
| 24 Nov to 4 Dec | Writing in parallel; full draft 2 Dec; internal adversarial review 4 Dec |
| 5 Dec | Submit |

- **Tier A (must ship):** Gate 0; T1, T2, T4, T5, T7; C1, C3, C4, C5, C6; E1 (reduced if needed),
  E2, E3, E4, E5.
- **Tier B (only if ahead by 26 Nov):** C2 in full, C7, E6, E7, E8, Qwen3-8B.
- **Tier C (follow-up paper):**
  - dataset-scale circuit statistics;
  - gpt-oss-20b;
  - structured or learned sensing;
  - attribution through attention patterns (QK);
  - group-testing interventions.

**Fallback.** If C4 or C5 is not solid by 29 November, submit the GO-STRUCTURE version. Keep
whole-trace science for a separate follow-up with a genuinely different contribution, so the two
papers do not overlap.

---

## 14. Paper outline (9 pages) and figures

Working titles (choose after results; no overclaiming words such as "neural computation"):
- "Tracing Every Token: Attribution Graphs for Whole Reasoning Traces at the Cost of One"
- "Attribution Graphs Are Compressible"

Outline:
1. **Introduction** (1.25 pages): the barrier (Section 2.1), the question, the answer, the
   contributions, Figure 1.
2. **Background** (0.5): attribution graphs and the local replacement model.
3. **The structure of attribution matrices** (1.75): path decomposition (T2, T3), the atlas (E1), the
   reuse exponent β.
4. **Method** (1.5): CCT-v2, the guarantees (T4), the cost law (T5), whole-trace exactness (T7).
5. **Experiments** (2): short prompts reported honestly, whole-trace cost and fidelity, scaling,
   ablations.
6. **What whole-trace graphs show about reasoning** (1.5): dependency maps, decorative steps,
   intervention validation against baselines.
7. **Related work, limitations, conclusion** (0.5).

Appendix: proofs, preregistrations, extra models, the reference-graph convergence protocol.

Figures:
- **Fig 1 (headline):** cost per explained token against the number of explained tokens, log-log.
  Exact full is flat and high; union-truncation rises with slope β; CCT is flat and low.
- **Fig 2:** the path decomposition: energy share and sparsity of A_0 against A_attn, with one
  example column before and after subtraction.
- **Fig 3:** the atlas: k_eff against length and model size; hubs by node kind.
- **Fig 4:** fidelity against cost, for short prompts and for whole traces.
- **Fig 5:** the step dependency map of one trace, with intervention results overlaid.
- **Fig 6:** graph dependency against real-model intervention effect; AUROC against S1 to S3.
- **Table 1:** methods × cost × fidelity × coverage. **Table 2:** ablations.

---

## 15. Related work and honest novelty (verify every entry; complete it in docs/related_work_notes.md)

| Work | What it does | Our difference |
|---|---|---|
| Ameisen et al. 2025; Lindsey et al. 2025 | Attribution graphs, local replacement model, case studies including chain-of-thought faithfulness cases (verify) | Same graphs, whole traces, structure explained |
| circuit-tracer (PyPI) and Neuronpedia | Open implementation; defaults above | Drop-in |
| CRV, Zhao et al. ICLR 2026 (2510.09312) | Per-step graphs to verify reasoning; too costly for practical use | Every step at roughly one graph's cost; E6 |
| CIE (2605.25603) | Traces selected tokens to save cost | Full coverage instead of selection; baseline B8 |
| GraphGhost (2510.08613) | Token-level tracing on Qwen3-0.6B reasoning | Scale and cost |
| DifFRACT (2606.15796) | Budgeted greedy expansion in diffusion models | Truncation-type method; our B1 and B1U |
| Ge et al. 2024 (2405.13868), linear computation graphs | One forward and one backward pass for one target | One target, not the node-to-node matrix |
| Syed et al. 2023 (edge attribution patching) | One backward pass scores edges for one metric | Different object |
| Choromanski and Sindhwani, NeurIPS 2017 | Compressed sensing of Jacobians | The technique is known; B6 |
| TRAK (Park et al. 2023), LoGra (Choe et al. 2024), step-resolved data attribution (2602.10097, uses count-sketch inside backward passes) | Random projection or sketching of gradients for data attribution | They sketch the parameter side of gradients; we sketch the target side of the node-to-node matrix and subtract an exact local part |
| Charikar, Chen and Farach-Colton 2002 | Count-sketch | Decoder D2 |
| Elhage et al. 2021 | Path expansion of transformer circuits | Basis of T2, applied to attribution graphs |
| Abnar and Zuidema 2020 | Attention rollout | Reachability screening; baseline S1 |
| Lanham et al. 2023; Chen et al. 2025 (verify) | Chain-of-thought faithfulness by behavioural tests | Our interventions; mechanistic maps |
| Li 2026 (2611.00606); CLT-Forge (2603.21014); HAGD (2601.12879) | Graph accuracy; CLT training; search over graphs | Orthogonal; cite |

**Novelty statement:** compressed sensing of Jacobians (2017) and gradient sketching (data
attribution) are known. New in this paper:
1. target-side sketching of the attribution matrix, with an exact closed-form local part and
   reachability screening;
2. the measured and explained structure of attribution matrices;
3. the cost law, including the reuse exponent β;
4. whole-trace attribution and what it shows about reasoning.

Novelty gate (repeat in the final week): search arXiv and OpenReview, including ICLR 2027
submissions, for "attribution graph" or "circuit tracing" combined with "sketch", "compressed
sensing", "count sketch", "every token", "whole trace" or "long chain of thought". If a match
appears, stop and discuss (PD7).

---

## 16. New risks and mitigations

| Risk | Mitigation |
|---|---|
| β is small: models reuse the same rows, so union-truncation is cheap | Gate 0 measures β first; GO-STRUCTURE path |
| The closed-form A_0 does not match circuit-tracer's definition | Derive from the source; K0.7 numerical check |
| k_eff grows with length | Subtract A_0, screen by reachability, report k_eff against length; if it fails, claim short traces only |
| Reference graphs at long lengths are themselves budgeted | Convergence protocol (Section 10); say so in the paper |
| Memory and time at T, N of 1e5 to 1e6 | D2 decoder, chunking, sparse storage; L ≤ 400 in Tier A |
| Transcoders are poor on thinking text (large error nodes) | Report completeness and replacement per trace; limit claims accordingly |
| Interventions confounded by shifted positions or attention | Equal-length neutral replacement and deletion, both; several variants |
| A competing paper appears (CRV or CIE groups extending their work) | Novelty gate; PD7 |
| Time | Tiers; two abstract variants; fallback |

---

## 17. Decision register

**Taken by this file (Dawar can override):**

| ID | Decision |
|---|---|
| V1 | The headline moves from regime D to whole-trace attribution; regime D is reported honestly |
| V2 | B1U (union-truncation) is the key baseline |
| V3 | A_0 is computed in closed form and subtracted before sensing |
| V4 | D2 (count-sketch) is the decoder for whole traces; D1 (Rademacher + OMP) for short prompts |
| V5 | E5 (reasoning science with real-model interventions) is Tier A |
| V6 | The global certificate moves to Tier B; Tier A uses fresh-probe error estimates |
| V7 | Gate 0 replaces the v1 premise check as the first GPU job |
| V8 | Qwen3-1.7B and 4B with thinking mode are the main whole-trace models |

**Pending:**

| ID | Decision | Who, when |
|---|---|---|
| PD-G0 | GO-BIG, GO-STRUCTURE or NO-GO | Dawar, 17 Nov |
| PD-C | Compute cap: 20 or 40-plus H100-hours | Dawar, before Gate 0 |
| PD-T | Title and abstract variant | Dawar, 22 Nov |
| PD-B | Tier B scope | Dawar, 26 Nov |
| PD-V | Fallback venue if C4 or C5 fails | Dawar, 29 Nov |
| PD7 | Overlapping paper found | Any time |

---

## 18. Rules (v1 Section 12 still applies; additions below)

- No headline number goes anywhere (README, abstract, slides) until it is measured. Hypotheses are
  labelled as hypotheses.
- Every claim has a kill condition (Section 6), and the kill condition is checked before writing.
- Regime D results are reported even if CCT loses there.
- Intervention experiments run on the real model, never only on the replacement model.
- Writing: British spelling, plain language, no em-dashes, no emoji, full names for papers and
  projects.

---

## 19. Tasks for Claude Code, in order

1. Read PROJECT.md (v1), PHASE1_KILL_TEST.md and this file. Write PREREG_GATE0.md from Section 7,
   with every threshold, and commit it.
2. `cct/local_part.py`: closed-form A_0 (T2), with CPU tests on a tiny pure-PyTorch model against
   the attention-zeroed backward pass.
3. `cct/columns_jvp.py`: exact columns by forward mode (or double backward), with CPU tests against
   the transpose of the exact rows.
4. `cct/reach.py`: attention reachability ρ and candidate sets Ω_s, with CPU tests.
5. `scripts/gate0_short.py`: G0-A, steps 1 to 5.
6. `scripts/gen_traces.py`: Qwen3 thinking traces. Commit the text and the settings.
7. `scripts/gate0_union.py`: G0-B step 2 (union growth and the β fit).
8. `scripts/gate0_columns.py`: G0-B steps 3 and 4.
9. `scripts/gate0_report.py`: results/GATE0_REPORT.md and results/gate0.json, with the decision table
   applied automatically. **Stop and wait for Dawar.**
10. After GO-BIG:
    - modules: `cct/inject.py` (multi-location seeds; K0.2), `cct/sketch.py` (D1 and D2),
      `cct/decode.py`, `cct/hubs.py`, `cct/check.py`, `cct/export.py` (circuit-tracer JSON);
    - then K0 to K3 with K0.7 to K0.9; then E2 pilot, E1, E2, E3, E4, E5;
    - every result file stores config, seed, git hash, package versions, GPU name and wall time.
