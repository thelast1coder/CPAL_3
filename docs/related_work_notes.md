# Related work notes and novelty gate (CCT, Phase 0)

Written 10 November 2026 during P0 (laptop). Sources were read on that date unless stated. Anything
not checked against a primary source is marked (verify). Quotes are copied exactly; locations are
given as section, appendix, table or source line.

## Novelty gate outcome

**PASSED (with one caveat).** No work found that measures attribution-graph rows or columns with randomly mixed (or any sensing) passes and recovers the graph, and none that measures the compressibility of attribution matrices (claim C1).
Closest overlaps: Choromanski and Sindhwani 2017 (the general technique, Jacobian sensing by compressed sensing plus colouring, already cited as prior art); Tan et al. arXiv 2604.17761 (packs targets into separate batch slots, no mixing, no recovery); ADAG arXiv 2604.07615 (one backward pass from the sum of top-K logits, for node scores, not edges).
Caveat: Google Scholar could not be queried directly (general web search, arXiv API and OpenReview site search were used); a paper behind a paywall or posted in the last few days could be missed, so repeat the search before submission.

---

## 1. Ameisen et al. 2025, "Circuit Tracing: Revealing Computational Graphs in Language Models"

Source: https://transformer-circuits.pub/2025/attribution-graphs/methods.html (read 10 Nov 2026).

- **What it does.** Trains cross-layer transcoders (CLTs), builds a per-prompt local replacement
  model (attention patterns and normalisation denominators frozen, error nodes added), and computes
  an attribution graph of direct linear effects between features, token embeddings, error nodes and
  output logits. Prunes the graph by influence on the logits and evaluates it with replacement and
  completeness scores.
- **Object computed.** Edge weights "A_{s→t} := a_s w_{s→t}" (section "Constructing an Attribution
  Graph for a Prompt"): source activation times a virtual weight built from the source decoder, the
  frozen Jacobian and the target encoder. "Edges originate from feature, embedding, and error nodes,
  and terminate at feature and output nodes" (same section).
- **Cost profile.** Appendix "Attribution Graph Computation": "The cost to compute the graph is
  linear in the number of active features in the prompt" and is "dominated by the cost of the
  underlying model backwards pass". "To compute the graph edges in practice, we iterate over target
  nodes in the graph (output or feature nodes)" (one backward pass per target row). "To economize,
  we sometimes compute the graph adaptively" (queue of the most important nodes first); for long
  prompts "We terminate when a target number of nodes have been explored" (Appendix "Graph Pruning").
  No per-graph wall-clock time or GPU count is given in the text read.
- **Pruning and influence.** Appendix "Graph Pruning": "We employ a two-step algorithm which first
  prunes the nodes and then prunes the edges of the remaining nodes." Edge weights are replaced by
  absolute values and each node's input edges are normalised to sum to 1. Node threshold: "We use a
  threshold of 0.8 unless otherwise noted." Edge threshold: "using a higher cutoff 0.98 unless
  otherwise noted." Logits: "If this would result in K > 10, we clamp K to 10." "Embedding nodes and
  error nodes are not pruned." Indirect influence is the Neumann series, which "can be efficiently
  computed as B = (I - A)^{-1} - I" (section "Evaluating Node Influence").
- **Numbers (copied).**
  - "on the order of a hundred active features per token position" (section "Constructing an
    Attribution Graph for a Prompt").
  - "the number of edges can grow to the millions even for short prompts" (same section).
  - "A pruned graph often contains hundreds of nodes and tens of thousands of edges" (same section).
  - "the number of nodes typically decreases by ~10× and the number of edges typically decreases by
    ~500×" "(the exact numbers are sensitive to prompt length)" (Appendix "Graph Pruning").
  - "With our default parameters, we typically reduce the number of nodes by a factor of 10"
    (section "Evaluating Graph Pruning"); the pruning table lists "0.8 (default)" with 55 nodes
    (same section; table row read through a summariser, (verify) the 55 against the table).
  - "the graphs typically contain hundreds of features and thousands of edges" (Limitations, "Graph
    Complexity").
  - "Each feature receives many small inputs from many other features" (Limitations, "Graph
    Complexity"). Relevant to C1: it says rows (in-edges) are spread out; it says nothing about
    columns (out-edges), which is what CCT recovers.
  - "In general we targeted L0 values in the low hundreds" (Appendix "CLT Implementation Details",
    ML details).
- **How CCT differs.** Same object (the same exact edge matrix, same pruning). Ameisen et al.
  enumerate one target row per backward pass (or truncate adaptively). CCT measures signed mixtures
  of rows and recovers columns by sparse recovery, so the pass count scales with the sparsity of
  columns rather than the number of targets. The adaptive truncation described here is our baseline
  B1.

## 2. Lindsey et al. 2025, "On the Biology of a Large Language Model"

Source: https://transformer-circuits.pub/2025/attribution-graphs/biology.html (read 10 Nov 2026).

- **What it does.** Case studies of Claude 3.5 Haiku using the attribution graphs of Ameisen et al.
  (multi-hop reasoning, planning in poems, addition, multilingual circuits, refusals, hidden goals and
  more).
- **Object computed.** Attribution graphs on a production model, with a CLT of up to "10M features"
  in one section ("We trained a cross-layer transcoder with 10M features on the model described
  above", Hidden Goals section).
- **Cost profile.** No per-graph compute figure in the text read. Scale limits stated:
  "we have not scaled our method to apply to prompts longer than about a hundred tokens"
  (Limitations, "Long Prompts"). Human cost: "This is a slow manual process that can take over an
  hour for one of our researchers" (Limitations, "Graph Complexity").
- **Numbers (copied).** "we've found that our attribution graphs provide us with satisfying insight
  for about a quarter of the prompts we've tried" (Introduction, note on approach and limitations).
  "generated attribution graphs using a set of 200 diverse harmful prompts" (Refusals). "the overall
  picture is usually clear within 1–2 hours" (Discussion).
- **How CCT differs.** Not a method paper. CCT targets the compute side of the long-prompt limit
  quoted above; we reproduce canonical cases (on open models, not Haiku) at lower pass counts.

## 3. circuit-tracer library (Hanna, Piotrowski, Lindsey, Ameisen) and its defaults

Sources: PyPI JSON API for `circuit-tracer`; the 0.5.0 wheel downloaded without dependencies and
read (SHA256 e457e6558e1c9eb0d3d7c54ce2298492b4e919a4f7f2b8b7e217427284fdc802); GitHub
`decoderesearch/circuit-tracer` (README and tag v0.5.0); paper "Circuit-Tracer: A New Library for
Finding Feature Circuits", BlackboxNLP 2025, pages 239–249, DOI 10.18653/v1/2025.blackboxnlp-1.14.

- **Repository location (checked).** `github.com/safety-research/circuit-tracer` returns HTTP 301 to
  `github.com/decoderesearch/circuit-tracer`. PyPI lists exactly one release, **0.5.0, uploaded
  2026-03-29**; authors "Michael Hanna, Mateusz Piotrowski, Decode Research". Its project URLs point
  to a fork, `github.com/Caerii/circuit-tracer` (GitHub API: fork of decoderesearch), with
  "Original Repository" = decoderesearch. The decoderesearch repo has tags v0.2.0 to **v0.5.3**, so
  GitHub is ahead of PyPI. The PyPI 0.5.0 wheel's `attribution/attribute.py` is a unified,
  refactored file that differs from the GitHub v0.5.0 tag's `attribute.py` (which routes to
  backend-specific implementations). Defaults agree (below). **Decision needed at pin time:** pin the
  PyPI wheel by hash, or a decoderesearch tag by commit, and record which (see docs/decisions.md).
- **What it does.** Open implementation of Ameisen et al.: loads a model plus transcoders as a
  `ReplacementModel` (TransformerLens default, NNSight optional), runs attribution, prunes, and
  serves the graph viewer; also interventions. Paper: tested "with models up to 14B" parameters
  (abstract, via search summary, (verify)); "We recommend a GPU with at least 15 GB VRAM for circuit
  tracing with Gemma-2 (2B), and up to 40 GB for larger models" (paper, section 3.2.3).
- **Object computed.** `Graph.adjacency_matrix`, dense, rows are targets, columns are sources
  (`edge_matrix[target, source]`; feature rows are filled in the order visited, then logits). Node
  influence uses `normalize_matrix` (absolute values, each row normalised to sum 1) and
  `compute_influence`, which sums `w @ A + w @ A^2 + ...` "until `current_influence.any()` is false"
  (finite for a DAG; `max_iter` 1000) (`graph.py`, wheel 0.5.0).
- **Cost profile (from source).** Wheel 0.5.0, `attribute.py` `_run_attribution`: Phase 3 runs one
  batched backward per batch of logit targets; Phase 4 loops over feature targets, each batch of
  `batch_size` targets is one batched backward (`ctx.compute_batch`). If `max_feature_nodes` is below
  the number of active features, it re-ranks pending features by partial influence every
  `update_interval` (default 4) batches and visits the highest first. So cost is
  ceil(n_targets / batch_size) batched backward passes, plus one forward pass.
- **Defaults (verified in code, wheel 0.5.0).**

  | Parameter | Python `attribute()` default | CLI default (`__main__.py`) | Pruning default |
  |---|---|---|---|
  | max_feature_nodes | `None` (all active features) | `DEFAULT_MAX_FEATURE_NODES = 7500` | |
  | batch_size | `512` | `DEFAULT_BATCH_SIZE = 256` | |
  | max_n_logits | `10` | `DEFAULT_MAX_N_LOGITS = 10` | |
  | desired_logit_prob | `0.95` | `DEFAULT_DESIRED_LOGIT_PROB = 0.95` | |
  | node_threshold | | `DEFAULT_NODE_THRESHOLD = 0.8` | `prune_graph(..., node_threshold=0.8)` (`graph.py`) |
  | edge_threshold | | `DEFAULT_EDGE_THRESHOLD = 0.98` | `prune_graph(..., edge_threshold=0.98)` (`graph.py`); `create_graph_files` 0.8 / 0.98 |
  | offload | `None` | `None` | |
  | update_interval | `4` | | |

  Locations: `circuit_tracer/attribution/attribute.py` lines 51 to 54 and 56 to 58 (signature);
  `circuit_tracer/__main__.py` lines 8 to 13; `circuit_tracer/graph.py` line 240;
  `circuit_tracer/utils/create_graph_files.py` lines 165 to 166. The GitHub v0.5.0 tag has the same
  values (CLI `default=7500`, `256`, `10`, `0.8`, `0.98`; `attribute()` `512`, `None`). The README
  (decoderesearch) states the same CLI defaults: "`--batch_size` (default: 256)", "`--max_feature_nodes`
  ... (defaults to 7500)", "`--node_threshold` (default: 0.8)", "`--edge_threshold` (default: 0.98)",
  "`--max_n_logits` (default: 10)", "`--desired_logit_prob` (default: 0.95)".
  **Important for the kill test:** the "max_feature_nodes 7,500, batch 256" setting in PROJECT.md
  section 11 is the CLI default; calling `attribute()` from Python gives no cap and batch 512.
  The B0 configuration must state which one it uses. Also, the BlackboxNLP paper's own example uses
  `edge_threshold=0.95` (section 3.1.3, listing), not the 0.98 default.
- **How CCT differs.** Drop-in alternative to Phase 4 (and optionally Phase 3) with the same
  `Graph` output; exact graphs from the pinned version stay the ground truth.

## 4. Choromanski and Sindhwani 2017, "On Blackbox Backpropagation and Jacobian Sensing"

Source: NIPS 2017 (Advances in Neural Information Processing Systems 30), PDF from
papers.nips.cc (read 10 Nov 2026). Pages 6521–6529 per proceedings listing (verify).

- **What it does.** Recovers the Jacobian of a blackbox from a few calls on random input
  perturbations, combining compressed sensing with a probabilistic graph-colouring step ("rainbow"),
  exploiting sparsity and symmetry; applications: backpropagation through noisy blackbox layers,
  linearising rigid-body dynamics.
- **Object computed.** A full Jacobian J (or J^T v) of a function given only forward evaluations.
- **Cost profile / numbers.** Theorem 3.3 (section 3): an approximation with ||J^ − J||_F <= E(n) is
  found "with probability p = 1 − 1/spoly(n)" using "k = O(min(A, B)) calls", where A depends on the
  intrinsic dimensionality d_int of a convex prior set and B on a combinatorial parameter
  m(J, G^weak_int) of the sparsity pattern, both times a log factor in mn / E(n) (exact formula in
  Theorem 3.3; the extracted text was garbled, (verify) the log argument).
- **How CCT differs.** The core idea (sense a Jacobian with random mixtures, recover by sparse
  recovery) is theirs; CCT does not claim it. CCT is different in: (a) the object (attribution
  matrices between transcoder features in a frozen replacement model, exact linearity, no
  finite-difference noise); (b) the measurement side (mixed reverse-mode seeds over targets rather
  than forward perturbations of inputs); (c) attribution-specific machinery: causal masks, free
  energy screening, hub detection with exact forward-mode columns, and a DAG influence certificate;
  (d) the empirical claim C1 that these matrices are compressible. Baseline B6 is plain Jacobian
  sensing in their spirit.

## 5. Curtis, Powell and Reid 1974, "On the estimation of sparse Jacobian matrices"

Source: STFC record and citing works (paper itself not read). J. Inst. Maths Applics 13 (1974),
117–119 or 117–120 (sources disagree, (verify)); DOI 10.1093/imamat/13.1.117 (verify).

- **What it does.** With a known sparsity pattern, groups structurally orthogonal columns so one
  finite-difference (or forward-mode) evaluation measures several columns at once (the "CPR method").
- **Object / cost.** Full sparse Jacobian in as many evaluations as groups (colours).
- **How CCT differs.** CPR needs the exact zero pattern in advance. Attribution matrices have only a
  causal zero pattern (later layer, same or later position), which leaves columns dense in pattern
  even when small in value, so colouring gives little saving. This is baseline B4.

## 6. Gebremedhin, Manne and Pothen 2005, "What Color Is Your Jacobian? Graph Coloring for Computing Derivatives"

Source: SIAM Review 47(4), 629–705, 2005, DOI 10.1137/S0036144504444711 (checked via SIAM and the
authors' page header; full text not read).

- **What it does.** Survey and unifying framework: partitioning a matrix into structurally
  orthogonal column groups is distance-2 colouring of a graph; covers Jacobians and Hessians, direct
  and substitution methods, algorithms and results.
- **How CCT differs.** Same as CPR: structural, exact-pattern methods. Cited for B4.

## 7. Halko, Martinsson and Tropp 2011, "Finding structure with randomness"

Source: arXiv 0909.4061v2, Section 4.3 "A posteriori error estimation" (pages 23 to 24 of that
version), text extracted from the PDF on 10 Nov 2026. Journal version: SIAM Review 53(2), 217–288,
2011 (verify section and page numbering in the journal version).

- **Exact statement (copied, notation restored).** "Draw a sequence {ω^(i) : i = 1, 2, . . . , r} of
  standard Gaussian vectors, where r is a small integer that balances computational cost and
  reliability. Then

      ||(I − QQ*)A|| ≤ 10 √(2/π) max_{i=1,...,r} ||(I − QQ*)A ω^(i)||        (4.3)

  with probability at least 1 − 10^(−r)." This follows from **Lemma 4.1**: "Let B be a real m × n
  matrix. Fix a positive integer r and a real number α > 1. Draw an independent family
  {ω^(i) : i = 1, 2, . . . , r} of standard Gaussian vectors. Then ||B|| ≤ α √(2/π) max_{i=1,...,r}
  ||B ω^(i)|| except with probability α^(−r)." (Proof cited to their reference [137], §3.4.) Remark
  4.1: "The estimate (4.3) is actually somewhat crude."
- **Conditions to note.** (a) Probes are **standard Gaussian**, not Rademacher. (b) The norm is the
  spectral (operator) norm of the whole error matrix, bounded from above by a few matrix-vector
  products with the error. (c) It is a one-sided upper bound with a large constant
  (10 √(2/π) ≈ 7.98).
- **How CCT uses it and the gap.** CCT's check Z = Φ'(A − Â) applies probes on the target side and
  wants per-column norms ||a_s − â_s||, then feeds row-sum errors into the influence bound. Lemma 4.1
  as stated bounds ||B|| (spectral) for B ω with Gaussian ω; applying it to B^T = (A − Â)^T with
  probes on the left is fine for the spectral norm, but per-column estimates and Rademacher probes
  need a different result (for example a Johnson–Lindenstrauss or Hutchinson-type concentration bound
  for Rademacher vectors; source and constant (verify)). Simplest safe option: draw the certification
  probes Φ' as Gaussian so Lemma 4.1 applies exactly, and state the per-column estimator separately.
  Log this choice in docs/decisions.md.

## 8. Dunefsky, Chlenski and Nanda 2024, "Transcoders Find Interpretable LLM Feature Circuits"

Source: arXiv 2406.11944 (v2, 6 Nov 2024), NeurIPS 2024 (per arXiv comments).

- **What it does.** Introduces transcoders (a wider, sparsely active approximation of an MLP
  sublayer) and a weights-based circuit analysis through MLPs where circuits "neatly factorize into
  input-dependent and input-invariant terms" (abstract). Models of 120M, 410M and 1.4B parameters;
  greater-than circuit in GPT-2 small.
- **Cost profile.** No compute statement in the abstract. Circuit analysis is per-feature,
  greedy path search (full-text details not read, (verify)).
- **How CCT differs.** Provides the per-layer transcoder (PLT) idea that circuit-tracer uses; no
  attempt to cut the number of passes for a full edge matrix.

## 9. Marks et al. 2024/2025, "Sparse Feature Circuits"

Source: arXiv 2403.19647 (v3, 27 Mar 2025), ICLR 2025.

- **What it does.** Finds circuits of SAE features that explain a behaviour on a dataset, using
  linear approximations of indirect effects (attribution patching, integrated gradients), then edits
  them (SHIFT).
- **Object computed.** Node and edge indirect effects on a scalar metric, averaged over a dataset;
  not the per-prompt direct-effect matrix between all nodes.
- **Cost profile (copied).** Attribution patching "can be computed for many a in parallel using
  only two forward and one backward pass" (section 2). Integrated gradients uses "N=10
  equally-spaced" steps and its extra cost "scales linearly in N and the serial depth of m's
  computation graph" (section 2). Edge effects (Appendix A.1, Eq. 5) use stop-gradients; "the naive
  approach to computing is expression involves performing dmodel backwards passes; fortunately we can
  still compute the product in a single backwards pass" (Appendix A.1); Appendix A.3 says the edge
  term can be computed "with only two backwards passes" (per downstream node, (verify) the scaling
  over downstream nodes).
- **How CCT differs.** Different object (metric-weighted indirect effects on a dataset) and a
  different cost profile; CCT recovers the exact per-prompt direct-effect matrix.

## 10. Syed, Rager and Conmy 2023, "Attribution Patching Outperforms Automated Circuit Discovery"

Source: arXiv 2310.10348 (v2, 20 Nov 2023), NeurIPS 2023 ATTRIB workshop.

- **What it does.** Edge attribution patching (EAP): linear approximation of activation patching
  scores every edge of the computational graph for one metric, then prunes; higher circuit recovery
  AUC than ACDC on average (abstract).
- **Cost (copied).** "just two forward passes and a backward pass" (abstract).
- **Object computed.** One score per edge for one scalar metric (a gradient times an activation
  difference), not the matrix of direct effects of every node on every other node.
- **How CCT differs.** EAP gets a single metric-specific vector in one backward pass; an attribution
  graph needs a separate read per target node, which is the cost CCT reduces. B5 is the closest
  analogue (one-pass node ranking).

## 11. CLT-Forge, arXiv 2603.21014

Source: arXiv abstract page (read 10 Nov 2026). **ID checks out.** "CLT-Forge: A Scalable Library
for Cross-Layer Transcoders and Attribution Graphs", Draye, Palit, Harrasse, Wu, Liu, Pandey, Wu,
Hsu, Zhang, Jin, Schölkopf; v1 22 Mar 2026, v2 21 Jul 2026.

- **What it does.** End-to-end CLT training library: distributed training with model sharding,
  compressed activation caching, automated interpretability, attribution graph computation (the
  abstract cites "attribution graph computation using Circuit-Tracer"), visualisation.
- **Cost profile.** The abstract discusses training scale and that graphs are "large and redundant"
  (paraphrase); it does not discuss the cost of computing graphs. Whether the full text reduces
  passes: not checked (verify; abstract suggests not).
- **How CCT differs.** Orthogonal: makes CLTs cheaper to train; CCT makes graphs cheaper to compute.

## 12. HAGD, arXiv 2601.12879

Source: arXiv abstract page. **ID checks out.** "Hierarchical Sparse Circuit Extraction from
Billion-Parameter Language Models through Scalable Attribution Graph Decomposition", Uddin, Alam,
Pasha; v1 19 Jan 2026, v2 20 Jun 2026.

- **What it does.** Four stages: CLT training, spectral coarsening of attribution graphs,
  GNN-guided hierarchical traversal, causal intervention checks; GPT-2, Pythia, Llama.
- **Cost profile.** Claims to cut circuit search from "O(2^n)" exhaustive enumeration to
  "O(n^2 log n)" (abstract). Edges are "gradient-activation products" (abstract), so attribution is
  still computed in full; the speed-up is in search over the graph, not in building it. Numbers in a
  secondary write-up (91% behaviour preservation, 117M to 70B parameters) were not checked against
  the paper (verify).
- **How CCT differs.** Orthogonal: HAGD searches a computed graph faster; CCT computes the graph
  with fewer passes. The two could be combined.

## 13. Li, arXiv 2611.00606

Source: arXiv abstract page. **ID checks out.** "Sparse Autoencoder Attribution Graphs Are Decided
at the Margin", Karl Li; submitted 31 Oct 2026 (v1), cs.LG.

- **What it does.** Studies attribution graphs over TopK SAE features (five public TopK SAE pairs,
  three Pythia models, largest pythia-160m). Finds that zeroing one feature usually pushes a
  downstream feature across its activation cutoff and these crossings explain most of the response;
  a margin-based graph with a first-order preactivation correction does better "at the same cost"
  (paraphrased from the abstract); cross-seed comparisons of graph energy.
- **How CCT differs.** Concerns graph accuracy for TopK SAEs, not graph cost; orthogonal. Its
  cross-seed "energy" analysis is about feature matching, not about compressibility of columns.
  PROJECT.md's description ("accuracy for TopK SAEs at the margin; same cost") matches.

## 14. arXiv 2510.09312, "Verifying Chain-of-Thought Reasoning via Its Computational Graph"

Source: arXiv HTML v2 (read 10 Nov 2026). **ID checks out**, and it is chain-of-thought verification
with attribution graphs. Zhao et al., FAIR at Meta and University of Edinburgh; v2 dated Feb 2026;
venue listed as ICLR 2026 by a third-party site only (verify).

- **What it does.** Circuit-based Reasoning Verification (CRV): trains TopK per-layer transcoders
  for Llama 3.1 8B Instruct, builds an attribution graph for every reasoning step, extracts
  structural features, and trains a classifier to flag wrong steps.
- **Graph settings (copied, Appendix B.2).** "We use the implementation from Hanna et al. (2025) to
  compute attribution graphs." "a maximum of 4096 feature nodes, attribution traced from a maximum of
  10 logit nodes" "and a batch size of 16 for backward passes." "All other parameters follow the
  repository defaults." Pruning keeps components above an influence threshold, 80% as the example
  (section 3.2.3).
- **Cost.** No per-graph number. Appendix E "Computational Intensity": "A primary limitation of CRV
  is its computational intensity", "is orders of magnitude more resource-intensive than black-box or
  gray-box verification methods". Section 1: "too computationally intensive to be intended as a
  practical, drop-in verifier". Transcoder training used "4 nodes, each with 8 Nvidia H200 GPUs"
  (Appendix B).
- **How CCT differs.** A consumer of many graphs (one per step) that names graph cost as its main
  limit: a motivating use case for CCT.

## 15. Other works using attribution graphs at scale that state cost (found in the searches)

- **Yang et al., "Circuit Tracing in Vision–Language Models", arXiv 2602.20330.** Section 4.2:
  "Computing an attribution graph for a simple image-text QA task takes 20 minutes on a single H100
  GPU." Section 4.3: "Computing feature activation on a single attribution graph with approx. 1000
  features takes 20 H100 GPU hours" (the second figure is feature-dashboard cost, not graph cost).
  Uses circuit-tracer extended to Gemma-3-4B-it with TopK (k = 48) PLTs. The clearest stated
  per-graph cost found.
- **Arora, Wu, Steinhardt, Schwettmann, "ADAG: Automatically Describing Attribution Graphs", arXiv
  2604.07615.** Appendix B: a reimplementation "more efficiently" (removing CUDA syncs, "Better
  batching for Jacobian computations", data parallelism); H100 80GB cluster; for 10,000 math
  examples "we were able to trace all circuits in ≈6 hours with data paralellism over 4[4] GPUs"
  (one reading gave 44 GPUs, a search summary gave 4: (verify)). Section 3.1: attribution is
  backpropagated from a single scalar, the sum of the top-K logits (a deterministic mixture of
  targets used for node-level scores, not for recovering edges; (verify) exact use). Engineering
  speed-ups, no sensing.
- **Tan et al., "Contrastive Attribution in the Wild", arXiv 2604.17761.** Appendix A,
  "Batch-packed multi-target backpropagation": "batch element b injects relevance only at its
  corresponding target index"; reduces "the cost of constructing attribution-graph edges from" O(n)
  backward passes to O(⌈n/B⌉) passes. Appendix B Table 2 (Qwen3-0.6B, single A100): at 1508 tokens,
  B=1 6.23 s and 89.2 MB, B=8 2.67 s and 648.9 MB, B=16 3.45 s and 1290.0 MB. This is the same
  batching circuit-tracer already does (one slot per target); no mixing and no recovery, so the
  number of rows read is unchanged. Close in wording ("fewer backward passes") but not an overlap.
- **DifFRACT, arXiv 2606.15796** (attribution graphs for the FLUX.1 diffusion model): raw edge
  counts of about 1.3 to 4.3 million and pruned counts of about 8,000 to 65,000 per target, from a
  search snippet (verify table and location). Not about cost reduction.
- **Hanna et al. BlackboxNLP 2025** (section 3 above) and **jasn9776/attribution-graph-triage**
  (GitHub, states "Circuit tracing gives useful insight on ~25% of prompts at hours of effort each",
  300 graphs on Gemma-2-2B; an unreviewed repo, human effort not compute; (verify)).

## 16. Works close to claim C1 (compressibility), none of which measure it

- Ameisen et al. (Limitations): "Each feature receives many small inputs from many other features"
  (in-edges dense). No measurement of out-edge concentration.
- "Sparsely-connected Cross-layer Transcoders" (LessWrong post, title and findings from a search
  snippet, (verify) author and date): trains CLTs so each latent depends on few upstream latents.
  Changes the model to be sparse; does not measure standard attribution matrices.
- Jacobian Sparse Autoencoders, arXiv 2502.18147: trains SAEs with an L1 penalty on the Jacobian
  between input and output latents of an MLP to induce sparse computation. Different object (one
  MLP's latent Jacobian, trained to be sparse); does not measure attribution-graph compressibility.
- Pruning statistics (Ameisen: nodes ~10×, edges ~500× after pruning) show influence is
  concentrated, which is consistent with C1 but is a statement about the pruned graph, not about
  per-column s95.

---

## Searches run (all on 10 November 2026)

Web search (extended mode unless noted):
1. `"attribution graph" "compressed sensing" transcoder circuit tracing`
2. `circuit tracing attribution graphs fewer backward passes efficient scalable computation`
3. `attribution graph sketching random projection Jacobian sensing language model interpretability`
4. `transcoder Jacobian sparse recovery interpretability`
5. `sparsity of attribution graph adjacency matrix edges concentrated few targets cross-layer transcoder`
6. `"circuit tracing" faster attribution graph computation speedup GPU hours per graph`
7. `openreview attribution graphs randomized backward pass mixture targets sparse recovery edges`
8. `attribution graph adjacency low-rank or sparse approximation measured "circuit-tracer" edges per feature distribution`
9. `site:openreview.net attribution graphs circuit tracing efficient computation transcoders 2026`
10. `"compressed sensing" OR "compressive sensing" mechanistic interpretability circuits backward passes random signs Rademacher Jacobian rows`
11. Standard mode, ID and citation checks: `arXiv 2603.21014`; `arXiv 2601.12879 HAGD attribution graph`;
    `arXiv 2611.00606 attribution graph TopK`; `arXiv 2510.09312 chain-of-thought attribution graph verification`;
    Halko `10 \sqrt{2/\pi}` Section 4.3; Choromanski and Sindhwani NeurIPS 2017; Gebremedhin et al. SIAM Review 2005;
    Curtis Powell Reid 1974; `"circuit-tracer: A New Library for Finding Feature Circuits"`.

arXiv API (export.arxiv.org, abstract-level search, up to 30 results each):
`"attribution graph" AND "compressed sensing"`; `"attribution graph" AND sketch`;
`"circuit tracing" AND "compressed sensing"`; `"attribution graph" AND "random projection"`;
`"Jacobian sensing"`; `transcoder AND "sparse recovery"`; `"attribution graph" AND "backward passes"`;
`"circuit tracing" AND efficient AND transcoder`; `"attribution graphs" AND scalable`;
`"attribution graph" AND compressible`; `"circuit tracing" AND faster`;
`"attribution graph" AND sparsity AND edges`; `transcoder AND Jacobian`; `"circuit tracing" AND sketching`;
`"attribution graph" AND compressive`. Relevant hits: only CLT-Forge (2603.21014), HAGD (2601.12879)
and "Cross-Layer TopK" transcoders (2611.09233, title only, not relevant to cost); the rest were
about attributed graphs in graph learning.

Google Scholar: not queried directly (no API access from this environment). Repeat by hand before
submission.

## Still marked (verify)

1. Ameisen et al. pruning table: "0.8 (default)" row with 55 nodes (read through a summariser).
2. circuit-tracer paper: "tested with models up to 14B" wording (from a search summary).
3. Which circuit-tracer source to pin: PyPI 0.5.0 wheel (Caerii fork metadata, refactored
   `attribute.py`) or decoderesearch tag (v0.5.0 or v0.5.3). Defaults agree; code paths differ.
4. Choromanski and Sindhwani: proceedings pages 6521–6529 and the exact log arguments in Theorem 3.3.
5. Curtis, Powell and Reid 1974: page range (117–119 or 117–120) and DOI; paper not read.
6. Gebremedhin, Manne and Pothen: full text not read (citation details checked).
7. Halko et al.: section and page numbering in the SIAM Review version (checked on arXiv v2 only).
8. A per-column, Rademacher-probe a posteriori bound with its constant (source not yet identified).
9. Dunefsky et al.: circuit-analysis cost details (full text not read).
10. Marks et al.: how the "two backwards passes" edge computation scales with the number of
    downstream nodes.
11. CLT-Forge: full text not checked for any pass reduction.
12. HAGD: 91% and 117M to 70B figures (secondary source only).
13. CRV (2510.09312): ICLR 2026 venue (third-party site only).
14. ADAG: GPU count in "≈6 hours ... over 4[4] GPUs", and exactly how the top-K logit sum is used.
15. DifFRACT edge counts and table location.
16. "Sparsely-connected Cross-layer Transcoders" post: author and date.
17. attribution-graph-triage repository claims.
