# PHASE0_SETUP.md: repository, literature check, all code and CPU tests (no GPU)

Goal: when Dawar switches on the VM, the kill test runs with three commands and no new code.
Everything below runs on the laptop (no GPU, about 0.5 GB free disk) unless marked VM.

**Exit criteria for P0**
1. Repository created and pushed, with the layout in 1.2.
2. `docs/related_work_notes.md` written and the novelty gate (Section 3) passed.
3. All modules in Section 2 written. The method code works on an abstract operator (2.1), so it is
   fully tested on CPU with the dense and tiny backends. The circuit-tracer backend is written against
   the pinned source and tested first thing on the VM (K0).
4. All tests in Section 4 pass on CPU (or, if the laptop cannot install the packages, they are
   written and run first on the VM).
5. `bash scripts/run_kill_test.sh --tiny` runs end to end and writes `results/tiny/KILL_REPORT.md`
   (labelled TINY, numbers meaningless).
6. `PREREG_KILL_TEST.md`, `configs/prereg_kill_test.yaml` and the prompt files committed; the test that
   checks the prereg copies agree passes.
7. README has the exact VM commands; `results/STATUS.md` says "P0 done, waiting for VM".
Then stop and report to Dawar.

---

## 1. Repository setup

### 1.1 Git
- Directory `CCT_CPAL`, `git init`, branch `main`. Ask Dawar for the private GitHub URL and add it as
  `origin`; if not given yet, work locally and ask at the end.
- Keep the specification files from the zip (CLAUDE.md, PROJECT.md, EXPERIMENTS.md, PHASE0_SETUP.md,
  PHASE1_KILL_TEST.md, PREREG_KILL_TEST.md, KICKOFF.md) in the repository root, unchanged.

### 1.2 Layout
```
CCT_CPAL/
  CLAUDE.md PROJECT.md EXPERIMENTS.md PHASE0_SETUP.md PHASE1_KILL_TEST.md PREREG_KILL_TEST.md KICKOFF.md
  README.md pyproject.toml requirements.txt requirements-cpu.txt .gitignore
  src/cct/
    operators/     base.py dense.py tiny.py circuit_tracer_backend.py
    sensing/       matrices.py measure.py screening.py
    recovery/      omp.py cosamp.py fista.py iht.py batched.py
    hubs.py
    certify/       posterior.py neumann.py refine.py
    graph/         influence.py prune.py metrics.py export.py
    baselines/     truncation.py random_rows.py lowrank.py colouring.py total_effect.py
    interventions/ group_testing.py
    cost.py pipeline.py
    utils/         gpu.py logging.py seed.py io.py status.py timing.py
  configs/
    models/        gemma2_2b_plt.yaml gemma2_2b_clt426k.yaml (others later)
    prompts/       kill_test.json validation.json long.json
    experiments/   kill_test.yaml
    prereg_kill_test.yaml
  scripts/
    vm_setup.sh download_models.py sanity_k0.py run_exact_graphs.py analyse_structure.py
    simulate_sensing.py run_real_cct.py make_kill_report.py run_kill_test.sh
  tests/  docs/ (related_work_notes.md decisions.md)  results/ (STATUS.md)  plots/
  artifacts/ (gitignored: exact matrices, visiting orders, recovered graphs)  logs/ (gitignored)
```

### 1.3 Laptop environment (small)
- Check `df -h .` and `python -c "import torch, sys; print(torch.__version__, sys.version)"`.
- `python -m venv --system-site-packages .venv` (reuses the system PyTorch).
- `PIP_NO_CACHE_DIR=1 pip install numpy scipy pyyaml pytest` (only what is missing).
- Do not install circuit-tracer, transformer-lens, nnsight or transformers on the laptop. To read
  circuit-tracer's source: `pip download circuit-tracer==<pinned> --no-deps -d /tmp/ct` and unzip it
  (check the wheel is small first), or read it on GitHub. Delete the download after reading.
- If free space drops below about 150 MB, stop installing and mark in STATUS.md that tests run first
  on the VM.

### 1.4 Pinning
- Pin one circuit-tracer version (the latest stable; v0.5.0 was released on 29 March 2026, verify
  whether there is a newer one) in `requirements.txt`, with the versions of its backend dependencies
  that it resolves on the VM. Record the choice in `docs/decisions.md`.
- Read its attribution code before writing the backend: how the forward pass is cached, how targets
  are chosen and ordered, how backward seeds are injected (per batch row: one location or many),
  where stop-gradients sit, how rows are read out against decoder vectors (PLT and CLT), how logit
  targets are defined (demeaned unembedding or not), the graph object and its fields, the pruning
  function, and any graph-score function. Write a one-page summary in `docs/circuit_tracer_notes.md`
  with file and function names. The backend must reproduce its numbers exactly (K0).

### 1.5 Files to write now
- `pyproject.toml` (package `cct`, src layout, Python >= 3.10).
- `requirements.txt` (VM): torch (CUDA build), circuit-tracer (pinned) and its resolved dependencies,
  huggingface_hub, numpy, scipy, pandas, matplotlib, pyyaml, pytest, tqdm.
- `requirements-cpu.txt`: the minimal laptop set.
- `.gitignore`: `.venv*`, `artifacts/`, `logs/`, `*.pt`, `*.safetensors`, `*.npz`, `__pycache__/`,
  `.env`, anything with tokens.
- `README.md`, `results/STATUS.md`, `docs/decisions.md`.

---

## 2. Code to write
General rules: PyTorch only; batched over columns or rows (no Python loops over the T targets or N
sources; loops over batches and layers are fine); float32 or float64 in all recovery, influence and
metric code; deterministic given a seed; type hints and short docstrings.

### 2.1 The operator interface (`operators/base.py`)
All method code talks to this interface only:
```python
class AttributionOperator(Protocol):
    T: int                         # number of targets (logits first, then features)
    N: int                         # number of sources (embeddings, features, errors)
    target_meta: TargetMeta        # per target: kind, layer, position, feature id, logit token
    source_meta: SourceMeta        # per source: kind, layer (or write layers), position, feature id, activation
    def rows(self, seeds: Tensor) -> Tensor          # seeds [b, T] -> [b, N] = seeds @ A   (one batched mixed backward)
    def cols(self, tangents: Tensor) -> Tensor       # tangents [b, N] -> [b, T] = tangents @ A.T (one batched forward-mode pass)
    def exact_rows(self, idx: Tensor) -> Tensor      # one-hot seeds; must equal rows(onehot)
    def causal_mask(self) -> CausalMask              # reachability of each target from each source (compact form)
    def logit_weights(self) -> Tensor                # w: logit probabilities on logit targets
    counters: CostCounters                           # rows done, forward-mode passes done, wall time per kind
```
- `dense.py`: `DenseOperator(A, meta)` for tests and for offline simulation from stored exact matrices.
  Its `rows` and `cols` are matrix products and its counters count them like real passes.
- `tiny.py`: a tiny transformer in pure PyTorch (2 to 4 layers, d = 32, 2 heads, vocab 64, random
  weights) with random JumpReLU transcoders, in both PLT and CLT form, with error nodes. It freezes
  attention patterns, normalisation denominators and MLP outputs exactly as circuit-tracer does, and
  computes `rows` by `torch.autograd` VJPs with arbitrary multi-location seeds and `cols` by
  `torch.func.jvp` (or `torch.autograd.forward_ad`). Also provide `brute_force_A()` that builds A from
  the definition in PROJECT.md 5.1 with explicit Jacobians (tiny sizes only).
- `circuit_tracer_backend.py` (VM): wraps circuit-tracer's `ReplacementModel` and its cached forward
  state. `rows(seeds)` injects, for each batch row, the sum over targets of seed weight times the
  target's read vector at the target's layer and position. If circuit-tracer injects only one location
  per batch row, extend the injection (subclass or a small patch) to sums over locations; this is
  bookkeeping, since the backward pass is linear. `cols` uses forward-mode autodiff through the same
  frozen graph; if hooks block forward mode, emulate the JVP with the double-backward trick
  (VJP of a VJP) and record the cost. `exact_rows` must reproduce circuit-tracer's own adjacency rows
  (K0.1). Regime D target sets come from the exact run's visiting order; regime F uses all active
  features.

### 2.2 Sensing (`sensing/`)
- `matrices.py`: Rademacher (default, entries +-1 / sqrt(m)), Gaussian, sparse +-1 with d nonzeros per
  column, position-blocked; all seeded; generated in chunks so T up to 10^6 fits in memory.
- `measure.py`: `sense(op, Phi, batch) -> Y [m, N]` by calling `op.rows` in batches; counts cost.
- `screening.py`: column energy estimates ||y_s||^2; keep the smallest set of columns holding a
  fraction q of the total estimated energy (q from config, tuned on validation prompts).

### 2.3 Recovery (`recovery/`)
Batched over columns, masked by the causal mask, GPU-friendly, chunked over columns.
- `omp.py`: masked OMP up to k_max atoms with a residual stopping rule; incremental QR or Cholesky
  updates so a step is cheap.
- `cosamp.py`, `iht.py`: standard algorithms with masks.
- `fista.py`: masked LASSO by FISTA; lambda by a residual-matching rule (discrepancy principle).
- `batched.py`: chunking, mixed precision policy (fp32 accumulations), and timing hooks.
Return the recovered columns in a sparse format (indices and values per column) plus residual norms.

### 2.4 Hubs (`hubs.py`)
Flag columns with relative residual above rho_hub or recovered support at k_max; optional class rules
(embeddings, error nodes, first-position features); cap at budget H (largest estimated energy first);
compute flagged columns exactly with `op.cols` in batches; replace them.

### 2.5 Certification (`certify/`)
- `posterior.py`: m' fresh probes with an independent seed; Z = Phi' A - Phi' A^ (the first term by
  real passes, the second by matrix products); per-column error estimates and a high-probability
  global bound. Derive the constant for Rademacher probes from the source (Halko, Martinsson and
  Tropp 2011, Section 4.3) and write the derivation in `docs/certificate.md`; test it by Monte Carlo.
- `neumann.py`: (a) the global bound of PROJECT.md 5.6; (b) the data-dependent per-node influence
  interval, propagated through the DAG in layer order from per-row error bounds (convert per-column
  estimates to per-row bounds conservatively, and say how).
- `refine.py`: exact rows for kept nodes and borderline nodes (interval crosses the threshold).

### 2.6 Graph (`graph/`)
- `influence.py`: row-normalised |A|, the finite sum B = A_bar + A_bar^2 + ... + A_bar^D computed by D
  sparse-dense products (never invert a dense T x T matrix when it is large), node influence w^T B.
- `prune.py`: node and edge pruning that reproduces circuit-tracer's pruning exactly (read its code;
  K0.4 checks equality on real graphs). Keep circuit-tracer's defaults unless a config says otherwise.
- `metrics.py`: influence-weighted node recall, node Jaccard, top-k overlap, Spearman of influence,
  relative Frobenius error on the exact pruned edge set, replacement and completeness scores (reuse
  circuit-tracer's implementation if present, otherwise implement from Ameisen et al. 2025 and test).
- `export.py`: write the recovered graph in circuit-tracer's graph format so its viewers work.

### 2.7 Baselines (`baselines/`)
- `truncation.py` (B1): given the exact run's visiting order, the graph that uses only the first c rows
  (others empty), for any c. On the VM also support a real run with `max_feature_nodes = c` and check
  that it equals the prefix (K0.6).
- `random_rows.py` (B2), `lowrank.py` (B3: randomised range finder with the same m, then A ~ Q (Q^T A)
  with rows from the sketch; check the exact construction gives a fair use of m passes),
  `colouring.py` (B4: greedy distance-2 colouring on the causal pattern; number of colours = passes),
  `total_effect.py` (B5: one backward pass from the logit through the linearised model with features
  unfrozen; node ranking only).
- B6 (plain Jacobian sensing) is CCT with masks, screening, hubs and certification switched off.

### 2.8 Group testing (`interventions/group_testing.py`)
Random group designs (Bernoulli inclusion with probability p), decoders (sparse regression with
non-negativity, and a standard non-adaptive group-testing decoder), and an exhaustive single-ablation
reference. Model calls go through a small `Ablator` interface (tiny backend on CPU; circuit-tracer
interventions on the VM).

### 2.9 Cost accounting (`cost.py`) and pipeline (`pipeline.py`)
- `cost.py`: one ledger per graph: mixed rows, exact rows, forward-mode passes, recovery seconds,
  certification rows, other seconds; conversion to row-equivalents with measured per-row time and
  measured forward-to-backward ratio kappa (both from K0.5, stored in the model config).
- `pipeline.py`: `run_cct(op, cfg) -> (graph, ledger, diagnostics)` doing sense, screen, recover, hubs,
  certify, optional refine, prune, export. Every switch in config so ablations are config changes.

### 2.10 Utilities (`utils/`)
`gpu.py` (`cap_memory`, `free_memory_gb`), `logging.py` (run record: git hash and dirty flag, package
versions, GPU, times), `seed.py`, `io.py` (atomic writes, resumable JSONL, fp16 matrix storage with an
fp32 checksum), `status.py`, `timing.py` (CUDA-synchronised timers).

### 2.11 Scripts and configs
Every script: `--config`, `--seed`, `--tiny`, `--force` (otherwise skip finished outputs), logs to
`logs/`, results with the run record.
- `run_exact_graphs.py`: exact graphs with circuit-tracer defaults; saves the full matrix A for the
  computed rows (fp16 on disk, fp32 in memory for analysis), the visiting order, active-feature
  metadata, logit weights, the pruned graph, wall time and peak memory.
- `analyse_structure.py` (K1), `simulate_sensing.py` (K2, on stored matrices through DenseOperator),
  `run_real_cct.py` (K3, real mixed passes), `sanity_k0.py`, `make_kill_report.py`.
- `run_kill_test.sh [--tiny] [--transcoders plt|clt|both]` chains them in order.
- `configs/prompts/kill_test.json`: the 32 test prompts; `validation.json`: 4 prompts; `long.json`:
  8 prompts of 128 to 256 tokens (PHASE1 Section 4). Commit them in P0.

---

## 3. Literature notes and the novelty gate (laptop, web allowed)
Write `docs/related_work_notes.md`, one block per work: what it does, cost profile, what object it
computes, numbers (copied exactly with location), and how CCT differs. Works: Ameisen et al. 2025
(methods paper, including its appendices on cost and pruning); Lindsey et al. 2025; circuit-tracer
documentation and source; Choromanski and Sindhwani 2017; sparse Jacobian colouring (Curtis, Powell
and Reid 1974; Gebremedhin, Manne and Pothen 2005, verify); Halko, Martinsson and Tropp 2011;
Dunefsky et al. 2024; Marks et al. 2024; Syed et al. 2023; CLT-Forge (2603.21014); HAGD
(2601.12879); Li (2611.00606); anything using attribution graphs at scale (for example
chain-of-thought verification work, arXiv 2510.09312, verify) that states per-graph cost. Then search
arXiv, OpenReview and Google Scholar for: "attribution graph" or "circuit tracing" with "compressed
sensing", "sketch", "random projection", "Jacobian sensing", "fewer backward passes", "efficient",
"scalable"; and "transcoder" with "Jacobian" and "sparse recovery".

**Novelty gate.** If any work already measures attribution-graph rows or columns with randomly mixed
passes (or any sensing scheme) and recovers the graph, stop and report to Dawar with the exact
overlap. Also report any work measuring the compressibility of attribution matrices (claim C1).

---

## 4. Tests (`tests/`, all CPU, no downloads)
| ID | Test | Pass condition |
|---|---|---|
| T1 | Dense operator | rows(S) = S A and cols(U) = U A^T; counters correct |
| T2 | Tiny backend against brute force | exact rows equal `brute_force_A()` (relative 1e-6, float64), PLT and CLT |
| T3 | Linearity of mixed passes | rows(sum of signed one-hots) equals the signed sum of exact rows (1e-6); multi-location seeds work |
| T4 | Forward-mode columns | cols(one-hot) equals the matching column from exact rows (1e-6) |
| T5 | Causal mask | every entry outside the mask is exactly zero in the tiny backend; the mask never hides a nonzero |
| T6 | Recovery, exact sparse | masked OMP, CoSaMP, FISTA recover k-sparse synthetic columns with m = 4 k log(T / k): relative error <= 1e-6 (noiseless OMP) in at least 99% of columns |
| T7 | Recovery, compressible | error within a constant of sigma_k(a)_1 / sqrt(k) on power-law columns; batched equals per-column |
| T8 | Screening | energy estimates unbiased (Monte Carlo); kept set holds at least fraction q of the true energy in at least 95% of trials at the configured m |
| T9 | Hubs | dense synthetic columns are flagged and replaced with exact columns; counters charge them |
| T10 | Influence | finite sum equals (I - A_bar)^(-1) - I on small DAGs; A_bar^(D+1) = 0 |
| T11 | Neumann bounds | global bound and data-dependent intervals always contain the true influence change for random perturbations |
| T12 | Posterior estimate | coverage of the high-probability bound at least the stated level (Monte Carlo, 1,000 trials) |
| T13 | Pruning and metrics | pruning matches a slow reference implementation; metrics correct on hand-built graphs; recall and Jaccard are 1 for identical graphs |
| T14 | Baselines | truncation at c = T equals exact; low-rank sketch recovers exactly low-rank matrices; colouring is valid (no two columns of a colour share a nonzero row) |
| T15 | Group testing | on a synthetic model with k effects among N, decoders recover them with C k log N tests in at least 95% of trials |
| T16 | Prereg consistency | the YAML block in PREREG_KILL_TEST.md equals `configs/prereg_kill_test.yaml` |
| T17 | End-to-end tiny | `run_kill_test.sh --tiny` writes every output and a TINY report with the decision logic exercised |

## 5. Tiny end-to-end run
`bash scripts/run_kill_test.sh --tiny`: tiny backend, random token prompts, "exact graphs", structure
analysis, sensing simulation, a real mixed-pass run on the tiny backend, metrics, and the report with
the decision table filled. Header: TINY RUN, NUMBERS ARE MEANINGLESS.

## 6. VM bootstrap scripts (write now, run on the VM)
- `scripts/vm_setup.sh`: venv, installs, `nvidia-smi`, free disk under `$HF_HOME`, versions,
  `pytest -q`.
- `scripts/download_models.py --set gemma2_2b_plt|gemma2_2b_clt426k|...`: `snapshot_download` of the
  model (weights, config, tokenizer only) and the transcoder repository into `$HF_HOME`; records
  revisions in the model config. No token in any script; the token comes from `hf auth login` on the
  VM (verify the command for the installed huggingface_hub).

## 7. Finish P0
1. Commit PREREG_KILL_TEST.md, `configs/prereg_kill_test.yaml` and the prompt files in their own
   commit; write the hash into `results/STATUS.md`.
2. Push; STATUS.md: "P0 done, waiting for VM".
3. Report to Dawar: what is ready, test results, the novelty-gate outcome in three lines, what you
   learnt from circuit-tracer's source that affects the plan (especially seed injection and forward
   mode), anything still marked (verify), and the exact first commands for the VM.
