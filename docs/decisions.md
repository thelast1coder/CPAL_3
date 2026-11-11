# Decisions log (implementation details; project-level decisions live in PROJECT.md Section 13)

## Open questions for Dawar (block or affect P1)
| ID | Question | Current default | Why it matters |
|---|---|---|---|
| Q1 | Which circuit-tracer source to pin? PyPI has only 0.5.0, whose metadata points to a fork (`Caerii/circuit-tracer`); the main repo `decoderesearch/circuit-tracer` has tags up to v0.5.3 with a refactored `attribute.py` (same defaults). | `circuit-tracer==0.5.0` from PyPI (the source the backend was written against) | Provenance of the ground truth. Installing from a git tag instead would need the backend re-read against that tag before K0. |
| Q2 | Gemma-2-2B PLT transcoder repository: PROJECT.md says `mntss/gemma-scope-transcoders`; circuit-tracer 0.5.0's preset `"gemma"` resolves to `mwhanna/gemma-scope-transcoders`. | config uses `mwhanna/gemma-scope-transcoders` (spec ID kept as `transcoder_set_spec`) | A model/transcoder ID change needs your approval (CLAUDE.md). Check on the VM with `download_models.py --dry-run` which exists. |
| Q3 | GitHub remote URL for the private repository. | local commits only | Push and the VM clone. |
| Q4 | Rhyme prompt `kill_test_20` ("He saw a carrot...") was not found verbatim in the Anthropic papers or demos. | kept, marked `unverified` in the prompt file | Prompts freeze at the PREREG commit. Replace it before then if you want only verified canonical strings. |

## Implementation decisions
| ID | Date | Decision | Reason |
|---|---|---|---|
| I1 | 10 Nov | Laptop: no system PyTorch existed and the disk had 32 GB free (not 0.5 GB), so CPU-only torch 2.x was installed in `.venv` (`--index-url .../whl/cpu`) with numpy, scipy, pyyaml, pytest. No model, transcoder or dataset was downloaded. | Needed to run the CPU tests; within the spirit of "small environment". |
| I2 | 10 Nov | Pin `circuit-tracer==0.5.0` (latest on PyPI on 10 Nov 2026), TransformerLens backend. See Q1. | Source read from that wheel. |
| I3 | 10 Nov | B0 uses the CLI defaults (cap 7,500, batch 256, 10 logits, p 0.95, thresholds 0.8 / 0.98), passed explicitly; `attribute()`'s own defaults (batch 512, no cap) differ. | PROJECT.md's stated defaults are the CLI ones. |
| I4 | 10 Nov | Mixed seeds use an additive multi-location injection (own tensor hooks) instead of circuit-tracer's `index_put_` replacement; scores are read with circuit-tracer's own buffer and hooks. | Replacement corrupts mixed rows (downstream gradient overwritten, always for CLTs). See circuit_tracer_notes.md. |
| I5 | 10 Nov | Forward mode by `torch.autograd.forward_ad` through the hooked model; fallback double backward. K0.3 records which. | Spec 2.1. |
| I6 | 10 Nov | Exact-graph runs reproduce `attribute.py`'s visiting loop with circuit-tracer's `compute_batch` and `compute_partial_influences`, saving the visiting order; regime F ranks by influence even with no cap (circuit-tracer visits in index order then). | Needed for B1 prefixes; the matrix is unchanged, only the order. |
| I7 | 10 Nov | Target order everywhere: [logits, features in visiting order]; source order: [selected features sorted, errors layer-major, tokens], equal to circuit-tracer's node order, so `to_square` reproduces its adjacency. B1 budget c counts all rows, logits included. | One layout for every method. |
| I8 | 10 Nov | CCT computes the (at most 10) logit rows exactly (cost counted) and senses the feature-target rows only. | Logit rows carry the weights w; tiny cost. |
| I9 | 10 Nov | Effective k_max = min(k_max, m // 2); a column is a hub if its relative residual exceeds rho_hub or its support reaches k_max. | Found on the tiny run: with k_max close to m, OMP interpolates any column with zero residual and no hub is ever flagged. |
| I10 | 10 Nov | Every estimate (CCT without mask, B3 low-rank) is multiplied by the causal mask before influence and pruning. | Found on the tiny run: without it the graph is not a DAG and influence never terminates. For B6 / A3 the mask is still not used inside recovery. |
| I11 | 10 Nov | Screening keeps the columns holding 1 - (1 - q) / safety of the estimated energy, safety = 2. | So the kept set holds at least q of the true energy with high probability (T8); safety = 1 is the plain rule. |
| I12 | 10 Nov | Certification probes are Rademacher with the Paley-Zygmund bound derived in certificate.md (constant 7.5 at m' = 16, delta 0.05; 10.9 at delta 0.01). Halko et al.'s 10 sqrt(2/pi) is for Gaussian probes. | Literature check flagged the mismatch; T12 confirms coverage. |
| I13 | 10 Nov | Per-node influence intervals from interval arithmetic on the row-normalised matrix, propagated through the finite Neumann series (monotone in nonnegative entries). | Spec 2.5(b); conservative column-to-entry conversion |A - A_hat|[t,s] <= c_s. |
| I14 | 10 Nov | Matrix storage: fp16 of each row divided by its max, plus an fp32 row scale and checksums. | Plain fp16 flushes small edges to zero, which would bias K1 towards "sparse". |
| I15 | 10 Nov | B3 low-rank: m // 2 forward-mode probes for the range, the rest mixed backward passes, A ~ Q (Phi Q)^+ Y; cost m_b + kappa m_f. | Fair single-pass sketch at the same budget. |
| I16 | 10 Nov | B5 (total effect) implemented on the tiny backend only; the circuit-tracer version is a P2 item. | Not used in the kill test. |
| I17 | 10 Nov | K3: sensing seeds 0, 1, 2 are also the three timing repeats; CCT wall time includes building the operator (forward pass). Exact time is circuit-tracer's `attribute()`. | Prereg: fidelity on 2 of 3 seeds, median of 3 repeats. |
| I18 | 10 Nov | K2 tuning grid reduced to 54 settings (algo 3 x k_max 3 x q 2 x rho 1 x H 3), written before any run. | About 3,200 simulated runs; keeps Step 1 near 1 GPU-hour (estimate). |
| I19 | 10 Nov | S_exact = median over prompts of T_D, divided by CCT's c75 cost. | Reading of the prereg sentence; stated so it is fixed before data. |
| I20 | 10 Nov | Long prompts are original paragraphs plus a question (not WikiText-2). Token counts checked with the real tokenizer at K0 and reported; nothing is changed after the PREREG commit. | No dataset downloads in P0. |
| I21 | 10 Nov | If no setting meets fidelity on all validation prompts, Step 1 falls back to the first grid setting and logs it. | Rule needed a defined failure case. |
