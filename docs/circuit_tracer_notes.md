# circuit-tracer 0.5.0: what the attribution code does

Source read: the PyPI wheel `circuit_tracer-0.5.0-py3-none-any.whl` (downloaded without dependencies,
read, then deleted). File and function names below refer to that wheel. Caveat found in the literature
check: the PyPI metadata points to a fork (`Caerii/circuit-tracer`), while the main repository
(`decoderesearch/circuit-tracer`, formerly `safety-research`) has tags up to v0.5.3 and a refactored
`attribute.py`. Which source to pin is open question Q1 in `decisions.md`.

## Defaults
| Setting | `attribute()` (Python) | CLI (`__main__.py`) | Used for B0 |
|---|---|---|---|
| max_n_logits | 10 | 10 | 10 |
| desired_logit_prob | 0.95 | 0.95 | 0.95 |
| batch_size | 512 | 256 | 256 |
| max_feature_nodes | None (no cap) | 7500 | 7500 |
| update_interval | 4 | (not exposed) | 4 |
| node / edge threshold (`prune_graph`) | 0.8 / 0.98 | 0.8 / 0.98 | 0.8 / 0.98 |
PROJECT.md's "cap 7,500, batch 256" is the CLI default. We pass all values explicitly.

## Pipeline (`attribution/attribute.py`, `_run_attribution`)
1. `model.setup_attribution(tokens)` (`replacement_model_transformerlens.py`): one forward pass caching
   MLP inputs and outputs; `transcoders.compute_attribution_components` returns the sparse activation
   matrix (n_layers, n_pos, d_transcoder), **encoder vectors** (raw encoder rows, not scaled), **decoder
   vectors scaled by activation** (for CLTs one per (feature, output layer), with
   `encoder_to_decoder_map` and `decoder_locations`), error vectors (MLP output minus reconstruction)
   and token vectors `W_E[tokens]`. Position 0 features and errors are zeroed (`zero_positions`).
2. `model.run_forward_pass(tokens, batch_size, ctx)`: forward on the prompt **expanded to the batch
   size**, under `ctx.install_hooks`, caching `_resid_activations[l]` at each layer's
   `feature_input_hook` and `ln_final(residual)` as layer n_layers. The graph is retained for all
   later backward passes; backward score hooks are attached during this forward.
3. Targets (`attribution/targets.py`, `_from_salient`): top logits until cumulative probability 0.95,
   at most 10. Logit read vectors are **demeaned unembedding columns** (`W_U[:, i] - W_U.mean(-1)`).
4. Logit rows, then feature rows in batches (`ctx.compute_batch`). Visiting order: re-rank unvisited
   features by `compute_partial_influences` (power iteration on the rows computed so far) every
   `update_interval * batch_size` rows. **If the cap equals the number of active features (cap None),
   features are visited in index order, not influence order.**
5. Packaging: unvisited features are dropped as nodes entirely (rows and columns); the adjacency is
   square over [selected features, errors (layer-major), tokens, logits]; rows = targets,
   columns = sources; error and token rows are zero.

## Freezing (`_configure_gradient_flow`, `_configure_skip_connection`)
Stop-gradient (`.detach()`) on `attn.hook_pattern`, `ln1/ln2(/ln1_post/ln2_post).hook_scale`,
`ln_final.hook_scale`; MLP output replaced by `skip + (acts - skip).detach()` (skip = transcoder skip
connection if present, else 0), with a `hook_out_grad` HookPoint on the result, which is the
`feature_output_hook` where scores are read. All parameters have `requires_grad=False`; `hook_embed`
re-enables gradient.

## Seed injection (`attribution/context_transformerlens.py`, `compute_batch`)
- Per layer in the batch, a tensor hook on `_resid_activations[layer]` runs
  `grads.clone().index_put_((batch_idx, pos_idx), values)`: it **replaces** the gradient at one
  (batch row, position) per target. Backward starts from the highest layer in the batch with a zero
  gradient.
- Scores are read by backward hooks: `einsum(grads[:, positions], vectors)` at each layer's
  `feature_output_hook` (features via decoder vectors, errors via error vectors) and at `hook_embed`
  (tokens), accumulated into a [row_size, batch] buffer.
- **Consequence for CCT:** replace equals add only when nothing downstream in the same batch row
  injected. In a mixed row, later-layer targets send gradient into earlier injection points; for CLTs
  the input hook is `hook_resid_mid`, which carries that gradient, so replacing would corrupt the row.
  `CircuitTracerOperator._rows_impl` registers its own hooks that **add** a dense [B, n_pos, d]
  injection per layer, then reuses the context's buffer and score hooks unchanged. K0.1 checks one-hot
  rows against circuit-tracer's adjacency; K0.2 checks mixed rows against `seed @ A`.

## Forward mode
Not used by circuit-tracer. `CircuitTracerOperator._cols_fwad` adds dual tangents at the
feature-output hooks and `hook_embed` and reads them at the feature-input hooks and `ln_final`. The
`.detach()` stop-gradients also drop tangents, so the map is the same frozen linear map. Risks: a hook
or op without a forward-AD formula (for example in Gemma-2's soft-capping or the TransformerLens hook
machinery). Fallback `_cols_double_backward` (VJP of a VJP) is implemented; K0.3 records which runs.

## Pruning and scores (`graph.py`)
`normalize_matrix` (|A| row-normalised, clamp 1e-10), `compute_influence` (w A + w A^2 + ... until
zero), `find_threshold` (cumulative share), `prune_graph` (node mask at 0.8 of influence, logits and
tokens always kept; edge scores = normalised pruned matrix times (influence + w), threshold 0.98;
iterative removal of features without incoming or outgoing edges), `compute_graph_scores`
(replacement and completeness). `cct.graph.prune` and `cct.graph.metrics.graph_scores` copy these;
K0.4 checks equality on real graphs.

## NNSight backend
`context_nnsight.py` restructures the backward (explicit per-layer loop, `.grad` on proxies). Our
backend targets the TransformerLens backend only (Gemma-2, Llama-3.2). Gemma-3 (optional) would need a
second implementation.
