"""Attention reachability and candidate target sets (PROJECT_V2.md Sections 3 and 5, step 4).

rho(q <- p) between a write layer a and a read layer b is attention rollout (Abnar and Zuidema 2020,
verify) over the attention layers a path from the write to the read passes: layers a+1 .. b for a
feature target read at layer b (the read is after layer b's attention), a+1 .. L-1 for logits
(b = L). Each layer's matrix is R_l = r I + (1 - r) sum_h w_lh pattern_lh, with w_lh proportional to
the Frobenius norm of the head's OV map (W_V W_O), normalised per layer, so each R_l is row-stochastic
(rows: query positions, columns: key positions).

The candidate set of a source s at position p with write layers W_s is
    Omega_s = {t : pos(t) = p} union {t : max over a in W_s, a < l_t, of rho_{a -> l_t}(pos(t) <- p) >= tau}.
Same-position targets are always candidates (A_0 lives there and is known exactly).
"""
from __future__ import annotations

import torch
from torch import Tensor


def head_weights(W_V: Tensor, W_O: Tensor) -> Tensor:
    """W_V [L, H, d, dh], W_O [L, H, dh, d] -> [L, H] OV Frobenius norms, normalised per layer."""
    # ||W_V W_O||_F^2 = trace((W_V^T W_V)(W_O W_O^T))
    a = torch.einsum("lhdi,lhdj->lhij", W_V.double(), W_V.double())
    b = torch.einsum("lhid,lhjd->lhij", W_O.double(), W_O.double())
    nrm = torch.einsum("lhij,lhji->lh", a, b).clamp_min(0).sqrt()
    return nrm / nrm.sum(1, keepdim=True).clamp_min(1e-300)


def rollout_layers(patterns: Tensor, hw: Tensor, residual: float = 0.5) -> Tensor:
    """patterns [L, H, n, n] (query x key, rows sum to 1), hw [L, H] -> R [L, n, n] row-stochastic."""
    patterns = patterns.double()
    n = patterns.shape[-1]
    mix = torch.einsum("lh,lhqk->lqk", hw.double().to(patterns.device), patterns)
    eye = torch.eye(n, dtype=torch.float64, device=patterns.device)
    R = residual * eye[None] + (1 - residual) * mix
    return R / R.sum(-1, keepdim=True).clamp_min(1e-300)


def reach_from(R: Tensor, a: int, p: Tensor) -> Tensor:
    """For write layer a (-1 for tokens) and source positions p [S]: X [L + 1, n, S] where X[b, q, s] is
    rho_{a -> b}(q <- p_s); zero for b <= a (no causal path). b = L is the logit read."""
    L, n, _ = R.shape
    S = p.numel()
    X = torch.zeros(L + 1, n, S, dtype=R.dtype, device=R.device)
    x = torch.zeros(n, S, dtype=R.dtype, device=R.device)
    x[p.to(R.device), torch.arange(S, device=R.device)] = 1.0
    for j in range(a + 1, L):
        x = R[j] @ x  # through layer j's attention; a feature read at layer j sees it
        X[j] = x
    if a < L:
        X[L] = x  # logits read after the last layer (no further attention)
    return X


def reach_columns(R: Tensor, write_layers: list[list[int]], pos: Tensor) -> Tensor:
    """Max reach over each source's write layers: [L + 1, n, S]."""
    L, n, _ = R.shape
    S = len(write_layers)
    out = torch.zeros(L + 1, n, S, dtype=R.dtype, device=R.device)
    groups: dict[int, list[int]] = {}
    for s, wl in enumerate(write_layers):
        for a in wl:
            groups.setdefault(int(a), []).append(s)
    for a, ss in groups.items():
        idx = torch.tensor(ss, dtype=torch.long)
        X = reach_from(R, a, pos[idx])
        cur = out[:, :, idx.to(R.device)]
        out[:, :, idx.to(R.device)] = torch.maximum(cur, X)
    return out


def candidate_mask(R: Tensor, write_layers: list[list[int]], src_pos: Tensor, t_layer: Tensor, t_pos: Tensor,
                   tau: float) -> Tensor:
    """Bool [T, S]: target t is a candidate for source s (Omega_s, see module docstring)."""
    X = reach_columns(R, write_layers, src_pos)  # [L+1, n, S]
    tl = t_layer.to(R.device)
    tp = t_pos.to(R.device)
    rho = X[tl, tp]  # [T, S]
    same = tp[:, None] == src_pos.to(R.device)[None, :]
    # causal: some write layer below the read layer (rho is zero otherwise, but same-position
    # candidates must also respect it)
    min_w = torch.tensor([min(w) for w in write_layers], device=R.device)
    causal = tl[:, None] > min_w[None, :]
    return ((rho >= tau) | same) & causal


def screening_recall(A_cols: Tensor, mask: Tensor) -> dict:
    """A_cols [T, S] exact columns, mask [T, S]: energy-weighted recall and mean |Omega|/T."""
    e = A_cols.double().pow(2)
    tot = float(e.sum())
    inside = float((e * mask.to(e.device)).sum())
    per = (e * mask.to(e.device)).sum(0) / e.sum(0).clamp_min(1e-300)
    has = e.sum(0) > 0
    return {"recall": inside / tot if tot > 0 else float("nan"),
            "recall_median_per_column": float(per[has].median()) if bool(has.any()) else float("nan"),
            "omega_frac_T": float(mask.double().mean())}


def tiny_rollout(op, residual: float = 0.5) -> Tensor:
    m = op.model
    pats = torch.stack([op.cache["pattern"][l][0] for l in range(op.cfg.n_layers)])
    return rollout_layers(pats, head_weights(m.W_V, m.W_O), residual)


def source_write_layers(source_meta, transcoder: str, n_layers: int) -> list[list[int]]:
    """Write layers per source: PLT feature and error at their layer, CLT feature at its layer and every
    later layer, tokens at -1."""
    from .operators.base import KIND_FEATURE, KIND_TOKEN

    out = []
    for k, l in zip(source_meta.kind.tolist(), source_meta.layer.tolist()):
        if k == KIND_TOKEN:
            out.append([-1])
        elif k == KIND_FEATURE and transcoder == "clt":
            out.append(list(range(int(l), n_layers)))
        else:
            out.append([int(l)])
    return out
