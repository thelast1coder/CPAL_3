"""T21: attention reachability (rollout) and candidate sets on the tiny backend."""
import torch

from cct.local_part import from_tiny
from cct.operators.tiny import TinyConfig, TinyModel, TinyOperator
from cct.reach import (candidate_mask, head_weights, reach_from, rollout_layers, screening_recall,
                       source_write_layers, tiny_rollout)


def _op(tc="plt"):
    m = TinyModel(TinyConfig(n_layers=3, d_model=16, n_heads=2, d_head=8, vocab=32, n_features=24, transcoder=tc))
    toks = torch.randint(0, 32, (10,), generator=torch.Generator().manual_seed(3))
    return TinyOperator(m, toks)


def test_rollout_row_stochastic_and_products():
    op = _op()
    R = tiny_rollout(op)
    assert torch.allclose(R.sum(-1), torch.ones_like(R.sum(-1)))
    assert float(R.triu(1).abs().max()) == 0.0  # causal
    hw = head_weights(op.model.W_V, op.model.W_O)
    assert torch.allclose(hw.sum(1), torch.ones(hw.shape[0], dtype=hw.dtype))
    L, n, _ = R.shape
    p = torch.tensor([0, 4, 9])
    X = reach_from(R, 0, p)
    E = torch.zeros(n, 3, dtype=R.dtype)
    E[p, torch.arange(3)] = 1
    assert torch.allclose(X[2], R[2] @ R[1] @ E)
    assert torch.allclose(X[L], R[2] @ R[1] @ E)
    assert float(X[0].abs().max()) == 0.0 and float(X[1].abs().max()) > 0
    Xt = reach_from(R, -1, p)
    assert torch.allclose(Xt[0], R[0] @ E)


def test_candidate_mask_contains_causal_support_at_tau_zero():
    for tc in ("plt", "clt"):
        op = _op(tc)
        A = op.exact_matrix()
        R = tiny_rollout(op)
        wl = source_write_layers(op.source_meta, tc, op.cfg.n_layers)
        mask = candidate_mask(R, wl, op.source_meta.pos, op.target_meta.layer, op.target_meta.pos, tau=0.0)
        assert float(A[~mask].abs().max()) == 0.0  # every nonzero edge is reachable
        # A_0 is inside Omega at any tau (same position always kept)
        mask_hi = candidate_mask(R, wl, op.source_meta.pos, op.target_meta.layer, op.target_meta.pos, tau=0.9)
        A0 = from_tiny(op).dense()
        assert float(A0[~mask_hi].abs().max()) == 0.0
        r = screening_recall(A, mask)
        assert abs(r["recall"] - 1.0) < 1e-12
        r_hi = screening_recall(A - A0, mask_hi)
        assert 0.0 <= r_hi["recall"] <= 1.0 and r_hi["omega_frac_T"] <= r["omega_frac_T"]


def test_rollout_identity_residual():
    pats = torch.softmax(torch.randn(2, 3, 5, 5).masked_fill(~torch.ones(5, 5, dtype=torch.bool).tril(), -1e9), -1)
    R = rollout_layers(pats, torch.full((2, 3), 1 / 3), residual=1.0)
    assert torch.allclose(R, torch.eye(5, dtype=R.dtype).expand(2, 5, 5))
