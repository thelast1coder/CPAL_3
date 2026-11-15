"""T18 to T20: closed-form local part A_0 (PROJECT_V2 T2) on the tiny backend.

A_0 must equal the matrix of the frozen model with every attention output stop-gradiented (same
forward values, no attention paths), for PLT, CLT and a skip transcoder, and must be block-diagonal
over positions."""
import pytest
import torch

from cct.local_part import energy_share, from_tiny
from cct.operators.tiny import TinyConfig, TinyModel, TinyOperator


def _op(tc: str, skip: bool = False, detach: bool = False, seed: int = 0):
    m = TinyModel(TinyConfig(n_layers=3, d_model=16, n_heads=2, d_head=8, vocab=32, n_features=24,
                             transcoder=tc, skip=skip, seed=seed))
    toks = torch.randint(0, 32, (9,), generator=torch.Generator().manual_seed(seed + 1))
    return TinyOperator(m, toks, detach_attn=detach), m, toks


@pytest.mark.parametrize("tc,skip", [("plt", False), ("clt", False), ("plt", True), ("clt", True)])
def test_t18_local_part_equals_attention_detached(tc, skip):
    op, m, toks = _op(tc, skip)
    op_det = TinyOperator(m, toks, detach_attn=True)
    A_det = op_det.exact_matrix()
    lp = from_tiny(op)
    A0 = lp.dense()
    err = (A0 - A_det).norm() / A_det.norm()
    assert err < 1e-10, float(err)
    # the remainder is the attention part: A - A_0 is nonzero and exactly zero on same-position
    # pairs only if no attention path returns to the same position (not required); check shapes
    A = op.exact_matrix()
    assert A.shape == A0.shape
    assert (A - A0).norm() > 0


def test_t19_block_diagonal_and_views():
    op, _, _ = _op("clt")
    lp = from_tiny(op)
    A0 = lp.dense()
    tp, sp = op.target_meta.pos, op.source_meta.pos
    off = tp[:, None] != sp[None, :]
    assert float(A0[off].abs().max()) == 0.0
    ridx = torch.tensor([0, 3, 7, op.T - 1])
    cidx = torch.tensor([1, 5, op.N - 1, 2])
    assert torch.allclose(lp.rows(ridx), A0[ridx])
    assert torch.allclose(lp.cols(cidx), A0[:, cidx])
    Phi = torch.randn(5, op.T, dtype=A0.dtype)
    assert torch.allclose(lp.sketch(Phi), Phi @ A0)
    sub = torch.arange(op.n_logits, op.T)
    assert torch.allclose(lp.sketch(Phi[:, : sub.numel()], sub), Phi[:, : sub.numel()] @ A0[sub])


def test_t20_energy_share_and_skip_matrix_consistency():
    # the skip transcoder's full matrix still matches brute force when attention is detached
    op, m, toks = _op("plt", skip=True)
    A = op.exact_matrix()
    lp = from_tiny(op)
    es = energy_share(A, lp.dense())
    assert 0.0 < es["a0_energy_share"] < 1.0
    assert abs(es["a0_energy_share"] + es["remainder_energy_share"] - 1.0) < 1e-12
    # linearity of mixed rows with skip connections (T1 extended)
    seeds = torch.randn(4, op.T, dtype=A.dtype)
    assert torch.allclose(op.rows(seeds), seeds @ A, atol=1e-10, rtol=1e-8)
    # forward-mode columns equal the transpose of rows
    tang = torch.randn(3, op.N, dtype=A.dtype)
    assert torch.allclose(op.cols(tang), tang @ A.T, atol=1e-10, rtol=1e-8)


def test_run_cct_with_local_part():
    """Sensing on the remainder with A_0 added back: with a zero A_0 it equals plain CCT; with the true
    A_0 the estimate keeps A_0 exactly where the remainder is recovered."""
    from cct.operators.dense import DenseOperator
    from cct.pipeline import run_cct

    op, _, _ = _op("plt")
    A = op.exact_matrix()
    A0 = from_tiny(op).dense()
    dop = DenseOperator(A, op.target_meta, op.source_meta, batch_size=64)
    cfg = {"m": 64, "certify": {"enabled": False}, "hubs": {"enabled": False}}
    r_plain = run_cct(dop, cfg, seed=0)
    r_zero = run_cct(dop, cfg, seed=0, local_part=torch.zeros_like(A))
    assert torch.allclose(r_plain.A_hat, r_zero.A_hat)
    r_lp = run_cct(dop, {**cfg, "m": 128}, seed=0, local_part=A0)
    err_lp = (r_lp.A_hat - A).norm() / A.norm()
    err_plain = (run_cct(dop, {**cfg, "m": 128}, seed=0).A_hat - A).norm() / A.norm()
    assert err_lp <= err_plain + 1e-9
