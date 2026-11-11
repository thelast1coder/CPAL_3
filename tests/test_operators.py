"""T1 dense operator; T2 tiny vs brute force; T3 linearity; T4 forward mode; T5 causal mask."""
import torch

from cct.operators.dense import DenseOperator


def rel(a, b):
    return float((a - b).norm() / b.norm())


def test_T1_dense_operator():
    g = torch.Generator().manual_seed(0)
    A = torch.randn(30, 40, generator=g, dtype=torch.float64)
    op = DenseOperator.from_arrays(A, torch.ones(30), torch.zeros(30), torch.zeros(40), torch.zeros(40),
                                   batch_size=7, dtype=torch.float64)
    S = torch.randn(10, 30, generator=g, dtype=torch.float64)
    U = torch.randn(9, 40, generator=g, dtype=torch.float64)
    assert torch.allclose(op.rows(S), S @ A)
    assert torch.allclose(op.cols(U), U @ A.T)
    assert torch.allclose(op.exact_rows(torch.tensor([3, 5])), A[[3, 5]])
    c = op.counters
    assert c.mixed_rows == 10 and c.exact_rows == 2 and c.forward_passes == 9
    assert c.backward_calls == 2 + 1 and c.forward_calls == 2


def test_T2_tiny_vs_brute_force(tiny_op):
    A = tiny_op.exact_matrix()
    B = tiny_op.brute_force_A()
    assert A.abs().max() > 0
    assert rel(A, B) <= 1e-6
    assert float(((A - B).abs().amax(1) / B.abs().amax(1).clamp_min(1e-30)).max()) <= 1e-6


def test_T3_linearity_multilocation(tiny_op):
    A = tiny_op.exact_matrix()
    g = torch.Generator().manual_seed(2)
    S = (torch.randint(0, 2, (6, tiny_op.T), generator=g) * 2 - 1).double()
    assert rel(tiny_op.rows(S), S @ A) <= 1e-6
    # targets at many layers and positions in one seed (multi-location)
    assert tiny_op.target_meta.layer.unique().numel() > 2 and tiny_op.target_meta.pos.unique().numel() > 2
    # seeds summing several one-hots at the SAME location still add
    k = tiny_op.n_logits
    same = torch.nonzero((tiny_op.target_meta.layer == tiny_op.target_meta.layer[k]) &
                         (tiny_op.target_meta.pos == tiny_op.target_meta.pos[k])).flatten()
    s = torch.zeros(1, tiny_op.T, dtype=torch.float64)
    s[0, same] = torch.linspace(-1, 1, same.numel(), dtype=torch.float64)
    assert rel(tiny_op.rows(s), s @ A) <= 1e-6


def test_T4_forward_mode_columns(tiny_op):
    A = tiny_op.exact_matrix()
    idx = torch.arange(tiny_op.N)
    C = tiny_op.exact_cols(idx)
    assert rel(C, A.T) <= 1e-6
    assert tiny_op.counters.forward_passes >= tiny_op.N


def test_T5_causal_mask(tiny_op):
    A = tiny_op.exact_matrix()
    M = tiny_op.causal_mask().dense()
    assert float(A[~M].abs().max()) == 0.0  # nothing outside the mask
    assert bool((A != 0).sum() <= M.sum())
    # columns subset helper equals the dense mask
    idx = torch.tensor([0, 3, tiny_op.N - 1])
    assert torch.equal(tiny_op.causal_mask().columns(idx), M[:, idx])
