"""Count-sketch (D2) on synthetic sparse columns with a known tail (GATE0_PREMISES Section 6)."""
import torch

from cct.sketch import CountSketch, column_rel_err


def _sparse_cols(T=4000, S=40, k=10, tail=1e-3, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = tail * torch.randn(T, S, generator=g, dtype=torch.float64)
    for s in range(S):
        idx = torch.randperm(T, generator=g)[:k]
        A[idx, s] += torch.randn(k, generator=g, dtype=torch.float64) * 3
    return A


def test_apply_equals_dense_seeds():
    A = _sparse_cols(T=500, S=7)
    cs = CountSketch(500, 96, R=3, seed=1)
    assert torch.allclose(cs.apply(A), cs.seeds(torch.float64) @ A)


def test_recovers_heavy_entries_and_error_falls_with_m():
    A = _sparse_cols()
    errs = []
    for m in (100, 400, 1600):
        cs = CountSketch(A.shape[0], m, R=5, seed=2)
        A_hat, sec = cs.decode(cs.apply(A), keep=20)
        errs.append(float(column_rel_err(A_hat, A).nanmedian()))
        assert sec >= 0
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.05  # tail energy is about 4000 * 1e-6 against 10 * 9


def test_candidate_mask_restricts_support():
    A = _sparse_cols(T=1000, S=5)
    cand = torch.zeros_like(A, dtype=torch.bool)
    cand[:500] = True
    cs = CountSketch(1000, 300, seed=3)
    A_hat, _ = cs.decode(cs.apply(A), cand=cand)
    assert float(A_hat[500:].abs().max()) == 0.0
