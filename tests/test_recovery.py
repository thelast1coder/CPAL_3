"""T6 exact sparse recovery; T7 compressible recovery and batching; T8 screening; T9 hubs."""
import math

import pytest
import torch

from cct.operators.base import CausalMask
from cct.operators.dense import DenseOperator
from cct.pipeline import run_cct
from cct.recovery.batched import recover_columns
from cct.recovery.omp import omp
from cct.sensing.matrices import sensing_matrix
from cct.sensing.screening import screen_columns


def sparse_problem(T=600, C=200, k=5, seed=0, masked=True):
    g = torch.Generator().manual_seed(seed)
    t_layer = torch.randint(1, 6, (T,), generator=g)
    t_pos = torch.randint(0, 8, (T,), generator=g)
    s_layer = torch.randint(0, 5, (C,), generator=g)
    s_pos = torch.randint(0, 8, (C,), generator=g)
    mask = CausalMask(t_layer, t_pos, s_layer, s_pos)
    M = mask.dense() if masked else torch.ones(T, C, dtype=torch.bool)
    X = torch.zeros(T, C, dtype=torch.float64)
    for c in range(C):
        allowed = torch.nonzero(M[:, c]).flatten()
        idx = allowed[torch.randperm(allowed.numel(), generator=g)[: min(k, allowed.numel())]]
        X[idx, c] = torch.randn(idx.numel(), generator=g, dtype=torch.float64) + torch.sign(
            torch.randn(idx.numel(), generator=g, dtype=torch.float64))
    return X, mask if masked else None


@pytest.mark.parametrize("algo,tol", [("omp", 1e-6), ("cosamp", 1e-6), ("fista", 1e-6), ("iht", 1e-6)])
def test_T6_exact_sparse(algo, tol):
    T, C, k = 600, 200, 5
    X, mask = sparse_problem(T, C, k)
    m = int(math.ceil(4 * k * math.log(T / k)))
    Phi = sensing_matrix("rademacher", m, T, 0, dtype=torch.float64)
    Y = Phi @ X
    kmax = 2 * k if algo == "omp" else k
    rec, _ = recover_columns(Phi, Y, torch.arange(C), mask, algo, k_max=kmax, tol=1e-10, dtype=torch.float64)
    Xh = rec.to_dense(T)
    err = (Xh - X).norm(dim=0) / X.norm(dim=0)
    assert float((err <= tol).double().mean()) >= 0.99
    # masked recovery never puts mass outside the mask
    assert float(Xh[~mask.dense()].abs().max()) == 0.0


def test_T7_compressible_and_batched():
    T, C, m, k = 500, 120, 160, 20
    g = torch.Generator().manual_seed(3)
    X = torch.zeros(T, C, dtype=torch.float64)
    for c in range(C):
        perm = torch.randperm(T, generator=g)
        X[perm, c] = torch.sign(torch.randn(T, generator=g, dtype=torch.float64)) * \
            torch.arange(1, T + 1, dtype=torch.float64) ** -1.5
    Phi = sensing_matrix("rademacher", m, T, 1, dtype=torch.float64)
    Y = Phi @ X
    rec = omp(Phi, Y, None, k_max=k, tol=0.0)
    Xh = rec.to_dense(T)
    err = (Xh - X).norm(dim=0)
    tail = torch.sort(X.abs(), dim=0, descending=True).values[k:].sum(0)
    ratio = err / (tail / math.sqrt(k))
    assert float(ratio.quantile(0.95)) <= 5.0  # error within a constant of sigma_k(a)_1 / sqrt(k)
    # batched (chunked) equals per-column
    rb, _ = recover_columns(Phi, Y, torch.arange(C), None, "omp", k_max=k, tol=0.0, chunk=17, dtype=torch.float64)
    single = torch.cat([omp(Phi, Y[:, [c]], None, k_max=k, tol=0.0).to_dense(T) for c in range(0, C, 10)], 1)
    assert torch.allclose(rb.to_dense(T), Xh, atol=1e-10)
    assert torch.allclose(rb.to_dense(T)[:, ::10], single, atol=1e-10)


def test_T8_screening():
    T, C, m = 400, 300, 256
    g = torch.Generator().manual_seed(4)
    a = torch.randn(T, generator=g, dtype=torch.float64)
    est = torch.stack([(sensing_matrix("rademacher", m, T, s, dtype=torch.float64) @ a).pow(2).sum()
                       for s in range(400)])
    assert abs(float(est.mean()) / float(a.pow(2).sum()) - 1) < 0.02  # unbiased
    q, ok = 0.995, 0
    scales = torch.arange(1, C + 1, dtype=torch.float64) ** -1.2
    A = torch.randn(T, C, generator=g, dtype=torch.float64) * (torch.rand(T, C, generator=g) < 0.05) * scales
    true_e = A.pow(2).sum(0)
    trials = 100
    for s in range(trials):
        Y = sensing_matrix("rademacher", m, T, 100 + s, dtype=torch.float64) @ A
        keep = screen_columns(Y, q, safety=2.0)
        ok += float(true_e[keep].sum() / true_e.sum()) >= q
    assert ok / trials >= 0.95


def test_T9_hubs():
    g = torch.Generator().manual_seed(5)
    T, N = 300, 80
    A = torch.zeros(T, N, dtype=torch.float64)
    for c in range(N):
        idx = torch.randperm(T, generator=g)[:3]
        A[idx, c] = torch.randn(3, generator=g, dtype=torch.float64)
    hubs = torch.tensor([7, 21, 50])
    A[:, hubs] = torch.randn(T, 3, generator=g, dtype=torch.float64)  # dense columns
    op = DenseOperator.from_arrays(A, torch.ones(T), torch.zeros(T), torch.zeros(N), torch.zeros(N),
                                   dtype=torch.float64)
    from cct.operators.base import KIND_LOGIT
    op.target_meta.kind[:2] = KIND_LOGIT
    op.target_meta.layer[:2] = 2
    op.target_meta.extra[:2] = torch.tensor([0.5, 0.3])
    res = run_cct(op, {"m": 96, "screening": {"enabled": False}, "certify": {"enabled": False},
                       "recovery": {"k_max": 16, "dtype": "float64"}, "hubs": {"budget": 10, "rho_hub": 0.2}},
                  seed=0)
    assert set(hubs.tolist()) <= set(res.diagnostics["hub_ids"].tolist())  # flagged
    assert torch.allclose(res.A_hat[:, hubs], A[:, hubs])  # replaced by exact columns
    assert res.ledger.forward_passes == res.diagnostics["n_hubs"]  # charged
    assert res.ledger.mixed_rows == 96 and res.ledger.exact_rows == 2
