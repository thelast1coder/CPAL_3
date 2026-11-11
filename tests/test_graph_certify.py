"""T10 influence; T11 Neumann bounds; T12 posterior coverage; T13 pruning and metrics."""
import math

import torch

from cct.certify.neumann import entry_intervals, global_bound, influence_interval
from cct.certify.posterior import bound_constant, failure_prob, posterior_from_Z
from cct.graph.influence import compute_influence, neumann_B, nilpotency_index, normalize_matrix
from cct.graph.metrics import graph_scores, iw_node_recall, node_jaccard, spearman, topk_overlap
from cct.graph.prune import prune_adjacency, prune_reference
from cct.sensing.matrices import sensing_matrix

from conftest import random_dag


def test_T10_influence_finite_sum():
    for seed in range(10):
        A, nf, nt, probs, mask = random_dag(25, 0.4, seed)
        Ab = normalize_matrix(A)
        n = Ab.shape[0]
        B = neumann_B(Ab)
        inv = torch.linalg.inv(torch.eye(n, dtype=torch.float64) - Ab) - torch.eye(n, dtype=torch.float64)
        assert torch.allclose(B, inv, atol=1e-10)
        D = nilpotency_index(Ab) - 1  # longest path in edges
        assert not bool(torch.linalg.matrix_power(Ab, D + 1).any())
        assert D <= 7  # layers 0..6 used in random_dag
        w = torch.zeros(n, dtype=torch.float64)
        w[-2:] = probs
        assert torch.allclose(compute_influence(Ab, w)[0], w @ B)


def test_T11_neumann_bounds():
    worst = 0.0
    for seed in range(100):
        A, nf, nt, probs, mask = random_dag(20, 0.5, seed)
        n = A.shape[0]
        w = torch.zeros(n, dtype=torch.float64)
        w[-2:] = probs
        g = torch.Generator().manual_seed(1000 + seed)
        cb = 0.05 * torch.rand(n, generator=g, dtype=torch.float64) * A.abs().max()
        E = (2 * torch.rand(n, n, generator=g, dtype=torch.float64) - 1) * cb[None, :] * mask
        A2 = A + E  # true matrix; A is the estimate with |error| <= cb per column
        Ab, Ab2 = normalize_matrix(A), normalize_matrix(A2)
        infl, infl2 = compute_influence(Ab, w)[0], compute_influence(Ab2, w)[0]
        # (a) global bound
        eps = float((Ab2 - Ab).abs().sum(1).max())
        D = nilpotency_index((Ab + Ab2) != 0) - 1
        gb = global_bound(eps, D)
        diff = float((infl2 - infl).abs().sum())
        assert diff <= gb + 1e-12
        worst = max(worst, diff / max(gb, 1e-30))
        # (b) data-dependent intervals contain the true influence
        Lo, Hi = entry_intervals(A, cb, mask)
        lo, hi = influence_interval(Lo, Hi, w)
        assert bool((lo <= infl2 + 1e-12).all() and (infl2 <= hi + 1e-12).all())
        assert bool((lo <= infl + 1e-12).all() and (infl <= hi + 1e-12).all())


def test_T12_posterior_coverage():
    m_probe, delta = 16, 0.05
    c = bound_constant(m_probe, delta)
    assert failure_prob(m_probe, m_probe / c ** 2) <= delta + 1e-12
    g = torch.Generator().manual_seed(7)
    T = 200
    vectors = [torch.randn(T, generator=g, dtype=torch.float64)]
    two = torch.zeros(T, dtype=torch.float64)
    two[:2] = 1.0  # worst case for Rademacher anti-concentration
    vectors += [two, torch.nn.functional.one_hot(torch.tensor(5), T).double(),
                torch.arange(1, T + 1, dtype=torch.float64) ** -1.0]
    for e in vectors:
        cover, est = 0, []
        for t in range(1000):
            Phi = sensing_matrix("rademacher", m_probe, T, 50_000 + t, dtype=torch.float64)
            out = posterior_from_Z((Phi @ e)[:, None], delta)
            cover += float(out["col_bound"][0]) >= float(e.norm()) * (1 - 1e-12)
            est.append(float(out["col_est"][0]) ** 2)
        assert cover / 1000 >= 1 - delta
        assert abs(sum(est) / len(est) / float(e.pow(2).sum()) - 1) < 0.1  # unbiased in square


def test_T13_pruning_matches_reference():
    for seed in range(6):
        A, nf, nt, probs, mask = random_dag(18, 0.5, seed)
        fast = prune_adjacency(A, nf, nt, probs, 0.8, 0.98)
        nodes, edges = prune_reference(A, nf, nt, probs, 0.8, 0.98)
        assert torch.equal(fast.node_mask, nodes)
        assert torch.equal(fast.edge_mask, edges)


def test_T13_metrics_hand_built():
    A, nf, nt, probs, mask = random_dag(30, 0.4, 11)
    ex = prune_adjacency(A, nf, nt, probs)
    assert iw_node_recall(ex, ex.node_mask, nf) == 1.0
    assert node_jaccard(ex.node_mask, ex.node_mask, nf) == 1.0
    assert topk_overlap(ex.node_influence, ex.node_influence, nf, 5) == 1.0
    assert abs(spearman(ex.node_influence[:nf], ex.node_influence[:nf]) - 1) < 1e-12
    a = torch.tensor([True, True, False, False])
    b = torch.tensor([True, False, True, False])
    assert abs(node_jaccard(a, b, 4) - 1 / 3) < 1e-12
    from cct.graph.prune import PruneResult
    pr = PruneResult(a, None, None, torch.tensor([3.0, 1.0, 5.0, 0.0]))
    assert abs(iw_node_recall(pr, b, 4) - 0.75) < 1e-12  # 3 / (3 + 1)
    assert abs(spearman(torch.tensor([1.0, 2, 3]), torch.tensor([3.0, 2, 1])) + 1) < 1e-12
    # graph scores: a graph with no error edges has completeness 1
    n_err = 0
    r, c = graph_scores(A, nf, n_err, nt, probs)
    assert abs(c - 1.0) < 1e-12 and (math.isnan(r) or 0 <= r <= 1)
