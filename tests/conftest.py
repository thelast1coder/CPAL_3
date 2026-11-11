import warnings

import pytest
import torch

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session", params=["plt", "clt"])
def tiny_op(request):
    from cct.operators.tiny import TinyConfig, TinyModel, TinyOperator

    model = TinyModel(TinyConfig(transcoder=request.param, n_layers=3, n_features=32, seed=0))
    tokens = torch.randint(0, 64, (8,), generator=torch.Generator().manual_seed(1))
    return TinyOperator(model, tokens, batch_size=64)


def random_dag(n: int, density: float, seed: int, n_logits: int = 2, n_tokens: int = 2):
    """Random square adjacency in circuit-tracer order [features, errors(0), tokens, logits] that is a
    DAG: node i can only receive from nodes with a lower 'layer' index."""
    g = torch.Generator().manual_seed(seed)
    n_feat = n - n_logits - n_tokens
    layer = torch.cat([torch.randint(1, 5, (n_feat,), generator=g), torch.zeros(n_tokens, dtype=torch.long),
                       torch.full((n_logits,), 6)])
    mask = layer[:, None] > layer[None, :]
    A = torch.randn(n, n, generator=g, dtype=torch.float64) * (torch.rand(n, n, generator=g) < density) * mask
    A[n_feat : n_feat + n_tokens] = 0  # token rows are empty
    probs = torch.rand(n_logits, generator=g, dtype=torch.float64)
    probs = probs / probs.sum() * 0.9
    return A, n_feat, n_tokens, probs, mask
