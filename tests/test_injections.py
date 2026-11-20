"""The vectorised additive injection of CircuitTracerOperator equals the per-location reference loop
(no model needed: a stand-in object carries the attributes the method reads)."""
import types

import torch

from cct.operators.base import NodeMeta
from cct.operators.circuit_tracer_backend import CircuitTracerOperator


def _fake(T_f=60, k=3, L=4, n=7, d=5, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    op = types.SimpleNamespace()
    op.n_logits, op.n_layers, op.n_pos = k, L, n
    op.device, op.dtype = torch.device("cpu"), torch.float32
    op.enc_vecs = torch.randn(T_f, d, generator=g)
    op.logit_vecs = torch.randn(k, d, generator=g)
    lay = torch.randint(0, L, (T_f,), generator=g)
    pos = torch.randint(0, n, (T_f,), generator=g)
    op.target_meta = NodeMeta(kind=torch.zeros(k + T_f, dtype=torch.long),
                              layer=torch.cat([torch.full((k,), L), lay]),
                              pos=torch.cat([torch.full((k,), n - 1), pos]),
                              feature=torch.zeros(k + T_f, dtype=torch.long), extra=torch.zeros(k + T_f))
    op.ctx = types.SimpleNamespace(_resid_activations=[torch.zeros(B, n, d)])
    return op, k + T_f


def test_vectorised_injection_matches_loop():
    op, T = _fake()
    g = torch.Generator().manual_seed(1)
    for seeds in (torch.randn(6, T, generator=g), torch.eye(T)[:6] , torch.eye(T)[10:16]):
        a = CircuitTracerOperator._injections(op, seeds)
        b = CircuitTracerOperator._injections_loop(op, seeds)
        assert set(a) <= set(b)  # the loop also emits all-zero injections for unseeded layers
        for l in b:
            ref = b[l]
            got = a.get(l, torch.zeros_like(ref))
            assert torch.allclose(got, ref, atol=1e-5), l
