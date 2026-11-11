"""A posteriori error check (PROJECT.md 5.7) with m' fresh, independent Rademacher probes.

Z = Phi' A - Phi' A_hat = Phi' (A - A_hat). The first term costs m' real mixed passes; the second
is a matrix product. With Phi' scaled by 1/sqrt(m'):
- point estimate: ||z_s||^2 is unbiased for ||e_s||^2 (e_s = a_s - a_hat_s);
- high-probability bound (docs/certificate.md): with probability at least 1 - delta,
      ||e_s|| <= c * max_i |z_{i,s}|,   c = sqrt(m' / theta),
  delta = (1 - (1 - theta)^2 / 3)^m'. The same holds for ||E||_F with z replaced by the row norms
  of Z. Derived from Paley-Zygmund plus the Rademacher fourth-moment bound E(phi.e)^4 <= 3||e||^4.
  (Halko, Martinsson and Tropp 2011, Lemma 4.1, give 10 sqrt(2/pi) for Gaussian probes and the
  spectral norm; that constant does not transfer to Rademacher probes.)
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from ..sensing.matrices import sensing_matrix
from ..sensing.measure import sense


def failure_prob(m_probe: int, theta: float) -> float:
    return (1.0 - (1.0 - theta) ** 2 / 3.0) ** m_probe


def theta_for(m_probe: int, delta: float) -> float:
    """Largest theta in (0, 1) with failure_prob(m', theta) <= delta (bisection); maximising theta
    minimises the constant c = sqrt(m' / theta)."""
    lo, hi = 0.0, 1.0
    if failure_prob(m_probe, 1e-12) > delta:
        raise ValueError(f"m'={m_probe} probes cannot reach delta={delta}")
    for _ in range(100):
        mid = (lo + hi) / 2
        if failure_prob(m_probe, mid) <= delta:
            lo = mid
        else:
            hi = mid
    return lo


def bound_constant(m_probe: int, delta: float) -> float:
    return math.sqrt(m_probe / theta_for(m_probe, delta))


def posterior_from_Z(Z: Tensor, delta: float) -> dict:
    """Z [m', C] = Phi'(A - A_hat) with Phi' scaled 1/sqrt(m')."""
    m_probe = Z.shape[0]
    c = bound_constant(m_probe, delta)
    Z = Z.double()
    return {
        "col_est": Z.norm(dim=0),  # unbiased in square
        "col_bound": c * Z.abs().max(0).values,
        "fro_est": float(Z.norm()),
        "fro_bound": float(c * Z.norm(dim=1).max()),
        "constant": c,
        "delta": delta,
        "m_probe": m_probe,
    }


def posterior_check(op, A_hat: Tensor, target_idx: Tensor, m_probe: int = 16, seed: int = 10_000,
                    delta: float = 0.01, kind: str = "rademacher") -> dict:
    """A_hat [T_sensed, N] estimate of A[target_idx]. Costs m_probe real mixed rows."""
    Phi2 = sensing_matrix(kind, m_probe, target_idx.numel(), seed, dtype=torch.float64)
    Y2 = sense(op, Phi2.to(op.dtype), target_idx).double().cpu()
    Z = Y2 - Phi2 @ A_hat.double()
    out = posterior_from_Z(Z, delta)
    out["seed"] = seed
    return out
