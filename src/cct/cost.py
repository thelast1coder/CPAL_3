"""Cost ledger (PROJECT.md 9, D7). Unit: one backward row at circuit-tracer's batch size.
row-equivalents = mixed rows + exact rows + certification rows + kappa * forward-mode passes
                  + (recovery seconds + other method seconds) / row_seconds.
row_seconds and kappa are measured at K0.5 and stored in the model config."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Ledger:
    mixed_rows: int = 0
    exact_rows: int = 0  # exact logit rows and refinement rows
    refine_rows: int = 0  # subset of exact_rows spent on refinement (reported separately)
    cert_rows: int = 0
    forward_passes: int = 0
    recovery_seconds: float = 0.0
    other_seconds: float = 0.0  # screening, hub bookkeeping, influence for refinement
    pass_seconds: float = 0.0  # measured time inside operator calls (real runs)
    wall_seconds: float = 0.0
    row_seconds: float = 1.0
    kappa: float = 1.0

    def passes(self) -> float:
        return self.mixed_rows + self.exact_rows + self.cert_rows + self.kappa * self.forward_passes

    def row_equivalents(self) -> float:
        return self.passes() + (self.recovery_seconds + self.other_seconds) / self.row_seconds

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passes"] = self.passes()
        d["row_equivalents"] = self.row_equivalents()
        return d


def truncation_cost(rows: int) -> float:
    return float(rows)
