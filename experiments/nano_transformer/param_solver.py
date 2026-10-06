"""Generic FFN-hidden-width parameter-budget solver.

Every model in this experiment (`ModernTransformer1M`, `CellV03Transformer1M`,
`FixedConfidenceCellTransformer`) shares an identical backbone (embedding,
attention, norms) and differs only in FFN hidden width `d_hidden` -- see
`experiments/nano_transformer/models.py`. For a fixed backbone the total
parameter count is affine in `d_hidden`:

    total(d_hidden) = fixed_params + coef_per_hidden_unit * d_hidden

(`coef_per_hidden_unit` folds in `n_layers` since every layer uses the same
hidden width.) This mirrors the closed-form-plus-integer-correction approach
in `src/models/architecture_v2/param_count.py`, generalized to take the
affine coefficients as arguments instead of hard-coding one architecture's
formula, since three different FFN shapes need solving here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WidthSolution:
    hidden: int
    parameter_count: int
    target: int

    @property
    def deviation(self) -> int:
        return self.parameter_count - self.target

    @property
    def relative_deviation(self) -> float:
        return self.deviation / self.target


def solve_hidden_width_for_target(
    target: int,
    fixed_params: int,
    coef_per_hidden_unit: int,
    min_hidden: int = 1,
) -> WidthSolution:
    """Return the integer `d_hidden >= min_hidden` whose
    `fixed_params + coef_per_hidden_unit * d_hidden` is closest to `target`
    (ties broken toward the smaller width). Closed-form estimate, then a
    local integer search of +/-2 to correct for rounding -- exact against the
    real affine formula, not assumed."""
    if coef_per_hidden_unit <= 0:
        raise ValueError("coef_per_hidden_unit must be positive")

    raw = (target - fixed_params) / coef_per_hidden_unit
    candidate = max(min_hidden, round(raw))

    best = None
    for h in range(max(min_hidden, candidate - 2), candidate + 3):
        total = fixed_params + coef_per_hidden_unit * h
        dev = abs(total - target)
        if best is None or dev < best[0] or (dev == best[0] and h < best[1]):
            best = (dev, h, total)

    assert best is not None
    _, hidden, total = best
    return WidthSolution(hidden=hidden, parameter_count=total, target=target)
