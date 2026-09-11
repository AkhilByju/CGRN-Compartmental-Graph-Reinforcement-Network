"""Architecture V2 parameter-count formula and budget solver (Sec J).

For one `BeliefDendriteLayer` (identical layout for `ScalarDendriteLayer`):
`H*B*K` synaptic weights (`V_branch`) + `2*H*B` branch gain/bias + `H*B`
cable weights (`V_cable`) + `2*H` soma gain/bias:

    params(H, B, K) = H*B*K + 3*H*B + 2*H = H * (B*(K+3) + 2)

Exercised against the real `nn.Module` parameter count in
`tests/test_architecture_v2_param_count.py` -- this is a prediction to be
checked, not a substitute for `count_parameters`.
"""

from __future__ import annotations

from dataclasses import dataclass


def belief_dendrite_layer_param_count(hidden: int, branches: int, sources_per_branch: int) -> int:
    """`H * (B*(K+3) + 2)` -- exact parameter count of one
    `BeliefDendriteLayer`/`ScalarDendriteLayer(H, B, K)`."""
    return hidden * (branches * (sources_per_branch + 3) + 2)


def readout_param_count(hidden: int, out_features: int) -> int:
    """`nn.Linear(hidden, out_features)`'s parameter count."""
    return hidden * out_features + out_features


def two_layer_param_count(
    hidden: int,
    branches: int,
    sources_per_branch_1: int,
    sources_per_branch_2: int,
    out_features: int,
) -> int:
    """Total trainable parameters of the Sec F two-stage network built with
    a **uniform** hidden width `hidden` for both `BeliefDendriteLayer`s
    (`K1` for layer 1, `K2` for layer 2, fixed and given -- not re-derived
    from `hidden`) plus the linear readout."""
    return (
        belief_dendrite_layer_param_count(hidden, branches, sources_per_branch_1)
        + belief_dendrite_layer_param_count(hidden, branches, sources_per_branch_2)
        + readout_param_count(hidden, out_features)
    )


@dataclass(frozen=True)
class BudgetSolution:
    hidden: int
    parameter_count: int
    budget: int

    @property
    def slack(self) -> int:
        return self.budget - self.parameter_count


def solve_hidden_width_for_budget(
    budget: int,
    branches: int,
    sources_per_branch_1: int,
    sources_per_branch_2: int,
    out_features: int,
) -> BudgetSolution:
    """The largest integer `hidden` (uniform across both layers) whose
    `two_layer_param_count` does not exceed `budget`.

    `two_layer_param_count` is affine in `hidden` for fixed `(branches, K1,
    K2, out_features)` -- `coef * hidden + out_features` with
    `coef = branches*(K1+3) + branches*(K2+3) + 2 + 2 + out_features` -- so
    this is solved in closed form, then adjusted by at most one step to
    correct for integer rounding (checked exactly against
    `two_layer_param_count`, never assumed)."""
    coef = (
        branches * (sources_per_branch_1 + 3)
        + branches * (sources_per_branch_2 + 3)
        + 4
        + out_features
    )
    fixed = out_features
    if budget < fixed + coef:
        return BudgetSolution(
            hidden=0, parameter_count=fixed if budget >= fixed else 0, budget=budget
        )

    hidden = (budget - fixed) // coef
    hidden = max(1, int(hidden))
    # Closed-form gives a lower bound up to integer division; walk to the
    # exact largest-fitting integer against the real formula.
    while (
        two_layer_param_count(
            hidden + 1, branches, sources_per_branch_1, sources_per_branch_2, out_features
        )
        <= budget
    ):
        hidden += 1
    while (
        hidden > 0
        and two_layer_param_count(
            hidden, branches, sources_per_branch_1, sources_per_branch_2, out_features
        )
        > budget
    ):
        hidden -= 1

    return BudgetSolution(
        hidden=hidden,
        parameter_count=two_layer_param_count(
            hidden, branches, sources_per_branch_1, sources_per_branch_2, out_features
        ),
        budget=budget,
    )
