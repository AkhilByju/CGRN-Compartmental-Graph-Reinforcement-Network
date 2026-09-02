"""MLP baseline: a conventional feed-forward network used as the
parameter-/compute-matched comparison point for CellV0 (Q1/Q3 in
docs/hypotheses.md), starting with Experiment 002/003.
"""

from __future__ import annotations

import torch
from torch import nn


class MLPBaseline(nn.Module):
    """`in_features -> [Linear -> activation] * num_hidden_layers ->
    Linear(out_features)`, with no activation on the final layer (so
    regression outputs are unbounded, unlike `BeliefLayer`'s internal
    `tanh`)."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")

        dims = [in_features] + [hidden_dim] * num_hidden_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), activation()]
        layers.append(nn.Linear(dims[-1], out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPWithUncertaintyHead(nn.Module):
    """Conventional heteroscedastic-regression MLP: a shared trunk feeding
    two linear heads, `mean` and `log_var`, trained with Gaussian NLL
    (Nix & Weigend, 1994; `torch.nn.functional.gaussian_nll_loss`) -- the
    standard, architecture-agnostic way to get an explicit per-example
    uncertainty estimate out of a plain feed-forward network.

    Used starting Experiment 003B as the comparison point for whether
    `BeliefCell`'s unsupervised `uncertainty` channel tracks known noise
    levels any better than this conventional alternative
    (docs/experiment_protocol.md)."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        target_dim: int = 1,
        num_hidden_layers: int = 2,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")

        dims = [in_features] + [hidden_dim] * num_hidden_layers
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), activation()]
        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(dims[-1], target_dim)
        self.log_var_head = nn.Linear(dims[-1], target_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(mean, log_var)`, both shape `(batch, target_dim)`. Feed
        `log_var.exp()` as `var` to `F.gaussian_nll_loss(mean, target, var)`;
        `log_var` (not `var` directly) is parameterized so the head can
        output negative values freely instead of needing a positivity
        constraint on its raw output."""
        h = self.trunk(x)
        return self.mean_head(h), self.log_var_head(h)


def param_count_for_mlp(
    in_features: int, hidden_dim: int, out_features: int, num_hidden_layers: int = 2
) -> int:
    """Closed-form parameter count, used to search for a `hidden_dim` that
    approximately matches another architecture's parameter count."""
    dims = [in_features] + [hidden_dim] * num_hidden_layers + [out_features]
    return sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))


def match_hidden_dim(
    target_params: int,
    in_features: int,
    out_features: int,
    num_hidden_layers: int = 2,
    search_range: range = range(1, 2048),
) -> int:
    """Returns the `hidden_dim` (within `search_range`) whose `MLPBaseline`
    parameter count is closest to `target_params`, for an approximately
    parameter-matched baseline."""
    best_dim = search_range[0]
    best_diff: int | None = None
    for dim in search_range:
        diff = abs(
            param_count_for_mlp(in_features, dim, out_features, num_hidden_layers) - target_params
        )
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_dim = dim
    return best_dim
