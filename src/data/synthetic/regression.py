"""Synthetic regression datasets for the R0/R1/R2 progression used in
Experiment 002/003 (docs/benchmark_plan.md "Synthetic regression and
classification progressions")."""

from __future__ import annotations

import torch

from src.data.synthetic.utils import make_splits

REGRESSION_DATASETS: tuple[str, ...] = ("r0_linear", "r1_nonlinear", "r2_interaction")

_DOMAIN_RADIUS = {"r0_linear": 2.0, "r1_nonlinear": 3.0, "r2_interaction": 2.0}
_N_FEATURES = {"r0_linear": 1, "r1_nonlinear": 1, "r2_interaction": 4}


def r0_linear(
    n_samples: int, seed: int, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = 3x + 2 -- sanity check only. x in [-2, 2]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 1).uniform_(-2.0, 2.0, generator=g)
    noise = torch.randn(n_samples, 1, generator=g) * noise_std
    y = 3.0 * x + 2.0 + noise
    return x, y


def r1_nonlinear(
    n_samples: int, seed: int, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = sin(x) + 0.3*x^2. x in [-3, 3]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 1).uniform_(-3.0, 3.0, generator=g)
    noise = torch.randn(n_samples, 1, generator=g) * noise_std
    y = torch.sin(x) + 0.3 * x**2 + noise
    return x, y


def r2_interaction(
    n_samples: int, seed: int, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = x1*x2 + sin(x3) + x4^2 (docs/benchmark_plan.md "Feature
    interactions"). 4 features, each in [-2, 2]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 4).uniform_(-2.0, 2.0, generator=g)
    noise = torch.randn(n_samples, 1, generator=g) * noise_std
    y = (x[:, 0:1] * x[:, 1:2]) + torch.sin(x[:, 2:3]) + x[:, 3:4] ** 2 + noise
    return x, y


_GENERATORS = {
    "r0_linear": r0_linear,
    "r1_nonlinear": r1_nonlinear,
    "r2_interaction": r2_interaction,
}


def make_regression_splits(
    name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    noise_std: float = 0.3,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if name not in _GENERATORS:
        raise ValueError(
            f"Unknown regression dataset '{name}'. Expected one of {REGRESSION_DATASETS}."
        )
    return make_splits(_GENERATORS[name], n_train, n_val, n_test, seed, noise_std=noise_std)


def regression_ood_inputs(
    name: str, n_samples: int, seed: int, extra_radius: float = 3.0
) -> torch.Tensor:
    """Samples inputs outside the dataset's training domain -- magnitude
    beyond the training radius, random sign per feature. Used as an
    "extrapolation" diagnostic: does the model's uncertainty rise on inputs
    unlike anything it was trained on?"""
    if name not in _DOMAIN_RADIUS:
        raise ValueError(
            f"Unknown regression dataset '{name}'. Expected one of {REGRESSION_DATASETS}."
        )
    radius = _DOMAIN_RADIUS[name]
    n_features = _N_FEATURES[name]
    g = torch.Generator().manual_seed(seed)
    magnitude = torch.empty(n_samples, n_features).uniform_(
        radius, radius + extra_radius, generator=g
    )
    sign = torch.where(torch.rand(n_samples, n_features, generator=g) > 0.5, 1.0, -1.0)
    return magnitude * sign
