"""Synthetic classification datasets for the C0/C1/C2 progression used in
Experiment 002/003 (docs/benchmark_plan.md "Synthetic regression and
classification progressions")."""

from __future__ import annotations

import torch

from src.data.synthetic.utils import make_splits

CLASSIFICATION_DATASETS: tuple[str, ...] = ("c0_linear", "c1_xor", "c2_interaction")


def c0_linear(
    n_samples: int, seed: int, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly separable: y = 1[x1 - x2 + noise > 0]. 2 features in [-2, 2]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 2).uniform_(-2.0, 2.0, generator=g)
    noise = torch.randn(n_samples, generator=g) * noise_std
    score = x[:, 0] - x[:, 1] + noise
    y = (score > 0).long()
    return x, y


def c1_xor(n_samples: int, seed: int, noise_std: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
    """XOR: y = 1[(x1>0) xor (x2>0)], with noise applied before
    thresholding. 2 features in [-2, 2]."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 2).uniform_(-2.0, 2.0, generator=g)
    noisy = x + torch.randn(n_samples, 2, generator=g) * noise_std
    y = ((noisy[:, 0] > 0) ^ (noisy[:, 1] > 0)).long()
    return x, y


def c2_interaction(
    n_samples: int, seed: int, noise_std: float = 0.3
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interaction-heavy boundary: y = 1[x1*x2 + (x3^2 - 4/3) - 0.5*x4 +
    noise > 0]. 4 features in [-2, 2]; `-4/3` recenters `x3^2` (its mean
    for x3 ~ U(-2,2)) so classes come out roughly balanced without relying
    on a per-split statistic."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 4).uniform_(-2.0, 2.0, generator=g)
    noise = torch.randn(n_samples, generator=g) * noise_std
    score = x[:, 0] * x[:, 1] + (x[:, 2] ** 2 - 4.0 / 3.0) - 0.5 * x[:, 3] + noise
    y = (score > 0).long()
    return x, y


def c0_linear_margin(x: torch.Tensor) -> torch.Tensor:
    """Signed distance to the decision boundary (noiseless score)."""
    return x[:, 0] - x[:, 1]


def c1_xor_margin(x: torch.Tensor) -> torch.Tensor:
    """Distance to the nearest boundary axis (x1=0 or x2=0): small values
    are close to flipping the label."""
    return torch.minimum(x[:, 0].abs(), x[:, 1].abs())


def c2_interaction_margin(x: torch.Tensor) -> torch.Tensor:
    """Noiseless score; see `c2_interaction`'s docstring for the formula."""
    return x[:, 0] * x[:, 1] + (x[:, 2] ** 2 - 4.0 / 3.0) - 0.5 * x[:, 3]


_GENERATORS = {
    "c0_linear": c0_linear,
    "c1_xor": c1_xor,
    "c2_interaction": c2_interaction,
}

MARGIN_FUNCTIONS = {
    "c0_linear": c0_linear_margin,
    "c1_xor": c1_xor_margin,
    "c2_interaction": c2_interaction_margin,
}


def make_classification_splits(
    name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    noise_std: float = 0.3,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if name not in _GENERATORS:
        raise ValueError(
            f"Unknown classification dataset '{name}'. Expected one of {CLASSIFICATION_DATASETS}."
        )
    return make_splits(_GENERATORS[name], n_train, n_val, n_test, seed, noise_std=noise_std)
