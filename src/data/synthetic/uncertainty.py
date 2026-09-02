"""Synthetic regression datasets with known, ground-truth heteroscedastic
noise, for Experiment 003B (docs/experiment_protocol.md).

Unlike the R0/R1/R2 progression (`regression.py`), where "how uncertain is
this input" is only approximated after training via the "ambiguous vs.
clear" margin/extrapolation proxy (Experiment 002's harness), these
datasets are constructed so the true per-example noise level -- and
therefore the true irreducible uncertainty -- is known exactly, because it
is what generated the label. That makes it possible to ask directly
whether a model's predicted/derived uncertainty tracks a *real* signal,
not just a plausible-looking proxy.

Mean functions are reused unchanged from `r1_nonlinear`/`r2_interaction`
(`regression.py`) so that any predictive-performance difference from those
datasets isolates the effect of heteroscedastic noise, not a different
function class.
"""

from __future__ import annotations

import torch

U_REGRESSION_DATASETS: tuple[str, ...] = (
    "u1_heteroscedastic_1d",
    "u2_heteroscedastic_interaction",
)


def u1_heteroscedastic_1d(
    n_samples: int, seed: int, base_std: float = 0.05, slope: float = 0.35
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y = sin(x) + 0.3*x^2 + noise, noise_std(x) = base_std + slope*|x|.

    Same mean function as `r1_nonlinear`; noise grows away from the
    origin, so there is a real feature (|x|) a model could in principle
    learn to condition its uncertainty on. x in [-3, 3]. Returns
    `(x, y, true_std)`."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 1).uniform_(-3.0, 3.0, generator=g)
    true_std = base_std + slope * x.abs()
    noise = torch.randn(n_samples, 1, generator=g) * true_std
    y = torch.sin(x) + 0.3 * x**2 + noise
    return x, y, true_std


def u2_heteroscedastic_interaction(
    n_samples: int, seed: int, base_std: float = 0.05, slope: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same mean function as `r2_interaction`; noise_std depends only on
    `|x1|` (one of four features), so the ground-truth uncertainty signal
    is entangled with the same interaction-heavy input space where
    Experiment 002 found the aggregation methods struggled most. 4
    features in [-2, 2]. Returns `(x, y, true_std)`."""
    g = torch.Generator().manual_seed(seed)
    x = torch.empty(n_samples, 4).uniform_(-2.0, 2.0, generator=g)
    true_std = base_std + slope * x[:, 0:1].abs()
    noise = torch.randn(n_samples, 1, generator=g) * true_std
    y = (x[:, 0:1] * x[:, 1:2]) + torch.sin(x[:, 2:3]) + x[:, 3:4] ** 2 + noise
    return x, y, true_std


_GENERATORS = {
    "u1_heteroscedastic_1d": u1_heteroscedastic_1d,
    "u2_heteroscedastic_interaction": u2_heteroscedastic_interaction,
}


def make_uncertainty_splits(
    name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Like `src.data.synthetic.utils.make_splits`, but for the 3-tensor
    `(x, y, true_std)` generators above -- kept separate rather than
    generalizing `make_splits` itself, since every other dataset in this
    project returns `(x, y)` only."""
    if name not in _GENERATORS:
        raise ValueError(
            f"Unknown uncertainty dataset '{name}'. Expected one of {U_REGRESSION_DATASETS}."
        )
    n_total = n_train + n_val + n_test
    x, y, true_std = _GENERATORS[name](n_total, seed=seed, **kwargs)
    return {
        "train": (x[:n_train], y[:n_train], true_std[:n_train]),
        "val": (
            x[n_train : n_train + n_val],
            y[n_train : n_train + n_val],
            true_std[n_train : n_train + n_val],
        ),
        "test": (
            x[n_train + n_val :],
            y[n_train + n_val :],
            true_std[n_train + n_val :],
        ),
    }
