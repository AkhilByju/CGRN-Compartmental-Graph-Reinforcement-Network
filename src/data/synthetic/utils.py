"""Shared helpers for synthetic dataset generators."""

from __future__ import annotations

from collections.abc import Callable

import torch


def make_splits(
    generator_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    **kwargs,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Generates `n_train + n_val + n_test` examples in one seeded call and
    slices them into splits, so the whole dataset (and therefore each
    split) is reproducible from `seed` alone."""
    n_total = n_train + n_val + n_test
    x, y = generator_fn(n_total, seed=seed, **kwargs)
    return {
        "train": (x[:n_train], y[:n_train]),
        "val": (x[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (x[n_train + n_val :], y[n_train + n_val :]),
    }


def standardize(train_x: torch.Tensor, *other_xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Z-scores `train_x` and any additional tensors using train-set
    statistics only (no leakage from val/test into the standardization)."""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return tuple((x - mean) / std for x in (train_x, *other_xs))
