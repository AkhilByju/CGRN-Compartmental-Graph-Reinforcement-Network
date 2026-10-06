"""CIFAR-10 loading and splitting
(experiments/belief_dendrite/strong_baselines/datasets.py, spec Sec 2):
official split sizes, the seeded 45k/5k stratified carve-out, and
train-only per-channel standardization. Skipped if the CIFAR-10 tarball
isn't cached locally (no network access in this environment) -- these are
integration tests over real data, not corruption/model unit tests.
"""

from __future__ import annotations

import pytest
import torch

from experiments.belief_dendrite.strong_baselines.datasets import (
    _TARBALL_PATH,
    N_TRAIN,
    N_VAL,
    load_cifar10_raw,
    prepare_cifar10,
    stratified_train_val_split,
)

pytestmark = pytest.mark.skipif(
    not _TARBALL_PATH.exists(),
    reason="CIFAR-10 tarball not cached locally; skipping integration test",
)


def test_official_split_sizes() -> None:
    raw = load_cifar10_raw()
    assert raw["train_images"].shape == (50_000, 3, 32, 32)
    assert raw["test_images"].shape == (10_000, 3, 32, 32)
    assert raw["train_labels"].shape == (50_000,)
    assert raw["test_labels"].shape == (10_000,)
    assert set(raw["train_labels"].tolist()) == set(range(10))


def test_stratified_split_disjoint_and_correctly_sized() -> None:
    raw = load_cifar10_raw()
    split = stratified_train_val_split(raw["train_labels"], seed=0)
    assert split["train"].shape[0] == N_TRAIN
    assert split["val"].shape[0] == N_VAL
    assert set(split["train"].tolist()).isdisjoint(split["val"].tolist())


def test_different_seeds_give_different_val_partitions() -> None:
    raw = load_cifar10_raw()
    a = stratified_train_val_split(raw["train_labels"], seed=0)
    b = stratified_train_val_split(raw["train_labels"], seed=1)
    assert set(a["val"].tolist()) != set(b["val"].tolist())


def test_prepare_cifar10_shapes_and_stratification() -> None:
    ds = prepare_cifar10(seed=0)
    assert ds.x_train.shape == (N_TRAIN, 3, 32, 32)
    assert ds.x_val.shape == (N_VAL, 3, 32, 32)
    assert ds.x_test.shape == (10_000, 3, 32, 32)
    assert torch.equal(torch.bincount(ds.y_train), torch.full((10,), N_TRAIN // 10))
    assert torch.equal(torch.bincount(ds.y_val), torch.full((10,), N_VAL // 10))


def test_standardization_uses_training_statistics_only() -> None:
    ds = prepare_cifar10(seed=0)
    train_mean = ds.x_train.mean(dim=(0, 2, 3))
    train_std = ds.x_train.std(dim=(0, 2, 3))
    assert torch.allclose(train_mean, torch.zeros(3), atol=1e-4)
    assert torch.allclose(train_std, torch.ones(3), atol=1e-2)
    # val/test are transformed with the *training* mean/std, so they need not
    # be exactly zero-mean/unit-std themselves.
    assert ds.channel_mean.shape == (3,)
    assert ds.channel_std.shape == (3,)
