"""Paper A Phase 1 -- dataset determinism, split-leakage, and preprocessing
tests (Paper-A task Sec 15).

Only the sklearn-bundled datasets are exercised here (no network): the
loaders for California Housing / MNIST / Fashion-MNIST share the exact same
split and preprocessing code paths, tested through the bundled ones.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.paper_a.datasets import (
    BENCHMARK_DATASETS,
    PARAM_BUDGET,
    compute_split_indices,
    prepare_dataset,
)

_BUNDLED = ["breast_cancer", "wine", "digits", "diabetes"]


def test_benchmark_list_is_frozen():
    # This list was fixed before inspecting CellV0.1 performance; changing it
    # is an anti-cherry-picking violation (Paper-A task Sec 2/12).
    assert BENCHMARK_DATASETS == (
        "breast_cancer",
        "wine",
        "digits",
        "diabetes",
        "california_housing",
        "mnist",
        "fashion_mnist",
    )
    assert set(PARAM_BUDGET) == set(BENCHMARK_DATASETS)


@pytest.mark.parametrize("name", _BUNDLED)
@pytest.mark.parametrize("frac", [0.25, 1.0])
def test_split_is_deterministic_from_seed(name, frac):
    a = compute_split_indices(name, seed=0, train_fraction=frac)
    b = compute_split_indices(name, seed=0, train_fraction=frac)
    for key in ("train", "val", "test"):
        assert np.array_equal(a[key], b[key])

    d1 = prepare_dataset(name, seed=0, train_fraction=frac)
    d2 = prepare_dataset(name, seed=0, train_fraction=frac)
    assert torch.equal(d1.x_train, d2.x_train)
    assert torch.equal(d1.y_train, d2.y_train)
    assert torch.equal(d1.x_test, d2.x_test)
    assert torch.equal(d1.y_test, d2.y_test)


@pytest.mark.parametrize("name", _BUNDLED)
def test_different_seed_gives_a_different_split(name):
    a = compute_split_indices(name, seed=0, train_fraction=1.0)
    b = compute_split_indices(name, seed=1, train_fraction=1.0)
    assert not np.array_equal(a["test"], b["test"])


@pytest.mark.parametrize("name", _BUNDLED)
@pytest.mark.parametrize("frac", [0.25, 1.0])
def test_splits_are_disjoint(name, frac):
    idx = compute_split_indices(name, seed=1, train_fraction=frac)
    train, val, test = set(idx["train"]), set(idx["val"]), set(idx["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    # train subsample must be a subset of the full training pool -- never
    # drawn from val/test.
    assert set(idx["train"]).issubset(set(idx["train_pool"]))


@pytest.mark.parametrize("name", _BUNDLED)
def test_25pct_subsamples_only_the_training_pool(name):
    full = compute_split_indices(name, seed=2, train_fraction=1.0)
    low = compute_split_indices(name, seed=2, train_fraction=0.25)
    # val / test identical between the two regimes for a given seed.
    assert np.array_equal(full["val"], low["val"])
    assert np.array_equal(full["test"], low["test"])
    # ~25% of the pool, and a strict subset of it.
    assert set(low["train"]).issubset(set(full["train"]))
    ratio = len(low["train"]) / len(full["train_pool"])
    assert 0.2 <= ratio <= 0.3


@pytest.mark.parametrize("name", ["breast_cancer", "wine", "digits"])
def test_low_data_subsample_is_stratified(name):
    d_full = prepare_dataset(name, seed=0, train_fraction=1.0)
    d_low = prepare_dataset(name, seed=0, train_fraction=0.25)
    full_frac = torch.bincount(d_full.y_train).float()
    full_frac /= full_frac.sum()
    low_frac = torch.bincount(d_low.y_train, minlength=len(full_frac)).float()
    low_frac /= low_frac.sum()
    # stratification keeps class proportions close (tolerance is loose --
    # tiny datasets can't hit exact proportions).
    assert torch.max(torch.abs(full_frac - low_frac)) < 0.12


@pytest.mark.parametrize("name", _BUNDLED)
def test_preprocessing_uses_training_statistics_only(name):
    d = prepare_dataset(name, seed=0, train_fraction=1.0)
    # Train features are z-scored on their own statistics -> ~0 mean, ~1 std.
    assert torch.allclose(d.x_train.mean(dim=0), torch.zeros(d.n_features), atol=1e-4)
    train_std = d.x_train.std(dim=0)
    # (zero-variance columns are clamped, so allow a few < 1)
    assert (train_std <= 1.0 + 1e-4).all()
    # Val/test are NOT re-centred on their own stats -- they carry the shift.
    if name != "digits":  # digits has some all-equal columns per split
        assert not torch.allclose(d.x_test.mean(dim=0), torch.zeros(d.n_features), atol=1e-2)


@pytest.mark.parametrize("name", ["diabetes"])
def test_regression_target_inverse_transform_roundtrips(name):
    d = prepare_dataset(name, seed=0, train_fraction=1.0)
    assert d.task_type == "regression"
    recovered = d.inverse_transform_targets(d.y_test)
    assert torch.allclose(recovered, d.y_test_raw, atol=1e-3)
    # standardized targets are ~unit scale; raw ones are not.
    assert d.y_train.std() == pytest.approx(1.0, abs=1e-3)
    assert abs(d.target_std - 1.0) > 1e-3


@pytest.mark.parametrize("name", _BUNDLED)
def test_prepared_shapes_and_dtypes(name):
    d = prepare_dataset(name, seed=0, train_fraction=1.0)
    assert d.x_train.shape[1] == d.n_features
    assert d.x_train.dtype == torch.float32
    if d.task_type == "classification":
        assert d.y_train.dtype == torch.long
        assert d.out_features == d.n_classes
        assert int(d.y_train.max()) < d.n_classes
    else:
        assert d.y_train.dtype == torch.float32
        assert d.out_features == 1
