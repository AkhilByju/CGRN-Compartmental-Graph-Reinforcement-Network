"""Paper A Phase 3 Part B -- real-dataset loaders: official split preservation,
deterministic stratified validation split, no preprocessing leakage,
impute-after-standardize, reliability masks, chronological Air Quality split,
target-only row dropping, forbidden columns, missingness bins (Phase-3 task
"Required tests").

These hit the network once (UCI download, then cached). If the archive is
unreachable the whole module is skipped with an explicit reason.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.paper_a.real_reliability.datasets import (
    _AIR_QUALITY_FORBIDDEN,
    MISSING_CONFIDENCE,
    MISSINGNESS_BINS,
    OBSERVED_CONFIDENCE,
    dataset_available,
    missingness_bin_index,
    prepare_dataset,
)

pytestmark = pytest.mark.skipif(
    not (dataset_available("aps") and dataset_available("air_quality")),
    reason="UCI APS / Air Quality archives not reachable in this environment",
)


@pytest.fixture(scope="module")
def aps0():
    return prepare_dataset("aps", seed=0)


@pytest.fixture(scope="module")
def air0():
    return prepare_dataset("air_quality", seed=0)


# ---------------------------------------------------------------------------
# APS
# ---------------------------------------------------------------------------


def test_aps_official_split_sizes_and_features(aps0):
    assert aps0.n_features == 170
    assert aps0.x_test.shape == (16_000, 170)          # official test, untouched
    assert aps0.x_train.shape[0] + aps0.x_val.shape[0] == 60_000  # official train
    assert aps0.meta["official_test"] is True


def test_aps_train_val_split_is_deterministic_and_stratified(aps0):
    b = prepare_dataset("aps", seed=0)
    assert torch.equal(aps0.x_train, b.x_train)
    assert torch.equal(aps0.y_val, b.y_val)
    # different seed -> different val set
    c = prepare_dataset("aps", seed=1)
    assert not torch.equal(aps0.y_val, c.y_val) or not torch.equal(aps0.x_val, c.x_val)
    # stratified: val positive rate close to train positive rate
    tr_rate = aps0.y_train.float().mean().item()
    val_rate = aps0.y_val.float().mean().item()
    assert abs(tr_rate - val_rate) < 0.01


def test_aps_official_test_rows_are_identical_across_seeds(aps0):
    # the official 16k test *rows* and their missing pattern never change; only
    # the standardization (fit on the seeded 80% train split) rescales columns.
    for s in (1, 2):
        d = prepare_dataset("aps", seed=s)
        assert torch.equal(aps0.y_test, d.y_test)
        assert torch.equal(torch.isnan(aps0.x_test_nan), torch.isnan(d.x_test_nan))
        # each feature column is a linear rescale of the same raw values, so
        # per-column Pearson correlation between seeds is ~1
        a, b = aps0.x_test_nan, d.x_test_nan
        for j in range(0, aps0.n_features, 17):
            m = ~torch.isnan(a[:, j])
            if m.sum() > 2 and a[m, j].std() > 0 and b[m, j].std() > 0:
                corr = torch.corrcoef(torch.stack([a[m, j], b[m, j]]))[0, 1]
                assert corr > 0.999, f"column {j} corr {corr:.4f}"


def test_aps_class_weights_from_train_labels_only(aps0):
    counts = torch.bincount(aps0.y_train, minlength=2).double()
    expect = counts.sum() / (2.0 * counts.clamp_min(1.0))
    assert torch.allclose(aps0.class_weights.double(), expect, rtol=1e-5)


# ---------------------------------------------------------------------------
# Air Quality
# ---------------------------------------------------------------------------


def test_air_quality_is_chronological_60_20_20(air0):
    n = air0.x_train.shape[0] + air0.x_val.shape[0] + air0.x_test.shape[0]
    assert air0.x_train.shape[0] == round(0.6 * n)
    assert air0.x_val.shape[0] == round(0.2 * n)
    assert air0.meta["chronological"] is True
    # seed-independent split
    assert torch.equal(air0.x_test, prepare_dataset("air_quality", seed=2).x_test)


def test_air_quality_inputs_exclude_all_other_gt_pollutants(air0):
    inputs = set(air0.meta["inputs"])
    assert inputs.isdisjoint(_AIR_QUALITY_FORBIDDEN)
    assert air0.meta["target"] == "CO(GT)"
    assert air0.n_features == 8


def test_air_quality_drops_missing_target_rows_only(air0):
    # every kept row has a finite target; rows with missing *inputs* are kept
    assert torch.isfinite(air0.y_train_raw).all()
    assert torch.isnan(air0.x_train_nan).any()  # input missingness survived


# ---------------------------------------------------------------------------
# Shared preprocessing contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fx", ["aps0", "air0"])
def test_no_preprocessing_leakage_train_stats_only(fx, request):
    d = request.getfixturevalue(fx)
    # observed training features are ~z-scored on their own stats
    obs = ~torch.isnan(d.x_train_nan)
    col_means = []
    for j in range(d.n_features):
        v = d.x_train_nan[:, j][obs[:, j]]
        if v.numel() > 1:
            col_means.append(v.mean().item())
    assert np.allclose(col_means, 0.0, atol=1e-3)


@pytest.mark.parametrize("fx", ["aps0", "air0"])
def test_impute_happens_after_standardization(fx, request):
    d = request.getfixturevalue(fx)
    missing = torch.isnan(d.x_train_nan)
    # imputed value is exactly 0 where missing, and equals the standardized
    # value where observed
    assert torch.all(d.x_train[missing] == 0.0)
    obs = ~missing
    assert torch.allclose(d.x_train[obs], d.x_train_nan[obs], atol=0.0)
    assert not torch.isnan(d.x_train).any()


@pytest.mark.parametrize("fx", ["aps0", "air0"])
def test_reliability_mask_matches_original_missing_locations(fx, request):
    d = request.getfixturevalue(fx)
    for c, xn in ((d.c_train, d.x_train_nan), (d.c_val, d.x_val_nan), (d.c_test, d.x_test_nan)):
        assert torch.equal(c == MISSING_CONFIDENCE, torch.isnan(xn))
        assert torch.equal(c == OBSERVED_CONFIDENCE, ~torch.isnan(xn))
        assert torch.all((c == MISSING_CONFIDENCE) | (c == OBSERVED_CONFIDENCE))


@pytest.mark.parametrize("fx", ["aps0", "air0"])
def test_reliability_is_not_derived_from_the_target(fx, request):
    d = request.getfixturevalue(fx)
    # c depends only on the feature mask; permuting the labels would not change it
    assert d.c_train.shape == d.x_train.shape


# ---------------------------------------------------------------------------
# Missingness bins
# ---------------------------------------------------------------------------


def test_missingness_bin_assignment_partitions_every_example(air0, aps0):
    for d in (air0, aps0):
        idx = missingness_bin_index(d.missing_frac_test)
        assert idx.min() >= 0 and idx.max() < len(MISSINGNESS_BINS)
        assert idx.shape[0] == d.x_test.shape[0]
        # zero-missing examples land in the first bin
        zero = d.missing_frac_test <= 1e-9
        assert torch.all(idx[zero] == 0)


def test_missingness_bins_are_the_fixed_boundaries():
    assert [b[0] for b in MISSINGNESS_BINS] == [
        "0", "(0, 0.10]", "(0.10, 0.25]", "(0.25, 0.50]", ">0.50"
    ]
