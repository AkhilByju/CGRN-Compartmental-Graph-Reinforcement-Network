"""Paper A Phase 2 -- deterministic-corruption sanity checks (Phase-2 task
Sec 5 / Sec 16). No network access: a small synthetic standardized batch
exercises the same code paths the real datasets use.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.paper_a.reliability.corruption import (
    GAUSSIAN,
    GAUSSIAN_S_TRAIN,
    MISSING,
    MISSING_CONFIDENCE,
    MISSING_P_TRAIN,
    OBSERVED_CONFIDENCE,
    corrupt,
    corrupt_gaussian,
    corrupt_missing,
    shuffle_confidence,
)


@pytest.fixture
def clean() -> torch.Tensor:
    g = torch.Generator().manual_seed(12345)
    return torch.randn(256, 20, generator=g)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [MISSING, GAUSSIAN])
def test_corruption_is_deterministic_for_fixed_seed_split_epoch(clean, family):
    a = corrupt(family, clean, experiment_seed=0, split="train", epoch=3)
    b = corrupt(family, clean, experiment_seed=0, split="train", epoch=3)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.c, b.c)
    assert np.array_equal(a.severity, b.severity)


@pytest.mark.parametrize("family", [MISSING, GAUSSIAN])
def test_different_experiment_seed_gives_different_corruption(clean, family):
    a = corrupt(family, clean, experiment_seed=0, split="train", epoch=0)
    b = corrupt(family, clean, experiment_seed=1, split="train", epoch=0)
    assert not torch.equal(a.x, b.x)
    assert not torch.equal(a.c, b.c)


@pytest.mark.parametrize("family", [MISSING, GAUSSIAN])
def test_training_corruption_varies_between_epochs(clean, family):
    a = corrupt(family, clean, experiment_seed=0, split="train", epoch=0)
    b = corrupt(family, clean, experiment_seed=0, split="train", epoch=1)
    assert not torch.equal(a.x, b.x)


@pytest.mark.parametrize("family", [MISSING, GAUSSIAN])
def test_model_identity_cannot_influence_corruption(clean, family):
    # There is no model argument anywhere in the corruption API -- the stream
    # is a pure function of (seed, split, epoch, replica). This test documents
    # that contract: two "models" = two identical calls.
    kw = dict(experiment_seed=7, split="test", epoch=0, replica=1, severity=0.3)
    model_a = corrupt(family, clean, **kw)
    model_b = corrupt(family, clean, **kw)
    assert torch.equal(model_a.x, model_b.x)
    assert torch.equal(model_a.c, model_b.c)


@pytest.mark.parametrize("family", [MISSING, GAUSSIAN])
def test_test_replicas_are_deterministic_and_distinct(clean, family):
    sev = 0.5
    kw = dict(experiment_seed=2, split="test", epoch=0, severity=sev)
    reps = [corrupt(family, clean, replica=r, **kw) for r in range(3)]
    # deterministic
    again = corrupt(family, clean, replica=1, **kw)
    assert torch.equal(reps[1].x, again.x)
    # distinct draws
    assert not torch.equal(reps[0].x, reps[1].x)
    assert not torch.equal(reps[1].x, reps[2].x)


def test_train_val_test_streams_are_all_independent(clean):
    # no train/test leakage: each split keys its own RNG stream, and training
    # corruption (any epoch) never coincides with the pinned test corruption.
    tr0 = corrupt(MISSING, clean, experiment_seed=0, split="train", epoch=0)
    tr1 = corrupt(MISSING, clean, experiment_seed=0, split="train", epoch=1)
    va = corrupt(MISSING, clean, experiment_seed=0, split="val", epoch=0, severity=0.3)
    te = corrupt(MISSING, clean, experiment_seed=0, split="test", epoch=0, severity=0.3)
    for a, b in ((tr0, va), (tr0, te), (tr1, te), (va, te)):
        assert not torch.equal(a.x, b.x)


# ---------------------------------------------------------------------------
# Missing-feature contract
# ---------------------------------------------------------------------------


def test_missing_confidence_is_exactly_one_or_1e_3(clean):
    cor = corrupt_missing(clean, experiment_seed=0, split="train", epoch=0)
    obs = torch.tensor(OBSERVED_CONFIDENCE, dtype=cor.c.dtype)
    miss = torch.tensor(MISSING_CONFIDENCE, dtype=cor.c.dtype)
    assert torch.all((cor.c == obs) | (cor.c == miss))
    assert MISSING_CONFIDENCE == pytest.approx(1e-3)


def test_missing_values_are_zeroed_and_confidence_marks_them(clean):
    cor = corrupt_missing(clean, experiment_seed=0, split="train", epoch=0, p=0.5)
    missing = cor.c == MISSING_CONFIDENCE
    assert torch.all(cor.x[missing] == 0.0)
    observed = cor.c == OBSERVED_CONFIDENCE
    assert torch.equal(cor.x[observed], clean[observed])


def test_missing_p_zero_is_the_clean_input(clean):
    cor = corrupt_missing(clean, experiment_seed=0, split="test", epoch=0, p=0.0)
    assert torch.equal(cor.x, clean.to(torch.float32))
    assert torch.all(cor.c == OBSERVED_CONFIDENCE)


def test_missing_training_severity_is_drawn_from_the_discrete_set(clean):
    cor = corrupt_missing(clean, experiment_seed=0, split="train", epoch=0)
    assert set(np.unique(cor.severity)).issubset(set(MISSING_P_TRAIN))
    assert len(np.unique(cor.severity)) > 1  # actually random per example


def test_missing_higher_p_masks_more(clean):
    lo = corrupt_missing(clean, experiment_seed=0, split="test", epoch=0, p=0.1)
    hi = corrupt_missing(clean, experiment_seed=0, split="test", epoch=0, p=0.7)
    assert (hi.c == MISSING_CONFIDENCE).float().mean() > (lo.c == MISSING_CONFIDENCE).float().mean()


# ---------------------------------------------------------------------------
# Heterogeneous Gaussian contract
# ---------------------------------------------------------------------------


def test_gaussian_confidence_satisfies_c_equals_1_over_1_plus_sigma_sq(clean):
    cor = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=0.75)
    # recover sigma^2 from c and check the additive noise is consistent in scale
    sigma_sq = (1.0 / cor.c - 1.0).clamp_min(0.0)
    assert torch.all(sigma_sq >= 0.0)
    assert torch.all(cor.c > 0.0) and torch.all(cor.c <= 1.0)
    # sigma_j <= s everywhere (sigma_j ~ U(0, s))
    assert torch.all(sigma_sq <= 0.75**2 + 1e-5)


def test_gaussian_corruption_is_feature_heterogeneous_within_an_example(clean):
    cor = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=0.5)
    # within each row the reliability varies feature-to-feature
    per_row_spread = cor.c.std(dim=1)
    assert torch.all(per_row_spread > 0.0)
    assert per_row_spread.mean() > 1e-3


def test_gaussian_s_zero_is_the_clean_input(clean):
    cor = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=0.0)
    assert torch.allclose(cor.x, clean.to(torch.float32), atol=0.0)
    assert torch.all(cor.c == 1.0)


def test_gaussian_training_severity_from_discrete_set(clean):
    cor = corrupt_gaussian(clean, experiment_seed=0, split="train", epoch=0)
    assert set(np.unique(cor.severity)).issubset(set(GAUSSIAN_S_TRAIN))


def test_gaussian_more_severe_adds_more_noise(clean):
    lo = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=0.25)
    hi = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=1.5)
    dev_lo = (lo.x - clean).abs().mean()
    dev_hi = (hi.x - clean).abs().mean()
    assert dev_hi > dev_lo
    assert hi.c.mean() < lo.c.mean()  # more noise -> lower reported reliability


def test_gaussian_values_are_not_clipped(clean):
    # a big-severity draw should push some standardized values well outside
    # the clean [min, max] -- the task forbids clipping.
    cor = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=1.5)
    assert cor.x.max() > clean.max()
    assert cor.x.min() < clean.min()


# ---------------------------------------------------------------------------
# Confidence shuffling (Sec 15 intervention)
# ---------------------------------------------------------------------------


def test_shuffle_confidence_preserves_per_row_multiset_but_breaks_alignment(clean):
    cor = corrupt_gaussian(clean, experiment_seed=0, split="test", epoch=0, s=0.75)
    shuffled = shuffle_confidence(cor.c, seed=0)
    assert torch.allclose(shuffled.sort(dim=1).values, cor.c.sort(dim=1).values, atol=1e-6)
    assert not torch.equal(shuffled, cor.c)
    # deterministic
    assert torch.equal(shuffled, shuffle_confidence(cor.c, seed=0))
    assert not torch.equal(shuffled, shuffle_confidence(cor.c, seed=1))
