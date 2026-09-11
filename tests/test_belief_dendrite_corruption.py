"""Architecture V2 frozen-benchmark corruption generator
(experiments/belief_dendrite/corruption.py, docs/architecture_v2.md Sec
K/L/M): determinism, reliability values, and patch geometry for both
structured-corruption families.
"""

from __future__ import annotations

import torch

from experiments.belief_dendrite.corruption import (
    EVAL_PATCH_SIDES,
    EVAL_SIGMAS,
    MISSING_CONFIDENCE,
    MISSING_PATCH,
    NOISY_PATCH,
    OBSERVED_CONFIDENCE,
    corrupt,
)


def _x() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(24, 784, generator=g)


def test_corruption_is_a_pure_function_of_seed_split_epoch_replica_not_model() -> None:
    x = _x()
    a = corrupt(MISSING_PATCH, x, experiment_seed=3, split="test", epoch=0, replica=1, severity=10)
    b = corrupt(MISSING_PATCH, x, experiment_seed=3, split="test", epoch=0, replica=1, severity=10)
    assert torch.equal(a.x, b.x) and torch.equal(a.c, b.c) and torch.equal(a.mask, b.mask)


def test_different_replicas_give_different_corruption() -> None:
    x = _x()
    a = corrupt(MISSING_PATCH, x, experiment_seed=3, split="test", epoch=0, replica=0, severity=10)
    b = corrupt(MISSING_PATCH, x, experiment_seed=3, split="test", epoch=0, replica=1, severity=10)
    assert not torch.equal(a.x, b.x)


def test_missing_patch_pins_reliability_and_zeros_inside_patch() -> None:
    x = _x()
    for side in EVAL_PATCH_SIDES:
        cor = corrupt(
            MISSING_PATCH, x, experiment_seed=0, split="test", epoch=0, replica=0, severity=side
        )
        assert bool((cor.severity == side).all())
        assert cor.mask.sum(dim=1).max().item() == side * side
        assert torch.allclose(cor.c[cor.mask], torch.full_like(cor.c[cor.mask], MISSING_CONFIDENCE))
        assert torch.allclose(
            cor.c[~cor.mask], torch.full_like(cor.c[~cor.mask], OBSERVED_CONFIDENCE)
        )
        assert torch.allclose(cor.x[cor.mask], torch.zeros_like(cor.x[cor.mask]))
        assert torch.allclose(cor.x[~cor.mask], x[~cor.mask])


def test_noisy_patch_side_is_fixed_and_reliability_matches_sigma() -> None:
    x = _x()
    for sigma in EVAL_SIGMAS:
        cor = corrupt(
            NOISY_PATCH, x, experiment_seed=1, split="val", epoch=0, replica=0, severity=sigma
        )
        assert bool((cor.mask.sum(dim=1) == 100).all())  # side=10 always
        expected_c = 1.0 / (1.0 + sigma**2)
        assert torch.allclose(
            cor.c[cor.mask], torch.full_like(cor.c[cor.mask], expected_c), atol=1e-5
        )
        assert torch.allclose(cor.c[~cor.mask], torch.ones_like(cor.c[~cor.mask]))
        assert torch.allclose(cor.x[~cor.mask], x[~cor.mask])
        if sigma == 0.0:
            assert torch.allclose(cor.x, x)


def test_train_regime_draws_a_per_example_severity() -> None:
    x = _x()
    cor = corrupt(MISSING_PATCH, x, experiment_seed=0, split="train", epoch=0)
    assert cor.severity.min() >= 0
    assert len(set(cor.severity.tolist())) > 1  # not every example got the same side


def test_epoch_changes_the_training_draw() -> None:
    x = _x()
    a = corrupt(MISSING_PATCH, x, experiment_seed=0, split="train", epoch=0)
    b = corrupt(MISSING_PATCH, x, experiment_seed=0, split="train", epoch=1)
    assert not torch.equal(a.x, b.x)
