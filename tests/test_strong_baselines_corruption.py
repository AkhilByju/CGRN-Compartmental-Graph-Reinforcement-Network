"""Strong-baselines corruption generator
(experiments/belief_dendrite/strong_baselines/corruption.py, spec Sec 3):
determinism, cross-model identity, reliability formulas, and channel-shared
masking for the CIFAR-10 local-missing-patch family.
"""

from __future__ import annotations

import torch

from experiments.belief_dendrite.strong_baselines.corruption import (
    EVAL_PATCH_SIDES,
    IMAGE_SHAPE,
    MISSING_CONFIDENCE,
    OBSERVED_CONFIDENCE,
    corrupt_missing_patch,
    horizontal_flip,
    shuffle_reliability_spatially,
)


def _x(n: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, *IMAGE_SHAPE, generator=g)


def test_corruption_is_a_pure_function_of_seed_split_epoch_replica() -> None:
    x = _x()
    a = corrupt_missing_patch(x, experiment_seed=3, split="test", epoch=0, replica=1, severity=12)
    b = corrupt_missing_patch(x, experiment_seed=3, split="test", epoch=0, replica=1, severity=12)
    assert torch.equal(a.x, b.x)
    assert torch.equal(a.c, b.c)
    assert torch.equal(a.mask, b.mask)


def test_identical_corruption_across_independent_calls_i_e_across_models() -> None:
    """Two "models" (here: two independent call sites) requesting the same
    `(seed, split, epoch, replica, severity)` must see byte-identical
    corrupted inputs -- corruption never depends on model identity."""
    x = _x()
    kwargs = {"experiment_seed": 7, "split": "test", "epoch": 2, "replica": 0, "severity": 16}
    for _ in range(3):
        cor = corrupt_missing_patch(x, **kwargs)
        reference = corrupt_missing_patch(x, **kwargs)
        assert torch.equal(cor.x, reference.x) and torch.equal(cor.c, reference.c)


def test_different_replicas_give_different_corruption() -> None:
    x = _x()
    a = corrupt_missing_patch(x, experiment_seed=3, split="test", epoch=0, replica=0, severity=12)
    b = corrupt_missing_patch(x, experiment_seed=3, split="test", epoch=0, replica=1, severity=12)
    assert not torch.equal(a.x, b.x)


def test_reliability_and_zeroing_match_spec_formula() -> None:
    x = _x(16)
    for side in EVAL_PATCH_SIDES:
        cor = corrupt_missing_patch(
            x, experiment_seed=0, split="test", epoch=0, replica=0, severity=side
        )
        assert bool((cor.severity == side).all())
        assert cor.mask.sum(dim=(1, 2)).max().item() == side * side
        mask_c = cor.mask.unsqueeze(1).expand(-1, 3, -1, -1)
        assert torch.allclose(cor.c[mask_c], torch.full_like(cor.c[mask_c], MISSING_CONFIDENCE))
        assert torch.allclose(cor.c[~mask_c], torch.full_like(cor.c[~mask_c], OBSERVED_CONFIDENCE))
        assert torch.allclose(cor.x[mask_c], torch.zeros_like(cor.x[mask_c]))
        assert torch.allclose(cor.x[~mask_c], x[~mask_c])


def test_mask_is_identical_across_all_three_rgb_channels() -> None:
    x = _x()
    cor = corrupt_missing_patch(x, experiment_seed=0, split="test", epoch=0, replica=0, severity=12)
    # c is channel-constant: every channel of the reliability map matches every other.
    assert torch.equal(cor.c[:, 0], cor.c[:, 1])
    assert torch.equal(cor.c[:, 1], cor.c[:, 2])


def test_train_regime_draws_a_per_example_severity_from_the_train_grid() -> None:
    x = _x(32)
    cor = corrupt_missing_patch(x, experiment_seed=0, split="train", epoch=0)
    assert cor.severity.min() >= 0
    assert len(set(cor.severity.tolist())) > 1


def test_epoch_changes_the_training_draw() -> None:
    x = _x()
    a = corrupt_missing_patch(x, experiment_seed=0, split="train", epoch=0)
    b = corrupt_missing_patch(x, experiment_seed=0, split="train", epoch=1)
    assert not torch.equal(a.x, b.x)


def test_horizontal_flip_is_deterministic_and_only_flips_some_examples() -> None:
    x = _x(32)
    a = horizontal_flip(x, experiment_seed=0, split="train", epoch=0)
    b = horizontal_flip(x, experiment_seed=0, split="train", epoch=0)
    assert torch.equal(a, b)
    matches_original = (a == x).all(dim=(1, 2, 3))
    assert matches_original.any() and not matches_original.all()


def test_shuffle_reliability_spatially_preserves_histogram_and_channel_agreement() -> None:
    x = _x(8)
    cor = corrupt_missing_patch(x, experiment_seed=0, split="test", epoch=0, replica=0, severity=16)
    shuffled = shuffle_reliability_spatially(cor.c, seed=0)
    orig_sorted, _ = cor.c[:, 0].reshape(cor.c.shape[0], -1).sort(dim=-1)
    shuf_sorted, _ = shuffled[:, 0].reshape(shuffled.shape[0], -1).sort(dim=-1)
    assert torch.allclose(orig_sorted, shuf_sorted)
    assert torch.equal(shuffled[:, 0], shuffled[:, 1])
