"""Sec 10/11 mechanism diagnostics
(experiments/belief_dendrite/strong_baselines/evaluate.py), exercised
directly against freshly-initialized (untrained) models -- correctness of
the plumbing (shapes, key presence, correlation sign under a deliberately
constructed corruption) is independent of training quality.
"""

from __future__ import annotations

import torch

from experiments.belief_dendrite.strong_baselines.corruption import corrupt_missing_patch
from experiments.belief_dendrite.strong_baselines.evaluate import (
    belief_dendrite_diagnostics,
    confidence_vit_diagnostics,
    gated_vit_diagnostics,
)
from experiments.belief_dendrite.strong_baselines.models import build_model


def test_belief_dendrite_diagnostics_keys_and_finiteness_under_corruption() -> None:
    torch.manual_seed(0)
    x = torch.randn(48, 3, 32, 32)
    cor = corrupt_missing_patch(x, experiment_seed=0, split="test", epoch=0, replica=0, severity=24)
    model = build_model("belief_dendrite", seed=0).model
    out = belief_dendrite_diagnostics(model, cor.x, cor.c, cor.mask, chunk=16)

    for key in (
        "diag_layer1_pi_branch_mean",
        "diag_layer1_pi_soma_mean",
        "diag_layer1_u_soma_mean",
        "diag_layer1_effective_branches_mean",
        "localization_corr_overlap_vs_branch_pi",
        "localization_corr_overlap_vs_somatic_contribution",
    ):
        assert key in out, key
        assert out[key] == out[key], f"{key} is NaN under a non-trivial (severity=24) corruption"

    # By construction (Sec 4), a branch's precision cannot exceed the max
    # source precision, so mean branch pi over any batch is <= 1.0 + eps.
    assert out["diag_layer1_pi_branch_mean"] <= 1.0 + 1e-6


def test_confidence_vit_diagnostics_shape_and_key() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 3, 32, 32)
    c = torch.rand(16, 3, 32, 32)
    model = build_model("confidence_tiny_vit", seed=0).model
    out = confidence_vit_diagnostics(model, x, c, chunk=8)
    assert "reliability_to_image_embedding_norm_ratio_mean" in out
    assert out["reliability_to_image_embedding_norm_ratio_mean"] >= 0.0


def test_gated_vit_diagnostics_mean_patch_reliability_bounds() -> None:
    torch.manual_seed(0)
    x = torch.randn(16, 3, 32, 32)
    c = torch.rand(16, 3, 32, 32)
    model = build_model("reliability_gated_tiny_vit", seed=0).model
    out = gated_vit_diagnostics(model, x, c, chunk=8)
    assert 0.0 <= out["mean_patch_reliability_mean"] <= 1.0
