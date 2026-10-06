"""Reliability interventions
(experiments/belief_dendrite/strong_baselines/interventions.py, spec Sec 12):
condition construction and delta bookkeeping, independent of any trained
model (a fixed linear-in-reliability probe model stands in for a real
checkpoint so the test is fast and deterministic).
"""

from __future__ import annotations

import torch

from experiments.belief_dendrite.strong_baselines.interventions import (
    CONDITIONS,
    INTERVENTION_SEVERITIES,
    evaluate_intervention,
)


class _ReliabilitySensitiveProbe(torch.nn.Module):
    """`logits[:, 0] = mean(c)`, everything else 0 -- a synthetic model whose
    prediction (via argmax against class 0's threshold) is deliberately
    driven by the reliability map's *mean*, not its spatial layout, so
    "all_ones" and "shuffled" should score identically to "correct" (both
    preserve or exceed the mean), while a model that depended on spatial
    alignment would not."""

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        n = x.shape[0]
        mean_c = c.reshape(n, -1).mean(dim=-1)
        logits = torch.zeros(n, 10)
        logits[:, 0] = mean_c
        return logits


def test_evaluate_intervention_covers_all_three_conditions() -> None:
    torch.manual_seed(0)
    x = torch.randn(32, 3, 32, 32)
    y = torch.zeros(32, dtype=torch.long)  # class 0 is "correct" whenever logits[:,0] is largest
    model = _ReliabilitySensitiveProbe()
    result = evaluate_intervention(model, "confidence_cnn", x, y, seed=0, severity=12.0)
    assert set(result.accuracy_by_condition.keys()) == set(CONDITIONS)
    assert result.severity == 12.0


def test_all_ones_reliability_gives_the_highest_mean_and_thus_accuracy() -> None:
    """mean(all_ones) = 1 > mean(correct, mostly-1-with-a-1e-3-patch) --
    since the probe predicts class 0 iff mean_c is its largest logit (all
    other logits are exactly 0), "all_ones" must never score worse than
    "correct" for this probe."""
    torch.manual_seed(0)
    x = torch.randn(64, 3, 32, 32)
    y = torch.zeros(64, dtype=torch.long)
    model = _ReliabilitySensitiveProbe()
    result = evaluate_intervention(model, "confidence_cnn", x, y, seed=0, severity=24.0)
    assert result.accuracy_by_condition["all_ones"] >= result.accuracy_by_condition["correct"]


def test_shuffle_preserves_the_condition_but_not_spatial_alignment() -> None:
    """The probe only reads the reliability mean, which shuffling preserves
    exactly -- so "shuffled" must score identically to "correct" for this
    particular (mean-only) probe, distinguishing "uses reliability" from
    "uses *spatially aligned* reliability" (spec Sec 12's point)."""
    torch.manual_seed(0)
    x = torch.randn(32, 3, 32, 32)
    y = torch.zeros(32, dtype=torch.long)
    model = _ReliabilitySensitiveProbe()
    result = evaluate_intervention(model, "confidence_cnn", x, y, seed=0, severity=16.0)
    assert result.accuracy_by_condition["correct"] == result.accuracy_by_condition["shuffled"]
    assert result.delta("shuffled") == 0.0


def test_intervention_severities_are_12_and_24() -> None:
    assert INTERVENTION_SEVERITIES == (12.0, 24.0)
