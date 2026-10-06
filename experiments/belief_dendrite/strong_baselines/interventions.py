"""Reliability interventions (spec Sec 12): for the best checkpoints of the
four reliability-aware families, evaluate patch sizes 12 and 24 under three
reliability conditions -- the true map, all-ones, and a spatially-shuffled
(histogram-preserving) map -- to test whether each architecture's use of
reliability actually depends on correct spatial alignment, not merely on
its presence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from experiments.belief_dendrite.strong_baselines.corruption import (
    corrupt_missing_patch,
    shuffle_reliability_spatially,
)
from experiments.belief_dendrite.strong_baselines.models import (
    BELIEF_DENDRITE,
    CONFIDENCE_CNN,
    CONFIDENCE_TINY_VIT,
    RELIABILITY_GATED_TINY_VIT,
)
from experiments.belief_dendrite.strong_baselines.training import EVAL_CHUNK, forward_xc
from src.evaluation.classification import accuracy

INTERVENTION_FAMILIES: tuple[str, ...] = (
    CONFIDENCE_CNN,
    CONFIDENCE_TINY_VIT,
    RELIABILITY_GATED_TINY_VIT,
    BELIEF_DENDRITE,
)
INTERVENTION_SEVERITIES: tuple[float, ...] = (12.0, 24.0)
CONDITIONS: tuple[str, ...] = ("correct", "all_ones", "shuffled")


@dataclass
class InterventionResult:
    family: str
    severity: float
    accuracy_by_condition: dict[str, float] = field(default_factory=dict)

    def delta(self, condition: str) -> float:
        """`accuracy(correct) - accuracy(condition)` -- how much accuracy is
        lost when the model is denied correct spatial reliability."""
        return self.accuracy_by_condition["correct"] - self.accuracy_by_condition[condition]

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "severity": self.severity,
            "accuracy_by_condition": self.accuracy_by_condition,
            "delta_all_ones": self.delta("all_ones"),
            "delta_shuffled": self.delta("shuffled"),
        }


@torch.no_grad()
def evaluate_intervention(
    model: torch.nn.Module,
    family: str,
    x_test_clean: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
    severity: float,
    *,
    chunk: int = EVAL_CHUNK,
) -> InterventionResult:
    model.eval()
    cor = corrupt_missing_patch(
        x_test_clean, experiment_seed=seed, split="test", epoch=0, replica=0, severity=severity
    )
    conditions = {
        "correct": cor.c,
        "all_ones": torch.ones_like(cor.c),
        "shuffled": shuffle_reliability_spatially(cor.c, seed=seed * 10_000 + int(severity)),
    }
    acc_by_condition: dict[str, float] = {}
    for name, c_variant in conditions.items():
        pred = forward_xc(model, cor.x, c_variant, chunk=chunk)
        acc_by_condition[name] = accuracy(pred.argmax(dim=-1).cpu(), y_test.cpu())
    return InterventionResult(
        family=family, severity=float(severity), accuracy_by_condition=acc_by_condition
    )


def run_interventions(
    model: torch.nn.Module,
    family: str,
    x_test_clean: torch.Tensor,
    y_test: torch.Tensor,
    seed: int,
    *,
    severities: tuple[float, ...] = INTERVENTION_SEVERITIES,
    chunk: int = EVAL_CHUNK,
) -> list[InterventionResult]:
    if family not in INTERVENTION_FAMILIES:
        raise ValueError(
            f"{family!r} is not one of the Sec 12 intervention families {INTERVENTION_FAMILIES}"
        )
    return [
        evaluate_intervention(model, family, x_test_clean, y_test, seed, sev, chunk=chunk)
        for sev in severities
    ]
