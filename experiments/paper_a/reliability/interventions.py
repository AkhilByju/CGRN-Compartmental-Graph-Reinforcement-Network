"""Confidence-intervention test for the already-trained CellV0.3 checkpoints
(Phase-2 task Sec 15). **Evaluation only -- no retraining, nothing tuned.**

For the trained CellV0.3 models on Fashion-MNIST and California Housing, at the
maximum in-distribution corruption and the maximum OOD corruption, the
corrupted observations are held fixed and only the reliability map ``c`` fed to
the belief-state initialization is swapped:

* ``true``      -- the actual ``c`` (as in the real benchmark),
* ``all_ones``  -- ``c := 1`` everywhere (the model is told everything is
  reliable),
* ``shuffled``  -- ``c`` permuted across feature positions independently per
  example (same per-example reliability distribution, wrong alignment).

The question: does correct *alignment* between an observation and its stated
reliability matter to CellV0.3, or only the marginal distribution of ``c``?
Reported as the metric change relative to ``true``; not used to modify the
model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.paper_a.datasets import PreparedDataset  # noqa: E402
from experiments.paper_a.reliability.corruption import (  # noqa: E402
    MAX_OOD_SEVERITY,
    MAX_TRAIN_SEVERITY,
    N_TEST_REPLICAS,
    corrupt,
    shuffle_confidence,
)
from experiments.paper_a.reliability.evaluate import _score, primary_metric_name  # noqa: E402
from experiments.paper_a.reliability.models import ReliabilityCellV03  # noqa: E402
from experiments.paper_a.reliability.training import EVAL_CHUNK, forward_xc  # noqa: E402

INTERVENTION_DATASETS: tuple[str, ...] = ("fashion_mnist", "california_housing")
CONDITIONS: tuple[str, ...] = ("true", "all_ones", "shuffled")
REGIMES: tuple[str, ...] = ("in_dist_max", "ood_max")


def _regime_severity(corruption_family: str, regime: str) -> float:
    if regime == "in_dist_max":
        return MAX_TRAIN_SEVERITY[corruption_family]
    if regime == "ood_max":
        return MAX_OOD_SEVERITY[corruption_family]
    raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")


@dataclass
class InterventionResult:
    dataset: str
    corruption_family: str
    regime: str
    severity: float
    primary_metric: str
    condition_metric: dict[str, float]       # condition -> replica-averaged primary metric
    delta_vs_true: dict[str, float]          # (all_ones|shuffled) - true

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "corruption_family": self.corruption_family,
            "regime": self.regime,
            "severity": self.severity,
            "primary_metric": self.primary_metric,
            "condition_metric": self.condition_metric,
            "delta_vs_true": self.delta_vs_true,
        }


@torch.no_grad()
def run_intervention(
    model: ReliabilityCellV03,
    prepared: PreparedDataset,
    corruption_family: str,
    seed: int,
    regime: str,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> InterventionResult:
    if not isinstance(model, ReliabilityCellV03):
        raise TypeError(f"the confidence intervention is CellV0.3-only, got {type(model).__name__}")
    model.eval()
    severity = _regime_severity(corruption_family, regime)
    task_type = prepared.task_type
    primary = primary_metric_name(task_type)

    per_condition: dict[str, list[float]] = {cond: [] for cond in CONDITIONS}
    for replica in range(n_replicas):
        cor = corrupt(
            corruption_family, prepared.x_test,
            experiment_seed=seed, split="test", epoch=0, replica=replica, severity=severity,
        )
        xc = cor.x.to(device)
        c_true = cor.c.to(device)
        c_variants = {
            "true": c_true,
            "all_ones": torch.ones_like(c_true),
            # shuffle seed depends only on (seed, replica) -- never the model
            "shuffled": shuffle_confidence(c_true, seed=seed * 1000 + replica),
        }
        for cond, c in c_variants.items():
            pred = forward_xc(model, xc, c, chunk=chunk)
            per_condition[cond].append(_score(task_type, pred, prepared)[primary])

    cond_metric = {cond: float(np.mean(vals)) for cond, vals in per_condition.items()}
    delta = {
        cond: cond_metric[cond] - cond_metric["true"]
        for cond in ("all_ones", "shuffled")
    }
    return InterventionResult(
        dataset=prepared.name,
        corruption_family=corruption_family,
        regime=regime,
        severity=severity,
        primary_metric=primary,
        condition_metric=cond_metric,
        delta_vs_true=delta,
    )


def run_all_regimes(
    model: ReliabilityCellV03,
    prepared: PreparedDataset,
    corruption_family: str,
    seed: int,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> list[InterventionResult]:
    return [
        run_intervention(
            model, prepared, corruption_family, seed, regime,
            device=device, n_replicas=n_replicas, chunk=chunk,
        )
        for regime in REGIMES
    ]


