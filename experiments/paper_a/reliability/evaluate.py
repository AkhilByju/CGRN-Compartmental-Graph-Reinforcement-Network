"""Per-severity evaluation, robustness-curve summaries, and CellV0.3 mechanism
diagnostics for Paper A Phase 2.

Each best checkpoint is evaluated **once**, across the whole fixed severity
grid for its corruption family (never a separate model per severity). At every
severity, 3 deterministic corruption replicas are drawn (identical across
models) and the metrics are averaged over them -- corruption replicas are
evaluation noise and are collapsed *before* any across-seed statistic
(Sec 10).

Reported per ``(dataset, corruption_family, model, seed)``:

* the severity curve of the primary + secondary metric,
* ``corruption_AUC`` -- trapezoidal area under the severity/metric curve over
  the predefined grid (higher is better for accuracy / F1 / R^2),
* ``OOD_drop`` -- primary metric at the max training severity minus the metric
  at the most severe test condition (missingness ``p=.3 -> .7``, Gaussian
  ``s=.75 -> 1.5``); smaller is better,
* for CellV0.3, the per-layer belief-state diagnostics at each severity
  (Sec 13) and the ``severity -> mean hidden pi`` response (Sec 14).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
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
    eval_severities,
    in_dist_severities,
)
from experiments.paper_a.reliability.models import (  # noqa: E402
    CELLV03,
    ReliabilityCellV03,
    belief_diagnostics,
)
from experiments.paper_a.reliability.training import EVAL_CHUNK, forward_xc  # noqa: E402
from src.evaluation.classification import accuracy, f1  # noqa: E402
from src.evaluation.regression import r_squared, rmse  # noqa: E402

CLASSIFICATION_METRICS = ("accuracy", "macro_f1")
REGRESSION_METRICS = ("r2", "rmse")
# corruption_AUC and OOD_drop are "higher is better" for these; RMSE is the
# opposite and is reported but flagged, never mixed into a verdict.
HIGHER_IS_BETTER = frozenset({"accuracy", "macro_f1", "r2"})


def primary_metric_name(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def metric_names(task_type: str) -> tuple[str, ...]:
    return CLASSIFICATION_METRICS if task_type == "classification" else REGRESSION_METRICS


@dataclass
class SeverityResult:
    severity: float
    in_distribution: bool
    metrics: dict[str, float]                      # replica-averaged
    replica_metrics: list[dict[str, float]] = field(default_factory=list)
    belief_diag: dict[str, float] | None = None    # CellV0.3 only, replica-averaged

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "in_distribution": self.in_distribution,
            "metrics": self.metrics,
            "replica_metrics": self.replica_metrics,
            "belief_diag": self.belief_diag,
        }


@dataclass
class SweepResult:
    dataset: str
    corruption_family: str
    family: str
    task_type: str
    primary_metric: str
    severities: list[SeverityResult]
    corruption_auc: dict[str, float]
    ood_drop: dict[str, float]

    def by_severity(self, sev: float) -> SeverityResult:
        for s in self.severities:
            if abs(s.severity - sev) < 1e-9:
                return s
        raise KeyError(f"severity {sev} not in sweep")

    def metric_curve(self, metric: str) -> list[tuple[float, float]]:
        return [(s.severity, s.metrics[metric]) for s in self.severities]

    def reliability_response(self) -> dict[float, tuple[float, float]]:
        """``severity -> (layer1 mean pi, layer2 mean pi)`` -- CellV0.3 only
        (empty otherwise)."""
        out: dict[float, tuple[float, float]] = {}
        for s in self.severities:
            if s.belief_diag:
                out[s.severity] = (
                    s.belief_diag["diag_layer1_precision_mean"],
                    s.belief_diag["diag_layer2_precision_mean"],
                )
        return out

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "corruption_family": self.corruption_family,
            "family": self.family,
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            "severities": [s.to_dict() for s in self.severities],
            "corruption_auc": self.corruption_auc,
            "ood_drop": self.ood_drop,
        }


def _score(task_type: str, pred_std: torch.Tensor, prepared: PreparedDataset) -> dict[str, float]:
    """Metrics on the original target scale, mirroring the Phase-1 harness."""
    pred_std = pred_std.detach().cpu()
    if task_type == "classification":
        y = prepared.y_test.cpu()
        preds = pred_std.argmax(dim=-1)
        return {
            "accuracy": accuracy(preds, y),
            "macro_f1": f1(preds, y, average="macro"),
        }
    y_raw = prepared.y_test_raw.cpu().reshape(-1)
    pred_raw = prepared.inverse_transform_targets(pred_std.reshape(-1))
    return {"r2": r_squared(pred_raw, y_raw), "rmse": rmse(pred_raw, y_raw)}


@torch.no_grad()
def evaluate_severity(
    model: torch.nn.Module,
    family: str,
    prepared: PreparedDataset,
    corruption_family: str,
    seed: int,
    severity: float,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> SeverityResult:
    model.eval()
    x_test_clean = prepared.x_test
    task_type = prepared.task_type
    names = metric_names(task_type)

    replica_metrics: list[dict[str, float]] = []
    diag_accum: list[dict[str, float]] = []
    for replica in range(n_replicas):
        cor = corrupt(
            corruption_family, x_test_clean,
            experiment_seed=seed, split="test", epoch=0, replica=replica, severity=severity,
        )
        xc = cor.x.to(device)
        cc = cor.c.to(device)
        pred = forward_xc(model, xc, cc, chunk=chunk)
        replica_metrics.append(_score(task_type, pred, prepared))
        if family == CELLV03:
            assert isinstance(model, ReliabilityCellV03)
            diag_accum.append(belief_diagnostics(model, xc, cc, chunk=chunk))

    avg = {m: float(np.mean([rm[m] for rm in replica_metrics])) for m in names}
    belief_diag = None
    if diag_accum:
        belief_diag = {
            k: float(np.mean([d[k] for d in diag_accum])) for k in diag_accum[0]
        }

    return SeverityResult(
        severity=float(severity),
        in_distribution=severity in in_dist_severities(corruption_family),
        metrics=avg,
        replica_metrics=replica_metrics,
        belief_diag=belief_diag,
    )


def corruption_auc(points: list[tuple[float, float]]) -> float:
    """Trapezoidal area under the severity/metric curve over the (non-uniform)
    predefined grid. No normalization (Sec 11)."""
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    order = np.argsort(xs)
    return float(np.trapezoid(ys[order], xs[order]))


def ood_drop(corruption_family: str, sweep_severities: list[SeverityResult], metric: str) -> float:
    """``metric(max training severity) - metric(most severe test)``. Smaller
    degradation is better."""
    lo = MAX_TRAIN_SEVERITY[corruption_family]
    hi = MAX_OOD_SEVERITY[corruption_family]
    at = {s.severity: s.metrics[metric] for s in sweep_severities}
    return float(at[lo] - at[hi])


def sweep(
    model: torch.nn.Module,
    family: str,
    prepared: PreparedDataset,
    corruption_family: str,
    seed: int,
    *,
    device: torch.device,
    n_replicas: int = N_TEST_REPLICAS,
    chunk: int = EVAL_CHUNK,
) -> SweepResult:
    """Evaluate one trained checkpoint across the whole fixed severity grid."""
    task_type = prepared.task_type
    names = metric_names(task_type)
    results = [
        evaluate_severity(
            model, family, prepared, corruption_family, seed, sev,
            device=device, n_replicas=n_replicas, chunk=chunk,
        )
        for sev in eval_severities(corruption_family)
    ]
    auc = {m: corruption_auc([(r.severity, r.metrics[m]) for r in results]) for m in names}
    drop = {m: ood_drop(corruption_family, results, m) for m in names}
    return SweepResult(
        dataset=prepared.name,
        corruption_family=corruption_family,
        family=family,
        task_type=task_type,
        primary_metric=primary_metric_name(task_type),
        severities=results,
        corruption_auc=auc,
        ood_drop=drop,
    )
