"""Test-set evaluation, real-missingness stratification, and confidence
interventions for Paper A Phase 3 Part B.

* **APS** (imbalanced): PR-AUC / average precision (primary), ROC-AUC,
  balanced accuracy, F1, precision, recall, the official ``10*FP + 500*FN``
  test cost (at the frozen validation threshold) and ``cost_per_1000``.
* **Air Quality**: R^2 (primary), RMSE, MAE on the original target scale.

Every test example's ``missing_fraction`` buckets it into the fixed bins
`{0, (0, .10], (.10, .25], (.25, .50], >.50}`; performance and (for CellV0.3)
the belief-state statistics are reported per bin, empty bins skipped.

The confidence intervention (evaluation only, no retraining) re-scores every
trained CellV0.3 model with the true ``c``, with ``c := 1`` everywhere, and
with ``c`` permuted across feature positions per example.
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
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from experiments.paper_a.real_reliability.datasets import (  # noqa: E402
    MISSINGNESS_BINS,
    RealPreparedDataset,
    missingness_bin_index,
)
from experiments.paper_a.real_reliability.models import (  # noqa: E402
    CELLV03,
    ReliabilityCellV03,
    belief_diagnostics,
)
from experiments.paper_a.real_reliability.training import (  # noqa: E402
    _forward,
    model_inputs,
    official_cost,
)
from experiments.paper_a.reliability.corruption import shuffle_confidence  # noqa: E402
from src.evaluation.regression import mae as _mae  # noqa: E402
from src.evaluation.regression import r_squared, rmse  # noqa: E402

CLASSIFICATION_PRIMARY = "pr_auc"
REGRESSION_PRIMARY = "r2"


def primary_metric_name(task_type: str) -> str:
    return CLASSIFICATION_PRIMARY if task_type == "classification" else REGRESSION_PRIMARY


@torch.no_grad()
def _test_predictions(
    model: torch.nn.Module, prepared: RealPreparedDataset, family: str, device: torch.device
) -> torch.Tensor:
    x, c = (t.to(device) for t in model_inputs(prepared, "test", family))
    return _forward(model, x, c).reshape(-1).cpu()


def _classification_metrics(
    prob: np.ndarray, y: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    finite_auc = y.sum() not in (0, len(y))
    cost = official_cost(prob, y, threshold)
    return {
        "pr_auc": float(average_precision_score(y, prob)) if finite_auc else float("nan"),
        "roc_auc": float(roc_auc_score(y, prob)) if finite_auc else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "official_cost": float(cost),
        "cost_per_1000": float(cost / len(y) * 1000.0),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
    }


def _regression_metrics(pred_raw: torch.Tensor, y_raw: torch.Tensor) -> dict[str, float]:
    return {
        "r2": r_squared(pred_raw, y_raw),
        "rmse": rmse(pred_raw, y_raw),
        "mae": _mae(pred_raw, y_raw),
        "n": int(y_raw.numel()),
    }


@dataclass
class TestEvaluation:
    task_type: str
    primary_metric: str
    overall: dict[str, float]
    by_stratum: list[dict] = field(default_factory=list)          # one row per non-empty bin
    belief_by_stratum: list[dict] = field(default_factory=list)   # CellV0.3 only

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            "overall": self.overall,
            "by_stratum": self.by_stratum,
            "belief_by_stratum": self.belief_by_stratum,
        }


@torch.no_grad()
def evaluate_test(
    model: torch.nn.Module,
    prepared: RealPreparedDataset,
    family: str,
    *,
    device: torch.device,
    threshold: float | None = None,
) -> TestEvaluation:
    task = prepared.task_type
    raw_pred = _test_predictions(model, prepared, family, device)
    bins = missingness_bin_index(prepared.missing_frac_test)

    if task == "classification":
        prob = torch.sigmoid(raw_pred).numpy()
        y = prepared.y_test.reshape(-1).cpu().numpy()
        thr = 0.5 if threshold is None else float(threshold)
        overall = _classification_metrics(prob, y, thr)
        strata = []
        for i, (label, _lo, _hi) in enumerate(MISSINGNESS_BINS):
            m = (bins == i).numpy()
            if m.sum() == 0:
                continue
            row = {"bin": label, **_classification_metrics(prob[m], y[m], thr)}
            strata.append(row)
    else:
        pred_raw = prepared.inverse_transform_targets(raw_pred)
        y_raw = prepared.y_test_raw.reshape(-1).cpu()
        overall = _regression_metrics(pred_raw, y_raw)
        strata = []
        for i, (label, _lo, _hi) in enumerate(MISSINGNESS_BINS):
            m = bins == i
            if m.sum() == 0:
                continue
            strata.append({"bin": label, **_regression_metrics(pred_raw[m], y_raw[m])})

    belief_strata = []
    if family == CELLV03:
        belief_strata = belief_by_missingness_stratum(model, prepared, device=device)

    return TestEvaluation(
        task_type=task,
        primary_metric=primary_metric_name(task),
        overall=overall,
        by_stratum=strata,
        belief_by_stratum=belief_strata,
    )


@torch.no_grad()
def belief_by_missingness_stratum(
    model: ReliabilityCellV03, prepared: RealPreparedDataset, *, device: torch.device
) -> list[dict]:
    """Per missingness bin: mean hidden pi, mean hidden u, pi CoV (both layers),
    for the trained CellV0.3 model on the test set (Phase-3 task
    "Real-missingness diagnostics")."""
    x = prepared.x_test.to(device)
    c = prepared.c_test.to(device)
    bins = missingness_bin_index(prepared.missing_frac_test)
    rows = []
    for i, (label, _lo, _hi) in enumerate(MISSINGNESS_BINS):
        m = (bins == i).nonzero(as_tuple=True)[0]
        if m.numel() == 0:
            continue
        diag = belief_diagnostics(model, x[m], c[m], chunk=4096)
        rows.append({
            "bin": label,
            "n": int(m.numel()),
            "l1_pi_mean": diag["diag_layer1_precision_mean"],
            "l1_u_mean": diag["diag_layer1_u_mean"],
            "l1_pi_cv": diag["diag_layer1_precision_cv"],
            "l2_pi_mean": diag["diag_layer2_precision_mean"],
            "l2_u_mean": diag["diag_layer2_u_mean"],
            "l2_pi_cv": diag["diag_layer2_precision_cv"],
        })
    return rows


@dataclass
class InterventionResult:
    dataset: str
    primary_metric: str
    true: float
    all_ones: float
    shuffled: float

    @property
    def delta_all_ones(self) -> float:
        return self.true - self.all_ones

    @property
    def delta_shuffled(self) -> float:
        return self.true - self.shuffled

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "primary_metric": self.primary_metric,
            "true": self.true,
            "all_ones": self.all_ones,
            "shuffled": self.shuffled,
            "true_minus_all_ones": self.delta_all_ones,
            "true_minus_shuffled": self.delta_shuffled,
        }


@torch.no_grad()
def confidence_intervention(
    model: ReliabilityCellV03,
    prepared: RealPreparedDataset,
    *,
    device: torch.device,
    threshold: float | None = None,
    seed: int = 0,
) -> InterventionResult:
    """Re-score the trained CellV0.3 model on the test set with the true ``c``,
    with ``c := 1``, and with ``c`` shuffled across feature positions per
    example. Primary metric only; evaluation only."""
    if not isinstance(model, ReliabilityCellV03):
        raise TypeError(f"confidence intervention is CellV0.3-only, got {type(model).__name__}")
    x = prepared.x_test.to(device)
    c_true = prepared.c_test.to(device)
    task = prepared.task_type
    primary = primary_metric_name(task)

    variants = {
        "true": c_true,
        "all_ones": torch.ones_like(c_true),
        "shuffled": shuffle_confidence(c_true, seed=seed),
    }
    scores: dict[str, float] = {}
    y = prepared.y_test.reshape(-1).cpu()
    for name, c in variants.items():
        raw = _forward(model, x, c).reshape(-1).cpu()
        if task == "classification":
            prob = torch.sigmoid(raw).numpy()
            yn = y.numpy()
            scores[name] = (
                float(average_precision_score(yn, prob))
                if yn.sum() not in (0, len(yn)) else float("nan")
            )
        else:
            pred_raw = prepared.inverse_transform_targets(raw)
            scores[name] = r_squared(pred_raw, prepared.y_test_raw.reshape(-1).cpu())

    return InterventionResult(
        dataset=prepared.name, primary_metric=primary,
        true=scores["true"], all_ones=scores["all_ones"], shuffled=scores["shuffled"],
    )
