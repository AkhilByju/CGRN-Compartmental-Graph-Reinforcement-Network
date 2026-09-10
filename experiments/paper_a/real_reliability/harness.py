"""One Paper-A Phase-3 Part-B run = one ``(dataset, model_family, seed)`` cell.

Sweeps the shared LR grid (and, for NeuMiss, the depth grid) on train/validation
only, picks the best validation primary metric, evaluates the test set once
(full metric suite + missingness strata + -- for CellV0.3 -- belief-by-stratum
diagnostics and the confidence intervention), and writes a full-provenance
`RunRecord`.

`run_hgb_reference` runs the scikit-learn `HistGradientBoosting*` **reference**
baseline (native missing-value support, no parameter matching, not used to
select any neural hyperparameter).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.paper_a.real_reliability.datasets import (  # noqa: E402
    REAL_DATASETS,
    prepare_dataset,
)
from experiments.paper_a.real_reliability.evaluate import (  # noqa: E402
    confidence_intervention,
    evaluate_test,
)
from experiments.paper_a.real_reliability.models import CELLV03, MODEL_FAMILIES  # noqa: E402
from experiments.paper_a.real_reliability.training import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    LEARNING_RATES,
    _cost_optimal_threshold,
    official_cost,
    train_with_grid,
)
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

EXPERIMENT_ID = "paper_a_real_reliability"
HGB_EXPERIMENT_ID = "paper_a_real_reliability_hgb"
SEEDS: tuple[int, ...] = (0, 1, 2)


def run_one(
    dataset_name: str,
    family: str,
    seed: int,
    *,
    device: torch.device | None = None,
    results_dir: str | Path | None = None,
    experiment_id: str = EXPERIMENT_ID,
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    max_steps: int = DEFAULT_MAX_STEPS,
    write_record: bool = True,
) -> dict:
    if dataset_name not in REAL_DATASETS:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown family {family!r}")

    device = device or get_device()
    if results_dir is None:
        results_dir = (
            _REPO_ROOT / "experiments" / "paper_a" / "real_reliability" / "results" / "raw"
        )
    results_dir = Path(results_dir)

    set_seed(seed)
    prepared = prepare_dataset(dataset_name, seed=seed)

    t0 = time.perf_counter()
    outcome, model, built = train_with_grid(
        prepared, family,
        in_features=prepared.n_features, param_budget=prepared.param_budget,
        seed=seed, device=device, learning_rates=learning_rates, max_steps=max_steps,
    )
    train_wall = time.perf_counter() - t0

    ev = evaluate_test(
        model, prepared, family, device=device, threshold=outcome.cost_threshold
    )

    intervention = None
    if family == CELLV03:
        intervention = confidence_intervention(
            model, prepared, device=device, threshold=outcome.cost_threshold, seed=seed
        ).to_dict()

    param_count = count_parameters(model)
    efficiency = {
        "parameter_count": param_count,
        "hidden_size": built.hidden_size,
        "neumiss_depth": built.neumiss_depth,
        "selected_lr": outcome.lr,
        "best_val_metric": outcome.best_val_metric,
        "best_val_step": outcome.best_val_step,
        "total_steps": outcome.total_steps,
        "train_wall_clock_seconds": train_wall,
        "time_per_step_seconds": outcome.time_per_step_seconds,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
        "cost_threshold": outcome.cost_threshold,
    }

    extra = {
        "device": device.type,
        "task_type": prepared.task_type,
        "primary_metric": ev.primary_metric,
        "n_features": prepared.n_features,
        "param_budget": prepared.param_budget,
        "sizing": built.sizing,
        "selected_lr": outcome.lr,
        "neumiss_depth": built.neumiss_depth,
        "evaluation": ev.to_dict(),
        "intervention": intervention,
        "efficiency": efficiency,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
        "dataset_meta": prepared.meta,
    }

    config = ExperimentConfig(
        experiment_id=experiment_id, architecture=family, dataset=dataset_name, seed=seed,
        optimizer="adamw", learning_rate=outcome.lr, weight_decay=0.0,
        batch_size=128, max_steps=max_steps, extra=extra,
    )
    run_id = make_run_id(config)

    record = RunRecord(
        run_id=run_id, experiment_id=experiment_id, architecture=family,
        config=config.to_dict(), parameter_count=param_count, dataset=dataset_name, seed=seed,
        optimizer="adamw", learning_rate=outcome.lr,
        steps_completed=outcome.total_steps, examples_or_tokens_seen=outcome.total_steps * 128,
        refinement_iterations=None, approximate_flops=None,
        train_wall_clock_seconds=train_wall, inference_wall_clock_seconds=None,
        validation_metrics={
            "best_val_metric": outcome.best_val_metric,
            "best_val_step": outcome.best_val_step,
            "selected_lr": outcome.lr,
        },
        test_metrics={**ev.overall, **efficiency},
        git_commit=get_git_commit(), checkpoint_path=None,
    )
    if write_record:
        results_dir.mkdir(parents=True, exist_ok=True)
        write_run_record(record, results_dir)
        with (results_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(outcome.history, f, indent=2)

    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "family": family,
        "seed": seed,
        "task_type": prepared.task_type,
        "primary_metric": ev.primary_metric,
        "parameter_count": param_count,
        "hidden_size": built.hidden_size,
        "neumiss_depth": built.neumiss_depth,
        "selected_lr": outcome.lr,
        "sizing": built.sizing,
        "evaluation": ev.to_dict(),
        "intervention": intervention,
        "efficiency": efficiency,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
        "git_commit": get_git_commit(),
    }


# ---------------------------------------------------------------------------
# HistGradientBoosting reference baseline (Phase-3 task "Optional non-neural
# reference") -- native missing support, NOT parameter-matched, NOT used to
# select any neural hyperparameter.
# ---------------------------------------------------------------------------


def run_hgb_reference(
    dataset_name: str,
    seed: int,
    *,
    results_dir: str | Path | None = None,
    write_record: bool = True,
) -> dict:
    from sklearn.ensemble import (  # noqa: PLC0415
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    set_seed(seed)
    prepared = prepare_dataset(dataset_name, seed=seed)
    if results_dir is None:
        results_dir = (
            _REPO_ROOT / "experiments" / "paper_a" / "real_reliability" / "results" / "raw"
        )
    results_dir = Path(results_dir)

    # HGB consumes the raw NaN-preserving features (native missing support).
    x_tr = prepared.x_train_nan.numpy()
    x_va = prepared.x_val_nan.numpy()
    x_te = prepared.x_test_nan.numpy()

    t0 = time.perf_counter()
    if prepared.task_type == "classification":
        clf = HistGradientBoostingClassifier(
            random_state=seed, class_weight="balanced", early_stopping=True
        )
        clf.fit(x_tr, prepared.y_train.numpy())
        prob_va = clf.predict_proba(x_va)[:, 1]
        prob_te = clf.predict_proba(x_te)[:, 1]
        thr, _ = _cost_optimal_threshold(prob_va, prepared.y_val.numpy())
        y = prepared.y_test.numpy()
        from sklearn.metrics import (  # noqa: PLC0415
            average_precision_score,
            balanced_accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        pred = (prob_te >= thr).astype(int)
        cost = official_cost(prob_te, y, thr)
        overall = {
            "pr_auc": float(average_precision_score(y, prob_te)),
            "roc_auc": float(roc_auc_score(y, prob_te)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "official_cost": float(cost),
            "cost_per_1000": float(cost / len(y) * 1000.0),
        }
        primary = "pr_auc"
    else:
        reg = HistGradientBoostingRegressor(random_state=seed, early_stopping=True)
        reg.fit(x_tr, prepared.y_train_raw.numpy().reshape(-1))
        pred_te = torch.tensor(reg.predict(x_te), dtype=torch.float32)
        y_raw = prepared.y_test_raw.reshape(-1)
        from src.evaluation.regression import mae, r_squared, rmse  # noqa: PLC0415
        overall = {
            "r2": r_squared(pred_te, y_raw),
            "rmse": rmse(pred_te, y_raw),
            "mae": mae(pred_te, y_raw),
        }
        primary = "r2"
    wall = time.perf_counter() - t0

    config = ExperimentConfig(
        experiment_id=HGB_EXPERIMENT_ID, architecture="hist_gradient_boosting",
        dataset=dataset_name, seed=seed, optimizer="none", learning_rate=0.0,
        extra={"task_type": prepared.task_type, "primary_metric": primary,
               "evaluation": {"overall": overall}, "reference_only": True,
               "train_wall_clock_seconds": wall},
    )
    run_id = make_run_id(config)
    record = RunRecord(
        run_id=run_id, experiment_id=HGB_EXPERIMENT_ID, architecture="hist_gradient_boosting",
        config=config.to_dict(), parameter_count=0, dataset=dataset_name, seed=seed,
        optimizer="none", learning_rate=0.0, steps_completed=0, examples_or_tokens_seen=0,
        refinement_iterations=None, approximate_flops=None,
        train_wall_clock_seconds=wall, inference_wall_clock_seconds=None,
        validation_metrics={}, test_metrics=overall,
        git_commit=get_git_commit(), checkpoint_path=None,
    )
    if write_record:
        results_dir.mkdir(parents=True, exist_ok=True)
        write_run_record(record, results_dir)

    return {
        "run_id": run_id, "dataset": dataset_name, "family": "hist_gradient_boosting",
        "seed": seed, "primary_metric": primary, "evaluation": {"overall": overall},
        "reference_only": True, "train_wall_clock_seconds": wall,
    }


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError(f"not serializable: {type(o)}")
