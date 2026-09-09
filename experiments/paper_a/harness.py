"""One Paper-A Phase-1 run = one (dataset, train_fraction, model_family, seed)
cell. Loads the frozen split, builds the sized model, trains it under the
shared protocol, evaluates it on the untouched test set, and writes a
`RunRecord` (full provenance -- git commit, config, params, seed, train
fraction, optimizer, best-val step, test metrics, wall-clock, device) plus
the validation curve.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.paper_a.datasets import PreparedDataset, prepare_dataset  # noqa: E402
from experiments.paper_a.models import (  # noqa: E402
    BELIEF_FAMILIES,
    CELLV02_FAMILY,
    MODEL_FAMILIES,
    batched_forward,
    belief_diagnostics,
    build_model,
    precision_gain_diagnostics,
)
from experiments.paper_a.training import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_STEPS,
    DEFAULT_WEIGHT_DECAY,
    train_model,
)
from src.evaluation.classification import accuracy, auroc, f1  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.regression import mae, r_squared, rmse  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

ALL_FAMILIES: tuple[str, ...] = (
    *MODEL_FAMILIES,
    "cellv0.1_fixed_confidence",
    CELLV02_FAMILY,
)
EXPERIMENT_ID = "paper_a_phase1"

# Early-stopping cadence is derived from the validation-set size so that a
# large val set (MNIST/Fashion-MNIST/California: 4k-10k examples) isn't
# re-scored every 50 steps -- the patience window stays ~1500 steps either
# way, keeping the protocol uniform in what matters (Paper-A task Sec 7).
_LARGE_VAL_THRESHOLD = 2000
_SMALL_VAL_CADENCE = (50, 30)   # (val_every, patience_checks) -> 1500-step patience
_LARGE_VAL_CADENCE = (200, 8)   # -> 1600-step patience


def _cadence_for(n_val: int) -> tuple[int, int]:
    return _LARGE_VAL_CADENCE if n_val > _LARGE_VAL_THRESHOLD else _SMALL_VAL_CADENCE


@torch.no_grad()
def evaluate(model: torch.nn.Module, prepared: PreparedDataset, device: torch.device) -> dict:
    """Test-set metrics on the original target scale. Classification:
    accuracy + macro-F1 (+ ROC-AUC for binary). Regression: RMSE + R^2 (+ MAE)
    after inverse-transforming predictions back from the standardized scale."""
    model.eval()
    x_test = prepared.x_test.to(device)

    infer_start = time.perf_counter()
    logits = batched_forward(model, x_test)
    if device.type == "mps":
        torch.mps.synchronize()
    inference_wall_clock = time.perf_counter() - infer_start
    logits = logits.cpu()

    metrics: dict[str, float] = {}
    if prepared.task_type == "classification":
        y_test = prepared.y_test.cpu()
        preds = logits.argmax(dim=-1)
        metrics["accuracy"] = accuracy(preds, y_test)
        metrics["macro_f1"] = f1(preds, y_test, average="macro")
        if prepared.is_binary:
            probs_pos = torch.softmax(logits, dim=-1)[:, 1]
            metrics["roc_auc"] = auroc(probs_pos, y_test)
    else:
        y_raw = prepared.y_test_raw.cpu().reshape(-1)
        pred_raw = prepared.inverse_transform_targets(logits.reshape(-1))
        metrics["rmse"] = rmse(pred_raw, y_raw)
        metrics["r2"] = r_squared(pred_raw, y_raw)
        metrics["mae"] = mae(pred_raw, y_raw)

    metrics["_inference_wall_clock_seconds"] = inference_wall_clock
    return metrics


def headline_metric_name(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def run_one(
    dataset_name: str,
    train_fraction: float,
    family: str,
    seed: int,
    *,
    device: torch.device | None = None,
    results_dir: str | Path = None,
    experiment_id: str = EXPERIMENT_ID,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    val_every: int | None = None,
    patience_checks: int | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    write_record: bool = True,
) -> dict:
    if family not in ALL_FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Expected one of {ALL_FAMILIES}.")
    device = device or get_device()
    if results_dir is None:
        results_dir = _REPO_ROOT / "experiments" / "paper_a" / "results" / "raw"
    results_dir = Path(results_dir)

    set_seed(seed)
    prepared = prepare_dataset(dataset_name, seed=seed, train_fraction=train_fraction)

    default_val_every, default_patience = _cadence_for(prepared.meta["n_val"])
    val_every = default_val_every if val_every is None else val_every
    patience_checks = default_patience if patience_checks is None else patience_checks

    built = build_model(family, prepared.n_features, prepared.out_features, prepared.param_budget)
    model = built.model
    param_count = count_parameters(model)

    outcome = train_model(
        model,
        prepared,
        device=device,
        seed=seed,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        val_every=val_every,
        patience_checks=patience_checks,
        max_steps=max_steps,
    )

    test_metrics = evaluate(model, prepared, device)
    inference_wall_clock = test_metrics.pop("_inference_wall_clock_seconds")

    diagnostics: dict[str, float] = {}
    if family in BELIEF_FAMILIES:
        diagnostics = belief_diagnostics(model, prepared.x_test.to(device))
    elif family == CELLV02_FAMILY:
        diagnostics = precision_gain_diagnostics(model, prepared.x_test.to(device))

    headline = headline_metric_name(prepared.task_type)
    efficiency = {
        "parameter_count": param_count,
        "best_val_step": outcome.best_val_step,
        "total_steps": outcome.total_steps,
        "epochs_completed": outcome.epochs_completed,
        "train_wall_clock_seconds": outcome.train_wall_clock_seconds,
        "inference_wall_clock_seconds": inference_wall_clock,
        "time_per_step_seconds": outcome.time_per_step_seconds,
        "effective_batch_size": outcome.effective_batch_size,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
    }

    extra = {
        "train_fraction": train_fraction,
        "device": device.type,
        "weight_decay": weight_decay,
        "batch_size_requested": batch_size,
        "batch_size_effective": outcome.effective_batch_size,
        "val_every": val_every,
        "patience_checks": patience_checks,
        "max_steps": max_steps,
        "task_type": prepared.task_type,
        "is_binary": prepared.is_binary,
        "n_classes": prepared.n_classes,
        "n_features": prepared.n_features,
        "headline_metric": headline,
        "hidden_size": built.hidden_size,
        "sizing": built.sizing,
        "dataset_meta": prepared.meta,
        "efficiency": efficiency,
        "diagnostics": diagnostics,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
    }

    config = ExperimentConfig(
        experiment_id=experiment_id,
        architecture=family,
        dataset=dataset_name,
        seed=seed,
        optimizer="adamw",
        learning_rate=lr,
        weight_decay=weight_decay,
        batch_size=outcome.effective_batch_size,
        max_steps=max_steps,
        extra=extra,
    )
    run_id = make_run_id(config)

    record = RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        architecture=family,
        config=config.to_dict(),
        parameter_count=param_count,
        dataset=dataset_name,
        seed=seed,
        optimizer="adamw",
        learning_rate=lr,
        steps_completed=outcome.total_steps,
        examples_or_tokens_seen=outcome.examples_seen,
        refinement_iterations=None,
        approximate_flops=None,
        train_wall_clock_seconds=outcome.train_wall_clock_seconds,
        inference_wall_clock_seconds=inference_wall_clock,
        validation_metrics={
            "best_val_loss": outcome.best_val_loss,
            "best_val_metric": outcome.best_val_metric,
            "best_val_step": outcome.best_val_step,
        },
        test_metrics={**test_metrics, **efficiency, **diagnostics},
        git_commit=get_git_commit(),
        checkpoint_path=None,
    )

    if write_record:
        out_dir = Path(results_dir)
        write_run_record(record, out_dir)
        with (out_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(outcome.history, f, indent=2)

    flat = {
        "run_id": run_id,
        "dataset": dataset_name,
        "train_fraction": train_fraction,
        "family": family,
        "seed": seed,
        "task_type": prepared.task_type,
        "headline_metric": headline,
        "headline_value": test_metrics[headline],
        "parameter_count": param_count,
        "n_train_used": prepared.meta["n_train_used"],
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"eff_{k}": v for k, v in efficiency.items()},
        **diagnostics,
        "sizing": built.sizing,
        "git_commit": get_git_commit(),
    }
    return flat


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
