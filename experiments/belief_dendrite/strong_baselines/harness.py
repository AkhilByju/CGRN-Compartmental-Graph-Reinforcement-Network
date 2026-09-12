"""One strong-baselines benchmark run = one `(model_family, seed)` cell:
LR-grid training with validation-only selection, a full severity-grid
evaluation (with mechanism/Transformer diagnostics where applicable), the
Sec 12 reliability interventions for the four reliability-aware families,
and a full-provenance `RunRecord` (spec Sec 1's device/dtype/wall-time
metadata included).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from experiments.belief_dendrite.strong_baselines.datasets import prepare_cifar10  # noqa: E402
from experiments.belief_dendrite.strong_baselines.device_utils import (  # noqa: E402
    DTYPE,
    resolve_device,
)
from experiments.belief_dendrite.strong_baselines.evaluate import (  # noqa: E402
    PRIMARY_METRIC,
    sweep,
)
from experiments.belief_dendrite.strong_baselines.interventions import (  # noqa: E402
    INTERVENTION_FAMILIES,
    run_interventions,
)
from experiments.belief_dendrite.strong_baselines.models import (  # noqa: E402
    MODEL_FAMILIES,
    build_model,
)
from experiments.belief_dendrite.strong_baselines.param_budget import (  # noqa: E402
    TARGET_PARAM_BUDGET,
)
from experiments.belief_dendrite.strong_baselines.training import (  # noqa: E402
    LR_GRID,
    MAX_EPOCHS,
    PATIENCE_EPOCHS,
    WEIGHT_DECAY,
    select_lr_and_train,
)
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

EXPERIMENT_ID = "belief_dendrite_strong_baselines"
DATASET_NAME = "cifar10"
SEEDS: tuple[int, ...] = (0, 1, 2)


def run_one(
    family: str,
    seed: int,
    *,
    device: torch.device | None = None,
    results_dir: str | Path | None = None,
    batch_size: int = 256,
    lr_grid: tuple[float, ...] = LR_GRID,
    max_epochs: int = MAX_EPOCHS,
    patience_epochs: int = PATIENCE_EPOCHS,
    write_record: bool = True,
) -> dict:
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family {family!r}; expected one of {MODEL_FAMILIES}")

    device_report = resolve_device() if device is None else resolve_device(prefer=str(device))
    device = device_report.device
    if results_dir is None:
        results_dir = (
            _REPO_ROOT / "experiments" / "belief_dendrite" / "strong_baselines" / "results" / "raw"
        )
    results_dir = Path(results_dir)

    set_seed(seed)
    prepared = prepare_cifar10(seed=seed)

    built_ref: dict[str, object] = {}

    def factory() -> torch.nn.Module:
        built = build_model(family, seed)
        built_ref["built"] = built
        return built.model

    run_start = time.perf_counter()
    model, lr_result = select_lr_and_train(
        factory,
        prepared,
        device=device,
        seed=seed,
        lr_grid=lr_grid,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience_epochs=patience_epochs,
        weight_decay=WEIGHT_DECAY,
    )
    train_wall_clock = time.perf_counter() - run_start
    built = built_ref["built"]

    x_test = prepared.x_test.to(device=device, dtype=DTYPE)
    y_test = prepared.y_test.to(device)

    eval_start = time.perf_counter()
    sweep_result = sweep(model, family, x_test, y_test, seed, device=device)
    eval_wall_clock = time.perf_counter() - eval_start

    interventions_result = None
    if family in INTERVENTION_FAMILIES:
        interventions_result = run_interventions(model, family, x_test, y_test, seed)

    interventions_dicts = (
        [r.to_dict() for r in interventions_result] if interventions_result else None
    )
    chosen_outcome = lr_result.chosen_outcome
    lr_grid_summary = {
        str(lr): {
            "best_val_metric": o.best_val_metric,
            "epochs_completed": o.epochs_completed,
            "diverged": o.diverged,
        }
        for lr, o in lr_result.all_outcomes.items()
    }

    efficiency = {
        "parameter_count": built.parameter_count,
        "width": built.width,
        "sizing": built.sizing,
        "chosen_lr": lr_result.chosen_lr,
        "lr_grid_summary": lr_grid_summary,
        "best_val_epoch": chosen_outcome.best_val_epoch,
        "epochs_completed": chosen_outcome.epochs_completed,
        "total_steps": chosen_outcome.total_steps,
        "batch_size": chosen_outcome.batch_size,
        "train_wall_clock_seconds": train_wall_clock,
        "eval_wall_clock_seconds": eval_wall_clock,
        "time_per_step_seconds": chosen_outcome.time_per_step_seconds,
        "diverged": chosen_outcome.diverged,
        "cap_hit": chosen_outcome.cap_hit,
        "device": device_report.to_dict(),
    }

    extra = {
        "primary_metric": PRIMARY_METRIC,
        "sizing": built.sizing,
        "sweep": sweep_result.to_dict(),
        "interventions": interventions_dicts,
        "efficiency": efficiency,
        "dataset_meta": prepared.meta,
        "param_budget": TARGET_PARAM_BUDGET,
    }

    config = ExperimentConfig(
        experiment_id=EXPERIMENT_ID,
        architecture=family,
        dataset=DATASET_NAME,
        seed=seed,
        optimizer="adamw",
        learning_rate=lr_result.chosen_lr,
        weight_decay=WEIGHT_DECAY,
        batch_size=chosen_outcome.batch_size,
        max_epochs=max_epochs,
        extra=extra,
    )
    run_id = make_run_id(config)

    sev_primary = {
        f"{PRIMARY_METRIC}@{s.severity}": s.metrics[PRIMARY_METRIC] for s in sweep_result.severities
    }
    record = RunRecord(
        run_id=run_id,
        experiment_id=EXPERIMENT_ID,
        architecture=family,
        config=config.to_dict(),
        parameter_count=built.parameter_count,
        dataset=DATASET_NAME,
        seed=seed,
        optimizer="adamw",
        learning_rate=lr_result.chosen_lr,
        steps_completed=chosen_outcome.total_steps,
        examples_or_tokens_seen=chosen_outcome.examples_seen,
        refinement_iterations=None,
        approximate_flops=None,
        train_wall_clock_seconds=train_wall_clock,
        inference_wall_clock_seconds=eval_wall_clock,
        validation_metrics={
            "best_val_metric": chosen_outcome.best_val_metric,
            "best_val_epoch": chosen_outcome.best_val_epoch,
        },
        test_metrics={
            **sev_primary,
            "corruption_auc_primary": sweep_result.corruption_auc[PRIMARY_METRIC],
            "ood_drop_primary": sweep_result.ood_drop[PRIMARY_METRIC],
            **{
                k: v
                for k, v in efficiency.items()
                if k not in ("device", "sizing", "lr_grid_summary")
            },
        },
        git_commit=get_git_commit(),
        checkpoint_path=None,
    )

    if write_record:
        results_dir.mkdir(parents=True, exist_ok=True)
        write_run_record(record, results_dir)
        with (results_dir / f"{run_id}_history.json").open("w") as f:
            history_by_lr = {lr: o.history for lr, o in lr_result.all_outcomes.items()}
            json.dump(history_by_lr, f, indent=2, default=str)

    return {
        "run_id": run_id,
        "family": family,
        "seed": seed,
        "parameter_count": built.parameter_count,
        "chosen_lr": lr_result.chosen_lr,
        "sweep": sweep_result.to_dict(),
        "interventions": interventions_dicts,
        "efficiency": efficiency,
        "diverged": chosen_outcome.diverged,
        "git_commit": get_git_commit(),
    }
