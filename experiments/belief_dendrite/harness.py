"""One Architecture V2 benchmark run = one `(dataset, corruption_family,
model_family, seed)` cell (Sec K/N).

Loads the frozen clean MNIST/Fashion-MNIST split (Paper A's dataset
loader, `train_fraction=1.0`), builds the family at ~150k parameters,
trains under per-epoch corruption with corrupted-validation checkpoint
selection, evaluates the best checkpoint across the full fixed severity
grid (3 replicas/severity), and writes a full-provenance `RunRecord` (git
commit, config, params, seed, optimizer, best-val step, per-severity
metrics, corruption AUC, OOD drop, wall-clock, device) plus the validation
curve.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402

from experiments.belief_dendrite.corruption import CORRUPTION_FAMILIES  # noqa: E402
from experiments.belief_dendrite.evaluate import PRIMARY_METRIC, sweep  # noqa: E402
from experiments.belief_dendrite.models import MODEL_FAMILIES, build_model  # noqa: E402
from experiments.belief_dendrite.training import DEFAULT_MAX_STEPS, train_model  # noqa: E402
from experiments.paper_a.datasets import prepare_dataset  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

EXPERIMENT_ID = "belief_dendrite"
DATASETS: tuple[str, ...] = ("mnist", "fashion_mnist")
SEEDS: tuple[int, ...] = (0, 1, 2)


def run_one(
    dataset_name: str,
    corruption_family: str,
    family: str,
    seed: int,
    *,
    device: torch.device | None = None,
    results_dir: str | Path | None = None,
    experiment_id: str = EXPERIMENT_ID,
    max_steps: int = DEFAULT_MAX_STEPS,
    write_record: bool = True,
) -> dict:
    if dataset_name not in DATASETS:
        raise ValueError(f"unknown dataset {dataset_name!r}; expected one of {DATASETS}")
    if corruption_family not in CORRUPTION_FAMILIES:
        raise ValueError(f"unknown corruption family {corruption_family!r}")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unknown model family {family!r}")

    device = device or get_device()
    if results_dir is None:
        results_dir = _REPO_ROOT / "experiments" / "belief_dendrite" / "results" / "raw"
    results_dir = Path(results_dir)

    set_seed(seed)
    prepared = prepare_dataset(dataset_name, seed=seed, train_fraction=1.0)
    built = build_model(family, seed=seed)
    model = built.model
    param_count = count_parameters(model)

    outcome = train_model(
        model,
        prepared,
        corruption_family=corruption_family,
        device=device,
        seed=seed,
        max_steps=max_steps,
    )

    eval_start = time.perf_counter()
    sweep_result = sweep(model, family, prepared, corruption_family, seed, device=device)
    eval_wall_clock = time.perf_counter() - eval_start

    efficiency = {
        "parameter_count": param_count,
        "hidden_size": built.hidden_size,
        "best_val_step": outcome.best_val_step,
        "total_steps": outcome.total_steps,
        "epochs_completed": outcome.epochs_completed,
        "train_wall_clock_seconds": outcome.train_wall_clock_seconds,
        "eval_wall_clock_seconds": eval_wall_clock,
        "time_per_step_seconds": outcome.time_per_step_seconds,
        "effective_batch_size": outcome.effective_batch_size,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
    }

    extra = {
        "corruption_family": corruption_family,
        "device": device.type,
        "primary_metric": PRIMARY_METRIC,
        "sizing": built.sizing,
        "in_dist_severities": list(outcome.in_dist_severities),
        "sweep": sweep_result.to_dict(),
        "efficiency": efficiency,
        "dataset_meta": prepared.meta,
    }

    config = ExperimentConfig(
        experiment_id=experiment_id,
        architecture=family,
        dataset=dataset_name,
        seed=seed,
        optimizer="adamw",
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=outcome.effective_batch_size,
        max_steps=max_steps,
        extra=extra,
    )
    run_id = make_run_id(config)

    sev_primary = {
        f"{PRIMARY_METRIC}@{s.severity}": s.metrics[PRIMARY_METRIC] for s in sweep_result.severities
    }
    record = RunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        architecture=family,
        config=config.to_dict(),
        parameter_count=param_count,
        dataset=dataset_name,
        seed=seed,
        optimizer="adamw",
        learning_rate=1e-2,
        steps_completed=outcome.total_steps,
        examples_or_tokens_seen=outcome.examples_seen,
        refinement_iterations=None,
        approximate_flops=None,
        train_wall_clock_seconds=outcome.train_wall_clock_seconds,
        inference_wall_clock_seconds=eval_wall_clock,
        validation_metrics={
            "best_val_metric": outcome.best_val_metric,
            "best_val_loss": outcome.best_val_loss,
            "best_val_step": outcome.best_val_step,
        },
        test_metrics={
            **sev_primary,
            "corruption_auc_primary": sweep_result.corruption_auc[PRIMARY_METRIC],
            "ood_drop_primary": sweep_result.ood_drop[PRIMARY_METRIC],
            **efficiency,
        },
        git_commit=get_git_commit(),
        checkpoint_path=None,
    )

    if write_record:
        results_dir.mkdir(parents=True, exist_ok=True)
        write_run_record(record, results_dir)
        with (results_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(outcome.history, f, indent=2)

    return {
        "run_id": run_id,
        "dataset": dataset_name,
        "corruption_family": corruption_family,
        "family": family,
        "seed": seed,
        "parameter_count": param_count,
        "hidden_size": built.hidden_size,
        "sizing": built.sizing,
        "sweep": sweep_result.to_dict(),
        "efficiency": efficiency,
        "diverged": outcome.diverged,
        "cap_hit": outcome.cap_hit,
        "git_commit": get_git_commit(),
    }
