"""Shared logic for Experiment 003D -- evidence/uncertainty intervention
(docs/research_log.md "Experiment 002 initial results" follow-up 2;
docs/experiment_protocol.md Experiment 003). Trains a `precision`
`BeliefNetwork` exactly as Experiment 002 did (same datasets/hyperparameters,
so results are comparable), then at inference time perturbs the evidence/
uncertainty channels of the belief that the trained `layer1` hands to
`layer2` (`src.evaluation.intervention.perturb_belief`) and compares
prediction quality against the unperturbed baseline.

Answers: are `evidence`/`uncertainty` actually participating in `layer2`'s
aggregation in a way that matters for the final prediction, or has the
network learned to route around them ("decorative state")? If performance
barely changes under these perturbations, the latter.

The perturbation is inserted between `layer1` and `layer2` via
`model.layer1`/`model.layer2`/`model.readout` directly, NOT by modifying
`belief_network.py` (CLAUDE.md Sec 2) -- this is the only point in this
2-layer network where evidence/uncertainty are both (a) computed by the
model itself, rather than the fixed `evidence=1, uncertainty=1` of
`BeliefCell.from_observed_features`, and (b) still consumed downstream:
`layer2`'s own evidence/uncertainty outputs are never used again --
`readout` reads only `belief.mu`.

Not a `src/` module -- like `experiments/002_cell_v0/harness.py` and
`experiments/003_cell_ablation/uncertainty_harness.py`, this is specific to
this experiment's dataset/model choices, not generic infrastructure
(CLAUDE.md repository philosophy). Only `precision` is exercised (the model
this diagnostic was written for, per docs/research_log.md's "Experiment 002
initial results" -- `reliability`/`support_conflict` share the same
intervention point and could be added the same way later if useful).
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.synthetic.classification import (  # noqa: E402
    CLASSIFICATION_DATASETS,
    make_classification_splits,
)
from src.data.synthetic.regression import REGRESSION_DATASETS, make_regression_splits  # noqa: E402
from src.data.synthetic.utils import standardize  # noqa: E402
from src.evaluation.classification import accuracy as accuracy_metric  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.intervention import PERTURBATION_MODES, perturb_belief  # noqa: E402
from src.evaluation.regression import mae as mae_metric  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402
from src.evaluation.regression import rmse as rmse_metric  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.training.optimization import build_optimizer  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

DATASETS: tuple[str, ...] = REGRESSION_DATASETS + CLASSIFICATION_DATASETS


def _is_regression(dataset: str) -> bool:
    return dataset in REGRESSION_DATASETS


def _build_splits(dataset: str, seed: int, n_train: int, n_val: int, n_test: int):
    if _is_regression(dataset):
        splits = make_regression_splits(dataset, n_train, n_val, n_test, seed=seed)
    else:
        splits = make_classification_splits(dataset, n_train, n_val, n_test, seed=seed)
    return splits["train"], splits["val"], splits["test"]


def _forward_with_intervention(
    model: BeliefNetwork, x: torch.Tensor, mode: str, seed: int
) -> torch.Tensor:
    """Reimplements `BeliefNetwork.forward_with_beliefs` one layer at a time
    so `mode`'s perturbation can be inserted between `layer1` and `layer2`.
    See module docstring for why this is the right intervention point."""
    belief = BeliefCell.from_observed_features(x)
    belief = model.layer1(belief)
    belief = perturb_belief(belief, mode, seed)
    belief = model.layer2(belief)
    return model.readout(belief.mu)


def _compute_metrics(regression: bool, pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    if regression:
        return {"mae": mae_metric(pred, y), "rmse": rmse_metric(pred, y), "r2": r_squared(pred, y)}
    return {"accuracy": accuracy_metric(pred, y)}


def run_experiment(
    dataset: str,
    seed: int,
    hidden_cells: int = 16,
    steps: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 64,
    n_train: int = 1500,
    n_val: int = 300,
    n_test: int = 300,
    eval_every: int = 100,
    experiment_id: str = "003d_evidence_uncertainty_intervention",
    results_dir: str | Path = "results/raw",
) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(dataset)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _build_splits(
        dataset, seed, n_train, n_val, n_test
    )
    x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2

    model = BeliefNetwork(in_features, hidden_cells, out_features, aggregation="precision")
    model.to(device)
    architecture_name = "belief_network[precision]"

    config = ExperimentConfig(
        experiment_id=experiment_id,
        architecture=architecture_name,
        dataset=dataset,
        seed=seed,
        optimizer="adamw",
        learning_rate=lr,
        batch_size=batch_size,
        max_steps=steps,
        extra={"hidden_cells": hidden_cells},
    )
    optimizer = build_optimizer(model, config)
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_dev, y_val_dev = x_val.to(device), y_val.to(device)
    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)

    history: list[dict] = []
    step = 0
    start_time = time.perf_counter()
    model.train()
    while step < steps:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_every == 0 or step >= steps:
                model.eval()
                with torch.no_grad():
                    val_loss = loss_fn(model(x_val_dev), y_val_dev).item()
                history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
                model.train()
            if step >= steps:
                break
    train_wall_clock = time.perf_counter() - start_time

    model.eval()
    test_metrics: dict[str, float] = {}
    headline = "r2" if regression else "accuracy"
    with torch.no_grad():
        per_mode_headline: dict[str, float] = {}
        for mode in PERTURBATION_MODES:
            pred = _forward_with_intervention(model, x_test_dev, mode, seed=seed)
            metrics = _compute_metrics(regression, pred, y_test_dev)
            for name, value in metrics.items():
                test_metrics[f"{mode}__{name}"] = value
            per_mode_headline[mode] = metrics[headline]
        for mode in PERTURBATION_MODES:
            if mode == "baseline":
                continue
            test_metrics[f"{mode}__delta_{headline}"] = (
                per_mode_headline[mode] - per_mode_headline["baseline"]
            )

    run_id = make_run_id(config)
    record = RunRecord(
        run_id=run_id,
        experiment_id=config.experiment_id,
        architecture=architecture_name,
        config=config.to_dict(),
        parameter_count=count_parameters(model),
        dataset=dataset,
        seed=seed,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        steps_completed=step,
        examples_or_tokens_seen=step * batch_size,
        refinement_iterations=None,
        approximate_flops=None,
        train_wall_clock_seconds=train_wall_clock,
        inference_wall_clock_seconds=None,
        validation_metrics={"loss": history[-1]["val_loss"] if history else float("nan")},
        test_metrics=test_metrics,
        git_commit=get_git_commit(),
        checkpoint_path=None,
    )

    results_dir = Path(results_dir)
    write_run_record(record, results_dir)
    with (results_dir / f"{run_id}_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    return {
        "dataset": dataset,
        "seed": seed,
        "architecture": architecture_name,
        "parameter_count": record.parameter_count,
        "headline": headline,
        **test_metrics,
    }
