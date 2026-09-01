"""Shared logic for Experiment 002/003 (docs/experiment_protocol.md):
trains one model (an MLP baseline or a BeliefNetwork with one of the three
aggregation methods) on one synthetic dataset, evaluates it, and records a
RunRecord plus a training-curve history file.

Not a `src/` module: it is specific to this experiment (its dataset/model
choices, its "ambiguous vs. clear" diagnostic), not generic infrastructure
-- see CLAUDE.md's repository philosophy. `run.py` (single run) and
`run_all.py` (the full grid) both `import harness` directly, which works
because Python adds a script's own directory to `sys.path` when it's
executed. The `sys.path` bootstrap below does the same for this
directory's `src.*` imports, which is necessary here (unlike in `run.py`/
`run_all.py`) because this module can also be imported on its own.
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
    MARGIN_FUNCTIONS,
    make_classification_splits,
)
from src.data.synthetic.regression import (  # noqa: E402
    REGRESSION_DATASETS,
    make_regression_splits,
    regression_ood_inputs,
)
from src.evaluation.classification import accuracy as accuracy_metric  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.regression import mae as mae_metric  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402
from src.evaluation.regression import rmse as rmse_metric  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.baselines.mlp import MLPBaseline, match_hidden_dim  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.training.optimization import build_optimizer  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

DATASETS: tuple[str, ...] = REGRESSION_DATASETS + CLASSIFICATION_DATASETS
MODEL_CHOICES: tuple[str, ...] = ("mlp", "reliability", "support_conflict", "precision")


def _is_regression(dataset: str) -> bool:
    return dataset in REGRESSION_DATASETS


def _build_splits(dataset: str, seed: int, n_train: int, n_val: int, n_test: int):
    if _is_regression(dataset):
        splits = make_regression_splits(dataset, n_train, n_val, n_test, seed=seed)
    else:
        splits = make_classification_splits(dataset, n_train, n_val, n_test, seed=seed)
    return splits["train"], splits["val"], splits["test"]


def _standardize_with(train_x: torch.Tensor, *xs: torch.Tensor):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std, tuple((x - mean) / std for x in xs)


def _ambiguous_and_clear_raw_inputs(
    dataset: str, seed: int, n_samples: int = 400
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns `(ambiguous_x, clear_x)` in RAW (unstandardized) feature
    space.

    Classification: "ambiguous" = smallest-|margin| points (near the
    decision boundary); "clear" = largest-|margin| points (confidently one
    class). Regression: "ambiguous" = extrapolation inputs outside the
    training domain; "clear" = fresh in-distribution inputs. There is no
    formal notion of an "ambiguous" input for a deterministic regression
    function, so extrapolation is the closest analogue: the model has less
    reason to be confident about inputs unlike anything it was trained on.
    """
    if _is_regression(dataset):
        clear_x, _ = make_regression_splits(dataset, n_samples, 0, 0, seed=seed + 10_000)["train"]
        ambiguous_x = regression_ood_inputs(dataset, n_samples, seed=seed + 20_000)
        return ambiguous_x, clear_x

    x, _ = make_classification_splits(dataset, n_samples, 0, 0, seed=seed + 10_000)["train"]
    margin = MARGIN_FUNCTIONS[dataset](x).abs()
    order = torch.argsort(margin)
    k = max(1, n_samples // 10)
    ambiguous_x = x[order[:k]]
    clear_x = x[order[-k:]]
    return ambiguous_x, clear_x


def run_experiment(
    dataset: str,
    model_name: str,
    seed: int,
    hidden_cells: int = 16,
    steps: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 64,
    n_train: int = 1500,
    n_val: int = 300,
    n_test: int = 300,
    eval_every: int = 100,
    results_dir: str | Path = "results/raw",
) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {MODEL_CHOICES}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(dataset)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = _build_splits(
        dataset, seed, n_train, n_val, n_test
    )
    train_mean, train_std, (x_train, x_val, x_test) = _standardize_with(
        x_train, x_train, x_val, x_test
    )

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2

    if model_name == "mlp":
        reference = BeliefNetwork(
            in_features, hidden_cells, out_features, aggregation="reliability"
        )
        target_params = count_parameters(reference)
        hidden_dim = match_hidden_dim(target_params, in_features, out_features, num_hidden_layers=2)
        model = MLPBaseline(in_features, hidden_dim, out_features, num_hidden_layers=2)
        architecture_name = "mlp_baseline"
    else:
        model = BeliefNetwork(in_features, hidden_cells, out_features, aggregation=model_name)
        architecture_name = f"belief_network[{model_name}]"
    model.to(device)

    config = ExperimentConfig(
        experiment_id="002_cell_v0",
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
    with torch.no_grad():
        test_pred = model(x_test_dev)
        if regression:
            test_metrics = {
                "mae": mae_metric(test_pred, y_test_dev),
                "rmse": rmse_metric(test_pred, y_test_dev),
                "r2": r_squared(test_pred, y_test_dev),
            }
        else:
            test_metrics = {"accuracy": accuracy_metric(test_pred, y_test_dev)}

        if model_name != "mlp":
            _, test_belief = model.forward_with_beliefs(x_test_dev)
            test_metrics["evidence_mean"] = test_belief.evidence.mean().item()
            test_metrics["evidence_std"] = test_belief.evidence.std().item()
            test_metrics["uncertainty_mean"] = test_belief.uncertainty.mean().item()
            test_metrics["uncertainty_std"] = test_belief.uncertainty.std().item()

            ambiguous_x, clear_x = _ambiguous_and_clear_raw_inputs(dataset, seed)
            ambiguous_x = ((ambiguous_x - train_mean) / train_std).to(device)
            clear_x = ((clear_x - train_mean) / train_std).to(device)
            _, ambiguous_belief = model.forward_with_beliefs(ambiguous_x)
            _, clear_belief = model.forward_with_beliefs(clear_x)
            test_metrics["ambiguous_uncertainty_mean"] = ambiguous_belief.uncertainty.mean().item()
            test_metrics["clear_uncertainty_mean"] = clear_belief.uncertainty.mean().item()

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
        "model": model_name,
        "seed": seed,
        "architecture": architecture_name,
        "parameter_count": record.parameter_count,
        "train_wall_clock_seconds": train_wall_clock,
        **test_metrics,
    }
