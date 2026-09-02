"""Shared logic for Experiment 003B (ground-truth uncertainty) and 003C
(uncertainty supervision) -- docs/experiment_protocol.md. Trains one model
(MLP, MLP+uncertainty-head, or a BeliefNetwork under `reliability`/
`precision`) on one heteroscedastic-noise synthetic dataset
(`src/data/synthetic/uncertainty.py`, where the true per-example noise
level is known), optionally adds an explicit calibration loss term
(`lambda_calibration`, 003C only -- 0.0 reproduces 003A/003B's plain
prediction loss), evaluates predictive performance AND how well the
model's predicted uncertainty tracks the known noise level
(`src/evaluation/calibration.py`), and records a RunRecord.

Not a `src/` module: it is specific to this experiment pair's dataset/
model/loss choices, not generic infrastructure (CLAUDE.md's repository
philosophy) -- mirrors `experiments/002_cell_v0/harness.py`.

Only `reliability`/`precision` are offered here, not `support_conflict`:
Experiment 002 found it the clear outlier (evidence collapse, huge/noisy
"ambiguous vs. clear" gaps -- docs/research_log.md), and 003A's job is to
re-confirm the two survivors, not re-litigate `support_conflict`.

Isolation of variables (CLAUDE.md Sec 2, docs/research_thesis.md): the
calibration term added here is a plain supervised MSE regression of each
model's own predicted-uncertainty output onto the dataset's known
`true_std` -- not a new optimizer, and not a change to any of
`integration.py`'s three aggregation formulas, which stay byte-for-byte
what Experiment 002 tested. It is exactly the "add an explicit calibration
signal ... as a deliberate, isolated change" follow-up docs/research_log.md
flagged after Experiment 002's calibration direction-flip finding.
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
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.synthetic.uncertainty import (  # noqa: E402
    U_REGRESSION_DATASETS,
    make_uncertainty_splits,
)
from src.data.synthetic.utils import standardize  # noqa: E402
from src.evaluation.calibration import (  # noqa: E402
    calibration_mse,
    pearson_correlation,
    spearman_correlation,
)
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.regression import mae as mae_metric  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402
from src.evaluation.regression import rmse as rmse_metric  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.baselines.mlp import (  # noqa: E402
    MLPBaseline,
    MLPWithUncertaintyHead,
    match_hidden_dim,
)
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.training.optimization import build_optimizer  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

DATASETS: tuple[str, ...] = U_REGRESSION_DATASETS
MODEL_CHOICES: tuple[str, ...] = ("mlp", "mlp_uncertainty", "reliability", "precision")
# Models with a predicted-uncertainty output a calibration loss/metric can apply to.
UNCERTAINTY_CAPABLE_MODELS: tuple[str, ...] = ("mlp_uncertainty", "reliability", "precision")


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
    lambda_calibration: float = 0.0,
    experiment_id: str = "003b_ground_truth_uncertainty",
    results_dir: str | Path = "results/raw",
) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    if model_name not in MODEL_CHOICES:
        raise ValueError(f"Unknown model '{model_name}'. Expected one of {MODEL_CHOICES}.")
    if lambda_calibration != 0.0 and model_name not in UNCERTAINTY_CAPABLE_MODELS:
        raise ValueError(
            f"lambda_calibration > 0 requires a model with a predicted-uncertainty "
            f"signal ({UNCERTAINTY_CAPABLE_MODELS}), got '{model_name}'."
        )

    set_seed(seed)
    device = get_device()

    splits = make_uncertainty_splits(dataset, n_train, n_val, n_test, seed=seed)
    x_train, y_train, std_train = splits["train"]
    x_val, y_val, std_val = splits["val"]
    x_test, y_test, std_test = splits["test"]
    x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]

    if model_name == "mlp":
        reference = BeliefNetwork(in_features, hidden_cells, 1, aggregation="reliability")
        target_params = count_parameters(reference)
        hidden_dim = match_hidden_dim(target_params, in_features, 1, num_hidden_layers=2)
        model = MLPBaseline(in_features, hidden_dim, 1, num_hidden_layers=2)
        architecture_name = "mlp_baseline"
    elif model_name == "mlp_uncertainty":
        reference = BeliefNetwork(in_features, hidden_cells, 1, aggregation="reliability")
        target_params = count_parameters(reference)
        # out_features=2 matches the two heads' combined parameter count
        # (mean_head + log_var_head), each Linear(hidden_dim, 1).
        hidden_dim = match_hidden_dim(target_params, in_features, 2, num_hidden_layers=2)
        model = MLPWithUncertaintyHead(in_features, hidden_dim, target_dim=1, num_hidden_layers=2)
        architecture_name = "mlp_uncertainty_head"
    else:
        model = BeliefNetwork(in_features, hidden_cells, 1, aggregation=model_name)
        architecture_name = f"belief_network[{model_name}]"
    model.to(device)

    config = ExperimentConfig(
        experiment_id=experiment_id,
        architecture=architecture_name,
        dataset=dataset,
        seed=seed,
        optimizer="adamw",
        learning_rate=lr,
        batch_size=batch_size,
        max_steps=steps,
        extra={"hidden_cells": hidden_cells, "lambda_calibration": lambda_calibration},
    )
    optimizer = build_optimizer(model, config)

    def forward_and_loss(
        xb: torch.Tensor, yb: torch.Tensor, stdb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns `(prediction, loss, predicted_std)`; `predicted_std` is
        `None` for `mlp`, which has no uncertainty output."""
        if model_name == "mlp":
            pred = model(xb)
            return pred, F.mse_loss(pred, yb), None

        if model_name == "mlp_uncertainty":
            mean, log_var = model(xb)
            var = log_var.exp()
            loss = F.gaussian_nll_loss(mean, yb, var, eps=1e-6)
            predicted_std = var.clamp_min(1e-12).sqrt()
            if lambda_calibration > 0:
                loss = loss + lambda_calibration * F.mse_loss(predicted_std, stdb)
            return mean, loss, predicted_std

        # reliability / precision
        pred, belief = model.forward_with_beliefs(xb)
        predicted_std = belief.uncertainty.mean(dim=-1, keepdim=True)
        loss = F.mse_loss(pred, yb)
        if lambda_calibration > 0:
            loss = loss + lambda_calibration * F.mse_loss(predicted_std, stdb)
        return pred, loss, predicted_std

    train_loader = DataLoader(
        TensorDataset(x_train, y_train, std_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_dev, y_val_dev, std_val_dev = x_val.to(device), y_val.to(device), std_val.to(device)
    x_test_dev, y_test_dev, std_test_dev = x_test.to(device), y_test.to(device), std_test.to(device)

    history: list[dict] = []
    step = 0
    start_time = time.perf_counter()
    model.train()
    while step < steps:
        for xb, yb, stdb in train_loader:
            xb, yb, stdb = xb.to(device), yb.to(device), stdb.to(device)
            optimizer.zero_grad()
            _, loss, _ = forward_and_loss(xb, yb, stdb)
            loss.backward()
            optimizer.step()
            step += 1

            if step % eval_every == 0 or step >= steps:
                model.eval()
                with torch.no_grad():
                    _, val_loss, _ = forward_and_loss(x_val_dev, y_val_dev, std_val_dev)
                history.append(
                    {"step": step, "train_loss": loss.item(), "val_loss": val_loss.item()}
                )
                model.train()
            if step >= steps:
                break
    train_wall_clock = time.perf_counter() - start_time

    model.eval()
    with torch.no_grad():
        test_pred, _, predicted_std_test = forward_and_loss(x_test_dev, y_test_dev, std_test_dev)
        test_metrics = {
            "mae": mae_metric(test_pred, y_test_dev),
            "rmse": rmse_metric(test_pred, y_test_dev),
            "r2": r_squared(test_pred, y_test_dev),
        }
        if predicted_std_test is not None:
            test_metrics["uncertainty_pearson_corr"] = pearson_correlation(
                predicted_std_test, std_test_dev
            )
            test_metrics["uncertainty_spearman_corr"] = spearman_correlation(
                predicted_std_test, std_test_dev
            )
            test_metrics["uncertainty_calibration_mse"] = calibration_mse(
                predicted_std_test, std_test_dev
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
        "model": model_name,
        "seed": seed,
        "lambda_calibration": lambda_calibration,
        "architecture": architecture_name,
        "parameter_count": record.parameter_count,
        "train_wall_clock_seconds": train_wall_clock,
        **test_metrics,
    }
