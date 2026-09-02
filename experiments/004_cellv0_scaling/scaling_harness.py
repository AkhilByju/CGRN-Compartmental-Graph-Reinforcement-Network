"""Shared logic for Experiment 004A/C/D -- CellV0 model-size scaling
(docs/experiment_protocol.md Experiment 004). Everything stays completely
frozen relative to Experiments 002/003: `BeliefCell`'s state, `precision`'s
aggregation formula (`integration.py::_precision_fusion`), network depth (2
`BeliefLayer`s + linear readout, `belief_network.py`), AdamW, `tanh`
activations, and the plain-MSE/cross-entropy training procedure. The only
thing that varies is `hidden_cells` (and the parameter-matched MLP's
`hidden_dim`) -- i.e. model size.

At a fixed target parameter count, this:
  1. builds a `precision` `BeliefNetwork` sized (via `_match_hidden_cells`,
     a closed-form search -- no model changes) to land near the target, and
     an `MLPBaseline` parameter-matched to that model's *actual* parameter
     count (reusing `src.models.baselines.mlp.match_hidden_dim`, already
     used this way in `experiments/002_cell_v0/harness.py`);
  2. trains both with identical hyperparameters (004A -- "Model-size
     scaling");
  3. always logs per-layer internal state for the `precision` model --
     evidence/uncertainty/relevance-gate summary statistics, evidence-
     uncertainty correlation, and (on the uncertainty dataset) correlation
     of the final layer's uncertainty with the known noise level (004D --
     "Track internal state with scale");
  4. optionally (`run_ablation=True`, used at the scales Experiment 004C
     asks for) also runs Experiment 003D's evidence/uncertainty
     intervention (`src.evaluation.intervention.perturb_belief`) on the
     just-trained `precision` model, so 004C needs no separate training
     runs -- it reuses 004A's own trained models at matching scales.

Not a `src/` module -- experiment-specific composition of already-existing
generic pieces (CLAUDE.md repository philosophy), following the same
pattern as `experiments/002_cell_v0/harness.py` and
`experiments/003_cell_ablation/*_harness.py`.
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

from src.data.synthetic.classification import make_classification_splits  # noqa: E402
from src.data.synthetic.regression import make_regression_splits  # noqa: E402
from src.data.synthetic.uncertainty import make_uncertainty_splits  # noqa: E402
from src.data.synthetic.utils import standardize  # noqa: E402
from src.evaluation.calibration import pearson_correlation  # noqa: E402
from src.evaluation.classification import accuracy as accuracy_metric  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.intervention import PERTURBATION_MODES, perturb_belief  # noqa: E402
from src.evaluation.regression import mae as mae_metric  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402
from src.evaluation.regression import rmse as rmse_metric  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.architecture_v0.cell import BeliefCell  # noqa: E402
from src.models.baselines.mlp import MLPBaseline, match_hidden_dim  # noqa: E402
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

DATASETS: tuple[str, ...] = ("r2_interaction", "c2_interaction", "u2_heteroscedastic_interaction")

# Scale label -> target trainable-parameter count (docs/experiment_protocol.md
# Experiment 004A). S5 (~500K) is deliberately dropped: a single S5 precision
# run (100K examples, 6000 steps) measured ~285s locally, i.e. a 3-dataset x
# 5-seed sweep at S5 alone (~71 min) would cost more wall-clock than every
# other scale combined -- exactly the "if 500K becomes annoying, stop at
# 150K" contingency in the experiment spec.
SCALES: dict[str, int] = {
    "S0": 600,
    "S1": 2_500,
    "S2": 10_000,
    "S3": 40_000,
    "S4": 150_000,
}
# Scales at which Experiment 004C's evidence/uncertainty intervention is run
# on the just-trained precision model (per the spec: "not every scale --
# maybe ~600, ~10K, ~150K").
ABLATION_SCALES: tuple[str, ...] = ("S0", "S2", "S4")


def _is_regression(dataset: str) -> bool:
    return dataset != "c2_interaction"


def _is_uncertainty(dataset: str) -> bool:
    return dataset == "u2_heteroscedastic_interaction"


def _build_splits(dataset: str, seed: int, n_train: int, n_val: int, n_test: int):
    """Returns `(train, val, test)`, each a `(x, y)` pair, or `(x, y,
    true_std)` for the uncertainty dataset."""
    if _is_uncertainty(dataset):
        splits = make_uncertainty_splits(dataset, n_train, n_val, n_test, seed=seed)
    elif _is_regression(dataset):
        splits = make_regression_splits(dataset, n_train, n_val, n_test, seed=seed)
    else:
        splits = make_classification_splits(dataset, n_train, n_val, n_test, seed=seed)
    return splits["train"], splits["val"], splits["test"]


def _belief_total_params(hidden_cells: int, in_features: int, out_features: int) -> int:
    """Closed-form parameter count for a 2-`BeliefLayer` `BeliefNetwork`
    (`docs/architecture_v0.md` §7 item 6 gives one `BeliefLayer`'s count as
    `out_cells * (2*in_cells + 1)`; this sums both layers plus the linear
    readout). Verified against `count_parameters` on an instantiated model
    before use."""
    layer1 = hidden_cells * (2 * in_features + 1)
    layer2 = hidden_cells * (2 * hidden_cells + 1)
    readout = hidden_cells * out_features + out_features
    return layer1 + layer2 + readout


def _match_hidden_cells(
    target_params: int, in_features: int, out_features: int, search_range: range = range(1, 700)
) -> int:
    """Returns the `hidden_cells` (within `search_range`) whose `BeliefNetwork`
    parameter count is closest to `target_params` -- the `BeliefNetwork`
    analogue of `src.models.baselines.mlp.match_hidden_dim`."""
    best_hc, best_diff = search_range[0], None
    for hc in search_range:
        diff = abs(_belief_total_params(hc, in_features, out_features) - target_params)
        if best_diff is None or diff < best_diff:
            best_diff, best_hc = diff, hc
    return best_hc


def _compute_metrics(regression: bool, pred: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    if regression:
        return {"mae": mae_metric(pred, y), "rmse": rmse_metric(pred, y), "r2": r_squared(pred, y)}
    return {"accuracy": accuracy_metric(pred, y)}


def _forward_capture(model: BeliefNetwork, x: torch.Tensor):
    """Runs `x` through `model` one layer at a time (mirroring
    `BeliefNetwork.forward_with_beliefs`) and returns `(prediction,
    belief_after_layer1, belief_after_layer2)` so both layers' internal
    state can be inspected (004D) and/or perturbed between them (004C,
    reusing `experiments/003_cell_ablation`'s intervention point)."""
    belief = BeliefCell.from_observed_features(x)
    b1 = model.layer1(belief)
    b2 = model.layer2(b1)
    pred = model.readout(b2.mu)
    return pred, b1, b2


def _layer_state_stats(
    name: str, belief: BeliefCell, relevance_logit: torch.Tensor
) -> dict[str, float]:
    """004D: mean/std evidence, mean/std uncertainty, mean/distribution of
    the relevance gate `g = sigmoid(relevance_logit)` (a static per-
    connection parameter, not example-dependent -- no forward pass needed
    for it), and the evidence-uncertainty correlation, for one layer."""
    g = torch.sigmoid(relevance_logit)
    return {
        f"{name}_evidence_mean": belief.evidence.mean().item(),
        f"{name}_evidence_std": belief.evidence.std().item(),
        f"{name}_uncertainty_mean": belief.uncertainty.mean().item(),
        f"{name}_uncertainty_std": belief.uncertainty.std().item(),
        f"{name}_g_mean": g.mean().item(),
        f"{name}_g_std": g.std().item(),
        f"{name}_g_min": g.min().item(),
        f"{name}_g_max": g.max().item(),
        f"{name}_evidence_uncertainty_corr": pearson_correlation(
            belief.evidence, belief.uncertainty
        ),
    }


def _internal_state_metrics(
    model: BeliefNetwork, x: torch.Tensor, true_std: torch.Tensor | None
) -> dict[str, float]:
    _, b1, b2 = _forward_capture(model, x)
    metrics = {}
    metrics.update(_layer_state_stats("layer1", b1, model.layer1.relevance_logit))
    metrics.update(_layer_state_stats("layer2", b2, model.layer2.relevance_logit))
    if true_std is not None:
        final_uncertainty = b2.uncertainty.mean(dim=-1)
        metrics["final_uncertainty_true_std_corr"] = pearson_correlation(
            final_uncertainty, true_std.reshape(-1)
        )
    return metrics


def _ablation_metrics(
    model: BeliefNetwork, x: torch.Tensor, y: torch.Tensor, regression: bool, seed: int
) -> dict[str, float]:
    """004C: Experiment 003D's evidence/uncertainty intervention
    (`src.evaluation.intervention.perturb_belief`), reusing the just-trained
    model -- no retraining."""
    headline = "r2" if regression else "accuracy"
    metrics: dict[str, float] = {}
    per_mode_headline: dict[str, float] = {}
    for mode in PERTURBATION_MODES:
        belief = BeliefCell.from_observed_features(x)
        b1 = model.layer1(belief)
        b1 = perturb_belief(b1, mode, seed)
        b2 = model.layer2(b1)
        pred = model.readout(b2.mu)
        mode_metrics = _compute_metrics(regression, pred, y)
        for name, value in mode_metrics.items():
            metrics[f"ablation_{mode}__{name}"] = value
        per_mode_headline[mode] = mode_metrics[headline]
    for mode in PERTURBATION_MODES:
        if mode == "baseline":
            continue
        metrics[f"ablation_{mode}__delta_{headline}"] = (
            per_mode_headline[mode] - per_mode_headline["baseline"]
        )
    return metrics


def _train(
    model: torch.nn.Module,
    loss_fn,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
    steps: int,
    batch_size: int,
    lr: float,
    eval_every: int,
) -> tuple[list[dict], float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_dev, y_val_dev = x_val.to(device), y_val.to(device)

    history: list[dict] = []
    step = 0
    start_time = time.perf_counter()
    model.train()
    while step < steps:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
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
    return history, time.perf_counter() - start_time


def run_scale(
    dataset: str,
    scale_label: str,
    seed: int,
    steps: int = 6000,
    batch_size: int = 128,
    lr: float = 1e-2,
    n_train: int = 100_000,
    n_val: int = 2_000,
    n_test: int = 2_000,
    eval_every: int = 500,
    run_ablation: bool | None = None,
    aggregation: str = "precision",
    experiment_id: str = "004a_model_size_scaling",
    results_dir: str | Path = "results/raw",
) -> dict:
    """Trains a parameter-matched (`mlp`, `<aggregation>`) pair at
    `SCALES[scale_label]` on `dataset`, evaluates both, and (for the belief
    network) always logs internal state (004D) and optionally runs the
    evidence/uncertainty ablation (004C, defaults to `scale_label in
    ABLATION_SCALES` when `run_ablation` is not given explicitly). `aggregation`
    selects which `BeliefLayer` aggregation method to use (any of
    `integration.AGGREGATION_METHODS` -- default `"precision"`, matching
    004A's original run; pass e.g. `"scale_stable_precision"` to rerun the
    same 004A/004B grid with Experiment 004I's fix instead). Writes one
    `RunRecord` per architecture. Returns a flat dict with both models'
    results, prefixed `mlp__`/`<aggregation>__` (so switching `aggregation`
    also changes the result dict's key names -- callers that hardcode
    `precision__...` keys should pass `aggregation="precision"`, the
    default, to keep those keys unchanged)."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    if scale_label not in SCALES:
        raise ValueError(f"Unknown scale '{scale_label}'. Expected one of {tuple(SCALES)}.")
    if run_ablation is None:
        run_ablation = scale_label in ABLATION_SCALES

    set_seed(seed)
    device = get_device()
    regression = _is_regression(dataset)
    uncertainty = _is_uncertainty(dataset)
    target_params = SCALES[scale_label]

    train_split, val_split, test_split = _build_splits(dataset, seed, n_train, n_val, n_test)
    if uncertainty:
        x_train, y_train, std_train = train_split
        x_val, y_val, std_val = val_split
        x_test, y_test, std_test = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        std_test = None

    x_train, x_val, x_test = standardize(x_train, x_val, x_test)
    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    hidden_cells = _match_hidden_cells(target_params, in_features, out_features)
    belief_model = BeliefNetwork(
        in_features, hidden_cells, out_features, aggregation=aggregation
    )
    belief_model.to(device)
    belief_params = count_parameters(belief_model)

    hidden_dim = match_hidden_dim(belief_params, in_features, out_features, num_hidden_layers=2)
    mlp_model = MLPBaseline(in_features, hidden_dim, out_features, num_hidden_layers=2)
    mlp_model.to(device)
    mlp_params = count_parameters(mlp_model)

    results: dict[str, float | int | str] = {
        "dataset": dataset,
        "scale": scale_label,
        "target_params": target_params,
        "seed": seed,
        "aggregation": aggregation,
        f"{aggregation}__hidden_cells": hidden_cells,
        f"{aggregation}__params": belief_params,
        "mlp__hidden_dim": hidden_dim,
        "mlp__params": mlp_params,
        "param_match_pct_off": 100.0 * (mlp_params - belief_params) / belief_params,
    }
    headline = "r2" if regression else "accuracy"
    results["headline"] = headline

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)
    std_test_dev = std_test.to(device) if std_test is not None else None

    for arch_name, model in (("mlp", mlp_model), (aggregation, belief_model)):
        history, train_wall_clock = _train(
            model,
            loss_fn,
            x_train,
            y_train,
            x_val,
            y_val,
            device,
            seed,
            steps,
            batch_size,
            lr,
            eval_every,
        )
        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            if device.type == "mps":
                torch.mps.synchronize()
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

            if arch_name != "mlp":
                test_metrics.update(_internal_state_metrics(model, x_test_dev, std_test_dev))
                if run_ablation:
                    test_metrics.update(
                        _ablation_metrics(model, x_test_dev, y_test_dev, regression, seed)
                    )

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock

        architecture_name = (
            "mlp_baseline" if arch_name == "mlp" else f"belief_network[{aggregation}]"
        )
        extra: dict[str, object] = {"scale": scale_label, "target_params": target_params}
        if arch_name != "mlp":
            extra["hidden_cells"] = hidden_cells
            extra["ablation_run"] = run_ablation
        else:
            extra["hidden_dim"] = hidden_dim
        config = ExperimentConfig(
            experiment_id=experiment_id,
            architecture=architecture_name,
            dataset=dataset,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra=extra,
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
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=None,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={"loss": history[-1]["val_loss"] if history else float("nan")},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        out_dir = Path(results_dir)
        write_run_record(record, out_dir)
        with (out_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(history, f, indent=2)

    return results


# Cell-count labels reuse 004A's own S0/S2/S4 hidden_cells values (15/68/271
# -- identical for both `out_features=1` and `out_features=2` at these three
# sizes, verified via `_belief_total_params`) so the cell-count-matched
# comparison lines up with the same three points already used for 004C's
# ablation-at-scale, rather than introducing a fourth, unrelated grid.
CELL_COUNTS: dict[str, int] = {"N0": 15, "N1": 68, "N2": 271}


def run_cell_count_matched(
    dataset: str,
    count_label: str,
    seed: int,
    steps: int = 6000,
    batch_size: int = 128,
    lr: float = 1e-2,
    n_train: int = 100_000,
    n_val: int = 2_000,
    n_test: int = 2_000,
    eval_every: int = 500,
    experiment_id: str = "004_cell_count_matched",
    results_dir: str | Path = "results/raw",
) -> dict:
    """The second fairness regime: `MLPBaseline(hidden_dim=N)` vs.
    `BeliefNetwork(hidden_cells=N, aggregation='precision')` at the *same*
    `N` -- same number of computational units, not the same parameter count
    (`precision` has more parameters at matched `N`, since every connection
    carries both a content weight `w` and a relevance parameter `g`). Asks:
    is one `BeliefCell` computationally richer than one ordinary neuron,
    independent of what it costs in parameters? No parameter-matching search
    (unlike `run_scale`) -- `N` is used directly for both architectures."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    if count_label not in CELL_COUNTS:
        raise ValueError(
            f"Unknown count label '{count_label}'. Expected one of {tuple(CELL_COUNTS)}."
        )

    set_seed(seed)
    device = get_device()
    regression = _is_regression(dataset)
    uncertainty = _is_uncertainty(dataset)
    n_units = CELL_COUNTS[count_label]

    train_split, val_split, test_split = _build_splits(dataset, seed, n_train, n_val, n_test)
    if uncertainty:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, std_test = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        std_test = None

    x_train, x_val, x_test = standardize(x_train, x_val, x_test)
    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    precision_model = BeliefNetwork(in_features, n_units, out_features, aggregation="precision")
    precision_model.to(device)
    precision_params = count_parameters(precision_model)

    mlp_model = MLPBaseline(in_features, n_units, out_features, num_hidden_layers=2)
    mlp_model.to(device)
    mlp_params = count_parameters(mlp_model)

    results: dict[str, float | int | str] = {
        "dataset": dataset,
        "cell_count_label": count_label,
        "n_units": n_units,
        "seed": seed,
        "precision__params": precision_params,
        "mlp__params": mlp_params,
        "precision_params_pct_more_than_mlp": 100.0 * (precision_params - mlp_params) / mlp_params,
    }
    headline = "r2" if regression else "accuracy"
    results["headline"] = headline

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)
    std_test_dev = std_test.to(device) if std_test is not None else None

    for arch_name, model in (("mlp", mlp_model), ("precision", precision_model)):
        history, train_wall_clock = _train(
            model,
            loss_fn,
            x_train,
            y_train,
            x_val,
            y_val,
            device,
            seed,
            steps,
            batch_size,
            lr,
            eval_every,
        )
        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            if device.type == "mps":
                torch.mps.synchronize()
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)
            if arch_name == "precision":
                test_metrics.update(_internal_state_metrics(model, x_test_dev, std_test_dev))

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock

        architecture_name = "mlp_baseline" if arch_name == "mlp" else "belief_network[precision]"
        config = ExperimentConfig(
            experiment_id=experiment_id,
            architecture=architecture_name,
            dataset=dataset,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra={"cell_count_label": count_label, "n_units": n_units},
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
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=None,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={"loss": history[-1]["val_loss"] if history else float("nan")},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        out_dir = Path(results_dir)
        write_run_record(record, out_dir)
        with (out_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(history, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# 004G -- compact 3-way scale rerun: mlp / precision / normalized_precision.
# ---------------------------------------------------------------------------

THREE_WAY_SCALES: tuple[str, ...] = ("S0", "S2", "S4")


def run_scale_three_way(
    dataset: str,
    scale_label: str,
    seed: int,
    steps: int = 6000,
    batch_size: int = 128,
    lr: float = 1e-2,
    n_train: int = 100_000,
    n_val: int = 2_000,
    n_test: int = 2_000,
    eval_every: int = 500,
    experiment_id: str = "004g_normalized_precision_scale_check",
    results_dir: str | Path = "results/raw",
) -> dict:
    """Experiment 004G: `mlp`, `precision`, and `normalized_precision`
    (Experiment 004F, `integration.py::_normalized_precision_fusion`), all
    at the same target parameter count, same frozen hyperparameters as
    004A. `precision` and `normalized_precision` always use the *same*
    `hidden_cells` (parameter count is closed-form and identical across
    aggregation methods -- `integration.py`'s own docstring), so only one
    `_match_hidden_cells` search is needed. Always logs internal state
    (004D-style) for both belief variants, to compare directly whether
    `normalized_precision` actually keeps evidence/uncertainty scale-stable
    where `precision` doesn't. No ablation here (kept compact per the 004G
    spec -- 004C already covers `precision`'s ablation-at-scale).
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    if scale_label not in SCALES:
        raise ValueError(f"Unknown scale '{scale_label}'. Expected one of {tuple(SCALES)}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(dataset)
    uncertainty = _is_uncertainty(dataset)
    target_params = SCALES[scale_label]

    train_split, val_split, test_split = _build_splits(dataset, seed, n_train, n_val, n_test)
    if uncertainty:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, std_test = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        std_test = None

    x_train, x_val, x_test = standardize(x_train, x_val, x_test)
    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    hidden_cells = _match_hidden_cells(target_params, in_features, out_features)
    precision_model = BeliefNetwork(
        in_features, hidden_cells, out_features, aggregation="precision"
    )
    normalized_model = BeliefNetwork(
        in_features, hidden_cells, out_features, aggregation="normalized_precision"
    )
    precision_model.to(device)
    normalized_model.to(device)
    precision_params = count_parameters(precision_model)

    hidden_dim = match_hidden_dim(precision_params, in_features, out_features, num_hidden_layers=2)
    mlp_model = MLPBaseline(in_features, hidden_dim, out_features, num_hidden_layers=2)
    mlp_model.to(device)
    mlp_params = count_parameters(mlp_model)

    headline = "r2" if regression else "accuracy"
    results: dict[str, float | int | str] = {
        "dataset": dataset,
        "scale": scale_label,
        "target_params": target_params,
        "seed": seed,
        "headline": headline,
        "hidden_cells": hidden_cells,
        "hidden_dim": hidden_dim,
        "precision__params": precision_params,
        "normalized_precision__params": count_parameters(normalized_model),
        "mlp__params": mlp_params,
    }

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)
    std_test_dev = std_test.to(device) if std_test is not None else None

    models = (
        ("mlp", mlp_model, "mlp_baseline"),
        ("precision", precision_model, "belief_network[precision]"),
        ("normalized_precision", normalized_model, "belief_network[normalized_precision]"),
    )
    for arch_name, model, architecture_name in models:
        history, train_wall_clock = _train(
            model,
            loss_fn,
            x_train,
            y_train,
            x_val,
            y_val,
            device,
            seed,
            steps,
            batch_size,
            lr,
            eval_every,
        )
        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            if device.type == "mps":
                torch.mps.synchronize()
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)
            if arch_name != "mlp":
                test_metrics.update(_internal_state_metrics(model, x_test_dev, std_test_dev))

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock

        config = ExperimentConfig(
            experiment_id=experiment_id,
            architecture=architecture_name,
            dataset=dataset,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra={
                "scale": scale_label,
                "target_params": target_params,
                "hidden_cells_or_dim": hidden_cells if arch_name != "mlp" else hidden_dim,
            },
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
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=None,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={"loss": history[-1]["val_loss"] if history else float("nan")},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        out_dir = Path(results_dir)
        write_run_record(record, out_dir)
        with (out_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(history, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# 004H -- fair MLP optimization check: train/val/test loss + a small LR
# sweep, same tuning budget for mlp and precision.
# ---------------------------------------------------------------------------


def run_optimization_check(
    dataset: str,
    scale_label: str,
    seed: int,
    lr: float,
    steps: int = 6000,
    batch_size: int = 128,
    n_train: int = 100_000,
    n_val: int = 2_000,
    n_test: int = 2_000,
    eval_every: int = 500,
    experiment_id: str = "004h_optimization_check",
    results_dir: str | Path = "results/raw",
) -> dict:
    """Experiment 004H: at a fixed `lr` (a caller sweeps this across e.g.
    0.3x/1x/3x the usual `1e-2`), trains `mlp` and `precision` (only --
    `normalized_precision` is out of scope for this diagnostic, which is
    about *why* `mlp` declines relative to `precision` in 004A, not about
    the fix) and records train/val/test **loss** (not just the headline
    prediction metric) for both. Distinguishes an optimization story
    (training loss itself is worse) from a generalization story (training
    loss is fine, test loss/metric is worse) for `mlp`'s scale-decline seen
    in 004A.

    Train/val loss come from the same `_train` used everywhere else in this
    module (`history`'s last recorded step -- a single minibatch, matching
    existing convention); test loss and a full-training-set loss are
    computed separately here, over the complete set, for a less noisy
    comparison point.
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {DATASETS}.")
    if scale_label not in SCALES:
        raise ValueError(f"Unknown scale '{scale_label}'. Expected one of {tuple(SCALES)}.")

    set_seed(seed)
    device = get_device()
    regression = _is_regression(dataset)
    uncertainty = _is_uncertainty(dataset)
    target_params = SCALES[scale_label]

    train_split, val_split, test_split = _build_splits(dataset, seed, n_train, n_val, n_test)
    if uncertainty:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, _ = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split

    x_train, x_val, x_test = standardize(x_train, x_val, x_test)
    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()

    hidden_cells = _match_hidden_cells(target_params, in_features, out_features)
    precision_model = BeliefNetwork(
        in_features, hidden_cells, out_features, aggregation="precision"
    )
    precision_model.to(device)
    precision_params = count_parameters(precision_model)

    hidden_dim = match_hidden_dim(precision_params, in_features, out_features, num_hidden_layers=2)
    mlp_model = MLPBaseline(in_features, hidden_dim, out_features, num_hidden_layers=2)
    mlp_model.to(device)

    headline = "r2" if regression else "accuracy"
    results: dict[str, float | int | str] = {
        "dataset": dataset,
        "scale": scale_label,
        "target_params": target_params,
        "seed": seed,
        "lr": lr,
        "headline": headline,
        "hidden_cells": hidden_cells,
        "hidden_dim": hidden_dim,
    }

    x_train_dev, y_train_dev = x_train.to(device), y_train.to(device)
    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)

    for arch_name, model, architecture_name in (
        ("mlp", mlp_model, "mlp_baseline"),
        ("precision", precision_model, "belief_network[precision]"),
    ):
        history, train_wall_clock = _train(
            model,
            loss_fn,
            x_train,
            y_train,
            x_val,
            y_val,
            device,
            seed,
            steps,
            batch_size,
            lr,
            eval_every,
        )
        model.eval()
        with torch.no_grad():
            full_train_loss = loss_fn(model(x_train_dev), y_train_dev).item()
            val_loss = history[-1]["val_loss"] if history else float("nan")
            test_pred = model(x_test_dev)
            test_loss = loss_fn(test_pred, y_test_dev).item()
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        results[f"{arch_name}__train_loss"] = full_train_loss
        results[f"{arch_name}__val_loss"] = val_loss
        results[f"{arch_name}__test_loss"] = test_loss
        results[f"{arch_name}__{headline}"] = test_metrics[headline]
        results[f"{arch_name}__train_wall_clock_seconds"] = train_wall_clock

        config = ExperimentConfig(
            experiment_id=experiment_id,
            architecture=architecture_name,
            dataset=dataset,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=steps,
            extra={"scale": scale_label, "target_params": target_params},
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
            steps_completed=steps,
            examples_or_tokens_seen=steps * batch_size,
            refinement_iterations=None,
            approximate_flops=None,
            train_wall_clock_seconds=train_wall_clock,
            inference_wall_clock_seconds=None,
            validation_metrics={"loss": val_loss},
            test_metrics={
                **test_metrics,
                "test_loss": test_loss,
                "full_train_loss": full_train_loss,
            },
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        out_dir = Path(results_dir)
        write_run_record(record, out_dir)
        with (out_dir / f"{run_id}_history.json").open("w") as f:
            json.dump(history, f, indent=2)

    return results
