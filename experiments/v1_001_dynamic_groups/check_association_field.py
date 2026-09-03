#!/usr/bin/env python3
"""Narrow sanity check for the Learned Association Field
(`association_model.py`) -- NOT a full sweep, per the user's explicit
"I would not do another huge sweep." Single seed, two levels only (easy
and the previously-problematic very_hard), convergence-based training
(`harness.py::_train_until_convergence`, same protocol as
`run_complexity_scaling_convergence.py`). Compares CellV0.1, the
existing ORFF field at its convergence-fixing config
(field_local_global_t2, R=512), and the new
association_local_global_t2 -- does the cheaper, exact, learned
association kernel reach comparable accuracy at meaningfully lower
wall-clock cost?

Usage:
    python experiments/v1_001_dynamic_groups/check_association_field.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from harness import (
    ASSOCIATION_DIM,
    _build_splits,
    _compute_metrics,
    _is_regression,
    _match_belief_hidden_cells,
    _train_until_convergence,
)
from src.evaluation.efficiency import count_parameters
from src.models.architecture_v0.belief_network import BeliefNetwork
from src.models.architecture_v1.association_model import LearnedAssociationField
from src.models.architecture_v1.field_model import SelfOrganizingRefinementField
from src.models.architecture_v1.object_encoder import ObjectSeededEncoder
from src.utilities.device import get_device
from src.utilities.seeding import set_seed

TASK = "dynamic_groups_global"
SEED = 0
LEVELS: dict[str, dict[str, int]] = {
    "easy": {"n_objects": 24, "k_min": 2, "k_max": 5},
    "very_hard": {"n_objects": 192, "k_min": 12, "k_max": 20},
}


def run_level(level: str, val_every: int = 100, patience_steps: int = 500, max_steps: int = 5_000) -> dict:
    cfg = LEVELS[level]
    n_objects = cfg["n_objects"]
    set_seed(SEED)
    device = get_device()
    regression = _is_regression(TASK)

    train_split, val_split, test_split = _build_splits(
        TASK, SEED, 3_000, 500, 500, n_objects=n_objects, k_min=cfg["k_min"], k_max=cfg["k_max"]
    )
    x_train, y_train, _ = train_split
    x_val, y_val, _ = val_split
    x_test, y_test, _ = test_split
    in_features = x_train.shape[1]
    loss_fn = torch.nn.MSELoss()

    def make_encoder():
        return ObjectSeededEncoder(n_objects=n_objects, n_cells=n_objects, association_dim=ASSOCIATION_DIM)

    association = LearnedAssociationField(
        in_features=in_features, out_features=1, n_cells=n_objects, association_dim=ASSOCIATION_DIM,
        assoc_dim=32, num_steps=2, encoder=make_encoder(), use_global=True, global_dim=16,
    )
    field = SelfOrganizingRefinementField(
        in_features=in_features, out_features=1, n_cells=n_objects, association_dim=ASSOCIATION_DIM,
        num_features=512, num_steps=2, encoder=make_encoder(), use_global=True, global_dim=16,
    )
    target_params = count_parameters(association)
    hidden_cells = _match_belief_hidden_cells(target_params, in_features, 1)
    cellv0_1 = BeliefNetwork(in_features, hidden_cells, 1, aggregation="scale_stable_precision")

    models = {"cellv0.1": cellv0_1, "field_local_global_t2": field, "association_local_global_t2": association}
    result = {"level": level, "n_objects": n_objects}
    for name, model in models.items():
        model.to(device)
        convergence = _train_until_convergence(
            model, loss_fn, x_train, y_train, x_val, y_val, device, SEED, regression,
            batch_size=64, lr=1e-2, val_every=val_every, patience_steps=patience_steps, max_steps=max_steps,
        )
        model.eval()
        with torch.no_grad():
            test_metrics = _compute_metrics(regression, model(x_test.to(device)), y_test.to(device))
        result[f"{name}__params"] = count_parameters(model)
        result[f"{name}__r2"] = test_metrics["r2"]
        result[f"{name}__steps_to_convergence"] = convergence["steps_to_convergence"]
        result[f"{name}__wall_clock_to_convergence_seconds"] = convergence["wall_clock_to_convergence_seconds"]
        print(
            f"  [{level}] {name:28s} params={result[f'{name}__params']:5d} "
            f"r2={result[f'{name}__r2']:.4f} steps={result[f'{name}__steps_to_convergence']:5d} "
            f"wall_clock={result[f'{name}__wall_clock_to_convergence_seconds']:.2f}s"
        )
    return result


def main() -> None:
    results = [run_level(level) for level in LEVELS]
    out_path = Path("results/processed/v1_001_association_field_check.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
