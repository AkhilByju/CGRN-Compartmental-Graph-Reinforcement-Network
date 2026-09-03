#!/usr/bin/env python3
"""Isolated MLP-only run of the convergence-based complexity sweep
(`run_complexity_scaling_convergence.py`) -- same protocol (early
stopping via `harness.py::_train_until_convergence`), same parameter-
matching target (`field_local_global_t2`'s size at each level, via
`_build_global_comparison_models(include_mlp=True)`), but trains *only*
the MLP baseline. `cellv0.1`/`field_t2`/`field_local_global_t2` are
constructed for sizing only (cheap; construction, not training, is what
costs time) and never trained here -- their numbers from the prior run
(`results/processed/v1_001_complexity_scaling_convergence.json`) are
reused as-is for comparison, per the user's "just the MLP, isolated"
ask.

Answers the curiosity question directly: is a plain MLP, at the same
parameter count, faster to converge *and* more accurate than CellV0.1?

Usage:
    python experiments/v1_001_dynamic_groups/run_mlp_convergence_only.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from harness import _build_global_comparison_models, _build_splits, _compute_metrics, _is_regression, _train_until_convergence
from src.utilities.device import get_device
from src.utilities.seeding import set_seed

TASK = "dynamic_groups_global"
COMPLEXITY_LEVELS: dict[str, dict[str, int]] = {
    "easy": {"n_objects": 24, "k_min": 2, "k_max": 5},
    "medium": {"n_objects": 48, "k_min": 5, "k_max": 8},
    "hard": {"n_objects": 96, "k_min": 8, "k_max": 14},
    "very_hard": {"n_objects": 192, "k_min": 12, "k_max": 20},
}


def run_one(
    level: str,
    seed: int,
    val_every: int,
    patience_steps: int,
    max_steps: int,
    batch_size: int,
    lr: float,
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict:
    cfg = COMPLEXITY_LEVELS[level]
    n_objects = cfg["n_objects"]
    set_seed(seed)
    device = get_device()
    regression = _is_regression(TASK)

    train_split, val_split, test_split = _build_splits(
        TASK, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=cfg["k_min"], k_max=cfg["k_max"]
    )
    x_train, y_train, _ = train_split
    x_val, y_val, _ = val_split
    x_test, y_test, _ = test_split

    in_features = x_train.shape[1]
    loss_fn = torch.nn.MSELoss()

    # cellv0.1/field_t2/field_local_global_t2 are built here too (for the
    # sizing target) but never trained -- only models["mlp"] is used below.
    models, sizing = _build_global_comparison_models(
        TASK, in_features, 1, seed, n_cells=n_objects, n_objects=n_objects,
        include_field_t1=False, include_mlp=True,
    )
    model = models["mlp"].to(device)

    convergence = _train_until_convergence(
        model, loss_fn, x_train, y_train, x_val, y_val, device, seed, regression,
        batch_size, lr, val_every=val_every, patience_steps=patience_steps, max_steps=max_steps,
    )
    model.eval()
    with torch.no_grad():
        test_metrics = _compute_metrics(regression, model(x_test.to(device)), y_test.to(device))

    return {
        "level": level,
        "seed": seed,
        "n_objects": n_objects,
        "mlp_params": sizing["mlp__params"],
        "mlp_hidden_dim": sizing["mlp__hidden_dim"],
        **test_metrics,
        **convergence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=list(COMPLEXITY_LEVELS), choices=list(COMPLEXITY_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--patience-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=3_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--summary-out", default="results/processed/v1_001_mlp_convergence_only.json")
    args = parser.parse_args()

    results = []
    for level in args.levels:
        for seed in args.seeds:
            r = run_one(
                level, seed, args.val_every, args.patience_steps, args.max_steps,
                args.batch_size, args.lr, args.n_train, args.n_val, args.n_test,
            )
            results.append(r)
            print(
                f"level={level:9s} seed={seed} n_objects={r['n_objects']:3d} params={r['mlp_params']} "
                f"r2={r['r2']:.4f} steps_to_convergence={r['steps_to_convergence']} "
                f"wall_clock_to_convergence={r['wall_clock_to_convergence_seconds']:.2f}s "
                f"total_steps={r['total_steps_run']}"
            )

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} run summaries to {summary_path}")

    print("\n=== Summary (mean +/- std across seeds) ===")
    for level in args.levels:
        level_results = [r for r in results if r["level"] == level]
        r2_vals = [r["r2"] for r in level_results]
        steps_vals = [r["steps_to_convergence"] for r in level_results]
        conv_s_vals = [r["wall_clock_to_convergence_seconds"] for r in level_results]
        r2_mean = statistics.mean(r2_vals)
        r2_std = statistics.pstdev(r2_vals) if len(r2_vals) > 1 else 0.0
        print(
            f"  {level:9s} (n_objects={COMPLEXITY_LEVELS[level]['n_objects']:3d}): "
            f"r2={r2_mean:.4f}+/-{r2_std:.4f}  "
            f"steps_to_convergence={statistics.mean(steps_vals):.0f}  "
            f"wall_clock_to_convergence={statistics.mean(conv_s_vals):.2f}s"
        )


if __name__ == "__main__":
    main()
