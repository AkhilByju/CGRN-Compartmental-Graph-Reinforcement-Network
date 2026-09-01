#!/usr/bin/env python3
"""Experiment 002/003: single run.

Usage:
    python experiments/002_cell_v0/run.py --dataset r0_linear --model mlp --seed 0
    python experiments/002_cell_v0/run.py --dataset c1_xor --model reliability --seed 0

`--model` is one of: mlp, reliability, support_conflict, precision. See
docs/experiment_protocol.md Experiment 002/003 and docs/architecture_v0.md
Sec 1 "Test the cell before the graph". For the full dataset x model x
seed grid, use run_all.py instead.
"""

from __future__ import annotations

import argparse

from harness import DATASETS, MODEL_CHOICES, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    result = run_experiment(
        dataset=args.dataset,
        model_name=args.model,
        seed=args.seed,
        hidden_cells=args.hidden_cells,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        results_dir=args.results_dir,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
