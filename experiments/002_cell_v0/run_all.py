#!/usr/bin/env python3
"""Experiment 002/003: full grid -- every dataset x every model x several
seeds, run in-process. Writes one RunRecord (+ history file) per run to
results/raw/, and prints a summary table (mean +/- std across seeds).

Usage: python experiments/002_cell_v0/run_all.py [--seeds 0 1 2] [--steps 2000]
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from harness import DATASETS, MODEL_CHOICES, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--hidden-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    all_results = []
    for dataset in DATASETS:
        for model_name in MODEL_CHOICES:
            for seed in args.seeds:
                result = run_experiment(
                    dataset=dataset,
                    model_name=model_name,
                    seed=seed,
                    hidden_cells=args.hidden_cells,
                    steps=args.steps,
                    lr=args.lr,
                    results_dir=args.results_dir,
                )
                all_results.append(result)
                metric_key = "r2" if "r2" in result else "accuracy"
                print(
                    f"{dataset:16s} {model_name:16s} seed={seed} "
                    f"{metric_key}={result.get(metric_key):.4f} "
                    f"params={result['parameter_count']} "
                    f"time={result['train_wall_clock_seconds']:.1f}s"
                )

    print("\n=== Summary (mean +/- std across seeds) ===")
    grouped = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["model"])].append(r)
    for (dataset, model_name), runs in grouped.items():
        metric_key = "r2" if "r2" in runs[0] else "accuracy"
        values = [r[metric_key] for r in runs]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        params = runs[0]["parameter_count"]
        print(f"{dataset:16s} {model_name:16s} {metric_key}={mean:.4f}+/-{std:.4f} params={params}")


if __name__ == "__main__":
    main()
