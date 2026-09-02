#!/usr/bin/env python3
"""Experiment 004B -- data scaling (docs/experiment_protocol.md Experiment
004). Same frozen architecture/hyperparameters as 004A, same
parameter-matched (`mlp`, `<aggregation>`) pair, but at ONE fixed model size
(default `--scale S3`, ~40K parameters -- picked from the 40K-100K range the
spec asks for) while training-set size varies.

Tests sample efficiency: does the belief network reach a given performance
level with less data than a parameter-matched `mlp`, even if they converge
to the same ceiling at the largest data sizes?

`--aggregation` selects which `BeliefLayer` method to use (default
`"precision"`, matching the original 004B run -- pass
`--aggregation scale_stable_precision` to rerun this exact grid with
Experiment 004I's "CellV0.1" fix instead).

Usage:
    python experiments/004_cellv0_scaling/run_004b.py
    python experiments/004_cellv0_scaling/run_004b.py --scale S3 --seeds 0 1 2 3 4
    python experiments/004_cellv0_scaling/run_004b.py --aggregation scale_stable_precision
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scaling_harness import DATASETS, SCALES, run_scale

from src.models.architecture_v0.integration import AGGREGATION_METHODS

DATA_SCALES: dict[str, int] = {
    "D0": 1_000,
    "D1": 3_000,
    "D2": 10_000,
    "D3": 30_000,
    "D4": 100_000,
    "D5": 300_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--scale", default="S3", choices=list(SCALES))
    parser.add_argument(
        "--data-scales", nargs="+", default=list(DATA_SCALES), choices=list(DATA_SCALES)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--aggregation", default="precision", choices=list(AGGREGATION_METHODS)
    )
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()
    summary_out = args.summary_out or f"results/processed/004b_{args.aggregation}_summary.json"
    agg = args.aggregation

    all_results = []
    for data_scale in args.data_scales:
        n_train = DATA_SCALES[data_scale]
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_scale(
                    dataset=dataset,
                    scale_label=args.scale,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    n_train=n_train,
                    run_ablation=False,
                    aggregation=agg,
                    experiment_id="004b_data_scaling",
                    results_dir=args.results_dir,
                )
                r["data_scale"] = data_scale
                r["n_train"] = n_train
                all_results.append(r)
                headline = r["headline"]
                print(
                    f"{dataset:32s} {data_scale} (n_train={n_train:>6d}) seed={seed}  "
                    f"mlp_{headline}={r[f'mlp__{headline}']:.4f}  "
                    f"{agg}_{headline}={r[f'{agg}__{headline}']:.4f}"
                )

    summary_path = Path(summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["data_scale"])].append(r)

    print("\n=== Summary (mean +/- std across seeds) ===")
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        for data_scale in args.data_scales:
            runs = grouped.get((dataset, data_scale), [])
            if not runs:
                continue
            headline = runs[0]["headline"]
            mlp_vals = [r[f"mlp__{headline}"] for r in runs]
            belief_vals = [r[f"{agg}__{headline}"] for r in runs]
            mlp_mean = statistics.mean(mlp_vals)
            belief_mean = statistics.mean(belief_vals)
            print(
                f"    {data_scale} (n_train={DATA_SCALES[data_scale]:>6d})  "
                f"mlp={mlp_mean:.4f}  {agg}={belief_mean:.4f}  "
                f"delta={belief_mean - mlp_mean:+.4f}"
            )


if __name__ == "__main__":
    main()
