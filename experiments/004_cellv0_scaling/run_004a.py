#!/usr/bin/env python3
"""Experiment 004A/C/D -- CellV0 model-size scaling (docs/experiment_protocol.md
Experiment 004). Everything is frozen relative to Experiments 002/003
(`BeliefCell`, the chosen aggregation method's math, network depth, AdamW,
`tanh` activations, plain MSE/cross-entropy training -- no calibration
loss, no recurrence, no compartment changes); only model size
(`hidden_cells`/`hidden_dim`) varies.

For each (dataset, scale, seed): trains a parameter-matched
(`mlp_baseline`, `belief_network[<aggregation>]`) pair on 100K fixed
training examples, records prediction performance (R²/accuracy), efficiency
(exact parameter count, train/inference wall clock), and -- for the belief
network -- always logs per-layer internal state (004D) and, at
`scaling_harness.ABLATION_SCALES` (~600/~10K/~150K), the evidence/
uncertainty ablation from Experiment 003D (004C), reusing the just-trained
model instead of retraining.

`--aggregation` selects which `BeliefLayer` method to use (default
`"precision"`, matching the original 004A run -- pass
`--aggregation scale_stable_precision` to rerun this exact grid with
Experiment 004I's "CellV0.1" fix instead, or `normalized_precision` for
004F's).

Usage:
    python experiments/004_cellv0_scaling/run_004a.py
    python experiments/004_cellv0_scaling/run_004a.py --seeds 0 1 2 3 4 --scales S0 S1 S2 S3 S4
    python experiments/004_cellv0_scaling/run_004a.py --aggregation scale_stable_precision
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scaling_harness import DATASETS, SCALES, run_scale

from src.models.architecture_v0.integration import AGGREGATION_METHODS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--scales", nargs="+", default=list(SCALES), choices=list(SCALES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument(
        "--aggregation", default="precision", choices=list(AGGREGATION_METHODS)
    )
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()
    summary_out = args.summary_out or f"results/processed/004a_{args.aggregation}_summary.json"
    agg = args.aggregation

    all_results = []
    for scale in args.scales:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_scale(
                    dataset=dataset,
                    scale_label=scale,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    n_train=args.n_train,
                    aggregation=agg,
                    results_dir=args.results_dir,
                )
                all_results.append(r)
                headline = r["headline"]
                print(
                    f"{dataset:32s} {scale} (target={r['target_params']:>7d}) seed={seed} "
                    f"{agg}_params={r[f'{agg}__params']:>7d} "
                    f"mlp_params={r['mlp__params']:>7d} "
                    f"({r['param_match_pct_off']:+.2f}%)  "
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
        grouped[(r["dataset"], r["scale"])].append(r)

    print("\n=== Summary (mean +/- std across seeds) ===")
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        for scale in args.scales:
            runs = grouped.get((dataset, scale), [])
            if not runs:
                continue
            headline = runs[0]["headline"]
            mlp_vals = [r[f"mlp__{headline}"] for r in runs]
            belief_vals = [r[f"{agg}__{headline}"] for r in runs]
            mlp_mean, mlp_std = statistics.mean(mlp_vals), (
                statistics.pstdev(mlp_vals) if len(mlp_vals) > 1 else 0.0
            )
            belief_mean, belief_std = statistics.mean(belief_vals), (
                statistics.pstdev(belief_vals) if len(belief_vals) > 1 else 0.0
            )
            params = statistics.mean(r[f"{agg}__params"] for r in runs)
            print(
                f"    {scale} (~{params:.0f} params)  "
                f"mlp={mlp_mean:.4f}+/-{mlp_std:.4f}  "
                f"{agg}={belief_mean:.4f}+/-{belief_std:.4f}  "
                f"delta={belief_mean - mlp_mean:+.4f}"
            )


if __name__ == "__main__":
    main()
