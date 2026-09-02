#!/usr/bin/env python3
"""Experiment 003B -- ground-truth uncertainty (docs/research_log.md
"Experiment 002 initial results", follow-up 2). Trains
mlp / mlp_uncertainty / reliability / precision on the
u1_heteroscedastic_1d / u2_heteroscedastic_interaction datasets
(src/data/synthetic/uncertainty.py, where the true per-example noise level
is known), with NO calibration-loss supervision (lambda_calibration=0 --
sweeping that is Experiment 003C), and reports predictive performance plus
how well each model's predicted uncertainty tracks the known noise level
(src/evaluation/calibration.py).

The key comparison: if `mlp_uncertainty` (a conventional MLP with a
Gaussian-NLL-trained variance head -- Nix & Weigend, 1994) calibrates
equally well or better than `reliability`/`precision`'s unsupervised
`BeliefCell.uncertainty` channel, then merely carrying uncertainty through
every neuron isn't buying anything a standard, much simpler baseline
doesn't already provide.

Usage:
    python experiments/003_cell_ablation/run_003b.py
    python experiments/003_cell_ablation/run_003b.py --seeds 0 1 2 3 4 5 6 7 8 9 --steps 2000
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from uncertainty_harness import DATASETS, MODEL_CHOICES, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
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
                    lambda_calibration=0.0,
                    experiment_id="003b_ground_truth_uncertainty",
                    results_dir=args.results_dir,
                )
                all_results.append(result)
                corr = result.get("uncertainty_pearson_corr", float("nan"))
                print(
                    f"{dataset:28s} {result['model']:14s} seed={seed} "
                    f"r2={result['r2']:.4f} corr={corr:+.3f}"
                )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["model"])].append(r)

    print("\n=== Summary (mean +/- std across seeds) ===")
    for (dataset, model_name), runs in grouped.items():
        r2_vals = [r["r2"] for r in runs]
        r2_mean = statistics.mean(r2_vals)
        r2_std = statistics.pstdev(r2_vals) if len(r2_vals) > 1 else 0.0

        corr_vals = [r["uncertainty_pearson_corr"] for r in runs if "uncertainty_pearson_corr" in r]
        if corr_vals:
            corr_mean = statistics.mean(corr_vals)
            corr_std = statistics.pstdev(corr_vals) if len(corr_vals) > 1 else 0.0
            corr_str = f"corr={corr_mean:+.3f}+/-{corr_std:.3f}"
        else:
            corr_str = "corr=n/a"

        print(f"{dataset:28s} {model_name:14s} r2={r2_mean:.4f}+/-{r2_std:.4f} {corr_str}")


if __name__ == "__main__":
    main()
