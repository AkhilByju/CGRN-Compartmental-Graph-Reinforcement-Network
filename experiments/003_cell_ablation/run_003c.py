#!/usr/bin/env python3
"""Experiment 003C -- uncertainty supervision (docs/research_log.md
"Experiment 002 initial results", follow-up 2, and Experiment 003B).
Adds an explicit calibration term to reliability/precision's training
objective:

    loss = prediction_loss + lambda * MSE(predicted_uncertainty, true_std)

`prediction_loss` stays exactly what Experiment 002/003A used (MSE on
`mu`), and `integration.py`'s three aggregation formulas are NOT touched
(CLAUDE.md Sec 2) -- only this extra, switchable term is new, and
lambda=0 exactly reproduces 003B's plain-training numbers. Sweeps
lambda in {0, 0.01, 0.1, 1.0} on the ground-truth-uncertainty datasets
(u1_heteroscedastic_1d, u2_heteroscedastic_interaction) and asks: can the
belief cell's `uncertainty` channel be forced to track known noise levels
without degrading prediction quality?

Usage:
    python experiments/003_cell_ablation/run_003c.py
    python experiments/003_cell_ablation/run_003c.py --models reliability precision mlp_uncertainty
    python experiments/003_cell_ablation/run_003c.py --lambdas 0 0.01 0.1 1.0 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from uncertainty_harness import DATASETS, UNCERTAINTY_CAPABLE_MODELS, run_experiment

LAMBDAS: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["reliability", "precision"],
        choices=list(UNCERTAINTY_CAPABLE_MODELS),
    )
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(LAMBDAS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--hidden-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    all_results = []
    for dataset in DATASETS:
        for model_name in args.models:
            for lam in args.lambdas:
                for seed in args.seeds:
                    result = run_experiment(
                        dataset=dataset,
                        model_name=model_name,
                        seed=seed,
                        hidden_cells=args.hidden_cells,
                        steps=args.steps,
                        lr=args.lr,
                        lambda_calibration=lam,
                        experiment_id="003c_uncertainty_supervision",
                        results_dir=args.results_dir,
                    )
                    all_results.append(result)
                    corr = result.get("uncertainty_pearson_corr", float("nan"))
                    print(
                        f"{dataset:28s} {model_name:12s} lambda={lam:<6} seed={seed} "
                        f"r2={result['r2']:.4f} corr={corr:+.3f}"
                    )

    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["model"], r["lambda_calibration"])].append(r)

    print("\n=== Accuracy/calibration tradeoff across lambda (mean across seeds) ===")
    for dataset in DATASETS:
        for model_name in args.models:
            print(f"-- {dataset} / {model_name} --")
            for lam in args.lambdas:
                runs = grouped.get((dataset, model_name, lam), [])
                if not runs:
                    continue
                r2_mean = statistics.mean(r["r2"] for r in runs)
                corr_mean = statistics.mean(r["uncertainty_pearson_corr"] for r in runs)
                calib_mse_mean = statistics.mean(r["uncertainty_calibration_mse"] for r in runs)
                print(
                    f"    lambda={lam:<6} r2={r2_mean:.4f} corr={corr_mean:+.3f} "
                    f"calib_mse={calib_mse_mean:.4f}"
                )


if __name__ == "__main__":
    main()
