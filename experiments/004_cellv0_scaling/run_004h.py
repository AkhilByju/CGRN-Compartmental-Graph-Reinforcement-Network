#!/usr/bin/env python3
"""Experiment 004H -- fair MLP optimization check (docs/research_log.md
"Experiment 004H"). At S0/S2/S4, trains `mlp` and `precision` at three
learning rates (0.3x/1x/3x the usual 1e-2 -- same tuning budget for both
architectures) and records train/val/test loss plus the headline prediction
metric for each.

Distinguishes why `mlp` declines relative to `precision` as scale grows
(Experiment 004A): if `mlp`'s *training* loss itself gets worse at a given
LR, that's an optimization-difficulty story (needs more steps/better LR at
width); if training loss is fine (or better) but test loss/metric is worse,
that's a generalization/overfitting story -- a different conclusion.

Kept small per the spec: 3 seeds (not 5) by default, since this is a
targeted diagnostic, not a result meant to stand alone statistically.

Usage:
    python experiments/004_cellv0_scaling/run_004h.py
    python experiments/004_cellv0_scaling/run_004h.py --lr-multipliers 0.3 1.0 3.0 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scaling_harness import DATASETS, THREE_WAY_SCALES, run_optimization_check

BASE_LR = 1e-2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument(
        "--scales", nargs="+", default=list(THREE_WAY_SCALES), choices=list(THREE_WAY_SCALES)
    )
    parser.add_argument("--lr-multipliers", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default="results/processed/004h_summary.json")
    args = parser.parse_args()

    all_results = []
    for scale in args.scales:
        for dataset in args.datasets:
            for mult in args.lr_multipliers:
                lr = BASE_LR * mult
                for seed in args.seeds:
                    r = run_optimization_check(
                        dataset=dataset,
                        scale_label=scale,
                        seed=seed,
                        lr=lr,
                        steps=args.steps,
                        n_train=args.n_train,
                        results_dir=args.results_dir,
                    )
                    r["lr_multiplier"] = mult
                    all_results.append(r)
                    headline = r["headline"]
                    print(
                        f"{dataset:32s} {scale} lr={lr:.4f} ({mult}x) seed={seed}  "
                        f"mlp: train={r['mlp__train_loss']:.4f} val={r['mlp__val_loss']:.4f} "
                        f"test={r['mlp__test_loss']:.4f} "
                        f"{headline}={r[f'mlp__{headline}']:.4f}  |  "
                        f"precision: train={r['precision__train_loss']:.4f} "
                        f"val={r['precision__val_loss']:.4f} test={r['precision__test_loss']:.4f} "
                        f"{headline}={r[f'precision__{headline}']:.4f}"
                    )

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["scale"], r["lr_multiplier"])].append(r)

    print("\n=== Summary (mean across seeds): train/test loss + headline metric ===")
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        for scale in args.scales:
            for mult in args.lr_multipliers:
                runs = grouped.get((dataset, scale, mult), [])
                if not runs:
                    continue
                headline = runs[0]["headline"]
                mlp_train = statistics.mean(r["mlp__train_loss"] for r in runs)
                mlp_test = statistics.mean(r["mlp__test_loss"] for r in runs)
                mlp_metric = statistics.mean(r[f"mlp__{headline}"] for r in runs)
                prec_train = statistics.mean(r["precision__train_loss"] for r in runs)
                prec_test = statistics.mean(r["precision__test_loss"] for r in runs)
                prec_metric = statistics.mean(r[f"precision__{headline}"] for r in runs)
                print(
                    f"    {scale} lr={mult}x  "
                    f"mlp: train_loss={mlp_train:.4f} test_loss={mlp_test:.4f} "
                    f"{headline}={mlp_metric:.4f}  |  "
                    f"precision: train_loss={prec_train:.4f} test_loss={prec_test:.4f} "
                    f"{headline}={prec_metric:.4f}"
                )


if __name__ == "__main__":
    main()
