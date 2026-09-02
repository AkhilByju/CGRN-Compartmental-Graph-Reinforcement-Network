#!/usr/bin/env python3
"""Experiment 004 -- cell-count-matched comparison (docs/experiment_protocol.md
Experiment 004, "Parameter-matched vs. cell-count-matched"). The second
fairness regime, run at only 2-3 sizes (`scaling_harness.CELL_COUNTS`,
reusing 004A's own S0/S2/S4 hidden-unit counts: 15/68/271): `MLPBaseline`
and `precision` `BeliefNetwork` built with the SAME number of hidden units,
not the same parameter count -- `precision` ends up with more parameters
(every connection carries both a content weight `w` and a relevance
parameter `g`).

Asks: is one `BeliefCell` computationally richer than one ordinary neuron,
independent of what it costs in parameters? If parameter-matched (004A) is a
tie but `precision` wins here, the richer cell adds capability but currently
pays for it in extra parameters -- a different conclusion than "the cell
doesn't help."

Usage:
    python experiments/004_cellv0_scaling/run_004_cell_count_matched.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scaling_harness import CELL_COUNTS, DATASETS, run_cell_count_matched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument(
        "--counts", nargs="+", default=list(CELL_COUNTS), choices=list(CELL_COUNTS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument(
        "--summary-out", default="results/processed/004_cell_count_matched_summary.json"
    )
    args = parser.parse_args()

    all_results = []
    for count_label in args.counts:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_cell_count_matched(
                    dataset=dataset,
                    count_label=count_label,
                    seed=seed,
                    steps=args.steps,
                    n_train=args.n_train,
                    results_dir=args.results_dir,
                )
                all_results.append(r)
                headline = r["headline"]
                print(
                    f"{dataset:32s} {count_label} (n_units={r['n_units']:>4d}) seed={seed}  "
                    f"mlp_params={r['mlp__params']:>7d} "
                    f"precision_params={r['precision__params']:>7d} "
                    f"(+{r['precision_params_pct_more_than_mlp']:.1f}%)  "
                    f"mlp_{headline}={r[f'mlp__{headline}']:.4f}  "
                    f"precision_{headline}={r[f'precision__{headline}']:.4f}"
                )

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["cell_count_label"])].append(r)

    print("\n=== Summary (mean across seeds) ===")
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        for count_label in args.counts:
            runs = grouped.get((dataset, count_label), [])
            if not runs:
                continue
            headline = runs[0]["headline"]
            mlp_mean = statistics.mean(r[f"mlp__{headline}"] for r in runs)
            precision_mean = statistics.mean(r[f"precision__{headline}"] for r in runs)
            print(
                f"    {count_label} (n_units={runs[0]['n_units']})  "
                f"mlp={mlp_mean:.4f}  precision={precision_mean:.4f}  "
                f"delta={precision_mean - mlp_mean:+.4f}"
            )


if __name__ == "__main__":
    main()
