#!/usr/bin/env python3
"""Experiment 003D -- evidence/uncertainty intervention (docs/research_log.md
"Experiment 002 initial results", follow-up 2; docs/experiment_protocol.md
Experiment 003).

Trains a `precision` `BeliefNetwork` on each of Experiment 002's six
datasets (r0_linear/r1_nonlinear/r2_interaction/c0_linear/c1_xor/
c2_interaction), then at inference time perturbs the evidence/uncertainty
channels the trained `layer1` hands to `layer2`
(`src.evaluation.intervention.perturb_belief`) under four conditions and
compares against the unperturbed baseline:

  - evidence_ones                 -- every cell's evidence forced to 1
  - uncertainty_ones               -- every cell's uncertainty forced to 1
  - shuffle_evidence_uncertainty  -- (evidence, uncertainty) pairs randomly
                                      permuted across cells per example
  - uncertainty_random             -- uncertainty replaced with values drawn
                                      uniformly from the batch's own observed
                                      range, independent of input/cell

Answers: are evidence/uncertainty actually participating in `layer2`'s
aggregation in a way that matters for the final prediction, or has the
network learned to route around them ("decorative state")? If performance
(R^2 for regression, accuracy for classification) barely changes under a
perturbation, that channel isn't pulling weight.

Usage:
    python experiments/003_cell_ablation/run_003d.py
    python experiments/003_cell_ablation/run_003d.py --seeds 0 1 2 3 4 --steps 2000
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from evidence_uncertainty_harness import DATASETS, run_experiment

from src.evaluation.intervention import PERTURBATION_MODES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--hidden-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    all_results = []
    for dataset in DATASETS:
        for seed in args.seeds:
            result = run_experiment(
                dataset=dataset,
                seed=seed,
                hidden_cells=args.hidden_cells,
                steps=args.steps,
                lr=args.lr,
                results_dir=args.results_dir,
            )
            all_results.append(result)
            headline = result["headline"]
            baseline_val = result[f"baseline__{headline}"]
            deltas = " ".join(
                f"{mode}={result[f'{mode}__delta_{headline}']:+.4f}"
                for mode in PERTURBATION_MODES
                if mode != "baseline"
            )
            print(
                f"{dataset:16s} seed={seed} baseline_{headline}={baseline_val:.4f}  {deltas}"
            )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[r["dataset"]].append(r)

    print("\n=== Summary (mean +/- std across seeds) ===")
    for dataset, runs in grouped.items():
        headline = runs[0]["headline"]
        baseline_mean = statistics.mean(r[f"baseline__{headline}"] for r in runs)
        print(f"-- {dataset} (baseline {headline}={baseline_mean:.4f}) --")
        for mode in PERTURBATION_MODES:
            if mode == "baseline":
                continue
            deltas = [r[f"{mode}__delta_{headline}"] for r in runs]
            mean = statistics.mean(deltas)
            std = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
            print(f"    {mode:32s} delta_{headline}={mean:+.4f}+/-{std:.4f}")


if __name__ == "__main__":
    main()
