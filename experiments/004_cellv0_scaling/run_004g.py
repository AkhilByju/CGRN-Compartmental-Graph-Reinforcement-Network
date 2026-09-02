#!/usr/bin/env python3
"""Experiment 004G -- compact 3-way scale rerun (docs/research_log.md
"Experiment 004G"): `mlp`, `precision`, and `normalized_precision`
(Experiment 004F) at S0 (~600), S2 (~10K), S4 (~150K) params, on
r2_interaction/c2_interaction/u2_heteroscedastic_interaction, 5 seeds.
Reports performance for all three architectures and per-layer internal
state (evidence/uncertainty mean, by layer) for both belief variants, so
`normalized_precision`'s scale-stability can be checked directly against
`precision`'s known instability (Experiment 004D/004E).

Usage:
    python experiments/004_cellv0_scaling/run_004g.py
    python experiments/004_cellv0_scaling/run_004g.py --scales S0 S2 S4 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from scaling_harness import DATASETS, THREE_WAY_SCALES, run_scale_three_way


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument(
        "--scales", nargs="+", default=list(THREE_WAY_SCALES), choices=list(THREE_WAY_SCALES)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--n-train", type=int, default=100_000)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default="results/processed/004g_summary.json")
    args = parser.parse_args()

    all_results = []
    for scale in args.scales:
        for dataset in args.datasets:
            for seed in args.seeds:
                r = run_scale_three_way(
                    dataset=dataset,
                    scale_label=scale,
                    seed=seed,
                    steps=args.steps,
                    n_train=args.n_train,
                    results_dir=args.results_dir,
                )
                all_results.append(r)
                headline = r["headline"]
                print(
                    f"{dataset:32s} {scale} seed={seed}  "
                    f"mlp={r[f'mlp__{headline}']:.4f}  "
                    f"precision={r[f'precision__{headline}']:.4f}  "
                    f"normalized={r[f'normalized_precision__{headline}']:.4f}  "
                    f"| layer2_evidence precision={r['precision__layer2_evidence_mean']:.2f} "
                    f"normalized={r['normalized_precision__layer2_evidence_mean']:.2f}"
                )

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["scale"])].append(r)

    print("\n=== Performance summary (mean +/- std across seeds) ===")
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        for scale in args.scales:
            runs = grouped.get((dataset, scale), [])
            if not runs:
                continue
            headline = runs[0]["headline"]
            means = {}
            for arch in ("mlp", "precision", "normalized_precision"):
                vals = [r[f"{arch}__{headline}"] for r in runs]
                means[arch] = statistics.mean(vals)
            print(
                f"    {scale}  mlp={means['mlp']:.4f}  precision={means['precision']:.4f}  "
                f"normalized={means['normalized_precision']:.4f}"
            )

    print("\n=== Layer 2 internal-state stability (mean across seeds) ===")
    for dataset in args.datasets:
        print(f"-- {dataset} --")
        for scale in args.scales:
            runs = grouped.get((dataset, scale), [])
            if not runs:
                continue
            prec_ev = statistics.mean(r["precision__layer2_evidence_mean"] for r in runs)
            prec_un = statistics.mean(r["precision__layer2_uncertainty_mean"] for r in runs)
            norm_ev = statistics.mean(
                r["normalized_precision__layer2_evidence_mean"] for r in runs
            )
            norm_un = statistics.mean(
                r["normalized_precision__layer2_uncertainty_mean"] for r in runs
            )
            print(
                f"    {scale}  precision: evidence={prec_ev:.2f} uncertainty={prec_un:.3f}  |  "
                f"normalized: evidence={norm_ev:.2f} uncertainty={norm_un:.3f}"
            )


if __name__ == "__main__":
    main()
