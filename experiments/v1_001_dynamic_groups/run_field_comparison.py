#!/usr/bin/env python3
"""mlp / cellv0.1 / field_t1 / field_t2 on `dynamic_groups` --
CellV1.3 (Self-Organizing Refinement Field)'s first real comparison
against the existing baselines, per the user's exact spec: `field_t1`/
`field_t2` at `R=256`, `n_cells=128`, one seed, 3000 training examples,
1500 steps. No architecture-math changes, no global field routing, no
further evidence/uncertainty tuning -- this only wires the already-built
`SelfOrganizingRefinementField` (`harness.py::run_field_task`) into a
runnable comparison.

Usage:
    python experiments/v1_001_dynamic_groups/run_field_comparison.py
    python experiments/v1_001_dynamic_groups/run_field_comparison.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import FIELD_ARCHITECTURES, FIELD_NUM_FEATURES, N_CELLS, run_field_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="dynamic_groups")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=3_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-cells", type=int, default=N_CELLS)
    parser.add_argument("--num-features", type=int, default=FIELD_NUM_FEATURES)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default="results/processed/v1_001_field_comparison.json")
    args = parser.parse_args()

    all_results = []
    for seed in args.seeds:
        r = run_field_task(
            args.task,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            n_cells=args.n_cells,
            num_features=args.num_features,
            results_dir=args.results_dir,
        )
        all_results.append(r)
        headline = r["headline"]
        line = f"seed={seed} params(mlp/v0.1/t1/t2)=" + "/".join(
            str(r[f"{a}__params"]) for a in FIELD_ARCHITECTURES
        )
        line += "  " + "  ".join(f"{a}_{headline}={r[f'{a}__{headline}']:.4f}" for a in FIELD_ARCHITECTURES)
        line += "  " + "  ".join(
            f"{a}_ms/step={r[f'{a}__train_wall_clock_seconds']/args.steps*1000:.0f}" for a in FIELD_ARCHITECTURES
        )
        print(line)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    if len(all_results) > 1:
        headline = all_results[0]["headline"]
        print("\n=== Summary (mean +/- std across seeds) ===")
        for arch in FIELD_ARCHITECTURES:
            vals = [r[f"{arch}__{headline}"] for r in all_results]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"    {arch:10s} {headline}={mean:.4f}+/-{std:.4f}")


if __name__ == "__main__":
    main()
