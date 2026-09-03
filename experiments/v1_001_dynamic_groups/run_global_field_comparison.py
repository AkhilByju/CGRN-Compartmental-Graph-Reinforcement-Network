#!/usr/bin/env python3
"""cellv0.1 / field_t1 / field_t2 / field_local_global_t2 on
`dynamic_groups` -- the user's point-18 question: "does learned non-local
communication improve beyond the already-successful local self-organizing
refinement?" No mlp in this comparison (matching the user's exact
listing); `field_local_global_t2` is the parameter-matching target for
`cellv0.1` (see `harness.py::_build_global_comparison_models`'s
docstring for why -- global communication's added parameters are a much
bigger fraction of this task's model size than of a from-scratch large
model, so reusing `field_t2`'s smaller target would leave
`field_local_global_t2` unmatched).

Usage:
    python experiments/v1_001_dynamic_groups/run_global_field_comparison.py
    python experiments/v1_001_dynamic_groups/run_global_field_comparison.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import GLOBAL_DIM, GLOBAL_FIELD_ARCHITECTURES, N_CELLS, run_global_field_comparison


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
    parser.add_argument("--n-objects", type=int, default=None, help="Defaults to --n-cells if unset.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=5)
    parser.add_argument("--global-dim", type=int, default=GLOBAL_DIM)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument(
        "--summary-out", default="results/processed/v1_001_global_field_comparison.json"
    )
    args = parser.parse_args()

    n_objects = args.n_objects if args.n_objects is not None else args.n_cells

    all_results = []
    for seed in args.seeds:
        r = run_global_field_comparison(
            args.task,
            seed=seed,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            n_cells=args.n_cells,
            n_objects=n_objects,
            k_min=args.k_min,
            k_max=args.k_max,
            global_dim=args.global_dim,
            results_dir=args.results_dir,
        )
        all_results.append(r)
        headline = r["headline"]
        line = f"seed={seed} params(v0.1/t1/t2/t2+global)=" + "/".join(
            str(r[f"{a}__params"]) for a in GLOBAL_FIELD_ARCHITECTURES
        )
        line += "  " + "  ".join(
            f"{a}_{headline}={r[f'{a}__{headline}']:.4f}" for a in GLOBAL_FIELD_ARCHITECTURES
        )
        line += "  " + "  ".join(
            f"{a}_ms/step={r[f'{a}__train_wall_clock_seconds']/args.steps*1000:.0f}"
            for a in GLOBAL_FIELD_ARCHITECTURES
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
        for arch in GLOBAL_FIELD_ARCHITECTURES:
            vals = [r[f"{arch}__{headline}"] for r in all_results]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"    {arch:22s} {headline}={mean:.4f}+/-{std:.4f}")


if __name__ == "__main__":
    main()
