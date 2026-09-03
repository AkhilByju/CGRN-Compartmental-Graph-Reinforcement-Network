#!/usr/bin/env python3
"""v1_001_dynamic_groups -- "Does self-organizing computation actually
help?" (the user's Experiment 005; see this folder's README for the
number). Four architectures (`mlp`, `cellv0.1`, `cellv1_local`,
`cellv1_full`, the latter two now `SparseDynamicBeliefGraph` --
CellV1.1 -- not dense) x tasks x scales -- one experiment, not five
diagnostic projects.

**Scale sweep:** `--scales` (default all three) sweeps `n_cells` over the
user's `V1_SCALES` (`V1-S0`=128, `V1-S1`=256, `V1-S2`=512), `T=6` fixed
throughout -- mlp/cellv0.1 are re-parameter-matched to cellv1_full's
actual size at *each* scale. `--n-cells` overrides this with one custom
value instead (for a quick correctness check at a tiny scale).

Usage:
    # The real run the user asked for: dynamic_groups, all 3 scales, 3 seeds:
    python experiments/v1_001_dynamic_groups/run_v1_001.py --tasks dynamic_groups --seeds 0 1 2

    # All four tasks too:
    python experiments/v1_001_dynamic_groups/run_v1_001.py --seeds 0 1 2

    # One scale only:
    python experiments/v1_001_dynamic_groups/run_v1_001.py --tasks dynamic_groups --scales V1-S0

    # Quick correctness check (tiny, custom n_cells, overrides --scales):
    python experiments/v1_001_dynamic_groups/run_v1_001.py --n-cells 30 --steps 20 --n-train 60 --n-val 20 --n-test 20 --graph-eval-n 10
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from harness import ARCHITECTURES, ASSOCIATION_DIM, CELLV1_HIDDEN_DIM, NUM_STEPS, TASKS, V1_SCALES, run_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--scales", nargs="+", default=list(V1_SCALES), choices=list(V1_SCALES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=2_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--graph-eval-n", type=int, default=100)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default="results/processed/v1_001_summary.json")
    parser.add_argument(
        "--n-cells", type=int, default=None,
        help="Override -- one custom CellV1 population size instead of sweeping --scales "
             "(e.g. 30, for a quick correctness check). CellV1.1's own cost, not mlp/cellv0.1 "
             "(cheap regardless), drives wall-clock. Must be >= --n-objects if 'dynamic_groups' "
             "is among --tasks.",
    )
    parser.add_argument("--association-dim", type=int, default=ASSOCIATION_DIM, help="CellV1's z dimension.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS, help="CellV1 refinement steps T (fixed at 6 for the real experiment).")
    parser.add_argument("--hidden-dim", type=int, default=CELLV1_HIDDEN_DIM, help="Width of CellV1's shared MLPs.")
    parser.add_argument("--n-objects", type=int, default=24, help="dynamic_groups: objects per example.")
    parser.add_argument("--k-min", type=int, default=2, help="dynamic_groups: minimum hidden groups per example.")
    parser.add_argument("--k-max", type=int, default=5, help="dynamic_groups: maximum hidden groups per example.")
    args = parser.parse_args()

    scale_plan = [("custom", args.n_cells)] if args.n_cells is not None else [
        (label, V1_SCALES[label]) for label in args.scales
    ]

    all_results = []
    for scale_label, n_cells in scale_plan:
        for task in args.tasks:
            for seed in args.seeds:
                r = run_task(
                    task,
                    seed=seed,
                    steps=args.steps,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    n_train=args.n_train,
                    n_val=args.n_val,
                    n_test=args.n_test,
                    graph_eval_n=args.graph_eval_n,
                    results_dir=args.results_dir,
                    n_cells=n_cells,
                    association_dim=args.association_dim,
                    num_steps=args.num_steps,
                    hidden_dim=args.hidden_dim,
                    n_objects=args.n_objects,
                    k_min=args.k_min,
                    k_max=args.k_max,
                )
                r["scale"] = scale_label
                all_results.append(r)
                headline = r["headline"]
                line = f"{scale_label:6s} {task:32s} seed={seed} params(mlp/v0.1/local/full)=" + "/".join(
                    str(r[f"{a}__params"]) for a in ARCHITECTURES
                )
                line += "  " + "  ".join(f"{a}_{headline}={r[f'{a}__{headline}']:.4f}" for a in ARCHITECTURES)
                print(line)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["task"], r["scale"])].append(r)

    print("\n=== Summary (mean +/- std across seeds) ===")
    for scale_label, _n_cells in scale_plan:
        for task in args.tasks:
            runs = grouped.get((task, scale_label), [])
            if not runs:
                continue
            headline = runs[0]["headline"]
            print(f"-- {task} @ {scale_label} (n_cells={runs[0]['n_cells']}) --")
            for arch in ARCHITECTURES:
                vals = [r[f"{arch}__{headline}"] for r in runs]
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                params = runs[0][f"{arch}__params"]
                print(f"    {arch:12s} (~{params} params)  {headline}={mean:.4f}+/-{std:.4f}")
            if task == "dynamic_groups":
                for arch in ("cellv1_local", "cellv1_full"):
                    agreement = [r[f"{arch}__local_group_agreement_auroc_tT"] for r in runs]
                    print(f"    {arch:12s} local_group_agreement_auroc_tT={statistics.mean(agreement):.4f}")
                cross = [r["cellv1_full__global_cross_group_fraction_tT"] for r in runs]
                print(f"    cellv1_full  global_cross_group_fraction_tT={statistics.mean(cross):.4f}")


if __name__ == "__main__":
    main()
