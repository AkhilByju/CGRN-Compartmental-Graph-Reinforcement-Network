#!/usr/bin/env python3
"""Complexity scaling on `dynamic_groups_global`, under convergence-based
training -- the corrected protocol after the global-collapse diagnostic
(`diagnose_global_collapse.py`) showed the fixed-1500-step sweep
(`run_complexity_scaling.py`) confounded *capability* with *convergence
speed*: a 24-object and a 192-object problem don't converge on the same
optimization timescale, so comparing them at one arbitrary step count
isn't a fair comparison.

Same three architectures, same four levels, same `n_cells = n_objects`
convention as `run_complexity_scaling.py` -- only the training protocol
and `num_features` (R) change:

  - Early stopping (`harness.py::_train_until_convergence`): validate
    every `--val-every` steps, stop after `--patience-steps` steps
    without improvement, cap at `--max-steps`. Reports the *best*-
    validation checkpoint's test performance, not whatever the model
    looked like at a fixed cutoff.
  - `num_features` (R) defaults to 512, not the 256 used everywhere else
    in this experiment line -- the diagnostic found R=512 let the global
    pathway find its solution in roughly a third of the steps R=256
    needed at n_objects=192; this sweep is specifically about capability
    *and* efficiency as complexity grows, so it uses the field models'
    faster-converging configuration throughout.

The question this answers (the user's framing): "does the self-
organizing architecture maintain a capability advantage as complexity
increases, and what computational price does it pay for that
advantage?" -- reported as best test R^2, steps-to-convergence, and
wall-clock-to-convergence, per level.

Usage:
    python experiments/v1_001_dynamic_groups/run_complexity_scaling_convergence.py
    python experiments/v1_001_dynamic_groups/run_complexity_scaling_convergence.py --levels hard very_hard
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import run_convergence_field_comparison

ARCHITECTURES: tuple[str, ...] = ("cellv0.1", "field_t2", "field_local_global_t2")

COMPLEXITY_LEVELS: dict[str, dict[str, int]] = {
    "easy": {"n_objects": 24, "k_min": 2, "k_max": 5},
    "medium": {"n_objects": 48, "k_min": 5, "k_max": 8},
    "hard": {"n_objects": 96, "k_min": 8, "k_max": 14},
    "very_hard": {"n_objects": 192, "k_min": 12, "k_max": 20},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=list(COMPLEXITY_LEVELS), choices=list(COMPLEXITY_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--num-features", type=int, default=512)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--patience-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=3_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument(
        "--summary-out", default="results/processed/v1_001_complexity_scaling_convergence.json"
    )
    parser.add_argument(
        "--include-mlp", action="store_true",
        help="Add a plain MLPBaseline, param-matched to field_local_global_t2's size at each level -- "
        "faster/better than CellV0.1?",
    )
    args = parser.parse_args()
    architectures = ARCHITECTURES + ("mlp",) if args.include_mlp else ARCHITECTURES

    all_results = []
    for level in args.levels:
        cfg = COMPLEXITY_LEVELS[level]
        n_objects = cfg["n_objects"]
        for seed in args.seeds:
            r = run_convergence_field_comparison(
                "dynamic_groups_global",
                seed=seed,
                batch_size=args.batch_size,
                lr=args.lr,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                val_every=args.val_every,
                patience_steps=args.patience_steps,
                max_steps=args.max_steps,
                n_cells=n_objects,
                n_objects=n_objects,
                k_min=cfg["k_min"],
                k_max=cfg["k_max"],
                num_features=args.num_features,
                results_dir=args.results_dir,
                include_mlp=args.include_mlp,
            )
            r["complexity_level"] = level
            all_results.append(r)
            line = f"level={level:9s} seed={seed} n_objects={n_objects} params=" + "/".join(
                str(r[f"{a}__params"]) for a in architectures
            )
            line += "  " + "  ".join(f"{a}_r2={r[f'{a}__r2']:.4f}" for a in architectures)
            line += "  " + "  ".join(f"{a}_steps={r[f'{a}__steps_to_convergence']}" for a in architectures)
            line += "  " + "  ".join(
                f"{a}_conv_s={r[f'{a}__wall_clock_to_convergence_seconds']:.1f}" for a in architectures
            )
            print(line)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    print("\n=== Summary (mean +/- std across seeds), per complexity level ===")
    for level in args.levels:
        level_results = [r for r in all_results if r["complexity_level"] == level]
        n_objects = COMPLEXITY_LEVELS[level]["n_objects"]
        print(f"  {level} (n_objects={n_objects}):")
        for arch in architectures:
            r2_vals = [r[f"{arch}__r2"] for r in level_results]
            steps_vals = [r[f"{arch}__steps_to_convergence"] for r in level_results]
            conv_s_vals = [r[f"{arch}__wall_clock_to_convergence_seconds"] for r in level_results]
            r2_mean = statistics.mean(r2_vals)
            r2_std = statistics.pstdev(r2_vals) if len(r2_vals) > 1 else 0.0
            steps_mean = statistics.mean(steps_vals)
            conv_s_mean = statistics.mean(conv_s_vals)
            print(
                f"    {arch:22s} r2={r2_mean:.4f}+/-{r2_std:.4f}  "
                f"steps_to_convergence={steps_mean:.0f}  wall_clock_to_convergence={conv_s_mean:.1f}s"
            )


if __name__ == "__main__":
    main()
