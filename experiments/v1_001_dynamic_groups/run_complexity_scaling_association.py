#!/usr/bin/env python3
"""The frozen-architecture confirmation run: CellV0.1 vs
Association-Local+Global-T2 (the user's "leading architecture" --
`association_model.py`) across `easy`/`medium`/`hard`/`very_hard`, 3
seeds, convergence-based training. Not a giant diagnostic suite -- just
establishing whether the pattern from the single-seed check
(`check_association_field.py`) is real: simple tasks, CellV0.1 ~=
association field; complex tasks, association field begins pulling
ahead. No architecture math changes here -- this only runs the already-
implemented, already-tested model over the sweep.

Same `n_cells = n_objects` convention, same convergence protocol
(`harness.py::_train_until_convergence`: validate every `--val-every`
steps, stop after `--patience-steps` without improvement, cap at
`--max-steps`) as `run_complexity_scaling_convergence.py` -- the ORFF
field sweep that motivated this redesign. `field_t2`/
`field_local_global_t2` are not part of this run (retired from active
development; their numbers are already recorded in
`results/processed/v1_001_complexity_scaling_convergence.json`).

Usage:
    python experiments/v1_001_dynamic_groups/run_complexity_scaling_association.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import ASSOCIATION_ARCHITECTURES, run_convergence_association_comparison

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
        "--summary-out", default="results/processed/v1_001_complexity_scaling_association.json"
    )
    args = parser.parse_args()

    all_results = []
    for level in args.levels:
        cfg = COMPLEXITY_LEVELS[level]
        n_objects = cfg["n_objects"]
        for seed in args.seeds:
            r = run_convergence_association_comparison(
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
                results_dir=args.results_dir,
            )
            r["complexity_level"] = level
            all_results.append(r)
            line = f"level={level:9s} seed={seed} n_objects={n_objects} params=" + "/".join(
                str(r[f"{a}__params"]) for a in ASSOCIATION_ARCHITECTURES
            )
            line += "  " + "  ".join(f"{a}_r2={r[f'{a}__r2']:.4f}" for a in ASSOCIATION_ARCHITECTURES)
            line += "  " + "  ".join(f"{a}_steps={r[f'{a}__steps_to_convergence']}" for a in ASSOCIATION_ARCHITECTURES)
            line += "  " + "  ".join(
                f"{a}_conv_s={r[f'{a}__wall_clock_to_convergence_seconds']:.1f}" for a in ASSOCIATION_ARCHITECTURES
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
        for arch in ASSOCIATION_ARCHITECTURES:
            r2_vals = [r[f"{arch}__r2"] for r in level_results]
            steps_vals = [r[f"{arch}__steps_to_convergence"] for r in level_results]
            conv_s_vals = [r[f"{arch}__wall_clock_to_convergence_seconds"] for r in level_results]
            r2_mean = statistics.mean(r2_vals)
            r2_std = statistics.pstdev(r2_vals) if len(r2_vals) > 1 else 0.0
            steps_mean = statistics.mean(steps_vals)
            conv_s_mean = statistics.mean(conv_s_vals)
            print(
                f"    {arch:28s} r2={r2_mean:.4f}+/-{r2_std:.4f}  "
                f"steps_to_convergence={steps_mean:.0f}  wall_clock_to_convergence={conv_s_mean:.1f}s"
            )


if __name__ == "__main__":
    main()
