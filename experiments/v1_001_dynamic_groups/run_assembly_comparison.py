#!/usr/bin/env python3
"""CellV1.6 (Precision-Regulated Assembly, docs/architecture_v1.md Sec 17)
-- first validation experiment.

Tests the frozen hypothesis: does letting CellV0.1's internal
evidence/uncertainty dynamically regulate population participation improve
computation over CellV0.1 alone, without meaningfully increasing parameter
count or optimization cost?

Compares ONLY `cellv0.1` and `cellv1_6`, on the existing
`dynamic_groups_global` benchmark, at `hard` (96 objects) and `very_hard`
(192 objects), seeds 0/1/2 -- the same task, splits, optimizer family,
parameter budget, checkpoint selection, and convergence-based protocol as
the CellV0.1 complexity experiment
(`run_complexity_scaling_association.py` /
`harness.run_convergence_association_comparison`). No special learning
rate or schedule for `cellv1_6`.

`cellv1_6` is `cellv0.1`'s exact 2-BeliefLayer backbone (same
`hidden_cells`, sized to the same per-level param budget the complexity
experiment used) plus one 34-param `PrecisionRegulatedAssemblyGate`
between the layers.

Usage:
    python experiments/v1_001_dynamic_groups/run_assembly_comparison.py
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import (
    ASSEMBLY_ARCHITECTURES,
    ASSEMBLY_DIAGNOSTIC_KEYS,
    run_convergence_assembly_comparison,
)

COMPLEXITY_LEVELS: dict[str, dict[str, int]] = {
    "easy": {"n_objects": 24, "k_min": 2, "k_max": 5},
    "medium": {"n_objects": 48, "k_min": 5, "k_max": 8},
    "hard": {"n_objects": 96, "k_min": 8, "k_max": 14},
    "very_hard": {"n_objects": 192, "k_min": 12, "k_max": 20},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", default=["hard", "very_hard"], choices=list(COMPLEXITY_LEVELS))
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
        "--summary-out", default="results/processed/v1_001_assembly_comparison.json"
    )
    args = parser.parse_args()

    all_results = []
    for level in args.levels:
        cfg = COMPLEXITY_LEVELS[level]
        for seed in args.seeds:
            r = run_convergence_assembly_comparison(
                "dynamic_groups_global",
                complexity_level=level,
                seed=seed,
                batch_size=args.batch_size,
                lr=args.lr,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                val_every=args.val_every,
                patience_steps=args.patience_steps,
                max_steps=args.max_steps,
                n_objects=cfg["n_objects"],
                k_min=cfg["k_min"],
                k_max=cfg["k_max"],
                results_dir=args.results_dir,
            )
            all_results.append(r)
            params = "/".join(f"{a}={r[f'{a}__params']}" for a in ASSEMBLY_ARCHITECTURES)
            r2 = "  ".join(f"{a}_r2={r[f'{a}__r2']:.4f}" for a in ASSEMBLY_ARCHITECTURES)
            steps = "  ".join(
                f"{a}_best@{r[f'{a}__steps_to_convergence']}/{r[f'{a}__total_steps_run']}"
                for a in ASSEMBLY_ARCHITECTURES
            )
            af = (
                f"  af init/best/end="
                f"{r['cellv1_6__assembly_active_fraction__init']:.3f}/"
                f"{r['cellv1_6__assembly_active_fraction__best']:.3f}/"
                f"{r['cellv1_6__assembly_active_fraction__end']:.3f}"
            )
            print(f"level={level:9s} seed={seed} params[{params}]  {r2}  {steps}{af}")

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    print("\n=== Summary (mean +/- std across seeds), per complexity level ===")
    for level in args.levels:
        level_results = [r for r in all_results if r["complexity_level"] == level]
        if not level_results:
            continue
        n_objects = COMPLEXITY_LEVELS[level]["n_objects"]
        hidden_cells = level_results[0]["hidden_cells"]
        print(f"  {level} (n_objects={n_objects}, hidden_cells={hidden_cells}):")
        for arch in ASSEMBLY_ARCHITECTURES:
            r2_vals = [r[f"{arch}__r2"] for r in level_results]
            steps_vals = [r[f"{arch}__steps_to_convergence"] for r in level_results]
            total_vals = [r[f"{arch}__total_steps_run"] for r in level_results]
            wall_vals = [r[f"{arch}__total_wall_clock_seconds"] for r in level_results]
            params = level_results[0][f"{arch}__params"]
            r2_mean = statistics.mean(r2_vals)
            r2_std = statistics.pstdev(r2_vals) if len(r2_vals) > 1 else 0.0
            print(
                f"    {arch:10s} params={params:5d}  r2={r2_mean:.4f}+/-{r2_std:.4f}  "
                f"best_step={statistics.mean(steps_vals):.0f}  total_step={statistics.mean(total_vals):.0f}  "
                f"wall={statistics.mean(wall_vals):.1f}s"
            )
        for phase in ("init", "best", "end"):
            diag = "  ".join(
                f"{k.replace('assembly_', '')}="
                f"{statistics.mean(r[f'cellv1_6__{k}__{phase}'] for r in level_results):.4f}"
                for k in ASSEMBLY_DIAGNOSTIC_KEYS
            )
            print(f"    cellv1_6 gate @{phase:4s}: {diag}")


if __name__ == "__main__":
    main()
