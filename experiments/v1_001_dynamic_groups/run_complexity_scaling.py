#!/usr/bin/env python3
"""Complexity scaling on `dynamic_groups_global` -- the architectural
hypothesis, not another component comparison: "as information becomes
structurally more complex, a system that self-organizes locally and
selectively exchanges non-local information should degrade more
gracefully than a fixed feed-forward organization." Architecture is
frozen (per the user's explicit freeze -- BeliefCell, scale-stable
precision, ORFF local field, self-anchoring, mean-shift, T2 dynamics, the
global linear field, send/need near-off initialization, gated global
residual all stay exactly as already validated); only the *task's*
complexity moves.

Three architectures only (no `field_t1` -- on `dynamic_groups_global`,
T1 and local-T2 were already shown to be ~identical, so an extra
refinement step isn't the bottleneck when the task needs non-local
information; no `mlp` -- not part of this comparison): `cellv0.1`,
`field_t2` (local only), `field_local_global_t2`. `field_local_global_t2`
is the parameter-matching target at every level (same convention as
`harness.py::_build_global_comparison_models`); `n_cells = n_objects`
at every level (no spare filler cells) -- the cleanest way to scale
"problem size" without also scaling unrelated slack capacity, so a
larger level isn't confounded by more filler cells to recruit.

Reports R^2 *and* ms/step per level, per the user's explicit ask: a
result where accuracy holds up while runtime quietly goes quadratic
would just be "the earlier graph problem in a different form."

Usage:
    python experiments/v1_001_dynamic_groups/run_complexity_scaling.py
    python experiments/v1_001_dynamic_groups/run_complexity_scaling.py --levels easy medium
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from harness import run_global_field_comparison

ARCHITECTURES: tuple[str, ...] = ("cellv0.1", "field_t2", "field_local_global_t2")

# The user's exact table. k_min/k_max approximate the stated group-count
# ranges; the generator's own separated-center sampling (dynamic_groups.py)
# does the rest -- exact ranges aren't load-bearing, per the user ("can
# follow whatever generator is convenient").
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
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=3_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--summary-out", default="results/processed/v1_001_complexity_scaling.json")
    args = parser.parse_args()

    all_results = []
    for level in args.levels:
        cfg = COMPLEXITY_LEVELS[level]
        n_objects = cfg["n_objects"]
        for seed in args.seeds:
            r = run_global_field_comparison(
                "dynamic_groups_global",
                seed=seed,
                steps=args.steps,
                batch_size=args.batch_size,
                lr=args.lr,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                n_cells=n_objects,
                n_objects=n_objects,
                k_min=cfg["k_min"],
                k_max=cfg["k_max"],
                results_dir=args.results_dir,
                include_field_t1=False,
            )
            r["complexity_level"] = level
            all_results.append(r)
            headline = r["headline"]
            line = f"level={level:9s} seed={seed} n_objects={n_objects} params=" + "/".join(
                str(r[f"{a}__params"]) for a in ARCHITECTURES
            )
            line += "  " + "  ".join(f"{a}_{headline}={r[f'{a}__{headline}']:.4f}" for a in ARCHITECTURES)
            line += "  " + "  ".join(
                f"{a}_ms/step={r[f'{a}__train_wall_clock_seconds'] / args.steps * 1000:.1f}" for a in ARCHITECTURES
            )
            print(line)

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    headline = all_results[0]["headline"]
    print("\n=== Summary (mean +/- std across seeds), per complexity level ===")
    for level in args.levels:
        level_results = [r for r in all_results if r["complexity_level"] == level]
        n_objects = COMPLEXITY_LEVELS[level]["n_objects"]
        print(f"  {level} (n_objects={n_objects}):")
        for arch in ARCHITECTURES:
            r2_vals = [r[f"{arch}__{headline}"] for r in level_results]
            ms_vals = [r[f"{arch}__train_wall_clock_seconds"] / args.steps * 1000 for r in level_results]
            r2_mean = statistics.mean(r2_vals)
            r2_std = statistics.pstdev(r2_vals) if len(r2_vals) > 1 else 0.0
            ms_mean = statistics.mean(ms_vals)
            print(f"    {arch:22s} {headline}={r2_mean:.4f}+/-{r2_std:.4f}  ms/step={ms_mean:.1f}")


if __name__ == "__main__":
    main()
