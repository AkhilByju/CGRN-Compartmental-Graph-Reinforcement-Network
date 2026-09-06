#!/usr/bin/env python3
"""CellV1.5's first evaluation (`docs/architecture_v1.md` §16): does the
slow structural plasticity earn its keep?

Three arms, and only three: `cellv0.1` (parameter-matched), CellV1.5 with
the bootstrap topology **frozen for all of training**, and full CellV1.5
with §16.9's prune/grow schedule executing. The frozen arm is the point of
the experiment -- it isolates structural plasticity from everything else
CellV1.5 does, so a win over `cellv0.1` can be attributed to the right
part. Everything about the two CellV1.5 arms is identical except whether
rewiring runs (`structural_harness.py`'s docstring lists exactly how that
is enforced).

Two conditions, matching `run_complexity_scaling_association.py`'s
definitions so the numbers sit next to the CellV1.4 sweep already on
record: `hard` (96 objects, K in 8..14) and `very_hard` (192 objects,
K in 12..20), on `dynamic_groups_global` -- the argmax/argmin
aggregate-value pairing task, not the spatial-pairing one. 3 seeds each.

**Patience.** Defaults are `--patience-steps 2000 --max-steps 15000`,
against the `500`/`5000` used for the CellV1.3/1.4 sweeps. That older
setting produced a documented artifact (`docs/research_log.md`,
2026-09-03 CellV1.3.1 entry, finding 3): `field_t2` at `very_hard`/seed 0
early-stopped at R^2=0.0667 while the same architecture reached 0.79/0.82
on the other two seeds -- a model killed on a plateau and reported as an
architectural result. Every run here records `total_steps_run` and
`hit_step_cap`, so a budget-limited run is identifiable instead of being
silently read as converged.

Nothing in `src/models/architecture_v1/structural*.py` is modified by this
experiment -- CellV1.5 is frozen for this evaluation.

Usage:
    # The real run (~1-2h on CPU):
    python experiments/v1_001_dynamic_groups/run_structural_comparison.py

    # One condition only:
    python experiments/v1_001_dynamic_groups/run_structural_comparison.py --levels hard

    # Fast correctness check, NOT a result:
    python experiments/v1_001_dynamic_groups/run_structural_comparison.py \\
        --levels hard --seeds 0 --n-objects-override 24 --n-train 200 --n-val 100 \\
        --n-test 100 --val-every 20 --patience-steps 40 --max-steps 120 \\
        --results-dir /tmp/v1_002_smoke --graph-dir /tmp/v1_002_smoke/graphs \\
        --summary-out /tmp/v1_002_smoke/summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from structural_harness import (  # noqa: E402
    COMPLEXITY_LEVELS,
    STRUCTURAL_ARCHITECTURES,
    run_structural_comparison,
)

# Metrics printed in the per-level summary block, in the order the user
# asked for them. Structural ones are CellV1.5-only (`cellv0.1` has no
# substrate to measure), so the printer skips them for that arm.
_SUBSTRATE_KEYS = (
    "structural_edges_changed_fraction",
    "bootstrap_edges_retained_fraction",
    "edge_use_rate_mean__tau1.0",
    "edges_input_dependent_fraction__tau1.0",
    "edges_never_used_fraction__tau1.0",
    "mean_pairwise_jaccard__tau1.0",
    "mean_active_edge_fraction_per_input__tau1.0",
    "in_degree_mean",
    "in_degree_max",
    "in_degree_gini",
    "mean_effective_in_degree",
)


def _fmt(value: float | int | bool | str | None) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    levels = list(COMPLEXITY_LEVELS)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--levels", nargs="+", default=levels, choices=levels)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--task", default="dynamic_groups_global")
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument(
        "--patience-steps",
        type=int,
        default=2_000,
        help="Deliberately 4x the CellV1.3/1.4 sweeps' 500 -- see this script's docstring.",
    )
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--n-train", type=int, default=3_000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument(
        "--num-steps", type=int, default=None,
        help="CellV1.5's T; defaults to structural_harness.NUM_STEPS.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "CPU by default and on purpose: CellV1.5's LSH candidate retrieval builds its "
            "hyperplanes on the default device rather than phi's, so growth/bootstrap raise "
            "on MPS/CUDA. A one-line device fix in structural_plasticity.py, deliberately not "
            "applied while CellV1.5 is frozen for this evaluation."
        ),
    )
    parser.add_argument(
        "--n-objects-override",
        type=int,
        default=None,
        help="Smoke-test escape hatch: run a level's K range at a smaller object count.",
    )
    parser.add_argument("--results-dir", default="results/raw")
    parser.add_argument("--graph-dir", default="results/raw/structural_substrate")
    parser.add_argument(
        "--summary-out", default="results/processed/v1_002_structural_substrate.json"
    )
    args = parser.parse_args()

    extra = {} if args.num_steps is None else {"num_steps": args.num_steps}
    device = torch.device(args.device)

    all_results = []
    for level in args.levels:
        cfg = COMPLEXITY_LEVELS[level]
        n_objects = args.n_objects_override or cfg["n_objects"]
        for seed in args.seeds:
            r = run_structural_comparison(
                args.task,
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
                graph_dir=args.graph_dir,
                device=device,
                **extra,
            )
            r["complexity_level"] = level
            all_results.append(r)

            print(f"level={level:9s} seed={seed} n_objects={n_objects}")
            for arch in STRUCTURAL_ARCHITECTURES:
                cap = " CAP" if r.get(f"{arch}__hit_step_cap") else ""
                line = (
                    f"    {arch:18s} r2={r[f'{arch}__r2']:.4f}  params={r[f'{arch}__params']:>6d}  "
                    f"steps={r[f'{arch}__steps_to_convergence']:>6d}"
                    f"/{r[f'{arch}__total_steps_run']:<6d}{cap}  "
                    f"conv_s={r[f'{arch}__wall_clock_to_convergence_seconds']:.1f}"
                )
                if f"{arch}__structural_edges_changed_fraction" in r:
                    line += (
                        f"\n        edges={int(r[f'{arch}__n_edges_final'])} "
                        "changed_vs_bootstrap="
                        f"{r[f'{arch}__structural_edges_changed_fraction']:.4f} "
                        f"events={r[f'{arch}__plasticity_events_at_best_checkpoint']} "
                        f"use_rate={r[f'{arch}__edge_use_rate_mean__tau1.0']:.4f} "
                        f"input_dep={r[f'{arch}__edges_input_dependent_fraction__tau1.0']:.4f} "
                        f"jaccard={r[f'{arch}__mean_pairwise_jaccard__tau1.0']:.4f} "
                        f"in_deg={r[f'{arch}__in_degree_mean']:.2f}"
                        f"(max {int(r[f'{arch}__in_degree_max'])}, "
                        f"gini {r[f'{arch}__in_degree_gini']:.3f})"
                    )
                print(line)
            sys.stdout.flush()

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {len(all_results)} run summaries to {summary_path}")

    print("\n=== Summary (mean +/- population std across seeds), per condition ===")
    for level in args.levels:
        rows = [r for r in all_results if r["complexity_level"] == level]
        if not rows:
            continue
        print(f"\n  {level} (n_objects={rows[0]['n_objects']}, n_cells={rows[0]['n_cells']}):")
        for arch in STRUCTURAL_ARCHITECTURES:
            r2 = [r[f"{arch}__r2"] for r in rows]
            steps = [r[f"{arch}__steps_to_convergence"] for r in rows]
            wall = [r[f"{arch}__wall_clock_to_convergence_seconds"] for r in rows]
            std = statistics.pstdev(r2) if len(r2) > 1 else 0.0
            caps = sum(1 for r in rows if r.get(f"{arch}__hit_step_cap"))
            print(
                f"    {arch:18s} params={rows[0][f'{arch}__params']:>6d}  "
                f"r2={statistics.mean(r2):.4f}+/-{std:.4f}  "
                f"steps_to_conv={statistics.mean(steps):.0f}  "
                f"wall_clock_to_conv={statistics.mean(wall):.1f}s"
                + (f"  [{caps}/{len(rows)} hit step cap]" if caps else "")
            )
            print(f"        per-seed r2: {', '.join(f'{v:.4f}' for v in r2)}")
            if f"{arch}__structural_edges_changed_fraction" in rows[0]:
                for key in _SUBSTRATE_KEYS:
                    vals = [r[f"{arch}__{key}"] for r in rows]
                    per_seed = ", ".join(_fmt(v) for v in vals)
                    print(f"        {key:38s} {statistics.mean(vals):.4f}  ({per_seed})")


if __name__ == "__main__":
    main()
