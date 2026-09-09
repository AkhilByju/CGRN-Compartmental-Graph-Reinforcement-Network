#!/usr/bin/env python3
"""Paper A Sec 11 -- the one cheap mechanistic ablation: CellV0.1 vs
CellV0.1-fixed-confidence.

`cellv0.1_fixed_confidence` keeps CellV0.1's exact content pathway and
parameter count but forces the evidence/uncertainty of every belief entering
a fusion to `e = u = 1`, so the dynamically propagated `e`/`u` state cannot
influence anything downstream. If CellV0.1's advantage (where it has one)
survives this, the advantage is coming from the content pathway, not the
propagated confidence state.

Phase 1 scope (fixed -- do not expand): **Digits, Diabetes, Fashion-MNIST**,
**25% training data**, seeds **0, 1, 2**. Both arms run here with the exact
same harness so the comparison is airtight; the `cellv0.1` arm reproduces the
Phase-1 grid's own numbers for those cells (deterministic).

Usage:
    python experiments/paper_a/run_ablation.py
    python experiments/paper_a/run_ablation.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from experiments.paper_a.harness import run_one  # noqa: E402

ABLATION_EXPERIMENT_ID = "paper_a_phase1_fixed_confidence"
ABLATION_DATASETS: tuple[str, ...] = ("digits", "diabetes", "fashion_mnist")
ABLATION_FAMILIES: tuple[str, ...] = ("cellv0.1", "cellv0.1_fixed_confidence")
ABLATION_FRACTION = 0.25
ABLATION_SEEDS: tuple[int, ...] = (0, 1, 2)
RESULTS_RAW = _HERE / "results" / "raw"
RESULTS_PROCESSED = _HERE / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(ABLATION_DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(ABLATION_SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--results-dir", default=str(RESULTS_RAW))
    parser.add_argument(
        "--summary-out", default=str(RESULTS_PROCESSED / "phase1_ablation_runs.json")
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    grid = [
        (ds, fam, seed)
        for ds in args.datasets
        for fam in ABLATION_FAMILIES
        for seed in args.seeds
    ]
    print(f"Running {len(grid)} fixed-confidence ablation runs...")

    all_results = []
    for i, (ds, fam, seed) in enumerate(grid, 1):
        t = time.time()
        r = run_one(
            ds,
            ABLATION_FRACTION,
            fam,
            seed,
            device=device,
            results_dir=args.results_dir,
            experiment_id=ABLATION_EXPERIMENT_ID,
            max_steps=args.max_steps,
        )
        all_results.append(r)
        flags = " ".join(
            name for name, on in
            [("DIVERGED", r["eff_diverged"]), ("CAP_HIT", r["eff_cap_hit"])]
            if on
        )
        print(
            f"[{i:2d}/{len(grid)}] {ds:16s} {fam:26s} seed={seed} "
            f"{r['headline_metric']}={r['headline_value']:.4f} "
            f"beststep={r['eff_best_val_step']:>6d} {time.time()-t:5.1f}s {flags}"
        )
        with open(args.summary_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nWrote {len(all_results)} ablation run summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
