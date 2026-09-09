#!/usr/bin/env python3
"""Paper A Phase 1 -- the public-benchmark screening grid.

7 datasets x {25%, 100%} train fraction x 3 model families x 3 seeds
= at most 126 runs. California Housing is skipped (with an explicit notice)
only if it cannot be downloaded in this environment.

This is a *screening* experiment, not the final paper suite. Once it starts,
nothing about CellV0.1, the dataset list, or the train fractions changes
(Paper-A task Sec 12).

Usage:
    python experiments/paper_a/run_phase1.py
    python experiments/paper_a/run_phase1.py --datasets digits diabetes --seeds 0
    python experiments/paper_a/run_phase1.py --device cpu
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

from experiments.paper_a.datasets import BENCHMARK_DATASETS, dataset_available  # noqa: E402
from experiments.paper_a.harness import run_one  # noqa: E402
from experiments.paper_a.models import MODEL_FAMILIES  # noqa: E402

TRAIN_FRACTIONS: tuple[float, ...] = (0.25, 1.0)
SEEDS: tuple[int, ...] = (0, 1, 2)
RESULTS_RAW = _HERE / "results" / "raw"
RESULTS_PROCESSED = _HERE / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--fractions", type=float, nargs="+", default=list(TRAIN_FRACTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="torch CPU thread cap (set when running shards in parallel)",
    )
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--results-dir", default=str(RESULTS_RAW))
    parser.add_argument("--summary-out", default=str(RESULTS_PROCESSED / "phase1_runs.json"))
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else None
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for name in args.datasets:
        if name == "california_housing" and not dataset_available(name):
            print("SKIP california_housing: could not be downloaded in this environment.")
            continue
        datasets.append(name)

    grid = [
        (ds, frac, fam, seed)
        for ds in datasets
        for frac in args.fractions
        for fam in args.families
        for seed in args.seeds
    ]
    print(f"Running {len(grid)} runs on device={args.device or 'auto'}...")

    all_results = []
    for i, (ds, frac, fam, seed) in enumerate(grid, 1):
        t = time.time()
        r = run_one(
            ds,
            frac,
            fam,
            seed,
            device=device,
            results_dir=args.results_dir,
            max_steps=args.max_steps,
        )
        all_results.append(r)
        flags = []
        if r["eff_diverged"]:
            flags.append("DIVERGED")
        if r["eff_cap_hit"]:
            flags.append("CAP_HIT")
        print(
            f"[{i:3d}/{len(grid)}] {ds:18s} f={frac:<4} {fam:16s} seed={seed} "
            f"{r['headline_metric']}={r['headline_value']:.4f} "
            f"p={r['parameter_count']:>7d} beststep={r['eff_best_val_step']:>6d} "
            f"steps={r['eff_total_steps']:>6d} {time.time()-t:5.1f}s "
            f"{' '.join(flags)}"
        )
        # incremental write so a long sweep is recoverable
        with open(args.summary_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nWrote {len(all_results)} run summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
