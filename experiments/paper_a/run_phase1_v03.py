#!/usr/bin/env python3
"""Paper A -- CellV0.3 on the frozen Phase-1 public-benchmark protocol.

CellV0.3 (the Conflict-Normalized Belief Cell,
`src/models/architecture_v0/conflict_normalized.py`, docs/architecture_v0.md
Sec 10) is run through the **exact same** frozen protocol as the original
Phase-1 screen and the CellV0.2 follow-up -- same 7 datasets, same seeded
60/20/20 (or official 60k/10k) splits, same {25%, 100%} conditions, same
seeds 0/1/2, same AdamW settings (lr=1e-2, wd=0), same best-validation
checkpoint restore and early stopping, same 15000-step cap.

Nothing about preprocessing, hyperparameters, datasets, or fractions changes.
This adds **42 CellV0.3 runs** (7 x 2 x 3). The CellV0.1, matched-MLP and
CellV0.2 arms are NOT re-run: reuse the already-recorded `paper_a_phase1` and
`paper_a_phase1_cellv02` records for comparison (`summarize_v03.py`).

CellV0.3's hidden width is fitted to the same per-dataset parameter budget as
CellV0.1 (`PARAM_BUDGET` in `datasets.py`); because its parameterization is
identical to CellV0.2's, the resulting hidden-cell count matches CellV0.2's
-- that is reported, not equalized against CellV0.1.

Usage:
    python experiments/paper_a/run_phase1_v03.py
    python experiments/paper_a/run_phase1_v03.py --datasets digits diabetes --seeds 0
    python experiments/paper_a/run_phase1_v03.py --device cpu --threads 4
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
from experiments.paper_a.models import CELLV03_FAMILY  # noqa: E402

CELLV03_EXPERIMENT_ID = "paper_a_phase1_cellv03"
TRAIN_FRACTIONS: tuple[float, ...] = (0.25, 1.0)
SEEDS: tuple[int, ...] = (0, 1, 2)
RESULTS_RAW = _HERE / "results" / "raw"
RESULTS_PROCESSED = _HERE / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(BENCHMARK_DATASETS))
    parser.add_argument("--fractions", type=float, nargs="+", default=list(TRAIN_FRACTIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="torch CPU thread cap (set when running shards in parallel)",
    )
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--results-dir", default=str(RESULTS_RAW))
    parser.add_argument(
        "--summary-out", default=str(RESULTS_PROCESSED / "phase1_cellv03_runs.json")
    )
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
        (ds, frac, seed)
        for ds in datasets
        for frac in args.fractions
        for seed in args.seeds
    ]
    print(f"Running {len(grid)} CellV0.3 runs on device={args.device or 'auto'}...")

    all_results = []
    for i, (ds, frac, seed) in enumerate(grid, 1):
        t = time.time()
        r = run_one(
            ds,
            frac,
            CELLV03_FAMILY,
            seed,
            device=device,
            results_dir=args.results_dir,
            experiment_id=CELLV03_EXPERIMENT_ID,
            max_steps=args.max_steps,
        )
        all_results.append(r)
        flags = []
        if r["eff_diverged"]:
            flags.append("DIVERGED")
        if r["eff_cap_hit"]:
            flags.append("CAP_HIT")
        print(
            f"[{i:2d}/{len(grid)}] {ds:18s} f={frac:<4} seed={seed} "
            f"{r['headline_metric']}={r['headline_value']:.4f} "
            f"p={r['parameter_count']:>7d} hc={r['sizing']['hidden_cells']:>4d} "
            f"cv={r.get('diag_layer2_precision_cv', float('nan')):.3f} "
            f"beststep={r['eff_best_val_step']:>6d} steps={r['eff_total_steps']:>6d} "
            f"{time.time()-t:6.1f}s {' '.join(flags)}"
        )
        with open(args.summary_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nWrote {len(all_results)} CellV0.3 run summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
