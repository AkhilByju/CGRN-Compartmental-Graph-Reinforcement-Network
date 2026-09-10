#!/usr/bin/env python3
"""Paper A Phase 2 -- the reliability / corruption benchmark grid.

Frozen grid (Phase-2 task Sec 3 / Sec 6 / Sec 8):

    4 datasets   x  2 corruption families  x  4 model families  x  3 seeds
    {mnist, fashion_mnist, digits, california_housing}
    {missing, gaussian}
    {plain_mlp, confidence_mlp, reliability_gated_mlp, cellv0.3}
    {0, 1, 2}
    = 96 training runs

Each trained checkpoint is evaluated at every severity level for its corruption
family (never one model per severity). CellV0.1 / CellV0.2 / CellV0.3 and every
Phase-1 result are untouched -- this only *composes* the frozen Phase-1 split /
preprocessing / parameter budgets with the deterministic corruption layer.

Usage:
    python experiments/paper_a/reliability/run_reliability.py
    python experiments/paper_a/reliability/run_reliability.py --datasets digits --seeds 0
    python experiments/paper_a/reliability/run_reliability.py --device cpu --threads 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
for p in (str(_REPO_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from experiments.paper_a.datasets import dataset_available  # noqa: E402
from experiments.paper_a.reliability.corruption import CORRUPTION_FAMILIES  # noqa: E402
from experiments.paper_a.reliability.harness import (  # noqa: E402
    RELIABILITY_DATASETS,
    SEEDS,
    run_one,
)
from experiments.paper_a.reliability.models import MODEL_FAMILIES  # noqa: E402

RESULTS_RAW = _HERE / "results" / "raw"
RESULTS_PROCESSED = _HERE / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(RELIABILITY_DATASETS))
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTION_FAMILIES))
    parser.add_argument("--models", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    parser.add_argument(
        "--threads", type=int, default=None,
        help="torch CPU thread cap (set when running shards in parallel)",
    )
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--results-dir", default=str(RESULTS_RAW))
    parser.add_argument(
        "--summary-out", default=str(RESULTS_PROCESSED / "reliability_runs.json")
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
        (ds, cf, fam, seed)
        for ds in datasets
        for cf in args.corruptions
        for fam in args.models
        for seed in args.seeds
    ]
    print(f"Running {len(grid)} reliability runs on device={args.device or 'auto'}...")

    all_results: list[dict] = []
    for i, (ds, cf, fam, seed) in enumerate(grid, 1):
        t = time.time()
        r = run_one(
            ds, cf, fam, seed,
            device=device, results_dir=args.results_dir, max_steps=args.max_steps,
        )
        all_results.append(r)
        sw = r["sweep"]
        primary = r["primary_metric"]
        clean = sw["severities"][0]["metrics"][primary]
        worst = sw["severities"][-1]["metrics"][primary]
        flags = []
        if r["diverged"]:
            flags.append("DIVERGED")
        if r["cap_hit"]:
            flags.append("CAP_HIT")
        if fam == "cellv0.3":
            for s in r["sweep"]["severities"]:
                d = s["belief_diag"] or {}
                if min(
                    d.get("diag_layer1_precision_min", 1.0),
                    d.get("diag_layer2_precision_min", 1.0),
                ) < 1e-6:
                    flags.append(f"tiny_pi@{s['severity']}")
        print(
            f"[{i:2d}/{len(grid)}] {ds:18s} {cf:8s} {fam:22s} seed={seed} "
            f"{primary} {clean:.4f}->{worst:.4f} "
            f"AUC={sw['corruption_auc'][primary]:.4f} OODdrop={sw['ood_drop'][primary]:+.4f} "
            f"p={r['parameter_count']:>7d} beststep={r['efficiency']['best_val_step']:>6d} "
            f"{time.time()-t:6.1f}s {' '.join(flags)}"
        )
        with open(args.summary_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nWrote {len(all_results)} reliability run summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
