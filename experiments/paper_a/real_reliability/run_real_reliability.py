#!/usr/bin/env python3
"""Paper A Phase 3 Part B -- the real missing-sensor benchmark grid.

    2 datasets  x  5 model families  x  3 seeds   = 30 run_one calls
    {aps, air_quality}
    {plain_mlp, confidence_mlp, confidence_mlp_same_width, cellv0.3, neumiss}
    {0, 1, 2}

Each `run_one` internally sweeps the shared LR grid {1e-3, 3e-3, 1e-2} (and the
NeuMiss depth grid {1,3,5}) on train/validation only. Plus the optional
`HistGradientBoosting` reference (2 datasets x 3 seeds).

Usage:
    python experiments/paper_a/real_reliability/run_real_reliability.py
    python experiments/paper_a/real_reliability/run_real_reliability.py --datasets air_quality
    python experiments/paper_a/real_reliability/run_real_reliability.py --no-hgb
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

from experiments.paper_a.real_reliability.datasets import (  # noqa: E402
    REAL_DATASETS,
    dataset_available,
)
from experiments.paper_a.real_reliability.harness import (  # noqa: E402
    SEEDS,
    run_hgb_reference,
    run_one,
)
from experiments.paper_a.real_reliability.models import (  # noqa: E402
    MODEL_FAMILIES,
    NeuMissIntegrationError,
)

RESULTS_RAW = _HERE / "results" / "raw"
RESULTS_PROCESSED = _HERE / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(REAL_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--aps-max-steps", type=int, default=8_000)
    parser.add_argument("--air-max-steps", type=int, default=4_000)
    parser.add_argument("--no-hgb", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_RAW))
    parser.add_argument(
        "--summary-out", default=str(RESULTS_PROCESSED / "real_reliability_runs.json")
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else None
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    datasets = []
    for name in args.datasets:
        if not dataset_available(name):
            print(f"SKIP {name}: UCI archive not reachable in this environment.")
            continue
        datasets.append(name)

    grid = [(ds, fam, seed) for ds in datasets for fam in args.models for seed in args.seeds]
    print(f"Running {len(grid)} Part-B runs (device={args.device or 'auto'})...")

    all_results: list[dict] = []
    for i, (ds, fam, seed) in enumerate(grid, 1):
        t = time.time()
        max_steps = args.aps_max_steps if ds == "aps" else args.air_max_steps
        try:
            r = run_one(
                ds, fam, seed, device=device, results_dir=args.results_dir, max_steps=max_steps
            )
        except NeuMissIntegrationError as exc:
            print(f"[{i:2d}/{len(grid)}] {ds} {fam} seed={seed}  NEUMISS INTEGRATION FAILED: {exc}")
            all_results.append({"dataset": ds, "family": fam, "seed": seed,
                                "neumiss_integration_error": str(exc)})
            continue
        all_results.append(r)
        ov = r["evaluation"]["overall"]
        primary = r["primary_metric"]
        extra = ""
        if primary == "pr_auc":
            extra = f"cost={ov['official_cost']:.0f} roc={ov['roc_auc']:.3f}"
        flags = [f for f, v in (("DIVERGED", r["diverged"]), ("CAP_HIT", r["cap_hit"])) if v]
        print(
            f"[{i:2d}/{len(grid)}] {ds:12s} {fam:24s} seed={seed} "
            f"{primary}={ov[primary]:.4f} {extra} "
            f"lr={r['selected_lr']:.0e} d={r['neumiss_depth']} "
            f"p={r['parameter_count']:>7d} {time.time()-t:6.1f}s {' '.join(flags)}"
        )
        with open(args.summary_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    if not args.no_hgb:
        print("\nHistGradientBoosting reference...")
        for ds in datasets:
            for seed in args.seeds:
                t = time.time()
                h = run_hgb_reference(ds, seed, results_dir=args.results_dir)
                ov = h["evaluation"]["overall"]
                p = h["primary_metric"]
                print(f"  HGB {ds:12s} seed={seed} {p}={ov[p]:.4f} {time.time()-t:5.1f}s")
                all_results.append(h)
                with open(args.summary_out, "w") as f:
                    json.dump(all_results, f, indent=2, default=str)

    print(f"\nWrote {len(all_results)} summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
