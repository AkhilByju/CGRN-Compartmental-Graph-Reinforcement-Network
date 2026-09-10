#!/usr/bin/env python3
"""Paper A Phase 3 — Part A: capacity-stress the Confidence MLP.

Phase 2 parameter-matched every baseline to CellV0.3's actual trainable count.
Because the Confidence MLP's input is ``concat(x, c)`` (twice as wide), matching
parameters forced it to a *narrower* hidden layer than CellV0.3. Part A removes
that confound with **one** new baseline, ``confidence_mlp_same_width``:

* identical architecture and activation to the Phase-2 Confidence MLP,
* hidden width set to **CellV0.3's exact hidden width** on each dataset
  (not parameter-matched -> more parameters than CellV0.3, intentionally; it
  is an over-powered capacity control and is **not** shrunk or tuned
  differently),

run through the **exact frozen Phase-2 protocol** (same train corruption
distribution, validation setup, test severity grids, corruption replicas,
preprocessing, optimizer, LR, early stopping, checkpoint selection) on

    {mnist, fashion_mnist, digits} x {missing, gaussian} x seeds {0, 1, 2}
    = 18 runs

CellV0.3's recorded Phase-2 results are **not** regenerated — the combined
summarizer (`experiments/paper_a/publication_validation.py`) reads them back.

Usage:
    python experiments/paper_a/capacity_stress.py
    python experiments/paper_a/capacity_stress.py --datasets digits --seeds 0
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

from experiments.paper_a.reliability.corruption import CORRUPTION_FAMILIES  # noqa: E402
from experiments.paper_a.reliability.harness import run_one  # noqa: E402
from experiments.paper_a.reliability.models import CONFIDENCE_MLP_SAME_WIDTH  # noqa: E402

CAPACITY_EXPERIMENT_ID = "paper_a_capacity_stress"
CAPACITY_DATASETS: tuple[str, ...] = ("mnist", "fashion_mnist", "digits")
SEEDS: tuple[int, ...] = (0, 1, 2)
RESULTS_RAW = _REPO_ROOT / "experiments" / "paper_a" / "reliability" / "results" / "raw"
RESULTS_PROCESSED = _REPO_ROOT / "experiments" / "paper_a" / "reliability" / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(CAPACITY_DATASETS))
    parser.add_argument("--corruptions", nargs="+", default=list(CORRUPTION_FAMILIES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=15_000)
    parser.add_argument("--results-dir", default=str(RESULTS_RAW))
    parser.add_argument(
        "--summary-out", default=str(RESULTS_PROCESSED / "capacity_stress_runs.json")
    )
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device) if args.device else None
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    grid = [
        (ds, cf, seed)
        for ds in args.datasets
        for cf in args.corruptions
        for seed in args.seeds
    ]
    print(f"Running {len(grid)} capacity-stress runs (device={args.device or 'auto'})...")

    all_results = []
    for i, (ds, cf, seed) in enumerate(grid, 1):
        t = time.time()
        r = run_one(
            ds, cf, CONFIDENCE_MLP_SAME_WIDTH, seed,
            device=device, results_dir=args.results_dir,
            experiment_id=CAPACITY_EXPERIMENT_ID, max_steps=args.max_steps,
        )
        all_results.append(r)
        sw = r["sweep"]
        primary = r["primary_metric"]
        ratio = r["sizing"].get("param_ratio_vs_cellv03", float("nan"))
        flags = []
        if r["diverged"]:
            flags.append("DIVERGED")
        if r["cap_hit"]:
            flags.append("CAP_HIT")
        print(
            f"[{i:2d}/{len(grid)}] {ds:14s} {cf:8s} seed={seed} "
            f"{primary} {sw['severities'][0]['metrics'][primary]:.4f}"
            f"->{sw['severities'][-1]['metrics'][primary]:.4f} "
            f"AUC={sw['corruption_auc'][primary]:.4f} "
            f"p={r['parameter_count']:>7d} (x{ratio:.2f} CellV0.3) "
            f"hidden={r['hidden_size']} {time.time()-t:5.1f}s {' '.join(flags)}"
        )
        with open(args.summary_out, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    print(f"\nWrote {len(all_results)} capacity-stress summaries to {args.summary_out}")


if __name__ == "__main__":
    main()
