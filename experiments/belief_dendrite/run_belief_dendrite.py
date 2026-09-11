"""Driver for the frozen Architecture V2 benchmark grid (Sec K/N):
2 datasets x 2 corruption families x 6 model families x 3 seeds = 72 runs.
Each cell is trained once and evaluated across its full severity grid --
never a separate model per severity (Sec N).

Usage:
    python -m experiments.belief_dendrite.run_belief_dendrite
    python -m experiments.belief_dendrite.run_belief_dendrite --max-steps 500  # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.belief_dendrite.corruption import CORRUPTION_FAMILIES  # noqa: E402
from experiments.belief_dendrite.harness import DATASETS, SEEDS, run_one  # noqa: E402
from experiments.belief_dendrite.models import MODEL_FAMILIES  # noqa: E402
from experiments.belief_dendrite.training import DEFAULT_MAX_STEPS  # noqa: E402
from src.utilities.device import get_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--corruption-families", nargs="+", default=list(CORRUPTION_FAMILIES))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"device={device}")

    cells = [
        (d, cf, fam, seed)
        for d in args.datasets
        for cf in args.corruption_families
        for fam in args.families
        for seed in args.seeds
    ]
    print(f"{len(cells)} run cells queued")

    t_start = time.perf_counter()
    for i, (dataset, corruption_family, family, seed) in enumerate(cells, start=1):
        t0 = time.perf_counter()
        result = run_one(
            dataset,
            corruption_family,
            family,
            seed,
            device=device,
            results_dir=args.results_dir,
            max_steps=args.max_steps,
        )
        elapsed = time.perf_counter() - t0
        acc0 = result["sweep"]["severities"][0]["metrics"]["accuracy"]
        print(
            f"[{i}/{len(cells)}] {dataset:14s} {corruption_family:14s} {family:35s} seed={seed} "
            f"params={result['parameter_count']:7d} acc@clean={acc0:.4f} "
            f"steps={result['efficiency']['total_steps']:5d} diverged={result['diverged']} "
            f"({elapsed:.1f}s)"
        )

    print(f"done in {time.perf_counter() - t_start:.1f}s total")


if __name__ == "__main__":
    main()
