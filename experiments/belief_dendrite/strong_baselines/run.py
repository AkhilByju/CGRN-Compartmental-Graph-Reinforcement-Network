"""Driver for the strong-baselines benchmark (spec Sec 8/16):
1. MPS speed preflight for all seven families (also picks the common batch
   size, spec Sec 7/8) -- saved to `results/processed/preflight.json`.
2. The primary sweep: `len(families) x len(seeds)` cells, each an LR-grid
   training run + full severity-grid evaluation + (where applicable) Sec 12
   interventions -- one `RunRecord` per cell under `results/raw/`.

Usage:
    python -m experiments.belief_dendrite.strong_baselines.run
    # smoke test:
    python -m experiments.belief_dendrite.strong_baselines.run --max-epochs 2 --patience-epochs 1
    python -m experiments.belief_dendrite.strong_baselines.run --skip-preflight --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.belief_dendrite.strong_baselines.device_utils import (  # noqa: E402
    require_mps,
    resolve_device,
)
from experiments.belief_dendrite.strong_baselines.harness import SEEDS, run_one  # noqa: E402
from experiments.belief_dendrite.strong_baselines.models import MODEL_FAMILIES  # noqa: E402
from experiments.belief_dendrite.strong_baselines.preflight import run_preflight  # noqa: E402
from experiments.belief_dendrite.strong_baselines.training import (  # noqa: E402
    MAX_EPOCHS,
    PATIENCE_EPOCHS,
)

_BASE_DIR = _REPO_ROOT / "experiments" / "belief_dendrite" / "strong_baselines"
DEFAULT_RESULTS_DIR = _BASE_DIR / "results" / "raw"
DEFAULT_PROCESSED_DIR = _BASE_DIR / "results" / "processed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience-epochs", type=int, default=PATIENCE_EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--families", nargs="+", default=list(MODEL_FAMILIES))
    parser.add_argument(
        "--lr-grid",
        nargs="+",
        type=float,
        default=None,
        help="override the spec Sec 7 LR grid {1e-3, 3e-3}, e.g. --lr-grid 0.001 for a single LR",
    )
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--allow-cpu", action="store_true", help="debug only -- spec requires MPS")
    args = parser.parse_args()

    device_report = resolve_device(prefer="cpu" if args.allow_cpu else None)
    print(f"device report: {device_report.to_dict()}")
    if not args.allow_cpu:
        require_mps(device_report)

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    processed_dir = Path(args.processed_dir) if args.processed_dir else DEFAULT_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_preflight:
        batch_size = args.batch_size or 256
        print(f"skipping preflight; batch_size={batch_size}")
    else:
        pf = run_preflight(device_report.device, seed=0, families=tuple(args.families))
        (processed_dir / "preflight.json").write_text(json.dumps(pf.to_dict(), indent=2))
        batch_size = pf.batch_size
        print(f"preflight complete; batch_size={batch_size}")
        for r in pf.results:
            print(
                f"  {r.family:28s} params={r.parameter_count:7d} ms/step={r.ms_per_step:8.2f} "
                f"ex/s={r.examples_per_sec:9.1f} oom={r.oom}"
            )
        for flag in pf.slowdown_flags:
            print(f"  SLOWDOWN FLAG: {flag}")

    lr_grid = tuple(args.lr_grid) if args.lr_grid else None
    cells = [(fam, seed) for fam in args.families for seed in args.seeds]
    n_lrs = len(lr_grid) if lr_grid else 2
    print(f"{len(cells)} run cells queued (families x seeds; each cell tries {n_lrs} LR(s))")

    t_start = time.perf_counter()
    for i, (family, seed) in enumerate(cells, start=1):
        t0 = time.perf_counter()
        kwargs = {} if lr_grid is None else {"lr_grid": lr_grid}
        result = run_one(
            family,
            seed,
            device=device_report.device,
            results_dir=results_dir,
            batch_size=batch_size,
            max_epochs=args.max_epochs,
            patience_epochs=args.patience_epochs,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0
        clean_acc = result["sweep"]["severities"][0]["metrics"]["accuracy"]
        print(
            f"[{i}/{len(cells)}] {family:28s} seed={seed} lr={result['chosen_lr']:<6} "
            f"params={result['parameter_count']:7d} clean_acc={clean_acc:.4f} "
            f"diverged={result['diverged']} ({elapsed:.1f}s)"
        )

    print(f"done in {time.perf_counter() - t_start:.1f}s total")


if __name__ == "__main__":
    main()
