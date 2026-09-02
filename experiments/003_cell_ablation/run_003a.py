#!/usr/bin/env python3
"""Experiment 003A -- verify the small performance signal (docs/research_log.md
"Experiment 002 initial results"). A narrowed re-run of Experiment 002's
grid, not a new architecture/training change: only the two datasets where
Experiment 002 showed something worth checking (r1_nonlinear's tiny
`reliability`/`precision` edge; r2_interaction where `support_conflict`
diverged) plus c2_interaction (`precision`'s largest accuracy edge), only
`reliability`/`precision` (not `support_conflict` -- Experiment 002 found
it the clear pathological outlier: evidence collapse, huge/noisy
"ambiguous vs. clear" gaps), and 10 seeds instead of 3, to tell a real
effect from seed noise. Reuses `experiments/002_cell_v0/harness.py`
directly -- no new training logic, just a narrower grid plus a
delta-vs.-MLP significance summary.

Usage:
    python experiments/003_cell_ablation/run_003a.py
    python experiments/003_cell_ablation/run_003a.py --seeds 0 1 2 3 4 5 6 7 8 9 --steps 2000
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[1] / "002_cell_v0"
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

from harness import run_experiment  # noqa: E402

DATASETS: tuple[str, ...] = ("r1_nonlinear", "r2_interaction", "c2_interaction")
MODELS: tuple[str, ...] = ("mlp", "reliability", "precision")


def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Two-sided p-value via a normal approximation to Welch's t-test.
    scipy is not a declared project dependency (pyproject.toml), so this is
    a deliberately simple stand-in -- treat the result as a rough
    significance indicator at n~10 per group, not an exact p-value."""
    mean_a, mean_b = statistics.mean(a), statistics.mean(b)
    var_a = statistics.variance(a) if len(a) > 1 else 0.0
    var_b = statistics.variance(b) if len(b) > 1 else 0.0
    se = math.sqrt(var_a / len(a) + var_b / len(b))
    if se == 0:
        return float("nan"), float("nan")
    t = (mean_a - mean_b) / se
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--hidden-cells", type=int, default=16)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--results-dir", default="results/raw")
    args = parser.parse_args()

    all_results = []
    for dataset in DATASETS:
        for model_name in MODELS:
            for seed in args.seeds:
                result = run_experiment(
                    dataset=dataset,
                    model_name=model_name,
                    seed=seed,
                    hidden_cells=args.hidden_cells,
                    steps=args.steps,
                    lr=args.lr,
                    results_dir=args.results_dir,
                )
                all_results.append(result)
                metric_key = "r2" if "r2" in result else "accuracy"
                print(
                    f"{dataset:16s} {model_name:12s} seed={seed} "
                    f"{metric_key}={result.get(metric_key):.4f}"
                )

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_results:
        grouped[(r["dataset"], r["model"])].append(r)

    print("\n=== Summary (mean +/- std across seeds) ===")
    for (dataset, model_name), runs in grouped.items():
        metric_key = "r2" if "r2" in runs[0] else "accuracy"
        values = [r[metric_key] for r in runs]
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        print(
            f"{dataset:16s} {model_name:12s} {metric_key}={mean:.4f}+/-{std:.4f} "
            f"(n={len(values)})"
        )

    print("\n=== reliability/precision vs. mlp (Welch's t-test, approx p-value) ===")
    for dataset in DATASETS:
        baseline_runs = grouped[(dataset, "mlp")]
        metric_key = "r2" if "r2" in baseline_runs[0] else "accuracy"
        baseline_values = [r[metric_key] for r in baseline_runs]
        baseline_var = statistics.variance(baseline_values) if len(baseline_values) > 1 else 0.0
        for model_name in ("reliability", "precision"):
            values = [r[metric_key] for r in grouped[(dataset, model_name)]]
            delta = statistics.mean(values) - statistics.mean(baseline_values)
            t, p = _welch_t_test(values, baseline_values)
            var_ratio = (
                statistics.variance(values) / baseline_var if baseline_var > 0 else float("nan")
            )
            flag = "**" if p < 0.05 else ("*" if p < 0.10 else "")
            print(
                f"{dataset:16s} {model_name:10s} delta={delta:+.4f} t={t:+.2f} p~={p:.3f} "
                f"{flag:2s} variance_ratio(vs mlp)={var_ratio:.2f}"
            )


if __name__ == "__main__":
    main()
