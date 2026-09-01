#!/usr/bin/env python3
"""Benchmark entry point: run train + evaluate across multiple configs and
collect results for comparison (docs/benchmark_plan.md "Compute
efficiency" -- parameter-matched, compute-matched, data-matched, and
wall-clock comparisons all read from the RunRecords this would produce).

Usage: python scripts/benchmark.py --configs configs/regression/*.yaml

Not yet implemented: depends on scripts/train.py and scripts/evaluate.py,
which in turn depend on a model implementation existing (see
docs/experiment_protocol.md, Experiment 001).
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train+evaluate across multiple configs.")
    parser.add_argument(
        "--configs", nargs="+", required=True, help="Paths to YAML configs under configs/."
    )
    args = parser.parse_args()

    print(f"Would benchmark {len(args.configs)} config(s): {args.configs}")
    raise NotImplementedError(
        "Benchmarking depends on scripts/train.py and scripts/evaluate.py, which "
        "depend on a model implementation existing (see docs/experiment_protocol.md, "
        "Experiment 001)."
    )


if __name__ == "__main__":
    main()
