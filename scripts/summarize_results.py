#!/usr/bin/env python3
"""Aggregate RunRecord JSON files under results/raw/ into a single table
under results/processed/.

Usage: python scripts/summarize_results.py [--raw-dir results/raw] \
    [--out results/processed/summary.csv]

Unlike train.py/evaluate.py/benchmark.py, this script only depends on the
RunRecord schema (src/training/logging.py), not on any model
implementation, so it is fully functional today -- it will just produce an
empty summary until the first run exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_run_records(raw_dir: Path) -> pd.DataFrame:
    records = []
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r") as f:
            records.append(json.load(f))
    if not records:
        return pd.DataFrame()
    return pd.json_normalize(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize RunRecord JSON files into a CSV table.")
    parser.add_argument("--raw-dir", default="results/raw", type=Path)
    parser.add_argument("--out", default="results/processed/summary.csv", type=Path)
    args = parser.parse_args()

    df = load_run_records(args.raw_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    if df.empty:
        print(f"No run records found under {args.raw_dir}. Wrote empty summary to {args.out}.")
    else:
        print(f"Summarized {len(df)} run(s) from {args.raw_dir} into {args.out}.")


if __name__ == "__main__":
    main()
