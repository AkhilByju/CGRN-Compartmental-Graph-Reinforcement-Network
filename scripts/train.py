#!/usr/bin/env python3
"""Training entry point.

Usage: python scripts/train.py --config configs/<family>/<name>.yaml

Currently loads and validates the config, sets the seed, and selects the
device -- then stops. Model construction and wiring up
`src.training.trainer` are `NotImplementedError` until at least one model
exists to train (see docs/experiment_protocol.md, Experiment 001).
"""

from __future__ import annotations

import argparse

from src.utilities.config import load_config, make_run_id
from src.utilities.device import get_device
from src.utilities.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a model from a config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config under configs/.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    device = get_device()
    run_id = make_run_id(config)

    print(
        f"Loaded config for experiment '{config.experiment_id}' "
        f"(architecture={config.architecture})"
    )
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")

    raise NotImplementedError(
        "No model implementations exist yet (see docs/experiment_protocol.md, "
        "Experiment 001). Config loading, seeding, and device selection above "
        "are functional; wire up model construction and src.training.trainer "
        "once a model exists."
    )


if __name__ == "__main__":
    main()
