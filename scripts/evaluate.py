#!/usr/bin/env python3
"""Evaluation entry point.

Usage: python scripts/evaluate.py --config configs/<family>/<name>.yaml \
    --checkpoint <path to a checkpoint from scripts/train.py>

Currently loads the config, selects the device, and loads the checkpoint
dict -- then stops. Wiring this up to an actual model +
`src.training.trainer.evaluate_loop` is `NotImplementedError` until a model
implementation exists (see docs/experiment_protocol.md, Experiment 001).
"""

from __future__ import annotations

import argparse

from src.training.checkpointing import load_checkpoint
from src.utilities.config import load_config
from src.utilities.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model from a config + checkpoint."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config under configs/.")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a checkpoint produced by scripts/train.py."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))

    print(f"Loaded config for experiment '{config.experiment_id}'")
    print(f"Loaded checkpoint for run '{checkpoint.get('run_id')}'")

    raise NotImplementedError(
        "No model implementations exist yet (see docs/experiment_protocol.md, "
        "Experiment 001). Wire up model construction and "
        "src.training.trainer.evaluate_loop once a model exists."
    )


if __name__ == "__main__":
    main()
