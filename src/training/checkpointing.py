"""Checkpoint naming and I/O, keyed by run ID.

Checkpoint filenames are always derived from a run ID (see
`src.utilities.config.make_run_id`) -- never hand-edited
(docs/experiment_protocol.md "Config conventions").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def checkpoint_path(run_id: str, checkpoint_dir: str | Path, step: int | None = None) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    name = f"{run_id}.pt" if step is None else f"{run_id}_step{step}.pt"
    return checkpoint_dir / name


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    run_id: str,
    checkpoint_dir: str | Path,
    step: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Saves model (and optionally optimizer) state under a run-ID-derived
    filename. `extra` can carry anything else worth restoring (e.g. RNG
    state, iteration count) but should not carry the config -- that belongs
    in the corresponding `RunRecord` (see `src.training.logging`)."""
    path = checkpoint_path(run_id, checkpoint_dir, step)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model_state_dict": model.state_dict(), "run_id": run_id}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | None = None) -> dict[str, Any]:
    """Loads a checkpoint written by `save_checkpoint`. `weights_only=False`
    is safe here because checkpoints are produced locally by this project's
    own training code, not loaded from untrusted third-party sources."""
    return torch.load(Path(path), map_location=map_location, weights_only=False)
