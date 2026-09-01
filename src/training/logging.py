"""Run-metadata logging.

Every run must record what docs/experiment_protocol.md calls out under
"What every run must record". `RunRecord` is that schema; `write_run_record`
persists it as JSON alongside the run's checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any


def get_git_commit() -> str | None:
    """Returns the current HEAD commit hash, or None outside a git repo /
    before the first commit -- callers should treat that as "unrecorded",
    not fail the run."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclasses.dataclass
class RunRecord:
    """One row of run provenance. Field set is fixed by
    docs/experiment_protocol.md -- do not add ad hoc fields here; put
    architecture-specific detail inside `config` instead."""

    run_id: str
    experiment_id: str
    architecture: str
    config: dict[str, Any]
    parameter_count: int
    dataset: str
    seed: int
    optimizer: str
    learning_rate: float
    steps_completed: int
    examples_or_tokens_seen: int
    refinement_iterations: int | None
    approximate_flops: float | None
    train_wall_clock_seconds: float
    inference_wall_clock_seconds: float | None
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    git_commit: str | None
    checkpoint_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def write_run_record(record: RunRecord, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{record.run_id}.json"
    with out_path.open("w") as f:
        json.dump(record.to_dict(), f, indent=2, sort_keys=True)
    return out_path
