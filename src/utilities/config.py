"""Experiment configuration loading, validation, and content-hashing.

Every run in this project is defined by a YAML config and identified by a
hash of that config's contents plus a timestamp -- never by a hand-edited
checkpoint name like `final_model_v2_REAL_fixed.pt`. See
docs/experiment_protocol.md "Config conventions" and "What every run must
record".
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass
class ExperimentConfig:
    """Required metadata for any training run in this project.

    Fields mirror docs/experiment_protocol.md. `extra` holds architecture-
    or task-specific hyperparameters that don't warrant their own top-level
    field (e.g. compartment count, cluster topology) -- keeping this
    dataclass stable means the run-logging/checkpointing code never needs to
    change when a new architecture variant adds hyperparameters.
    """

    experiment_id: str  # e.g. "002_cell_v0"
    architecture: str  # e.g. "mlp_baseline", "cellv0" -- a name, not an implementation detail
    dataset: str  # dataset identifier, e.g. "synthetic_regression_linear"
    seed: int
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 32
    max_steps: int | None = None
    max_epochs: int | None = None
    refinement_iterations: int | None = None  # None if not applicable to this architecture
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML config into an `ExperimentConfig`. Keys not matching a
    declared field are folded into `extra` rather than rejected."""
    path = Path(path)
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    known_field_names = {f.name for f in dataclasses.fields(ExperimentConfig)}
    declared = {k: v for k, v in raw.items() if k in known_field_names}
    overflow = {k: v for k, v in raw.items() if k not in known_field_names}

    declared_extra = declared.pop("extra", {}) or {}
    declared["extra"] = {**declared_extra, **overflow}

    return ExperimentConfig(**declared)


def config_hash(config: ExperimentConfig, length: int = 10) -> str:
    """Deterministic short hash identifying a config's contents. Two configs
    with identical field values (including `extra`) hash identically
    regardless of key order, so this is safe to use as (part of) a run ID."""
    canonical = json.dumps(config.to_dict(), sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


def make_run_id(config: ExperimentConfig) -> str:
    """`<experiment_id>_<config_hash>_<UTC timestamp>` -- unique, sortable,
    and never hand-edited. Use this, not a manually chosen name, for
    checkpoint and log filenames."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.experiment_id}_{config_hash(config)}_{timestamp}"
