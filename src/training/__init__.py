"""Generic, architecture-agnostic training infrastructure: optimizer
construction, a supervised train/eval loop, checkpointing, and run-metadata
logging. Nothing here should ever import from `src.models.architecture_v0`
or contain architecture-specific logic."""
