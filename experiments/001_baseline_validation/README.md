# Experiment 001 — Baseline Validation

**Status:** Not started.

**Purpose:** Implement the tiny MLP, tiny recurrent (GRU/LSTM), and tiny
Transformer baselines (`src/models/baselines/`), plus the common
training/evaluation framework they'll share with the future novel
architecture. Establish trusted baselines before any architecture-specific
claim is possible.

**Prerequisite:** None — this is the first experiment.

**Depends on:** `src/training/`, `src/evaluation/` (generic infra, already
scaffolded), Level 0 sanity-check tasks (`src/data/synthetic/`, not yet
implemented).

**Hypotheses tested:** None directly — this experiment exists to make later
hypothesis tests trustworthy.
