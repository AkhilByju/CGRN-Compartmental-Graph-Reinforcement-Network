# Experiment 005 — Recurrent Reuse

**Status:** Blocked — depends on Experiment 002/003 (Experiment 004 is a
scaling study of the same unclustered CellV0 and doesn't gate this one).

**Purpose:** Apply the same computational system (shared parameters)
repeatedly across refinement iterations. Determine whether parameter reuse
plus additional inference computation improves learning or reasoning.
Sweep iteration counts `T = 1, 2, 4, 8, 16, 32` at evaluation time (see
`docs/architecture_v0.md` Sec 2, `docs/benchmark_plan.md` "Repeated
reasoning compute").

**Prerequisite:** A validated CellV0 (`src/models/architecture_v0/dynamics.py`
depends on `cell.py`).

**Hypotheses tested:** H2 (Recurrent Computation Hypothesis).
