# Experiment 002 — CellV0

**Status:** Blocked — waiting on Architecture Specification V0.1
(`docs/architecture_v0.md` Sec 7).

**Purpose:** Introduce the compartmental cell alone. No graph, no
stochasticity, no dynamic connections, no novel learning rule. Determine
whether the richer computational cell itself deserves to exist.

**Prerequisite:** Experiment 001 (trusted baselines). CellV0's math must be
specified in `docs/architecture_v0.md` before `src/models/architecture_v0/cell.py`,
`compartment.py`, and `integration.py` can be implemented.

**Hypotheses tested:** H1 (Computational Unit Hypothesis). Answers
Q1–Q4 in `docs/hypotheses.md`.
