# Experiment 002 — CellV0

**Status:** Initial results in — mixed/negative on uncertainty calibration,
inconclusive-to-negative on predictive performance. See
`docs/research_log.md` (2026-09-01, "Experiment 002 initial results") for
the full write-up. Not yet a final verdict: no calibration loss has been
tried, and only 3 seeds / one hidden-cell width have been run.

**Purpose:** Introduce the compartmental cell alone. No graph, no
stochasticity, no dynamic connections, no novel learning rule. Determine
whether the richer computational cell itself deserves to exist.

**What was run:** `harness.py`/`run.py`/`run_all.py` train an MLP baseline
and a `BeliefNetwork` (`src/models/architecture_v0/belief_network.py`,
`Input -> BeliefLayer -> BeliefLayer -> linear readout`) under each of the
three aggregation methods (`reliability`, `support_conflict`, `precision`)
on the R0/R1/R2 regression and C0/C1/C2 classification datasets
(`src/data/synthetic/`), 3 seeds each, ~parameter-matched
(`match_hidden_dim`). Test performance, wall-clock time, parameter count,
evidence/uncertainty distributions, and an "ambiguous vs. clear" test
(near-decision-boundary vs. far, for classification; extrapolation vs.
in-distribution, for regression) were recorded per run
(`results/raw/*.json`, gitignored — see `docs/research_log.md` for the
numbers).

**Reproduce:**
```bash
python experiments/002_cell_v0/run_all.py --seeds 0 1 2 --steps 2000
```

**Prerequisite:** Experiment 001 (trusted baselines) has not been done as
a separate step — the MLP baseline used here (`src/models/baselines/mlp.py`)
was implemented directly for this experiment instead. CellV0's *state* and
*aggregation candidates* are specified (`docs/architecture_v0.md` Sec 1);
this experiment is exactly the comparison that section says should decide
among them.

**Hypotheses tested:** H1 (Computational Unit Hypothesis). Answers
Q1–Q4, Q6 in `docs/hypotheses.md`.
