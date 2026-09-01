# Experiment 003 — Cell Ablations

**Status:** Blocked on further Experiment 002 analysis, not on missing
implementation — `BeliefNetwork`, all three aggregation methods, and the
harness (`experiments/002_cell_v0/harness.py`) already exist and are
reusable here directly. See `docs/research_log.md` ("Experiment 002
initial results") for concrete candidate ablations raised by that run:
sweep `hidden_cells`/steps/learning rate before concluding the
uncertainty-calibration direction-flip is intrinsic to the formulas rather
than a scale artifact; investigate whether `support_conflict`'s evidence
collapse correlates with `relevance_logit` being driven very negative.

**Purpose:** Vary number of compartments, integration mechanism,
hidden-state dimensions, compartment sharing, and gating. Determine which
aspects of CellV0 actually matter.

**Prerequisite:** Experiment 002 (CellV0 exists and is trainable).

**Hypotheses tested:** H1. Answers Q5–Q6 in `docs/hypotheses.md`.
