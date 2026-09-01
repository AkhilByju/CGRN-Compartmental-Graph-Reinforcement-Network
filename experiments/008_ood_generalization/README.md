# Experiment 008 — OOD Generalization

**Status:** Blocked — depends on Experiment 007.

**Purpose:** Increase problem complexity beyond training (Level 4 of the
benchmark ladder) — e.g. train on 5-node graphs, test on 10/20/40 nodes;
train on 2-step relational inference, test on 5/10 steps. Determine whether
the model learned a reusable procedure or memorized the training
distribution.

**Prerequisite:** Experiment 007's task generators, extended to
out-of-distribution scales (`src/evaluation/generalization.py`, not yet
implemented).

**Hypotheses tested:** H2.
