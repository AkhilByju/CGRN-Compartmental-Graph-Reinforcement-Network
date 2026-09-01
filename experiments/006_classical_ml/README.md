# Experiment 006 — Classical Regression and Classification

**Status:** Blocked — depends on Experiment 001 (baselines) and at least
Experiment 002 (a working CellV0-based model) for the architecture to be
compared. Note this experiment does not strictly require clusters/recurrence
(003/004/005) and could run with whatever level of ArchitectureV0 exists at
the time.

**Purpose:** Evaluate the architecture broadly on Track A (regression) and
Track B (classification) against linear/logistic regression, random forest,
gradient boosting/XGBoost, and MLP. Characterize the architecture as a
general machine-learning model — the goal is not to beat boosted trees, it's
to find where the architecture works. See `docs/benchmark_plan.md` Tracks A
and B.

**Extra dependency:** the optional `classical-ml` dependency group
(`pip install -e ".[classical-ml]"`) for XGBoost.

**Hypotheses tested:** H5 (via the sample-efficiency sweep), general
characterization (not a specific H).
