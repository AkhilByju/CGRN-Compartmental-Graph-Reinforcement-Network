# Experiment 003 — Cell Ablations

**Status:** Scripts written for a narrowed three-part follow-up to
Experiment 002 (003A/003B/003C below); not yet run. This replaces running
the full ablation grid (`hidden_cells`/steps/learning-rate sweep across all
six datasets and all three aggregation methods) with a targeted set of
runs chosen to answer the two open questions Experiment 002 actually
raised: is the small `reliability`/`precision` edge on the harder datasets
real, and is the uncertainty-calibration direction-flip fixable or
intrinsic. `support_conflict` is dropped from all three parts —
Experiment 002 found it the clear pathological outlier (evidence collapse;
huge, noisy "ambiguous vs. clear" gaps) and re-litigating it isn't what
either open question is about.

**Prerequisite:** Experiment 002 (CellV0 exists and is trainable;
`BeliefNetwork`, all three aggregation methods, and
`experiments/002_cell_v0/harness.py` already exist and are reused
directly here).

## 003A — Verify the small performance signal

Re-runs Experiment 002's grid narrowed to the two datasets/model pairs
that showed something worth checking (`r1_nonlinear`'s tiny
`reliability`/`precision` edge, `r2_interaction`, `c2_interaction`'s
`precision` accuracy edge), `mlp`/`reliability`/`precision` only, 10 seeds
instead of 3 — enough to tell a real effect from seed noise via a
per-comparison Welch's t-test. No new training logic; reuses
`experiments/002_cell_v0/harness.py` directly
(`experiments/003_cell_ablation/run_003a.py`).

```bash
python experiments/003_cell_ablation/run_003a.py
```

## 003B — Ground-truth uncertainty

Experiment 002's "ambiguous vs. clear" calibration check only ever had a
*proxy* for which inputs are uncertain (margin/extrapolation). 003B
introduces datasets where the true per-example noise level is known
exactly (`src/data/synthetic/uncertainty.py`,
`u1_heteroscedastic_1d`/`u2_heteroscedastic_interaction` — same mean
functions as `r1_nonlinear`/`r2_interaction`, heteroscedastic noise on
top), and compares `reliability`/`precision`'s unsupervised
`BeliefCell.uncertainty` against a conventional MLP-plus-uncertainty-head
baseline (`MLPWithUncertaintyHead`, `src/models/baselines/mlp.py`, trained
with Gaussian NLL) on how well predicted uncertainty tracks that known
value (`src/evaluation/calibration.py`). If the conventional baseline
calibrates as well or better, carrying `(evidence, uncertainty)` through
every neuron isn't buying calibration over the standard alternative.

```bash
python experiments/003_cell_ablation/run_003b.py
```

## 003C — Add uncertainty supervision

Adds an explicit, switchable calibration term to `reliability`/
`precision`'s training loss —
`loss = prediction_loss + lambda * MSE(predicted_uncertainty, true_std)`
— and sweeps `lambda in {0, 0.01, 0.1, 1.0}` on the same ground-truth
datasets as 003B (`lambda=0` reproduces 003B's plain-training numbers
exactly). `integration.py`'s three aggregation formulas are not touched —
only this auxiliary loss term is new (`docs/research_log.md`'s own
follow-up after Experiment 002 called for exactly this: "add an explicit
calibration signal ... as a deliberate, isolated change"). Asks whether
`u` can be forced to acquire its intended meaning without degrading
prediction quality.

```bash
python experiments/003_cell_ablation/run_003c.py
```

Shared harness for 003B/003C: `experiments/003_cell_ablation/uncertainty_harness.py`.

## 003D — Evidence/uncertainty intervention

**Status: done** (`docs/research_log.md` "Experiment 003D results").
Experiment 002 found `reliability`/`precision` track a plain MLP baseline
closely on predictive performance (Finding 1) while their uncertainty
channel doesn't calibrate to true ambiguity (Finding 2) — raising the
question of whether `(evidence, uncertainty)` are doing any real
computational work at all, or whether the network learned to route around
them ("decorative state"). 003D answers this directly: trains a `precision`
`BeliefNetwork` on Experiment 002's six datasets, then at inference time
perturbs the evidence/uncertainty the trained `layer1` hands to `layer2`
(`src/evaluation/intervention.py::perturb_belief` — `evidence_ones`,
`uncertainty_ones`, `shuffle_evidence_uncertainty`,
`uncertainty_random`) and measures the resulting change in test R²/accuracy
relative to the unperturbed baseline. `layer1`/`layer2`/`readout` are called
directly from the harness (`evidence_uncertainty_harness.py`); no changes to
`belief_network.py` or `integration.py`.

```bash
python experiments/003_cell_ablation/run_003d.py
python experiments/003_cell_ablation/run_003d.py --seeds 0 1 2 3 4 5 6 7 8 9 --steps 2000
```

**Hypotheses tested:** H1. Answers Q5–Q6 in `docs/hypotheses.md`.
