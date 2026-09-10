# Paper A — Phase 3 Part B: real missing-sensor benchmarks

CellV0.3 is **permanently frozen**. This is implementation + experiment only —
no CellV0.3 change, no CellV0.4, no equation / confidence / loss tuning.

## Purpose

Phase 2 (synthetic corruption) gave a real but non-uniform signal for CellV0.3.
Part B tests it where the missingness is **real**: two public datasets with
naturally missing sensor measurements, compared against an established
missing-data neural architecture (**NeuMiss**).

## The two frozen datasets

| | APS Failure at Scania Trucks (UCI 421) | UCI Air Quality (UCI 360) |
|---|---|---|
| Task | imbalanced binary classification | sensor regression |
| Split | official 60k/16k train/test; deterministic **stratified** 80/20 train/val per seed; test never touched | **chronological** 60/20/20 (documented drift is part of the setting); no shuffle before splitting |
| Features | all 170 operational features | 8 sensor/environment inputs only (`PT08.S1/S2/S3/S4/S5`, `T`, `RH`, `AH`) — no other GT pollutant column, no `CO(GT)` as input |
| Target | `class` (pos = failure) | `CO(GT)` |
| Missingness | real dataset blanks | `-200` sentinel → missing; rows with a missing *target* dropped, rows with missing *inputs* kept |
| Primary metric | **PR-AUC** (never accuracy) | **R²** |
| Other metrics | ROC-AUC, balanced acc, F1, precision, recall, official `10·FP + 500·FN` cost @ frozen validation threshold, cost/1000 | RMSE, MAE |

Preprocessing: per-feature mean/std from **observed training values only**;
standardize observed; missing standardized value → `0` for the MLP / CellV0.3
inputs (`x_imputed`) and kept `NaN` for NeuMiss (`x_nan`). Reliability
`c_j = 1.0` observed / `1e-3` missing, from the mask alone (never the target).

## Five models (Phase-3 task Part B)

Parameter budgets frozen **before** running: APS ≈ 100k, Air Quality ≈ 50k.
CellV0.3's hidden width is the largest that fits; A/B are parameter-matched to
CellV0.3's actual count within 2%.

| | Model | Input | Sizing |
|---|---|---|---|
| A | Plain MLP | `x_imputed` | param-matched |
| B | Confidence MLP (matched) | `concat(x_imputed, c)` | param-matched |
| C | Confidence MLP (same width) | `concat(x_imputed, c)` | CellV0.3's hidden width; ratio reported, allowed to exceed the budget |
| D | CellV0.3 | belief `(mu=x_imputed, e=c, u=0)` | the target |
| E | **NeuMiss** | `x_nan` (NaN kept, handled natively) | official `marineLM/NeuMiss_sota` package, pinned commit `7902b8d`; `NeuMissBlock` + a `NeuMissMLP(mlp_depth=1)`-style head; depth grid `{1, 3, 5}` chosen on validation; count reported not matched |

Every model emits one output — a logit for APS (shared
`BCEWithLogitsLoss(pos_weight = n_neg/n_pos)`), a scalar for Air Quality
(`MSELoss`).

**Optional non-neural reference:** `HistGradientBoostingClassifier` /
`Regressor` (native missing support). Reference only — not parameter-matched,
never used to select a neural hyperparameter.

## Optimization fairness

Every neural model sweeps the **same** predeclared LR grid `{1e-3, 3e-3, 1e-2}`
(NeuMiss also sweeps its depth grid). Per `(dataset, model, seed)`: train every
config on train/validation only, select the best validation primary metric,
evaluate the test set once. AdamW, `weight_decay = 0`, batch 128,
best-checkpoint restore, ~1500-step early-stop patience. The selected LR (and
NeuMiss depth) is recorded. APS: the `10·FP + 500·FN` threshold is chosen on
the **validation** set, frozen, applied once to the test set.

## Diagnostics

* **Missingness strata:** every test example bucketed by
  `missing_fraction = missing_features / total_features` into the fixed bins
  `{0, (0, .10], (.10, .25], (.25, .50], >.50}` (empty bins skipped);
  performance and — for CellV0.3 — mean hidden `π` / `u` / `π` CoV per bin.
* **Confidence interventions** (evaluation only): re-score every trained
  CellV0.3 model with the true `c`, with `c := 1`, and with `c` permuted across
  feature positions per example; report `true − all_ones` and `true − shuffled`
  for the primary metric.

## Layout / running

```bash
pip install -e ".[paper-a-phase3]"   # installs the pinned NeuMiss

# Part A (18 runs, reuses the Phase-2 harness):
python experiments/paper_a/capacity_stress.py

# Part B (30 neural runs + 6 HGB reference):
python experiments/paper_a/real_reliability/run_real_reliability.py

# combined report:
python experiments/paper_a/publication_validation.py
```

Raw `RunRecord`s land in `results/raw/` (gitignored). The committed artifact is
`experiments/paper_a/publication_validation_results.md` plus the dated
`docs/research_log.md` entry.

## Predeclared stopping rule (Phase-3 task "Interpretation / stopping rules")

A result is genuinely strong only if **both** survive:

1. **Capacity control** — CellV0.3's Phase-2 robustness advantage remains
   against the deliberately larger same-width Confidence MLP.
2. **Real reliability** — CellV0.3 competitive with / better than NeuMiss (and
   the same-width MLP) on ≥ 1 real dataset, **and** meaningful degradation when
   correct reliability is removed / shuffled.

A V0.3 win only over the Plain MLP is insufficient; a win only over the
param-matched (not the same-width) Confidence MLP is weak. If NeuMiss and the
same-width Confidence MLP consistently match/beat V0.3, record the result and
**stop the Cell paper** rather than creating CellV0.4. Do not claim
calibration. Do not write the paper yet.
