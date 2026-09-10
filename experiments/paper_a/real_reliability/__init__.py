"""Paper A -- Phase 3 Part B: real missing-sensor benchmarks for CellV0.3.

Two public datasets with *naturally* missing measurements (no synthetic
corruption):

* **APS Failure at Scania Trucks** (UCI 421) -- imbalanced binary
  classification, 170 operational features, real dataset missingness, the
  official 60k/16k train/test split, the official ``10*FP + 500*FN`` cost.
* **UCI Air Quality** (UCI 360) -- sensor regression for ``CO(GT)`` from 8
  sensor/environment inputs, ``-200`` sentinel converted to missing, a
  chronological 60/20/20 split (documented drift is part of the setting).

Five neural models/controls -- Plain MLP, Confidence MLP (parameter-matched),
Confidence MLP (same width as CellV0.3), the **frozen** CellV0.3 with input
belief ``(mu = x_imputed, e = c, u = 0)``, and the official **NeuMiss**
(`marineLM/NeuMiss_sota`, handles NaN natively) -- plus optional
`HistGradientBoosting` reference baselines.

This package only *composes* external data + `src/` infrastructure; it does
not modify CellV0.3. See `experiments/paper_a/publication_validation_results.md`.
"""
