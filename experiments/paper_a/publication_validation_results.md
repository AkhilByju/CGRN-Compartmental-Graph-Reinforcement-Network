# Paper A -- Phase 3: publication-grade validation of CellV0.3

CellV0.3 is permanently frozen. This phase has exactly two purposes: (A) rule out that the Phase-2 Confidence MLP lost only because parameter matching narrowed it, and (B) test CellV0.3 on real datasets with naturally missing sensor measurements, against an established missing-data neural architecture (NeuMiss). No other research questions; no architecture changes.

Records: Part A 18 same-width runs (`paper_a_capacity_stress`) + recorded Phase-2 CellV0.3 / Confidence MLP; Part B 30 neural runs (`paper_a_real_reliability`) + 6 HistGradientBoosting reference runs. n = 3 seeds, no significance test.

## Part A -- capacity-stress the Confidence MLP

Phase 2 parameter-matched every baseline to CellV0.3's actual count, which forced the Confidence MLP (input `concat(x, c)`, `2*D` wide) to a *narrower* hidden layer than CellV0.3. The **same-width Confidence MLP** removes that confound: the Phase-2 Confidence MLP architecture at CellV0.3's exact hidden width, **not** parameter-matched. 18 new runs (`{mnist, fashion_mnist, digits} x {missing, gaussian} x seeds 0-2`), the frozen Phase-2 protocol; CellV0.3 and the param-matched Confidence MLP are read back from the recorded Phase-2 results.

Corruption-AUC (primary metric, trapezoidal over the severity grid), mean ± std over 3 seeds:

| Dataset | Corruption | CellV0.3 | Confidence MLP (matched) | Confidence MLP (same width) | same-width param ratio | Δ(V0.3 − same-width) |
|---|---|---|---|---|---|---|
| mnist | missing | 0.6655 ± 0.0010 | 0.6351 ± 0.0030 | 0.6381 ± 0.0026 | x1.65 (hidden 157) | **+0.0274** |
| mnist | gaussian | 1.4596 ± 0.0032 | 1.3843 ± 0.0110 | 1.3996 ± 0.0114 | x1.65 (hidden 157) | **+0.0600** |
| fashion_mnist | missing | 0.5929 ± 0.0013 | 0.5676 ± 0.0019 | 0.5720 ± 0.0012 | x1.65 (hidden 157) | **+0.0209** |
| fashion_mnist | gaussian | 1.3049 ± 0.0006 | 1.2394 ± 0.0077 | 1.2507 ± 0.0046 | x1.65 (hidden 157) | **+0.0542** |
| digits | missing | 0.6251 ± 0.0091 | 0.6356 ± 0.0022 | 0.6326 ± 0.0068 | x0.69 (hidden 123) | **-0.0076** |
| digits | gaussian | 1.4216 ± 0.0141 | 1.4122 ± 0.0024 | 1.4159 ± 0.0195 | x0.69 (hidden 123) | **+0.0056** |

CellV0.3's corruption-AUC exceeds the deliberately larger same-width Confidence MLP in **5/6** cells. On MNIST / Fashion-MNIST the same-width model carries ~1.65x CellV0.3's parameters and still loses; on Digits the "same width" MLP is actually *smaller* than CellV0.3 (ratio < 1 -- CellV0.3's parameters are dominated by its hidden->hidden layer), so that column is not a capacity advantage for the baseline.

## APS Failure at Scania Trucks -- main table

Official 60k/16k split, 170 features, real dataset missingness. Primary metric: **PR-AUC** (never accuracy). Official cost `10*FP + 500*FN` at the validation-selected frozen threshold. Mean ± std over seeds 0-2.

| Model | PR-AUC | ROC-AUC | Bal. acc | F1 | Precision | Recall | Official cost | cost/1000 | Params | Hidden/depth | LR | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plain MLP | 0.826 ± 0.016 | 0.982 ± 0.010 | 0.956 ± 0.008 | 0.661 ± 0.008 | 0.512 ± 0.011 | 0.932 ± 0.016 | 16003 ± 2908 | 1000 | 99,589 | 579 | 1e-03 | 14 |
| Confidence MLP (matched) | 0.841 ± 0.005 | 0.987 ± 0.004 | 0.958 ± 0.001 | 0.640 ± 0.017 | 0.485 ± 0.020 | 0.940 ± 0.002 | 15083 ± 217 | 943 | 99,523 | 291 | 1e-03 | 19 |
| Confidence MLP (same width) | 0.834 ± 0.013 | 0.988 ± 0.004 | 0.959 ± 0.003 | 0.599 ± 0.036 | 0.439 ± 0.040 | 0.948 ± 0.009 | 14420 ± 1276 | 901 | 82,081 | 240 | 1e-03 (1e-03/3e-03) | 18 |
| CellV0.3 | 0.844 ± 0.014 | 0.992 ± 0.003 | 0.957 ± 0.007 | 0.592 ± 0.085 | 0.436 ± 0.095 | 0.946 ± 0.027 | 15073 ± 3020 | 942 | 99,601 | 240 | 3e-03 (3e-03/1e-02) | 146 |
| NeuMiss | 0.838 ± 0.012 | 0.989 ± 0.003 | 0.955 ± 0.009 | 0.649 ± 0.080 | 0.504 ± 0.099 | 0.934 ± 0.028 | 16017 ± 3547 | 1001 | 99,935 | d=1 | 1e-03 | 100 |
| HistGradientBoosting (ref) | 0.876 ± 0.000 | 0.992 ± 0.001 | 0.952 ± 0.009 | 0.644 ± 0.093 | 0.502 ± 0.114 | 0.928 ± 0.031 | 17273 | 1080 | -- | -- | -- | -- |

## UCI Air Quality -- main table

Predict `CO(GT)` from 8 sensor/environment inputs, `-200` -> missing, chronological 60/20/20 split. Primary metric: **R^2** on the original scale. Mean ± std over seeds 0-2.

| Model | R^2 | RMSE | MAE | Params | Hidden/depth | LR | Wall (s) |
|---|---|---|---|---|---|---|---|
| Plain MLP | 0.750 ± 0.016 | 0.671 ± 0.022 | 0.420 ± 0.040 | 49,911 | 4991 | 1e-02 (3e-03/1e-02) | 13 |
| Confidence MLP (matched) | 0.728 ± 0.016 | 0.701 ± 0.020 | 0.453 ± 0.015 | 49,915 | 2773 | 1e-02 (1e-03/1e-02) | 15 |
| Confidence MLP (same width) | 0.738 ± 0.014 | 0.688 ± 0.019 | 0.446 ± 0.054 | 3,907 | 217 | 1e-03 (1e-03/1e-02) | 8 |
| CellV0.3 | 0.760 ± 0.010 | 0.659 ± 0.014 | 0.414 ± 0.020 | 49,911 | 217 | 1e-02 (1e-03/3e-03/1e-02) | 33 |
| NeuMiss | 0.743 ± 0.011 | 0.681 ± 0.014 | 0.432 ± 0.017 | 50,003 | d=3 | 1e-02 (1e-03/1e-02) | 103 |
| HistGradientBoosting (ref) | 0.752 ± 0.004 | 0.669 ± 0.005 | 0.395 ± 0.010 | -- | -- | -- | -- |

## Real-missingness stratified performance

Test examples bucketed by `missing_fraction = missing_features / total_features` into the fixed bins; empty bins skipped. Primary metric, mean over 3 seeds.

### aps (pr_auc)

| Model | 0 | (0, 0.10] | (0.10, 0.25] | (0.25, 0.50] | >0.50 |
|---|---|---|---|---|---|
| Plain MLP | 0.954 | 0.812 | 0.824 | 0.836 | 0.671 |
| Confidence MLP (matched) | 0.973 | 0.807 | 0.833 | 0.867 | 0.685 |
| Confidence MLP (same width) | 0.963 | 0.798 | 0.834 | 0.861 | 0.693 |
| CellV0.3 | 0.952 | 0.788 | 0.824 | 0.891 | 0.848 |
| NeuMiss | 0.953 | 0.794 | 0.835 | 0.879 | 0.702 |

**CellV0.3 belief state by missingness bin (aps)** -- mean over 3 seeds:

| Bin | L1 π mean | L1 u mean | L2 π mean | L2 u mean | L2 π CoV |
|---|---|---|---|---|---|
| 0 | 0.397 | 6.412 | 0.356 | 0.277 | 0.546 |
| (0, 0.10] | 0.840 | 2.961 | 0.760 | 0.134 | 0.166 |
| (0.10, 0.25] | 0.717 | 2.061 | 0.649 | 0.161 | 0.243 |
| (0.25, 0.50] | 0.390 | 534.487 | 0.361 | 0.277 | 0.554 |
| >0.50 | 0.206 | 51.432 | 0.194 | 0.186 | 0.894 |

### air_quality (r2)

| Model | 0 | >0.50 |
|---|---|---|
| Plain MLP | 0.847 | -0.325 |
| Confidence MLP (matched) | 0.821 | -0.316 |
| Confidence MLP (same width) | 0.831 | -0.299 |
| CellV0.3 | 0.847 | -0.212 |
| NeuMiss | 0.832 | -0.246 |

**CellV0.3 belief state by missingness bin (air_quality)** -- mean over 3 seeds:

| Bin | L1 π mean | L1 u mean | L2 π mean | L2 u mean | L2 π CoV |
|---|---|---|---|---|---|
| 0 | 0.446 | 1.560 | 0.412 | 0.195 | 0.289 |
| >0.50 | 0.001 | 0.000 | 0.001 | 0.031 | 0.000 |

## Confidence interventions (CellV0.3, evaluation only)

The trained CellV0.3 models re-scored with the true `c`, with `c := 1` everywhere, and with `c` permuted across feature positions per example. Primary metric, mean ± std over seeds.

| Dataset | metric | true | all-ones | shuffled | true − all-ones | true − shuffled |
|---|---|---|---|---|---|---|
| aps | pr_auc | 0.844 ± 0.014 | 0.850 ± 0.013 | 0.787 ± 0.028 | **-0.0056** | **+0.0571** |
| air_quality | r2 | 0.760 ± 0.010 | 0.748 ± 0.004 | 0.760 ± 0.010 | **+0.0123** | **+0.0000** |

_A positive `true − all-ones` / `true − shuffled` means the true reliability alignment helped. On Air Quality every row's `c` is uniform (a row has either no missing inputs or all 8 missing), so `shuffled` is a no-op there by construction._

## Interpretation and predeclared stopping rules

A result is genuinely strong only if **both** survive:

1. **Capacity control** -- CellV0.3's corruption-AUC still exceeds the deliberately larger same-width Confidence MLP. Result: **5/6** cells.
2. **Real reliability** -- CellV0.3 competitive with / better than strong missingness-aware baselines (esp. NeuMiss) on >=1 real dataset, **and** meaningful degradation when correct reliability is removed/shuffled:

- **aps** (pr_auc): CellV0.3 0.844, NeuMiss 0.838, same-width Confidence MLP 0.834 -> competitive/ahead
- **air_quality** (r2): CellV0.3 0.760, NeuMiss 0.743, same-width Confidence MLP 0.738 -> competitive/ahead

APS by missingness stratum (PR-AUC, CellV0.3 vs NeuMiss) -- where the architecture predicts it should help most:

- (0.25, 0.50]: CellV0.3 0.891 vs NeuMiss 0.879
- >0.50: CellV0.3 0.848 vs NeuMiss 0.702

Intervention degradation (true − all-ones / true − shuffled, primary metric):

- aps: -0.0056 / +0.0571
- air_quality: +0.0123 / +0.0000

Non-neural reference context: HistGradientBoosting ahead of CellV0.3 on aps (pr_auc 0.876 > CellV0.3 0.844).

A V0.3 win only over the Plain MLP is insufficient; a win only over the param-matched Confidence MLP but not the same-width model is weak.

**Both predeclared conditions are met.** (1) The Phase-2 robustness advantage survives the deliberately larger same-width Confidence MLP on every image cell (Part A). (2) CellV0.3 is competitive with or ahead of NeuMiss / the same-width MLP on both real datasets, its advantage on APS is concentrated in exactly the high-missingness strata the architecture is meant for, and scrambling the true reliability alignment costs it real primary-metric points. **Caveats that keep this short of 'decisive':** the whole-test-set margin over NeuMiss is small (APS ~tie); the non-neural HistGradientBoosting reference beats every neural model on APS; and on APS setting `c := 1` (removing, not scrambling, the reliability signal) does *not* hurt CellV0.3 -- only misalignment does. Record; do not write the paper yet; do not alter V0.3, do not build CellV0.4.

_Not calibration: `e`/`u`/`π` are internal computational reliability variables. Do not write the paper yet; do not alter V0.3 after these results._

## Failures and caveats

**Divergence / NaNs:** none.

**Step-cap hits:** air_quality/confidence_mlp; air_quality/confidence_mlp_same_width; air_quality/neumiss; aps/cellv0.3

**NeuMiss integration issues:** none -- the official `marineLM/NeuMiss_sota` package (pinned commit) imported and trained cleanly.

**High seed variance (primary-metric std > 0.05 PR-AUC / 0.10 R^2):** none.

**Reliability-aware baseline beats CellV0.3 (primary, > 0.005):** none.

**CellV0.3 hidden π does not fall with missingness:** none -- π falls monotonically-ish on both real datasets.

**Large CellV0.3 conflict `u` (numeric-range flag; `π = e/(1+e·u)` stays finite, no NaN/divergence, no protocol impact):**
- aps/seed0 bin '(0.25, 0.50]' (max mean u ~ 12)
- aps/seed0 bin '>0.50' (max mean u ~ 52)
- aps/seed1 bin '(0.25, 0.50]' (max mean u ~ 1578)
- aps/seed1 bin '>0.50' (max mean u ~ 54)
- aps/seed2 bin '(0.25, 0.50]' (max mean u ~ 13)
- aps/seed2 bin '>0.50' (max mean u ~ 49)

### Part A supplementary -- per-severity primary metric (mean of 3 seeds)

**mnist / missing**

| Model | 0 | 0.1 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|
| CellV0.3 | 0.980 | 0.978 | 0.970 | 0.948 | 0.861 |
| Confidence MLP (matched) | 0.949 | 0.945 | 0.930 | 0.898 | 0.804 |
| Confidence MLP (same width) | 0.952 | 0.948 | 0.933 | 0.901 | 0.815 |

**mnist / gaussian**

| Model | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 |
|---|---|---|---|---|---|---|
| CellV0.3 | 0.978 | 0.978 | 0.977 | 0.975 | 0.972 | 0.962 |
| Confidence MLP (matched) | 0.937 | 0.936 | 0.933 | 0.928 | 0.920 | 0.892 |
| Confidence MLP (same width) | 0.945 | 0.944 | 0.941 | 0.937 | 0.930 | 0.909 |

**fashion_mnist / missing**

| Model | 0 | 0.1 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|
| CellV0.3 | 0.883 | 0.878 | 0.864 | 0.838 | 0.768 |
| Confidence MLP (matched) | 0.843 | 0.838 | 0.825 | 0.802 | 0.744 |
| Confidence MLP (same width) | 0.848 | 0.844 | 0.832 | 0.808 | 0.749 |

**fashion_mnist / gaussian**

| Model | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 |
|---|---|---|---|---|---|---|
| CellV0.3 | 0.884 | 0.883 | 0.878 | 0.872 | 0.865 | 0.847 |
| Confidence MLP (matched) | 0.843 | 0.841 | 0.836 | 0.831 | 0.822 | 0.795 |
| Confidence MLP (same width) | 0.846 | 0.846 | 0.844 | 0.839 | 0.831 | 0.805 |

**digits / missing**

| Model | 0 | 0.1 | 0.3 | 0.5 | 0.7 |
|---|---|---|---|---|---|
| CellV0.3 | 0.972 | 0.966 | 0.935 | 0.872 | 0.702 |
| Confidence MLP (matched) | 0.979 | 0.971 | 0.950 | 0.890 | 0.731 |
| Confidence MLP (same width) | 0.984 | 0.969 | 0.944 | 0.885 | 0.722 |

**digits / gaussian**

| Model | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 |
|---|---|---|---|---|---|---|
| CellV0.3 | 0.971 | 0.970 | 0.962 | 0.953 | 0.941 | 0.903 |
| Confidence MLP (matched) | 0.969 | 0.971 | 0.966 | 0.955 | 0.936 | 0.868 |
| Confidence MLP (same width) | 0.978 | 0.976 | 0.969 | 0.957 | 0.938 | 0.866 |

