# Paper A — CellV0.2 (Conservative Precision-Gain Cell) on the frozen Phase-1 protocol

CellV0.2 runs loaded: 42 (experiment `paper_a_phase1_cellv02`). Comparison arms read from the frozen `paper_a_phase1` records: 42 CellV0.1, 42 matched MLP.
Datasets present (CellV0.2): breast_cancer, wine, digits, diabetes, california_housing, mnist, fashion_mnist.

CellV0.2 = `BeliefNetworkV02` — two `PrecisionGainLayer`s (one signed connection matrix `V` + per-output `gain_raw` + `bias`, **no** relevance gate) + a confidence-scaled linear readout. Input belief `e=1, u=0`. Hidden width fitted to the **same** per-dataset parameter budget as CellV0.1; the resulting (larger) hidden-cell count is reported. Identical shared protocol: AdamW, lr=1e-2, weight_decay=0, best-validation restore, ~1500-step early-stop patience, 15000-step cap. Nothing tuned; equations frozen before the run (docs/architecture_v0.md Sec 10).

## Headline signal (per dataset) — CellV0.2 vs each recorded arm

| Dataset | Δ vs matched MLP (25% / 100%) | Δ vs CellV0.1 (25% / 100%) | Read (vs matched) |
|---|---|---|---|
| breast_cancer | -0.0058 / +0.0000 | +0.0029 / -0.0029 | tie |
| wine | -0.0093 / -0.0093 | +0.0093 / +0.0000 | tie |
| digits | +0.0046 / +0.0000 | +0.0574 / +0.0398 | tie |
| diabetes | +0.2131 / +0.0378 | -0.1458 / -0.0430 | win (both fractions) |
| california_housing | +0.0422 / +0.1002 | +0.0252 / +0.0202 | win (both fractions) |
| mnist | +0.0172 / +0.0173 | +0.0100 / +0.0026 | win (both fractions) |
| fashion_mnist | +0.0199 / +0.0189 | +0.0098 / +0.0047 | win (both fractions) |

Across all 14 (dataset, fraction) cells vs the parameter-matched MLP: CellV0.2 ahead by >0.01 in 8, behind by >0.01 in 0, within ±0.01 in 6. Reporting threshold only; no significance test (n=3).

## A. Main table — CellV0.2

| Dataset | Frac | Hidden cells | Params | Headline (mean ± std) | 2nd metric | Best-val step | Total steps | Train wall-clock (s) | Time/step (ms) |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 25% | 83 | 9,879 | accuracy: 0.9503 ± 0.0253 | F1: 0.9472 ± 0.0251 | 2467 ± 4186 | 3883 | 4.3 ± 4.5 | 1.10 |
| breast_cancer | 100% | 83 | 9,879 | accuracy: 0.9678 ± 0.0051 | F1: 0.9653 ± 0.0052 | 850 ± 1386 | 2083 | 2.7 ± 1.1 | 1.28 |
| wine | 25% | 90 | 9,903 | accuracy: 0.9537 ± 0.0160 | F1: 0.9561 ± 0.0159 | 1233 ± 2006 | 2233 | 1.9 ± 0.9 | 0.83 |
| wine | 100% | 90 | 9,903 | accuracy: 0.9815 ± 0.0160 | F1: 0.9829 ± 0.0148 | 2283 ± 2011 | 2783 | 3.5 ± 1.6 | 1.24 |
| digits | 25% | 123 | 24,733 | accuracy: 0.9222 ± 0.0100 | F1: 0.9213 ± 0.0103 | 200 ± 173 | 1700 | 2.3 ± 0.3 | 1.33 |
| digits | 100% | 123 | 24,733 | accuracy: 0.9685 ± 0.0042 | F1: 0.9683 ± 0.0043 | 100 ± 0 | 1600 | 2.4 ± 0.1 | 1.50 |
| diabetes | 25% | 92 | 9,845 | r2: 0.2301 ± 0.0962 | RMSE: 63.9044 ± 3.1355 | 50 ± 0 | 1550 | 1.2 ± 0.0 | 0.77 |
| diabetes | 100% | 92 | 9,845 | r2: 0.3376 ± 0.0537 | RMSE: 59.3687 ± 2.7533 | 133 ± 144 | 1633 | 1.5 ± 0.1 | 0.95 |
| california_housing | 25% | 93 | 9,859 | r2: 0.7746 ± 0.0120 | RMSE: 0.5455 ± 0.0109 | 3067 ± 1361 | 4667 | 5.3 ± 1.5 | 1.13 |
| california_housing | 100% | 93 | 9,859 | r2: 0.7971 ± 0.0063 | RMSE: 0.5176 ± 0.0054 | 6667 ± 1677 | 8267 | 9.2 ± 1.9 | 1.12 |
| mnist | 25% | 157 | 149,945 | accuracy: 0.9607 ± 0.0075 | F1: 0.9604 ± 0.0076 | 1200 ± 1058 | 2800 | 14.2 ± 5.8 | 5.05 |
| mnist | 100% | 157 | 149,945 | accuracy: 0.9772 ± 0.0011 | F1: 0.9770 ± 0.0011 | 3333 ± 611 | 4933 | 25.2 ± 2.4 | 5.12 |
| fashion_mnist | 25% | 157 | 149,945 | accuracy: 0.8529 ± 0.0071 | F1: 0.8532 ± 0.0075 | 1000 ± 400 | 2600 | 13.2 ± 1.6 | 5.12 |
| fashion_mnist | 100% | 157 | 149,945 | accuracy: 0.8753 ± 0.0069 | F1: 0.8749 ± 0.0074 | 3067 ± 1747 | 4667 | 20.0 ± 8.0 | 4.26 |

## Paired comparison — CellV0.2 minus the parameter-matched MLP (headline metric)

| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |
|---|---|---|---|---|---|
| breast_cancer | 25% | +0.0000 | +0.0000 | -0.0175 | **-0.0058** |
| breast_cancer | 100% | +0.0000 | +0.0088 | -0.0088 | **+0.0000** |
| wine | 25% | -0.0278 | +0.0000 | +0.0000 | **-0.0093** |
| wine | 100% | +0.0000 | +0.0000 | -0.0278 | **-0.0093** |
| digits | 25% | +0.0056 | -0.0111 | +0.0194 | **+0.0046** |
| digits | 100% | +0.0000 | -0.0083 | +0.0083 | **+0.0000** |
| diabetes | 25% | +0.1556 | +0.4622 | +0.0216 | **+0.2131** |
| diabetes | 100% | +0.0700 | +0.0059 | +0.0375 | **+0.0378** |
| california_housing | 25% | +0.0180 | +0.0324 | +0.0762 | **+0.0422** |
| california_housing | 100% | +0.1911 | +0.0466 | +0.0629 | **+0.1002** |
| mnist | 25% | +0.0092 | +0.0343 | +0.0082 | **+0.0172** |
| mnist | 100% | +0.0189 | +0.0143 | +0.0188 | **+0.0173** |
| fashion_mnist | 25% | +0.0177 | +0.0167 | +0.0253 | **+0.0199** |
| fashion_mnist | 100% | +0.0219 | +0.0164 | +0.0184 | **+0.0189** |

_Positive = CellV0.2 better than the parameter-matched MLP. n=3 seeds; no significance test._

## Paired comparison — CellV0.2 minus CellV0.1 (recorded) (headline metric)

| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |
|---|---|---|---|---|---|
| breast_cancer | 25% | +0.0000 | +0.0088 | +0.0000 | **+0.0029** |
| breast_cancer | 100% | +0.0088 | -0.0088 | -0.0088 | **-0.0029** |
| wine | 25% | +0.0000 | +0.0278 | +0.0000 | **+0.0093** |
| wine | 100% | +0.0000 | +0.0000 | +0.0000 | **+0.0000** |
| digits | 25% | +0.0472 | +0.0389 | +0.0861 | **+0.0574** |
| digits | 100% | +0.0333 | +0.0306 | +0.0556 | **+0.0398** |
| diabetes | 25% | -0.1839 | -0.1356 | -0.1178 | **-0.1458** |
| diabetes | 100% | +0.0826 | -0.1790 | -0.0326 | **-0.0430** |
| california_housing | 25% | +0.0317 | +0.0049 | +0.0389 | **+0.0252** |
| california_housing | 100% | +0.0135 | +0.0202 | +0.0269 | **+0.0202** |
| mnist | 25% | +0.0063 | +0.0197 | +0.0039 | **+0.0100** |
| mnist | 100% | +0.0027 | +0.0017 | +0.0034 | **+0.0026** |
| fashion_mnist | 25% | +0.0071 | +0.0026 | +0.0197 | **+0.0098** |
| fashion_mnist | 100% | +0.0096 | -0.0051 | +0.0097 | **+0.0047** |

_Positive = CellV0.2 better than CellV0.1 (recorded). n=3 seeds; no significance test._

## Data-efficiency — headline at 25% → 100% training data

| Dataset | Model | 25% | 100% | Δ (100% − 25%) |
|---|---|---|---|---|
| breast_cancer | CellV0.2 | 0.9503 ± 0.0253 | 0.9678 ± 0.0051 | +0.0175 |
| breast_cancer | CellV0.1 (recorded) | 0.9474 ± 0.0232 | 0.9708 ± 0.0134 | +0.0234 |
| breast_cancer | MLP matched (recorded) | 0.9561 ± 0.0152 | 0.9678 ± 0.0051 | +0.0117 |
| wine | CellV0.2 | 0.9537 ± 0.0160 | 0.9815 ± 0.0160 | +0.0278 |
| wine | CellV0.1 (recorded) | 0.9444 ± 0.0000 | 0.9815 ± 0.0160 | +0.0370 |
| wine | MLP matched (recorded) | 0.9630 ± 0.0160 | 0.9907 ± 0.0160 | +0.0278 |
| digits | CellV0.2 | 0.9222 ± 0.0100 | 0.9685 ± 0.0042 | +0.0463 |
| digits | CellV0.1 (recorded) | 0.8648 ± 0.0339 | 0.9287 ± 0.0112 | +0.0639 |
| digits | MLP matched (recorded) | 0.9176 ± 0.0252 | 0.9685 ± 0.0080 | +0.0509 |
| diabetes | CellV0.2 | 0.2301 ± 0.0962 | 0.3376 ± 0.0537 | +0.1075 |
| diabetes | CellV0.1 (recorded) | 0.3759 ± 0.0750 | 0.3806 ± 0.0821 | +0.0047 |
| diabetes | MLP matched (recorded) | 0.0170 ± 0.3021 | 0.2998 ± 0.0273 | +0.2829 |
| california_housing | CellV0.2 | 0.7746 ± 0.0120 | 0.7971 ± 0.0063 | +0.0225 |
| california_housing | CellV0.1 (recorded) | 0.7494 ± 0.0088 | 0.7769 ± 0.0066 | +0.0274 |
| california_housing | MLP matched (recorded) | 0.7324 ± 0.0218 | 0.6969 ± 0.0786 | -0.0355 |
| mnist | CellV0.2 | 0.9607 ± 0.0075 | 0.9772 ± 0.0011 | +0.0165 |
| mnist | CellV0.1 (recorded) | 0.9507 ± 0.0030 | 0.9746 ± 0.0017 | +0.0238 |
| mnist | MLP matched (recorded) | 0.9435 ± 0.0088 | 0.9598 ± 0.0030 | +0.0164 |
| fashion_mnist | CellV0.2 | 0.8529 ± 0.0071 | 0.8753 ± 0.0069 | +0.0224 |
| fashion_mnist | CellV0.1 (recorded) | 0.8431 ± 0.0023 | 0.8706 ± 0.0047 | +0.0275 |
| fashion_mnist | MLP matched (recorded) | 0.8330 ± 0.0030 | 0.8564 ± 0.0066 | +0.0234 |

## B. Parameter count & hidden width — CellV0.2 vs CellV0.1 (same budget)

| Dataset | Budget | CellV0.2 hidden cells | CellV0.2 params | CellV0.1 hidden cells | CellV0.1 params | cell ratio |
|---|---|---|---|---|---|---|
| breast_cancer | 10,000 | 83 | 9,879 | 56 | 9,858 | 1.48× |
| wine | 10,000 | 90 | 9,903 | 63 | 9,894 | 1.43× |
| digits | 25,000 | 123 | 24,733 | 82 | 24,938 | 1.50× |
| diabetes | 10,000 | 92 | 9,845 | 65 | 9,946 | 1.42× |
| california_housing | 10,000 | 93 | 9,859 | 66 | 9,967 | 1.41× |
| mnist | 150,000 | 157 | 149,945 | 85 | 148,760 | 1.85× |
| fashion_mnist | 150,000 | 157 | 149,945 | 85 | 148,760 | 1.85× |

_CellV0.2 formula: `hc·(in+2) + hc·(hc+2) + hc·out + out`. CellV0.1 formula: `hc·(2·in+1) + hc·(2·hc+1) + hc·out + out`. Same budget, roughly 1.5× the hidden cells — its lower per-connection cost is part of the architecture, not equalized away._

## CellV0.2 internal diagnostics (observational only — not an evaluation target)

| Dataset | Frac | eff. precision L1 (mean/std, min–max) | eff. precision L2 (mean/std, min–max) | rel. gain L1 (mean/std) | rel. gain L2 (mean/std) |
|---|---|---|---|---|---|
| breast_cancer | 25% | 0.633 / 0.165, 7.08e-02–0.94 | 0.563 / 0.145, 9.74e-02–0.87 | 0.998 / 0.049 | 1.000 / 0.020 |
| breast_cancer | 100% | 0.633 / 0.165, 7.76e-02–0.96 | 0.538 / 0.141, 1.07e-01–0.87 | 0.997 / 0.050 | 1.000 / 0.014 |
| wine | 25% | 0.583 / 0.122, 1.84e-01–0.93 | 0.518 / 0.096, 2.81e-01–0.75 | 0.996 / 0.063 | 1.000 / 0.018 |
| wine | 100% | 0.597 / 0.124, 1.85e-01–0.93 | 0.527 / 0.093, 2.90e-01–0.76 | 0.995 / 0.067 | 1.000 / 0.021 |
| digits | 25% | 0.538 / 0.101, 3.24e-11–0.78 | 0.453 / 0.078, 1.67e-07–0.65 | 0.992 / 0.085 | 0.998 / 0.038 |
| digits | 100% | 0.541 / 0.090, 1.92e-02–0.80 | 0.453 / 0.068, 5.04e-02–0.64 | 0.996 / 0.059 | 1.000 / 0.017 |
| diabetes | 25% | 0.568 / 0.148, 1.66e-01–0.97 | 0.505 / 0.118, 2.50e-01–0.79 | 0.995 / 0.071 | 1.000 / 0.008 |
| diabetes | 100% | 0.580 / 0.140, 1.83e-01–0.95 | 0.523 / 0.117, 2.61e-01–0.79 | 0.996 / 0.061 | 1.000 / 0.006 |
| california_housing | 25% | 0.701 / 0.156, 2.09e-03–1.00 | 0.632 / 0.114, 1.10e-02–0.93 | 0.993 / 0.082 | 1.000 / 0.019 |
| california_housing | 100% | 0.717 / 0.156, 2.21e-03–1.00 | 0.672 / 0.118, 9.34e-03–0.96 | 0.993 / 0.082 | 0.999 / 0.031 |
| mnist | 25% | 0.432 / 0.087, 1.79e-01–0.81 | 0.327 / 0.052, 1.81e-01–0.58 | 0.998 / 0.043 | 1.000 / 0.007 |
| mnist | 100% | 0.430 / 0.088, 1.78e-01–0.82 | 0.325 / 0.051, 1.80e-01–0.57 | 0.998 / 0.045 | 1.000 / 0.007 |
| fashion_mnist | 25% | 0.531 / 0.111, 2.50e-01–0.91 | 0.384 / 0.068, 2.24e-01–0.70 | 1.000 / 0.020 | 1.000 / 0.009 |
| fashion_mnist | 100% | 0.529 / 0.110, 2.50e-01–0.91 | 0.383 / 0.067, 2.25e-01–0.67 | 1.000 / 0.020 | 1.000 / 0.008 |

_Effective precision `e/(1+e·u)` and relative gain `2π/(π+mean π)` of each hidden layer's output belief, on the test split. Logged per the task; nothing is tuned on these._

## C. Failures and caveats (CellV0.2)

**Divergence / NaNs:** none.

**Step-cap hits:** none.

**Numeric-range flag** (a single test example × cell with effective precision `< 1e-6` — large local disagreement `u`; all values finite, no NaN/divergence, no protocol impact):

- digits 25% seed0 layer1 (min π = 1.3e-11)
- digits 25% seed0 layer2 (min π = 8.2e-10)
- digits 25% seed1 layer1 (min π = 8.0e-11)
- digits 25% seed1 layer2 (min π = 5.0e-07)
- digits 25% seed2 layer1 (min π = 4.4e-12)
- digits 25% seed2 layer2 (min π = 3.9e-10)
- digits 100% seed0 layer1 (min π = 1.1e-10)
- digits 100% seed0 layer2 (min π = 8.4e-09)
- digits 100% seed2 layer1 (min π = 8.0e-11)
- digits 100% seed2 layer2 (min π = 9.6e-09)

