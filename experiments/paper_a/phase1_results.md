# Paper A — Phase 1 public-benchmark screen (results)

Runs loaded: 126 main + 9 fixed-confidence ablation. Datasets present: breast_cancer, wine, digits, diabetes, california_housing, mnist, fashion_mnist.

CellV0.1 = frozen `BeliefNetwork` with `scale_stable_precision` aggregation. MLP (matched) = 1-hidden-layer SiLU MLP, ≤2% off CellV0.1's parameter count. MLP (state-count) = 1-hidden-layer SiLU MLP, hidden width 3× CellV0.1's hidden cells (not parameter-matched). Shared protocol: AdamW, lr=1e-2, weight_decay=0, best-validation checkpoint restoration, early stopping (~1500-step patience), 15000-step cap.

## Headline signal (per dataset)

| Dataset | Δ vs matched (25% / 100%) | Δ vs state-count (25% / 100%) | Read |
|---|---|---|---|
| breast_cancer | -0.0088 / +0.0029 | -0.0117 / +0.0029 | mixed |
| wine | -0.0185 / -0.0093 | +0.0185 / -0.0093 | loss (one fraction) |
| digits | -0.0528 / -0.0398 | -0.0583 / -0.0417 | loss (both fractions) |
| diabetes | +0.3589 / +0.0808 | +0.2443 / +0.0378 | win (both fractions) |
| california_housing | +0.0170 / +0.0800 | +0.1108 / +0.0269 | win (both fractions) |
| mnist | +0.0073 / +0.0147 | +0.0091 / +0.0170 | win (one fraction) |
| fashion_mnist | +0.0101 / +0.0142 | +0.0070 / +0.0174 | win (both fractions) |

Across all 14 (dataset, fraction) cells: CellV0.1 ahead by >0.01 in 7, behind by >0.01 in 3, within ±0.01 in 4. The state-count column shows whether a wider (not parameter-matched) MLP erases the gap — compare its sign to the matched column.

## A. Main table

| Dataset | Frac | Model | Params | Headline (mean ± std) | Macro-F1 / RMSE | Best-val step | Total steps | Train wall-clock (s) |
|---|---|---|---|---|---|---|---|---|
| breast_cancer | 25% | CellV0.1 | 9,858 | accuracy: 0.9474 ± 0.0232 | F1: 0.9434 ± 0.0233 | 250 ± 304 | 1750 | 14.3 ± 1.9 |
| breast_cancer | 25% | MLP (matched) | 9,869 | accuracy: 0.9561 ± 0.0152 | F1: 0.9528 ± 0.0150 | 5033 ± 8631 | 6033 | 4.8 ± 6.3 |
| breast_cancer | 25% | MLP (state-count) | 5,546 | accuracy: 0.9591 ± 0.0183 | F1: 0.9561 ± 0.0185 | 5033 ± 8631 | 6033 | 3.6 ± 4.7 |
| breast_cancer | 100% | CellV0.1 | 9,858 | accuracy: 0.9708 ± 0.0134 | F1: 0.9685 ± 0.0145 | 167 ± 76 | 1667 | 17.4 ± 0.2 |
| breast_cancer | 100% | MLP (matched) | 9,869 | accuracy: 0.9678 ± 0.0051 | F1: 0.9651 ± 0.0056 | 83 ± 58 | 1583 | 1.7 ± 0.3 |
| breast_cancer | 100% | MLP (state-count) | 5,546 | accuracy: 0.9678 ± 0.0051 | F1: 0.9651 ± 0.0056 | 83 ± 58 | 1583 | 1.0 ± 0.1 |
| wine | 25% | CellV0.1 | 9,894 | accuracy: 0.9444 ± 0.0000 | F1: 0.9475 ± 0.0021 | 1933 ± 2158 | 2933 | 10.2 ± 3.8 |
| wine | 25% | MLP (matched) | 9,897 | accuracy: 0.9630 ± 0.0160 | F1: 0.9646 ± 0.0168 | 4317 ± 3106 | 5300 | 2.8 ± 2.0 |
| wine | 25% | MLP (state-count) | 3,216 | accuracy: 0.9259 ± 0.0321 | F1: 0.9290 ± 0.0279 | 2350 ± 2125 | 3150 | 1.5 ± 0.7 |
| wine | 100% | CellV0.1 | 9,894 | accuracy: 0.9815 ± 0.0160 | F1: 0.9829 ± 0.0148 | 3917 ± 3452 | 4400 | 31.6 ± 19.3 |
| wine | 100% | MLP (matched) | 9,897 | accuracy: 0.9907 ± 0.0160 | F1: 0.9906 ± 0.0163 | 5483 ± 5450 | 5983 | 3.7 ± 2.9 |
| wine | 100% | MLP (state-count) | 3,216 | accuracy: 0.9907 ± 0.0160 | F1: 0.9906 ± 0.0163 | 5667 ± 5920 | 6167 | 2.8 ± 2.4 |
| digits | 25% | CellV0.1 | 24,938 | accuracy: 0.8648 ± 0.0339 | F1: 0.8653 ± 0.0330 | 317 ± 76 | 1817 | 35.7 ± 3.1 |
| digits | 25% | MLP (matched) | 24,910 | accuracy: 0.9176 ± 0.0252 | F1: 0.9169 ± 0.0260 | 400 ± 606 | 1900 | 1.8 ± 0.6 |
| digits | 25% | MLP (state-count) | 18,460 | accuracy: 0.9231 ± 0.0098 | F1: 0.9226 ± 0.0093 | 67 ± 29 | 1567 | 1.3 ± 0.0 |
| digits | 100% | CellV0.1 | 24,938 | accuracy: 0.9287 ± 0.0112 | F1: 0.9287 ± 0.0112 | 467 ± 58 | 1967 | 44.8 ± 3.5 |
| digits | 100% | MLP (matched) | 24,910 | accuracy: 0.9685 ± 0.0080 | F1: 0.9685 ± 0.0078 | 633 ± 1010 | 2133 | 1.5 ± 0.7 |
| digits | 100% | MLP (state-count) | 18,460 | accuracy: 0.9704 ± 0.0085 | F1: 0.9703 ± 0.0087 | 67 ± 29 | 1567 | 1.0 ± 0.1 |
| diabetes | 25% | CellV0.1 | 9,946 | r2: 0.3759 ± 0.0750 | RMSE: 57.5529 ± 2.9147 | 50 ± 0 | 1550 | 5.1 ± 0.1 |
| diabetes | 25% | MLP (matched) | 9,949 | r2: 0.0170 ± 0.3021 | RMSE: 71.6968 ± 10.7291 | 50 ± 0 | 1550 | 0.8 ± 0.0 |
| diabetes | 25% | MLP (state-count) | 2,341 | r2: 0.1316 ± 0.2107 | RMSE: 67.5796 ± 7.7737 | 50 ± 0 | 1550 | 0.4 ± 0.0 |
| diabetes | 100% | CellV0.1 | 9,946 | r2: 0.3806 ± 0.0821 | RMSE: 57.3249 ± 3.2042 | 900 ± 229 | 2400 | 10.9 ± 1.1 |
| diabetes | 100% | MLP (matched) | 9,949 | r2: 0.2998 ± 0.0273 | RMSE: 61.0539 ± 1.3375 | 417 ± 289 | 1917 | 1.0 ± 0.1 |
| diabetes | 100% | MLP (state-count) | 2,341 | r2: 0.3428 ± 0.0296 | RMSE: 59.1509 ± 1.5949 | 367 ± 126 | 1867 | 0.6 ± 0.0 |
| california_housing | 25% | CellV0.1 | 9,967 | r2: 0.7494 ± 0.0088 | RMSE: 0.5752 ± 0.0109 | 6400 ± 5200 | 8000 | 56.1 ± 32.6 |
| california_housing | 25% | MLP (matched) | 9,971 | r2: 0.7324 ± 0.0218 | RMSE: 0.5943 ± 0.0291 | 2133 ± 1155 | 3733 | 2.9 ± 1.0 |
| california_housing | 25% | MLP (state-count) | 1,981 | r2: 0.6386 ± 0.1967 | RMSE: 0.6763 ± 0.1857 | 4267 ± 2859 | 5867 | 2.4 ± 1.1 |
| california_housing | 100% | CellV0.1 | 9,967 | r2: 0.7769 ± 0.0066 | RMSE: 0.5429 ± 0.0102 | 11467 ± 1222 | 13067 | 181.1 ± 50.6 |
| california_housing | 100% | MLP (matched) | 9,971 | r2: 0.6969 ± 0.0786 | RMSE: 0.6291 ± 0.0760 | 2800 ± 1970 | 4400 | 6.9 ± 3.3 |
| california_housing | 100% | MLP (state-count) | 1,981 | r2: 0.7500 ± 0.0191 | RMSE: 0.5743 ± 0.0199 | 4400 ± 3219 | 6000 | 3.4 ± 1.9 |
| mnist | 25% | CellV0.1 | 148,760 | accuracy: 0.9507 ± 0.0030 | F1: 0.9501 ± 0.0031 | 2400 ± 200 | 4000 | 427.8 ± 18.6 |
| mnist | 25% | MLP (matched) | 148,675 | accuracy: 0.9435 ± 0.0088 | F1: 0.9430 ± 0.0088 | 267 ± 115 | 1867 | 3.1 ± 0.2 |
| mnist | 25% | MLP (state-count) | 202,735 | accuracy: 0.9416 ± 0.0008 | F1: 0.9410 ± 0.0008 | 200 ± 0 | 1800 | 3.4 ± 0.0 |
| mnist | 100% | CellV0.1 | 148,760 | accuracy: 0.9746 ± 0.0017 | F1: 0.9743 ± 0.0018 | 7133 ± 1222 | 8733 | 769.3 ± 241.5 |
| mnist | 100% | MLP (matched) | 148,675 | accuracy: 0.9598 ± 0.0030 | F1: 0.9595 ± 0.0030 | 1200 ± 400 | 2800 | 3.4 ± 0.5 |
| mnist | 100% | MLP (state-count) | 202,735 | accuracy: 0.9575 ± 0.0056 | F1: 0.9571 ± 0.0056 | 867 ± 643 | 2467 | 3.3 ± 1.0 |
| fashion_mnist | 25% | CellV0.1 | 148,760 | accuracy: 0.8431 ± 0.0023 | F1: 0.8423 ± 0.0021 | 4000 ± 721 | 5600 | 556.6 ± 145.7 |
| fashion_mnist | 25% | MLP (matched) | 148,675 | accuracy: 0.8330 ± 0.0030 | F1: 0.8317 ± 0.0042 | 533 ± 115 | 2133 | 2.7 ± 0.1 |
| fashion_mnist | 25% | MLP (state-count) | 202,735 | accuracy: 0.8361 ± 0.0073 | F1: 0.8356 ± 0.0074 | 400 ± 200 | 2000 | 3.0 ± 0.3 |
| fashion_mnist | 100% | CellV0.1 | 148,760 | accuracy: 0.8706 ± 0.0047 | F1: 0.8695 ± 0.0049 | 10800 ± 2272 | 12400 | 956.7 ± 170.1 |
| fashion_mnist | 100% | MLP (matched) | 148,675 | accuracy: 0.8564 ± 0.0066 | F1: 0.8558 ± 0.0068 | 2000 ± 693 | 3600 | 3.0 ± 0.6 |
| fashion_mnist | 100% | MLP (state-count) | 202,735 | accuracy: 0.8532 ± 0.0032 | F1: 0.8530 ± 0.0027 | 1200 ± 0 | 2800 | 2.6 ± 0.0 |

## B. Paired comparison — CellV0.1 minus parameter-matched MLP (headline metric)

| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |
|---|---|---|---|---|---|
| breast_cancer | 25% | +0.0000 | -0.0088 | -0.0175 | **-0.0088** |
| breast_cancer | 100% | -0.0088 | +0.0175 | +0.0000 | **+0.0029** |
| wine | 25% | -0.0278 | -0.0278 | +0.0000 | **-0.0185** |
| wine | 100% | +0.0000 | +0.0000 | -0.0278 | **-0.0093** |
| digits | 25% | -0.0417 | -0.0500 | -0.0667 | **-0.0528** |
| digits | 100% | -0.0333 | -0.0389 | -0.0472 | **-0.0398** |
| diabetes | 25% | +0.3395 | +0.5979 | +0.1395 | **+0.3589** |
| diabetes | 100% | -0.0126 | +0.1849 | +0.0700 | **+0.0808** |
| california_housing | 25% | -0.0136 | +0.0274 | +0.0373 | **+0.0170** |
| california_housing | 100% | +0.1775 | +0.0264 | +0.0360 | **+0.0800** |
| mnist | 25% | +0.0029 | +0.0146 | +0.0043 | **+0.0073** |
| mnist | 100% | +0.0162 | +0.0126 | +0.0154 | **+0.0147** |
| fashion_mnist | 25% | +0.0106 | +0.0141 | +0.0056 | **+0.0101** |
| fashion_mnist | 100% | +0.0123 | +0.0215 | +0.0087 | **+0.0142** |

_Positive = CellV0.1 better. n=3 seeds; no significance test is implied (Paper-A task Sec 8)._

## C. Data-efficiency summary — headline metric at 25% → 100% training data

| Dataset | Model | 25% | 100% | Δ (100% − 25%) |
|---|---|---|---|---|
| breast_cancer | CellV0.1 | 0.9474 ± 0.0232 | 0.9708 ± 0.0134 | +0.0234 |
| breast_cancer | MLP (matched) | 0.9561 ± 0.0152 | 0.9678 ± 0.0051 | +0.0117 |
| breast_cancer | MLP (state-count) | 0.9591 ± 0.0183 | 0.9678 ± 0.0051 | +0.0088 |
| wine | CellV0.1 | 0.9444 ± 0.0000 | 0.9815 ± 0.0160 | +0.0370 |
| wine | MLP (matched) | 0.9630 ± 0.0160 | 0.9907 ± 0.0160 | +0.0278 |
| wine | MLP (state-count) | 0.9259 ± 0.0321 | 0.9907 ± 0.0160 | +0.0648 |
| digits | CellV0.1 | 0.8648 ± 0.0339 | 0.9287 ± 0.0112 | +0.0639 |
| digits | MLP (matched) | 0.9176 ± 0.0252 | 0.9685 ± 0.0080 | +0.0509 |
| digits | MLP (state-count) | 0.9231 ± 0.0098 | 0.9704 ± 0.0085 | +0.0472 |
| diabetes | CellV0.1 | 0.3759 ± 0.0750 | 0.3806 ± 0.0821 | +0.0047 |
| diabetes | MLP (matched) | 0.0170 ± 0.3021 | 0.2998 ± 0.0273 | +0.2829 |
| diabetes | MLP (state-count) | 0.1316 ± 0.2107 | 0.3428 ± 0.0296 | +0.2112 |
| california_housing | CellV0.1 | 0.7494 ± 0.0088 | 0.7769 ± 0.0066 | +0.0274 |
| california_housing | MLP (matched) | 0.7324 ± 0.0218 | 0.6969 ± 0.0786 | -0.0355 |
| california_housing | MLP (state-count) | 0.6386 ± 0.1967 | 0.7500 ± 0.0191 | +0.1114 |
| mnist | CellV0.1 | 0.9507 ± 0.0030 | 0.9746 ± 0.0017 | +0.0238 |
| mnist | MLP (matched) | 0.9435 ± 0.0088 | 0.9598 ± 0.0030 | +0.0164 |
| mnist | MLP (state-count) | 0.9416 ± 0.0008 | 0.9575 ± 0.0056 | +0.0159 |
| fashion_mnist | CellV0.1 | 0.8431 ± 0.0023 | 0.8706 ± 0.0047 | +0.0275 |
| fashion_mnist | MLP (matched) | 0.8330 ± 0.0030 | 0.8564 ± 0.0066 | +0.0234 |
| fashion_mnist | MLP (state-count) | 0.8361 ± 0.0073 | 0.8532 ± 0.0032 | +0.0172 |

## D. State-count control summary

Does a substantially wider ordinary MLP (hidden width ≈ 3 × CellV0.1 hidden cells, **not** parameter-matched) close or reverse any CellV0.1 edge over the parameter-matched MLP?

| Dataset | Frac | CellV0.1 params | State-count MLP params | ratio | CellV0.1 − matched | CellV0.1 − state-count |
|---|---|---|---|---|---|---|
| breast_cancer | 25% | 9,858 | 5,546 | 0.56× | -0.0088 | -0.0117 |
| breast_cancer | 100% | 9,858 | 5,546 | 0.56× | +0.0029 | +0.0029 |
| wine | 25% | 9,894 | 3,216 | 0.33× | -0.0185 | +0.0185 |
| wine | 100% | 9,894 | 3,216 | 0.33× | -0.0093 | -0.0093 |
| digits | 25% | 24,938 | 18,460 | 0.74× | -0.0528 | -0.0583 |
| digits | 100% | 24,938 | 18,460 | 0.74× | -0.0398 | -0.0417 |
| diabetes | 25% | 9,946 | 2,341 | 0.24× | +0.3589 | +0.2443 |
| diabetes | 100% | 9,946 | 2,341 | 0.24× | +0.0808 | +0.0378 |
| california_housing | 25% | 9,967 | 1,981 | 0.20× | +0.0170 | +0.1108 |
| california_housing | 100% | 9,967 | 1,981 | 0.20× | +0.0800 | +0.0269 |
| mnist | 25% | 148,760 | 202,735 | 1.36× | +0.0073 | +0.0091 |
| mnist | 100% | 148,760 | 202,735 | 1.36× | +0.0147 | +0.0170 |
| fashion_mnist | 25% | 148,760 | 202,735 | 1.36× | +0.0101 | +0.0070 |
| fashion_mnist | 100% | 148,760 | 202,735 | 1.36× | +0.0142 | +0.0174 |

**Verdict:** on the cells where CellV0.1 was >0.01 ahead of the parameter-matched MLP, the wider state-count MLP **does not** close that gap (diabetes 25%, diabetes 100%, california_housing 25%, california_housing 100%, mnist 100%, fashion_mnist 100%); it closes/reverses it on: fashion_mnist 25%.

_Note: for the low-feature tabular datasets the state-count MLP is actually **smaller** than CellV0.1 (CellV0.1 carries two parameters per connection — a content weight and a relevance gate — so its cell count buys fewer cells per parameter). The ratio column makes this explicit; the control is reported as specified, not resized._

## CellV0.1 internal diagnostics (Sec 9 — not an evaluation target)

| Dataset | Frac | mean e (L1/L2) | mean u (L1/L2) | mean e/(u²+ε) (L1/L2) | min/max e | min/max u |
|---|---|---|---|---|---|---|
| breast_cancer | 25% | 0.495 / 0.266 | 1.532 / 2.922 | 0.218 / 0.032 | 3.49e-01 / 0.60 | 1.32e+00 / 2.83 |
| breast_cancer | 100% | 0.498 / 0.262 | 1.500 / 2.896 | 0.226 / 0.031 | 3.66e-01 / 0.57 | 1.34e+00 / 2.31 |
| wine | 25% | 0.489 / 0.265 | 1.602 / 3.083 | 0.195 / 0.028 | 3.39e-01 / 0.61 | 1.32e+00 / 2.37 |
| wine | 100% | 0.488 / 0.282 | 1.679 / 3.113 | 0.183 / 0.030 | 2.78e-01 / 0.69 | 1.27e+00 / 3.08 |
| digits | 25% | 0.477 / 0.272 | 55.164 / 22.714 | 0.138 / 0.021 | 3.41e-01 / 0.60 | 1.51e+00 / 20971.77 |
| digits | 100% | 0.483 / 0.281 | 14.247 / 5.302 | 0.130 / 0.021 | 3.34e-01 / 0.65 | 1.51e+00 / 6757.92 |
| diabetes | 25% | 0.492 / 0.248 | 1.468 / 2.938 | 0.230 / 0.029 | 4.24e-01 / 0.56 | 1.35e+00 / 1.72 |
| diabetes | 100% | 0.478 / 0.239 | 1.559 / 3.171 | 0.200 / 0.024 | 3.89e-01 / 0.61 | 1.31e+00 / 3.36 |
| california_housing | 25% | 0.411 / 0.220 | 2.097 / 4.210 | 0.113 / 0.013 | 2.08e-01 / 0.80 | 1.15e+00 / 536.31 |
| california_housing | 100% | 0.419 / 0.235 | 2.276 / 4.460 | 0.098 / 0.013 | 2.01e-01 / 0.86 | 1.12e+00 / 528.83 |
| mnist | 25% | 0.494 / 0.284 | 2.777 / 5.247 | 0.072 / 0.011 | 2.82e-01 / 0.75 | 1.40e+00 / 6.67 |
| mnist | 100% | 0.541 / 0.339 | 3.546 / 6.073 | 0.052 / 0.010 | 2.07e-01 / 0.75 | 1.56e+00 / 9.35 |
| fashion_mnist | 25% | 0.508 / 0.313 | 2.393 / 4.413 | 0.095 / 0.017 | 3.26e-01 / 0.79 | 1.26e+00 / 5.07 |
| fashion_mnist | 100% | 0.557 / 0.357 | 2.959 / 5.163 | 0.071 / 0.014 | 3.01e-01 / 0.79 | 1.31e+00 / 6.27 |

_All finite; no zero/negative e or u observed — numerically stable._

## E. Fixed-confidence ablation — CellV0.1 vs e=u=1 control

Datasets: digits, diabetes, fashion_mnist · train fraction 25% · seeds 0,1,2. CellV0.1 arm = the Phase-1 grid record (identical deterministic conditions).

| Dataset | CellV0.1 (headline) | Fixed-confidence (headline) | Δ (CellV0.1 − fixed) | Δ by seed |
|---|---|---|---|---|
| digits | 0.8648 ± 0.0339 | 0.8657 ± 0.0343 | **-0.0009** | -0.0028, +0.0000, +0.0000 |
| diabetes | 0.3759 ± 0.0750 | 0.3765 ± 0.0742 | **-0.0006** | -0.0020, -0.0005, +0.0007 |
| fashion_mnist | 0.8431 ± 0.0023 | 0.8362 ± 0.0043 | **+0.0069** | +0.0063, +0.0051, +0.0094 |

_Positive Δ = the propagated evidence/uncertainty state helps beyond the content pathway; ≈0 = it does not, on that dataset._

## F. Duplication-invariance experiment (deterministic, no training)

Fixed source set (6 distinct beliefs): mu=[0.9, -0.4, 0.2, -0.8, 0.5, -0.1], e=[3.0, 0.5, 1.5, 2.0, 1.0, 0.8], u=[0.3, 1.8, 0.7, 1.2, 0.5, 1.0], g=[0.95, 0.08, 0.6, 0.3, 0.75, 0.15]. The whole multiset is duplicated ×m.

### `scale_stable_precision`

| m | out mu | out e | out u | e / e(×1) | u / u(×1) |
|---|---|---|---|---|---|
| 1 | 0.61490476 | 1.27669393 | 0.56301300 | 1.0000 | 1.0000 |
| 2 | 0.61490476 | 1.27669393 | 0.56301300 | 1.0000 | 1.0000 |
| 4 | 0.61490476 | 1.27669393 | 0.56301300 | 1.0000 | 1.0000 |
| 8 | 0.61490476 | 1.27669393 | 0.56301300 | 1.0000 | 1.0000 |
| 16 | 0.61490476 | 1.27669393 | 0.56301300 | 1.0000 | 1.0000 |

Verdict: mu invariant=True, e invariant=True, u invariant=True (max |Δmu|=1.13e-10, |Δe|=3.25e-09, |Δu|=2.98e-10; e ratio at ×16 = 1.000).

### `normalized_precision`

| m | out mu | out e | out u | e / e(×1) | u / u(×1) |
|---|---|---|---|---|---|
| 1 | 0.61490476 | 1.85865724 | 0.53119455 | 1.0000 | 1.0000 |
| 2 | 0.61490476 | 1.85865724 | 0.53119455 | 1.0000 | 1.0000 |
| 4 | 0.61490476 | 1.85865724 | 0.53119455 | 1.0000 | 1.0000 |
| 8 | 0.61490476 | 1.85865724 | 0.53119455 | 1.0000 | 1.0000 |
| 16 | 0.61490476 | 1.85865724 | 0.53119455 | 1.0000 | 1.0000 |

Verdict: mu invariant=True, e invariant=True, u invariant=True (max |Δmu|=1.13e-10, |Δe|=6.16e-09, |Δu|=1.89e-10; e ratio at ×16 = 1.000).

### `precision`

| m | out mu | out e | out u | e / e(×1) | u / u(×1) |
|---|---|---|---|---|---|
| 1 | 0.61490476 | 5.26000000 | 0.48247087 | 1.0000 | 1.0000 |
| 2 | 0.61490476 | 10.52000000 | 0.46827742 | 2.0000 | 0.9706 |
| 4 | 0.61490476 | 21.04000000 | 0.46101686 | 4.0000 | 0.9555 |
| 8 | 0.61490476 | 42.08000000 | 0.45734336 | 8.0000 | 0.9479 |
| 16 | 0.61490476 | 84.16000000 | 0.45549550 | 16.0000 | 0.9441 |

Verdict: mu invariant=True, e invariant=False, u invariant=False (max |Δmu|=1.13e-10, |Δe|=7.89e+01, |Δu|=2.70e-02; e ratio at ×16 = 16.000).

## G. Failures and caveats

**Divergence / NaNs:** none.

**Step-cap hits (early stopping did not terminate the run):**

- breast_cancer 25% MLP (matched) (1/3 seeds)
- breast_cancer 25% MLP (state-count) (1/3 seeds)

**Suspiciously high CellV0.1 seed variance (headline std > 3× matched-MLP std):**
 diabetes 100% (CellV0.1 std 0.0821 vs MLP 0.0273)

**Datasets where CellV0.1 clearly loses to the parameter-matched MLP (headline worse by > 0.01):**
 wine 25% (Δ -0.0185); digits 25% (Δ -0.0528); digits 100% (Δ -0.0398)

