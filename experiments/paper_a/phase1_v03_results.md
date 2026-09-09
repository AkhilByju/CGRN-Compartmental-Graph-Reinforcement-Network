# Paper A — CellV0.3 (Conflict-Normalized Belief Cell) on the frozen Phase-1 protocol

CellV0.3 runs loaded: 42 (experiment `paper_a_phase1_cellv03`). Comparison arms read verbatim from the frozen records: `paper_a_phase1` (CellV0.1, matched MLP) and `paper_a_phase1_cellv02` (CellV0.2).
Datasets present (CellV0.3): breast_cancer, wine, digits, diabetes, california_housing, mnist, fashion_mnist.

CellV0.3 = `BeliefNetworkV03` — two `ConflictNormalizedLayer`s (one signed connection matrix `V` + per-output `gain_raw` + `bias`, **no** relevance gate; CellV0.2's population-relative gain `2π/(π+mean π)` removed entirely) + a plain linear readout on `final_mu`. Each cell forms a precision-weighted signed consensus, takes the conflict (A-weighted variance of the signed messages) as `u`, derives `pi_out = e_out/(1+e_out·u_out)` and folds `sqrt(pi_out)` into its **own** `tanh` activation. Input belief `e=1, u=0`. Hidden width fitted to the **same** per-dataset parameter budget as CellV0.1 (identical to CellV0.2's). Identical shared protocol: AdamW, lr=1e-2, weight_decay=0, best-validation restore, ~1500-step early-stop patience, 15000-step cap. Nothing tuned; equations frozen before the run (docs/architecture_v0.md Sec 10).

## Compact summary — CellV0.3 headline Δ vs each recorded arm (mean of 3 seeds, 25% / 100%)

| Dataset | Δ vs MLP | Δ vs CellV0.1 | Δ vs CellV0.2 |
|---|---|---|---|
| breast_cancer | -0.0058 / +0.0000 | +0.0029 / -0.0029 | +0.0000 / +0.0000 |
| wine | +0.0000 / -0.0093 | +0.0185 / +0.0000 | +0.0093 / +0.0000 |
| digits | +0.0019 / -0.0028 | +0.0546 / +0.0370 | -0.0028 / -0.0028 |
| diabetes | +0.1559 / +0.0743 | -0.2031 / -0.0065 | -0.0573 / +0.0365 |
| california_housing | +0.0409 / +0.0937 | +0.0238 / +0.0137 | -0.0013 / -0.0065 |
| mnist | +0.0172 / +0.0169 | +0.0100 / +0.0022 | +0.0000 / -0.0004 |
| fashion_mnist | +0.0228 / +0.0247 | +0.0127 / +0.0105 | +0.0029 / +0.0058 |

_Positive = CellV0.3 better. n=3 seeds; no significance test._

## Headline signal — CellV0.3 vs the parameter-matched MLP

| Dataset | Δ (25% / 100%) | Read |
|---|---|---|
| breast_cancer | -0.0058 / +0.0000 | tie |
| wine | +0.0000 / -0.0093 | tie |
| digits | +0.0019 / -0.0028 | tie |
| diabetes | +0.1559 / +0.0743 | win (both fractions) |
| california_housing | +0.0409 / +0.0937 | win (both fractions) |
| mnist | +0.0172 / +0.0169 | win (both fractions) |
| fashion_mnist | +0.0228 / +0.0247 | win (both fractions) |

Across all 14 (dataset, fraction) cells vs the parameter-matched MLP: CellV0.3 ahead by >0.01 in 8, behind by >0.01 in 0, within ±0.01 in 6. Reporting threshold only; no significance test (n=3).

## A. Main table — CellV0.3

| Dataset | Frac | Hidden cells | Params | Headline (mean ± std) | 2nd metric | Best-val step | Total steps | Train wall-clock (s) | Time/step (ms) |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 25% | 83 | 9,879 | accuracy: 0.9503 ± 0.0253 | F1: 0.9471 ± 0.0249 | 233 ± 202 | 1717 | 1.5 ± 0.1 | 0.86 |
| breast_cancer | 100% | 83 | 9,879 | accuracy: 0.9678 ± 0.0134 | F1: 0.9656 ± 0.0142 | 1350 ± 2252 | 2833 | 3.1 ± 2.6 | 1.09 |
| wine | 25% | 90 | 9,903 | accuracy: 0.9630 ± 0.0160 | F1: 0.9646 ± 0.0168 | 1017 ± 1631 | 2017 | 1.3 ± 0.5 | 0.65 |
| wine | 100% | 90 | 9,903 | accuracy: 0.9815 ± 0.0160 | F1: 0.9829 ± 0.0148 | 850 ± 1300 | 1833 | 1.9 ± 0.5 | 1.03 |
| digits | 25% | 123 | 24,733 | accuracy: 0.9194 ± 0.0147 | F1: 0.9190 ± 0.0145 | 183 ± 104 | 1683 | 1.9 ± 0.2 | 1.14 |
| digits | 100% | 123 | 24,733 | accuracy: 0.9657 ± 0.0070 | F1: 0.9655 ± 0.0071 | 83 ± 29 | 1583 | 2.0 ± 0.1 | 1.26 |
| diabetes | 25% | 92 | 9,845 | r2: 0.1728 ± 0.1103 | RMSE: 66.2578 ± 4.1232 | 50 ± 0 | 1550 | 1.2 ± 0.0 | 0.76 |
| diabetes | 100% | 92 | 9,845 | r2: 0.3741 ± 0.0205 | RMSE: 57.7196 ± 0.2034 | 300 ± 50 | 1800 | 1.5 ± 0.1 | 0.85 |
| california_housing | 25% | 93 | 9,859 | r2: 0.7733 ± 0.0130 | RMSE: 0.5471 ± 0.0158 | 3867 ± 1724 | 5467 | 5.6 ± 1.7 | 1.03 |
| california_housing | 100% | 93 | 9,859 | r2: 0.7906 ± 0.0048 | RMSE: 0.5260 ± 0.0093 | 6467 ± 1890 | 8067 | 8.4 ± 1.8 | 1.05 |
| mnist | 25% | 157 | 149,945 | accuracy: 0.9607 ± 0.0077 | F1: 0.9604 ± 0.0077 | 1067 ± 987 | 2667 | 9.7 ± 2.9 | 3.71 |
| mnist | 100% | 157 | 149,945 | accuracy: 0.9767 ± 0.0015 | F1: 0.9766 ± 0.0015 | 2333 ± 115 | 3933 | 12.0 ± 0.7 | 3.05 |
| fashion_mnist | 25% | 157 | 149,945 | accuracy: 0.8558 ± 0.0060 | F1: 0.8572 ± 0.0053 | 1200 ± 200 | 2800 | 10.2 ± 0.8 | 3.64 |
| fashion_mnist | 100% | 157 | 149,945 | accuracy: 0.8811 ± 0.0063 | F1: 0.8811 ± 0.0069 | 4067 ± 1501 | 5667 | 18.3 ± 4.5 | 3.25 |

## Paired comparison — CellV0.3 minus the parameter-matched MLP (headline metric)

| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |
|---|---|---|---|---|---|
| breast_cancer | 25% | +0.0000 | +0.0000 | -0.0175 | **-0.0058** |
| breast_cancer | 100% | -0.0088 | +0.0175 | -0.0088 | **-0.0000** |
| wine | 25% | -0.0278 | +0.0000 | +0.0278 | **+0.0000** |
| wine | 100% | +0.0000 | +0.0000 | -0.0278 | **-0.0093** |
| digits | 25% | +0.0000 | -0.0083 | +0.0139 | **+0.0019** |
| digits | 100% | -0.0056 | -0.0111 | +0.0083 | **-0.0028** |
| diabetes | 25% | +0.1753 | +0.3392 | -0.0469 | **+0.1559** |
| diabetes | 100% | +0.0501 | +0.1008 | +0.0720 | **+0.0743** |
| california_housing | 25% | +0.0291 | +0.0276 | +0.0659 | **+0.0409** |
| california_housing | 100% | +0.1899 | +0.0432 | +0.0480 | **+0.0937** |
| mnist | 25% | +0.0099 | +0.0341 | +0.0077 | **+0.0172** |
| mnist | 100% | +0.0169 | +0.0153 | +0.0185 | **+0.0169** |
| fashion_mnist | 25% | +0.0175 | +0.0232 | +0.0277 | **+0.0228** |
| fashion_mnist | 100% | +0.0212 | +0.0340 | +0.0188 | **+0.0247** |

_Positive = CellV0.3 better than the parameter-matched MLP. n=3 seeds; no significance test._

## Paired comparison — CellV0.3 minus CellV0.1 (recorded) (headline metric)

| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |
|---|---|---|---|---|---|
| breast_cancer | 25% | +0.0000 | +0.0088 | +0.0000 | **+0.0029** |
| breast_cancer | 100% | +0.0000 | +0.0000 | -0.0088 | **-0.0029** |
| wine | 25% | +0.0000 | +0.0278 | +0.0278 | **+0.0185** |
| wine | 100% | +0.0000 | +0.0000 | +0.0000 | **+0.0000** |
| digits | 25% | +0.0417 | +0.0417 | +0.0806 | **+0.0546** |
| digits | 100% | +0.0278 | +0.0278 | +0.0556 | **+0.0370** |
| diabetes | 25% | -0.1642 | -0.2587 | -0.1864 | **-0.2031** |
| diabetes | 100% | +0.0627 | -0.0841 | +0.0019 | **-0.0065** |
| california_housing | 25% | +0.0427 | +0.0002 | +0.0286 | **+0.0238** |
| california_housing | 100% | +0.0123 | +0.0167 | +0.0120 | **+0.0137** |
| mnist | 25% | +0.0070 | +0.0195 | +0.0034 | **+0.0100** |
| mnist | 100% | +0.0007 | +0.0027 | +0.0031 | **+0.0022** |
| fashion_mnist | 25% | +0.0069 | +0.0091 | +0.0221 | **+0.0127** |
| fashion_mnist | 100% | +0.0089 | +0.0125 | +0.0101 | **+0.0105** |

_Positive = CellV0.3 better than CellV0.1 (recorded). n=3 seeds; no significance test._

## Paired comparison — CellV0.3 minus CellV0.2 (recorded) (headline metric)

| Dataset | Frac | Δ seed 0 | Δ seed 1 | Δ seed 2 | Δ mean |
|---|---|---|---|---|---|
| breast_cancer | 25% | +0.0000 | +0.0000 | +0.0000 | **+0.0000** |
| breast_cancer | 100% | -0.0088 | +0.0088 | +0.0000 | **-0.0000** |
| wine | 25% | +0.0000 | +0.0000 | +0.0278 | **+0.0093** |
| wine | 100% | +0.0000 | +0.0000 | +0.0000 | **+0.0000** |
| digits | 25% | -0.0056 | +0.0028 | -0.0056 | **-0.0028** |
| digits | 100% | -0.0056 | -0.0028 | +0.0000 | **-0.0028** |
| diabetes | 25% | +0.0197 | -0.1230 | -0.0685 | **-0.0573** |
| diabetes | 100% | -0.0199 | +0.0949 | +0.0345 | **+0.0365** |
| california_housing | 25% | +0.0110 | -0.0048 | -0.0102 | **-0.0013** |
| california_housing | 100% | -0.0012 | -0.0035 | -0.0149 | **-0.0065** |
| mnist | 25% | +0.0007 | -0.0002 | -0.0005 | **-0.0000** |
| mnist | 100% | -0.0020 | +0.0010 | -0.0003 | **-0.0004** |
| fashion_mnist | 25% | -0.0002 | +0.0065 | +0.0024 | **+0.0029** |
| fashion_mnist | 100% | -0.0007 | +0.0176 | +0.0004 | **+0.0058** |

_Positive = CellV0.3 better than CellV0.2 (recorded). n=3 seeds; no significance test._

## Data-efficiency — headline at 25% → 100% training data

| Dataset | Model | 25% | 100% | Δ (100% − 25%) |
|---|---|---|---|---|
| breast_cancer | CellV0.3 | 0.9503 ± 0.0253 | 0.9678 ± 0.0134 | +0.0175 |
| breast_cancer | CellV0.2 (recorded) | 0.9503 ± 0.0253 | 0.9678 ± 0.0051 | +0.0175 |
| breast_cancer | CellV0.1 (recorded) | 0.9474 ± 0.0232 | 0.9708 ± 0.0134 | +0.0234 |
| breast_cancer | MLP matched (recorded) | 0.9561 ± 0.0152 | 0.9678 ± 0.0051 | +0.0117 |
| wine | CellV0.3 | 0.9630 ± 0.0160 | 0.9815 ± 0.0160 | +0.0185 |
| wine | CellV0.2 (recorded) | 0.9537 ± 0.0160 | 0.9815 ± 0.0160 | +0.0278 |
| wine | CellV0.1 (recorded) | 0.9444 ± 0.0000 | 0.9815 ± 0.0160 | +0.0370 |
| wine | MLP matched (recorded) | 0.9630 ± 0.0160 | 0.9907 ± 0.0160 | +0.0278 |
| digits | CellV0.3 | 0.9194 ± 0.0147 | 0.9657 ± 0.0070 | +0.0463 |
| digits | CellV0.2 (recorded) | 0.9222 ± 0.0100 | 0.9685 ± 0.0042 | +0.0463 |
| digits | CellV0.1 (recorded) | 0.8648 ± 0.0339 | 0.9287 ± 0.0112 | +0.0639 |
| digits | MLP matched (recorded) | 0.9176 ± 0.0252 | 0.9685 ± 0.0080 | +0.0509 |
| diabetes | CellV0.3 | 0.1728 ± 0.1103 | 0.3741 ± 0.0205 | +0.2013 |
| diabetes | CellV0.2 (recorded) | 0.2301 ± 0.0962 | 0.3376 ± 0.0537 | +0.1075 |
| diabetes | CellV0.1 (recorded) | 0.3759 ± 0.0750 | 0.3806 ± 0.0821 | +0.0047 |
| diabetes | MLP matched (recorded) | 0.0170 ± 0.3021 | 0.2998 ± 0.0273 | +0.2829 |
| california_housing | CellV0.3 | 0.7733 ± 0.0130 | 0.7906 ± 0.0048 | +0.0173 |
| california_housing | CellV0.2 (recorded) | 0.7746 ± 0.0120 | 0.7971 ± 0.0063 | +0.0225 |
| california_housing | CellV0.1 (recorded) | 0.7494 ± 0.0088 | 0.7769 ± 0.0066 | +0.0274 |
| california_housing | MLP matched (recorded) | 0.7324 ± 0.0218 | 0.6969 ± 0.0786 | -0.0355 |
| mnist | CellV0.3 | 0.9607 ± 0.0077 | 0.9767 ± 0.0015 | +0.0160 |
| mnist | CellV0.2 (recorded) | 0.9607 ± 0.0075 | 0.9772 ± 0.0011 | +0.0165 |
| mnist | CellV0.1 (recorded) | 0.9507 ± 0.0030 | 0.9746 ± 0.0017 | +0.0238 |
| mnist | MLP matched (recorded) | 0.9435 ± 0.0088 | 0.9598 ± 0.0030 | +0.0164 |
| fashion_mnist | CellV0.3 | 0.8558 ± 0.0060 | 0.8811 ± 0.0063 | +0.0253 |
| fashion_mnist | CellV0.2 (recorded) | 0.8529 ± 0.0071 | 0.8753 ± 0.0069 | +0.0224 |
| fashion_mnist | CellV0.1 (recorded) | 0.8431 ± 0.0023 | 0.8706 ± 0.0047 | +0.0275 |
| fashion_mnist | MLP matched (recorded) | 0.8330 ± 0.0030 | 0.8564 ± 0.0066 | +0.0234 |

## B. Parameter count & hidden width — CellV0.3 vs CellV0.2 vs CellV0.1 (same budget)

| Dataset | Budget | V0.3 hidden | V0.3 params | V0.2 hidden | V0.1 hidden | V0.1 params | V0.3/V0.1 cell ratio |
|---|---|---|---|---|---|---|---|
| breast_cancer | 10,000 | 83 | 9,879 | 83 | 56 | 9,858 | 1.48× |
| wine | 10,000 | 90 | 9,903 | 90 | 63 | 9,894 | 1.43× |
| digits | 25,000 | 123 | 24,733 | 123 | 82 | 24,938 | 1.50× |
| diabetes | 10,000 | 92 | 9,845 | 92 | 65 | 9,946 | 1.42× |
| california_housing | 10,000 | 93 | 9,859 | 93 | 66 | 9,967 | 1.41× |
| mnist | 150,000 | 157 | 149,945 | 157 | 85 | 148,760 | 1.85× |
| fashion_mnist | 150,000 | 157 | 149,945 | 157 | 85 | 148,760 | 1.85× |

_CellV0.3's parameterization is identical to CellV0.2's (`hc·(in+2) + hc·(hc+2) + hc·out + out`), so its hidden width matches CellV0.2's exactly and is ~1.5× CellV0.1's at the same budget — its lower per-connection cost is part of the architecture, not equalized._

## Most important — does CellV0.3 use its belief state? (Sec 19)

Per (dataset, fraction), mean over 3 seeds, **untrained snapshot → best checkpoint**. `pi_out = e_out/(1+e_out·u_out)`; CoV = `std(pi_out)/(mean(pi_out)+ε)` across the whole test set × cells; `k = sqrt(pi_out)` is the factor folded into each cell's activation.

| Dataset | Frac | L2 pi_out mean | L2 CoV(pi_out) | L2 k=√pi_out mean | L2 conflict u_out mean | L1 pi_out mean | Read (L2 CoV) |
|---|---|---|---|---|---|---|---|
| breast_cancer | 25% | 0.566 → 0.591 | 0.277 → 0.245 | 0.743 → 0.762 | 0.106 → 0.142 | 0.598 → 0.638 | varies (0.24) |
| breast_cancer | 100% | 0.570 → 0.563 | 0.272 → 0.248 | 0.747 → 0.743 | 0.105 → 0.226 | 0.603 → 0.637 | varies (0.25) |
| wine | 25% | 0.520 → 0.543 | 0.193 → 0.173 | 0.718 → 0.734 | 0.134 → 0.176 | 0.557 → 0.595 | varies (0.17) |
| wine | 100% | 0.526 → 0.548 | 0.185 → 0.167 | 0.722 → 0.738 | 0.132 → 0.181 | 0.563 → 0.604 | varies (0.17) |
| digits | 25% | 0.537 → 0.473 | 0.192 → 0.171 | 0.726 → 0.683 | 0.111 → 0.258 | 0.571 → 0.539 | varies (0.17) |
| digits | 100% | 0.540 → 0.479 | 0.174 → 0.151 | 0.730 → 0.689 | 0.112 → 0.259 | 0.574 → 0.546 | varies (0.15) |
| diabetes | 25% | 0.520 → 0.520 | 0.233 → 0.224 | 0.716 → 0.716 | 0.148 → 0.175 | 0.560 → 0.568 | varies (0.22) |
| diabetes | 100% | 0.537 → 0.539 | 0.221 → 0.212 | 0.728 → 0.729 | 0.143 → 0.148 | 0.579 → 0.582 | varies (0.21) |
| california_housing | 25% | 0.636 → 0.636 | 0.210 → 0.178 | 0.792 → 0.793 | 0.112 → 0.169 | 0.679 → 0.710 | varies (0.18) |
| california_housing | 100% | 0.645 → 0.639 | 0.208 → 0.177 | 0.798 → 0.795 | 0.108 → 0.156 | 0.689 → 0.706 | varies (0.18) |
| mnist | 25% | 0.480 → 0.336 | 0.150 → 0.162 | 0.691 → 0.577 | 0.122 → 0.648 | 0.509 → 0.429 | varies (0.16) |
| mnist | 100% | 0.479 → 0.331 | 0.151 → 0.159 | 0.690 → 0.574 | 0.122 → 0.680 | 0.509 → 0.427 | varies (0.16) |
| fashion_mnist | 25% | 0.493 → 0.394 | 0.200 → 0.177 | 0.698 → 0.625 | 0.120 → 0.638 | 0.523 → 0.527 | varies (0.18) |
| fashion_mnist | 100% | 0.493 → 0.388 | 0.200 → 0.171 | 0.698 → 0.621 | 0.120 → 0.666 | 0.523 → 0.526 | varies (0.17) |

Of 14 (dataset, fraction) cells, layer-2 output precision **varies** (CoV ≥ 0.05) in 14 and is **effectively constant** (CoV < 0.05) in 0 at the best checkpoint. A near-constant precision pathway is not a success even where accuracy is good — see the verdict in `docs/research_log.md`.

_Layer 1's inherited support `e` is exactly 1 by construction (every input feature starts at precision 1 and `e_out` is a convex combination), so layer 1 only ever varies `pi_out` through its conflict `u`._

## C. Failures and caveats (CellV0.3)

**Divergence / NaNs:** none.

**Step-cap hits:** none.

**Numeric-range flag** (a single test example × cell with output precision `< 1e-6` — large local conflict `u`; all values finite, no NaN/divergence, no protocol impact):

- digits 25% seed0 layer1 (min π = 2.4e-11)
- digits 25% seed0 layer2 (min π = 1.0e-08)
- digits 25% seed1 layer1 (min π = 1.9e-10)
- digits 25% seed1 layer2 (min π = 1.6e-07)
- digits 25% seed2 layer1 (min π = 1.2e-11)
- digits 25% seed2 layer2 (min π = 1.0e-08)
- digits 100% seed0 layer1 (min π = 1.0e-10)
- digits 100% seed0 layer2 (min π = 1.0e-08)
- digits 100% seed2 layer1 (min π = 1.0e-10)
- digits 100% seed2 layer2 (min π = 1.0e-08)

