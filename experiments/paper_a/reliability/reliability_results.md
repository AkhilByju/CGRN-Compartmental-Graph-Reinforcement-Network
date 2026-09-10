# Paper A — Phase 2: reliability / corruption benchmark for CellV0.3

Runs loaded: 96 (experiment `paper_a_reliability`). Datasets present: mnist, fashion_mnist, digits, california_housing.

**Question.** When input information has heterogeneous and *known* reliability, does explicitly propagating reliability/conflict through CellV0.3 provide a useful inductive bias beyond conventional networks given the same corrupted observations and the same reliability information?

**Four families, all at the CellV0.3 parameter count** (fitted to the Phase-1 per-dataset budget):

| Family | Input | Reliability seen |
|---|---|---|
| Plain MLP (A) | `x_corrupted` | none |
| Confidence MLP (B) | `concat(x_corrupted, c)` | exact `c` |
| Reliability-Gated MLP (C) | `c * x_corrupted` | exploited directly |
| CellV0.3 (D) | belief `(mu=x_corrupted, e=c, u=0)` | exact `c` |

**Note — Model C ≡ Model A under missing-feature corruption.** A missing value is zero-imputed, and `c·0 = 0`; an observed value has `c = 1`. So `c · x_corrupted == x_corrupted` exactly, and with the same architecture, seed and corruption stream Model C's outputs are byte-identical to Model A's on missingness. Model C is only a distinct baseline under heterogeneous Gaussian noise (where `c_j < 1` attenuates noisy-but-nonzero features).

**Two corruption families**, applied after Phase-1 preprocessing: missing-feature (drop-to-zero, `c = 1.0`/`1e-3`) and heterogeneous Gaussian (`sigma_j ~ U(0, s)` per feature, `c_j = 1/(1+sigma_j^2)`). Training severity is drawn per example from the discrete training regime; evaluation uses the fixed grids `p ∈ {0, 0.1, 0.3, 0.5, 0.7}` and `s ∈ {0, 0.25, 0.5, 0.75, 1, 1.5}` (3 deterministic corruption replicas per severity, identical across models, averaged before any across-seed statistic).

Shared frozen protocol: AdamW, lr=1e-2, weight_decay=0, batch 128, best-checkpoint restore on the averaged in-distribution corrupted-validation primary metric, ~1500-step early-stop patience, 15000-step cap. Nothing tuned per model or dataset. CellV0.3's equations are frozen; only its *input belief* carries the reliability signal (Phase-2 task Sec 1). n=3 seeds; no significance test.

**Interpretation boundary (Sec 18).** A CellV0.3 win over the plain MLP is *not* sufficient — the plain MLP has no reliability signal. The meaningful comparisons are vs the Confidence MLP and the Reliability-Gated MLP. `e`/`u`/`pi` are internal computational reliability variables, not calibrated predictive probabilities.

## Compact summary — CellV0.3 corruption-AUC Δ vs each baseline (primary metric, mean of 3 seeds)

| Dataset | Corruption | Δ vs Plain | Δ vs Confidence MLP | Δ vs Reliability-Gated |
|---|---|---|---|---|
| mnist | missing | +0.0149 | +0.0303 | +0.0149 ‡ |
| mnist | gaussian | +0.0288 | +0.0753 | +0.0250 |
| fashion_mnist | missing | +0.0188 | +0.0253 | +0.0188 ‡ |
| fashion_mnist | gaussian | +0.0462 | +0.0655 | +0.0401 |
| digits | missing | +0.0065 | -0.0106 | +0.0065 ‡ |
| digits | gaussian | -0.0005 | +0.0094 | +0.0002 |
| california_housing | missing | +11.7941 † | +0.0251 | +11.7941 † |
| california_housing | gaussian | +32.6113 † | +0.2172 | +1.5466 † |

_Positive = CellV0.3 has the larger area under the severity/metric curve (primary metric: accuracy / R², both higher-is-better). The authoritative win/tie/loss counts on the stable, non-degenerate comparisons are in the go/no-go section below._
_† baseline diverged on some seeds (see Failures) — the Δ is not a CellV0.3 advantage. ‡ Reliability-Gated MLP is byte-identical to the Plain MLP under missingness (see header note)._

## A. Missingness

Primary metric (accuracy / R²), mean ± std over 3 seeds, at each severity, plus trapezoidal corruption-AUC over the grid.

| Dataset | Model | 0% | 10% | 30% | 50% | 70% | corruption AUC |
|---|---|---|---|---|---|---|---|
| mnist | Plain MLP | 0.965±0.002 | 0.962±0.003 | 0.950±0.001 | 0.922±0.000 | 0.834±0.002 | 0.6505±0.0010 |
| mnist | Confidence MLP | 0.949±0.005 | 0.945±0.005 | 0.930±0.006 | 0.898±0.007 | 0.804±0.011 | 0.6351±0.0030 |
| mnist | Reliability-Gated MLP | 0.965±0.002 | 0.962±0.003 | 0.950±0.001 | 0.922±0.000 | 0.834±0.002 | 0.6505±0.0010 |
| mnist | CellV0.3 | 0.980±0.001 | 0.978±0.001 | 0.970±0.001 | 0.948±0.002 | 0.861±0.005 | 0.6655±0.0010 |
| fashion_mnist | Plain MLP | 0.855±0.002 | 0.851±0.002 | 0.837±0.002 | 0.811±0.005 | 0.741±0.008 | 0.5742±0.0024 |
| fashion_mnist | Confidence MLP | 0.843±0.005 | 0.838±0.006 | 0.825±0.005 | 0.802±0.004 | 0.744±0.008 | 0.5676±0.0019 |
| fashion_mnist | Reliability-Gated MLP | 0.855±0.002 | 0.851±0.002 | 0.837±0.002 | 0.811±0.005 | 0.741±0.008 | 0.5742±0.0024 |
| fashion_mnist | CellV0.3 | 0.883±0.008 | 0.878±0.007 | 0.864±0.005 | 0.838±0.000 | 0.768±0.013 | 0.5929±0.0013 |
| digits | Plain MLP | 0.974±0.004 | 0.965±0.014 | 0.935±0.018 | 0.857±0.012 | 0.666±0.011 | 0.6185±0.0069 |
| digits | Confidence MLP | 0.979±0.007 | 0.971±0.007 | 0.950±0.009 | 0.890±0.005 | 0.731±0.014 | 0.6356±0.0022 |
| digits | Reliability-Gated MLP | 0.974±0.004 | 0.965±0.014 | 0.935±0.018 | 0.857±0.012 | 0.666±0.011 | 0.6185±0.0069 |
| digits | CellV0.3 | 0.972±0.005 | 0.966±0.004 | 0.935±0.011 | 0.872±0.017 | 0.702±0.026 | 0.6251±0.0091 |
| california_housing | Plain MLP | -22.468±40.060 | -22.533±40.063 | -23.005±40.606 | -7.624±13.674 | -7.759±13.606 | -11.4051±20.2290 |
| california_housing | Confidence MLP | 0.754±0.005 | 0.707±0.006 | 0.577±0.013 | 0.411±0.009 | 0.225±0.008 | 0.3639±0.0063 |
| california_housing | Reliability-Gated MLP | -22.468±40.060 | -22.533±40.063 | -23.005±40.606 | -7.624±13.674 | -7.759±13.606 | -11.4051±20.2290 |
| california_housing | CellV0.3 | 0.780±0.008 | 0.737±0.008 | 0.617±0.008 | 0.454±0.016 | 0.252±0.011 | 0.3890±0.0073 |

## B. Heterogeneous Gaussian noise

Primary metric (accuracy / R²), mean ± std over 3 seeds, at each severity, plus trapezoidal corruption-AUC over the grid.

| Dataset | Model | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.5 | corruption AUC |
|---|---|---|---|---|---|---|---|---|
| mnist | Plain MLP | 0.964±0.002 | 0.964±0.002 | 0.962±0.002 | 0.958±0.002 | 0.952±0.003 | 0.931±0.005 | 1.4308±0.0040 |
| mnist | Confidence MLP | 0.937±0.007 | 0.936±0.008 | 0.933±0.007 | 0.928±0.007 | 0.920±0.007 | 0.892±0.010 | 1.3843±0.0110 |
| mnist | Reliability-Gated MLP | 0.965±0.003 | 0.964±0.003 | 0.962±0.004 | 0.958±0.004 | 0.954±0.004 | 0.940±0.005 | 1.4346±0.0058 |
| mnist | CellV0.3 | 0.978±0.002 | 0.978±0.002 | 0.977±0.002 | 0.975±0.002 | 0.972±0.003 | 0.962±0.003 | 1.4596±0.0032 |
| fashion_mnist | Plain MLP | 0.857±0.004 | 0.855±0.003 | 0.851±0.004 | 0.844±0.003 | 0.834±0.002 | 0.806±0.001 | 1.2587±0.0031 |
| fashion_mnist | Confidence MLP | 0.843±0.003 | 0.841±0.003 | 0.836±0.003 | 0.831±0.005 | 0.822±0.006 | 0.795±0.010 | 1.2394±0.0077 |
| fashion_mnist | Reliability-Gated MLP | 0.858±0.002 | 0.857±0.002 | 0.852±0.002 | 0.846±0.003 | 0.837±0.004 | 0.819±0.009 | 1.2648±0.0041 |
| fashion_mnist | CellV0.3 | 0.884±0.000 | 0.883±0.001 | 0.878±0.001 | 0.872±0.001 | 0.865±0.001 | 0.847±0.001 | 1.3049±0.0006 |
| digits | Plain MLP | 0.980±0.003 | 0.978±0.002 | 0.969±0.004 | 0.963±0.005 | 0.941±0.006 | 0.877±0.010 | 1.4221±0.0075 |
| digits | Confidence MLP | 0.969±0.003 | 0.971±0.002 | 0.966±0.001 | 0.955±0.002 | 0.936±0.003 | 0.868±0.006 | 1.4122±0.0024 |
| digits | Reliability-Gated MLP | 0.974±0.003 | 0.971±0.007 | 0.966±0.008 | 0.956±0.010 | 0.940±0.009 | 0.896±0.007 | 1.4213±0.0115 |
| digits | CellV0.3 | 0.971±0.008 | 0.970±0.008 | 0.962±0.008 | 0.953±0.008 | 0.941±0.011 | 0.903±0.014 | 1.4216±0.0141 |
| california_housing | Plain MLP | -21.001±37.522 | -21.010±37.486 | -21.067±37.458 | -21.163±37.455 | -21.294±37.494 | -21.681±37.760 | -31.8405±56.2902 |
| california_housing | Confidence MLP | 0.640±0.070 | 0.590±0.077 | 0.502±0.088 | 0.414±0.104 | 0.311±0.125 | -0.078±0.256 | 0.5535±0.1859 |
| california_housing | Reliability-Gated MLP | 0.164±0.831 | 0.013±1.045 | -0.298±1.466 | -0.601±1.855 | -0.827±2.120 | -1.059±2.322 | -0.7758±2.5711 |
| california_housing | CellV0.3 | 0.726±0.018 | 0.662±0.010 | 0.579±0.001 | 0.500±0.008 | 0.433±0.016 | 0.331±0.027 | 0.7707±0.0142 |

## C. CellV0.3 vs the baselines — paired difference at every severity

Per (dataset, corruption, severity): mean over 3 seeds of `primary(CellV0.3, seed) − primary(baseline, seed)`. The last two columns are the scientifically meaningful ones (Sec 18).

| Dataset | Corruption | Severity | V03 − Plain | V03 − Confidence MLP | V03 − Reliability-Gated |
|---|---|---|---|---|---|
| mnist | missing | 0% | +0.0150 | +0.0311 | +0.0150 |
| mnist | missing | 10% | +0.0157 | +0.0331 | +0.0157 |
| mnist | missing | 30% | +0.0195 | +0.0396 | +0.0195 |
| mnist | missing | 50% | +0.0263 | +0.0508 | +0.0263 |
| mnist | missing | 70% | +0.0266 | +0.0572 | +0.0266 |
| mnist | gaussian | 0 | +0.0137 | +0.0412 | +0.0130 |
| mnist | gaussian | 0.25 | +0.0140 | +0.0420 | +0.0134 |
| mnist | gaussian | 0.5 | +0.0154 | +0.0440 | +0.0148 |
| mnist | gaussian | 0.75 | +0.0175 | +0.0471 | +0.0168 |
| mnist | gaussian | 1 | +0.0200 | +0.0515 | +0.0179 |
| mnist | gaussian | 1.5 | +0.0315 | +0.0704 | +0.0218 |
| fashion_mnist | missing | 0% | +0.0282 | +0.0404 | +0.0282 |
| fashion_mnist | missing | 10% | +0.0272 | +0.0401 | +0.0272 |
| fashion_mnist | missing | 30% | +0.0267 | +0.0384 | +0.0267 |
| fashion_mnist | missing | 50% | +0.0263 | +0.0361 | +0.0263 |
| fashion_mnist | missing | 70% | +0.0269 | +0.0241 | +0.0269 |
| fashion_mnist | gaussian | 0 | +0.0278 | +0.0416 | +0.0262 |
| fashion_mnist | gaussian | 0.25 | +0.0278 | +0.0415 | +0.0258 |
| fashion_mnist | gaussian | 0.5 | +0.0275 | +0.0420 | +0.0262 |
| fashion_mnist | gaussian | 0.75 | +0.0287 | +0.0414 | +0.0266 |
| fashion_mnist | gaussian | 1 | +0.0305 | +0.0427 | +0.0273 |
| fashion_mnist | gaussian | 1.5 | +0.0411 | +0.0521 | +0.0280 |
| digits | missing | 0% | -0.0019 | -0.0065 | -0.0019 |
| digits | missing | 10% | +0.0012 | -0.0052 | +0.0012 |
| digits | missing | 30% | -0.0006 | -0.0154 | -0.0006 |
| digits | missing | 50% | +0.0148 | -0.0176 | +0.0148 |
| digits | missing | 70% | +0.0358 | -0.0287 | +0.0358 |
| digits | gaussian | 0 | -0.0083 | +0.0019 | -0.0028 |
| digits | gaussian | 0.25 | -0.0077 | -0.0009 | -0.0003 |
| digits | gaussian | 0.5 | -0.0071 | -0.0043 | -0.0037 |
| digits | gaussian | 0.75 | -0.0099 | -0.0012 | -0.0022 |
| digits | gaussian | 1 | +0.0003 | +0.0052 | +0.0009 |
| digits | gaussian | 1.5 | +0.0262 | +0.0352 | +0.0071 |
| california_housing | missing | 0% | +23.2479 | +0.0259 | +23.2479 |
| california_housing | missing | 10% | +23.2703 | +0.0303 | +23.2703 |
| california_housing | missing | 30% | +23.6218 | +0.0398 | +23.6218 |
| california_housing | missing | 50% | +8.0784 | +0.0432 | +8.0784 |
| california_housing | missing | 70% | +8.0114 | +0.0269 | +8.0114 |
| california_housing | gaussian | 0 | +21.7265 | +0.0857 | +0.5619 |
| california_housing | gaussian | 0.25 | +21.6717 | +0.0719 | +0.6493 |
| california_housing | gaussian | 0.5 | +21.6461 | +0.0769 | +0.8765 |
| california_housing | gaussian | 0.75 | +21.6629 | +0.0857 | +1.1004 |
| california_housing | gaussian | 1 | +21.7265 | +0.1218 | +1.2599 |
| california_housing | gaussian | 1.5 | +22.0115 | +0.4090 | +1.3892 |

_Positive = CellV0.3 better. n=3 seeds; no significance test._

## D. OOD degradation — max training severity → most severe test

`OOD_drop = primary(max-train-severity) − primary(most-severe-test)` (missingness `p: .3→.7`, Gaussian `s: .75→1.5`). Smaller is better.

| Dataset | Corruption | Model | at max-train | at max-OOD | OOD_drop (mean ± std) |
|---|---|---|---|---|---|
| mnist | missing | Plain MLP | 0.950 | 0.834 | +0.1160 ± 0.0023 |
| mnist | missing | Confidence MLP | 0.930 | 0.804 | +0.1266 ± 0.0155 |
| mnist | missing | Reliability-Gated MLP | 0.950 | 0.834 | +0.1160 ± 0.0023 |
| mnist | missing | CellV0.3 | 0.970 | 0.861 | +0.1090 ± 0.0053 |
| mnist | gaussian | Plain MLP | 0.958 | 0.931 | +0.0268 ± 0.0031 |
| mnist | gaussian | Confidence MLP | 0.928 | 0.892 | +0.0362 ± 0.0059 |
| mnist | gaussian | Reliability-Gated MLP | 0.958 | 0.940 | +0.0179 ± 0.0009 |
| mnist | gaussian | CellV0.3 | 0.975 | 0.962 | +0.0129 ± 0.0015 |
| fashion_mnist | missing | Plain MLP | 0.837 | 0.741 | +0.0960 ± 0.0057 |
| fashion_mnist | missing | Confidence MLP | 0.825 | 0.744 | +0.0815 ± 0.0121 |
| fashion_mnist | missing | Reliability-Gated MLP | 0.837 | 0.741 | +0.0960 ± 0.0057 |
| fashion_mnist | missing | CellV0.3 | 0.864 | 0.768 | +0.0957 ± 0.0176 |
| fashion_mnist | gaussian | Plain MLP | 0.844 | 0.806 | +0.0376 ± 0.0033 |
| fashion_mnist | gaussian | Confidence MLP | 0.831 | 0.795 | +0.0359 ± 0.0051 |
| fashion_mnist | gaussian | Reliability-Gated MLP | 0.846 | 0.819 | +0.0267 ± 0.0064 |
| fashion_mnist | gaussian | CellV0.3 | 0.872 | 0.847 | +0.0252 ± 0.0014 |
| digits | missing | Plain MLP | 0.935 | 0.666 | +0.2694 ± 0.0231 |
| digits | missing | Confidence MLP | 0.950 | 0.731 | +0.2198 ± 0.0194 |
| digits | missing | Reliability-Gated MLP | 0.935 | 0.666 | +0.2694 ± 0.0231 |
| digits | missing | CellV0.3 | 0.935 | 0.702 | +0.2330 ± 0.0156 |
| digits | gaussian | Plain MLP | 0.963 | 0.877 | +0.0867 ± 0.0060 |
| digits | gaussian | Confidence MLP | 0.955 | 0.868 | +0.0870 ± 0.0076 |
| digits | gaussian | Reliability-Gated MLP | 0.956 | 0.896 | +0.0599 ± 0.0037 |
| digits | gaussian | CellV0.3 | 0.953 | 0.903 | +0.0506 ± 0.0095 |
| california_housing | missing | Plain MLP | -23.005 | -7.759 | -15.2454 ± 26.9999 |
| california_housing | missing | Confidence MLP | 0.577 | 0.225 | +0.3521 ± 0.0055 |
| california_housing | missing | Reliability-Gated MLP | -23.005 | -7.759 | -15.2454 ± 26.9999 |
| california_housing | missing | CellV0.3 | 0.617 | 0.252 | +0.3650 ± 0.0053 |
| california_housing | gaussian | Plain MLP | -21.163 | -21.681 | +0.5178 ± 0.3057 |
| california_housing | gaussian | Confidence MLP | 0.414 | -0.078 | +0.4924 ± 0.1582 |
| california_housing | gaussian | Reliability-Gated MLP | -0.601 | -1.059 | +0.4579 ± 0.4673 |
| california_housing | gaussian | CellV0.3 | 0.500 | 0.331 | +0.1691 ± 0.0195 |

## E. Efficiency

Per (dataset, model): parameter count, hidden width, % off the CellV0.3 target, best-validation step, total steps, train wall-clock, time/step. Mean over the runs per (dataset, model) — both corruption families × all seeds.

| Dataset | Model | Params | Hidden | Δ% vs target | Best-val step | Total steps | Train wall (s) | Time/step (ms) |
|---|---|---|---|---|---|---|---|---|
| mnist | Plain MLP | 150,265 | 189 | +0.21% | 3300 | 4900 | 11.9 | 2.46 |
| mnist | Confidence MLP | 150,015 | 95 | +0.05% | 2900 | 4500 | 11.1 | 2.49 |
| mnist | Reliability-Gated MLP | 150,265 | 189 | +0.21% | 4067 | 5467 | 13.5 | 2.47 |
| mnist | CellV0.3 | 149,945 | 157 | 0.00% | 4467 | 6067 | 31.9 | 5.27 |
| mnist | _(CellV0.3 target param count: 149,945)_ | | | | | | | |
| fashion_mnist | Plain MLP | 150,265 | 189 | +0.21% | 2100 | 3700 | 9.1 | 2.44 |
| fashion_mnist | Confidence MLP | 150,015 | 95 | +0.05% | 2300 | 3900 | 9.6 | 2.48 |
| fashion_mnist | Reliability-Gated MLP | 150,265 | 189 | +0.21% | 2800 | 4400 | 11.0 | 2.45 |
| fashion_mnist | CellV0.3 | 149,945 | 157 | 0.00% | 4800 | 6400 | 31.9 | 5.01 |
| fashion_mnist | _(CellV0.3 target param count: 149,945)_ | | | | | | | |
| digits | Plain MLP | 24,760 | 330 | +0.11% | 2067 | 3567 | 5.0 | 1.40 |
| digits | Confidence MLP | 24,752 | 178 | +0.08% | 1325 | 2733 | 3.8 | 1.40 |
| digits | Reliability-Gated MLP | 24,760 | 330 | +0.11% | 1517 | 3017 | 4.2 | 1.39 |
| digits | CellV0.3 | 24,733 | 123 | 0.00% | 2275 | 3775 | 14.0 | 3.71 |
| digits | _(CellV0.3 target param count: 24,733)_ | | | | | | | |
| california_housing | Plain MLP | 9,861 | 986 | +0.02% | 3500 | 5100 | 5.4 | 1.10 |
| california_housing | Confidence MLP | 9,865 | 548 | +0.06% | 5067 | 6667 | 7.1 | 1.07 |
| california_housing | Reliability-Gated MLP | 9,861 | 986 | +0.02% | 2567 | 4167 | 4.4 | 1.05 |
| california_housing | CellV0.3 | 9,859 | 93 | 0.00% | 8433 | 9900 | 30.4 | 3.07 |
| california_housing | _(CellV0.3 target param count: 9,859)_ | | | | | | | |

## F. CellV0.3 belief diagnostics across corruption severity (Sec 13/14)

Trained CellV0.3, replica- and seed-averaged. `pi = e/(1+e·u)`; `CoV(pi) = std/mean` over the test set × cells. Sec 14 asks whether mean hidden precision responds to increasing corruption — the curve is reported as-is, not repaired.

| Dataset | Corruption | Severity | L1 π mean | L2 π mean | L2 CoV(π) | L2 √π mean | L2 u mean | L2 |consensus| |
|---|---|---|---|---|---|---|---|---|
| mnist | missing | 0% | 0.439 | 0.335 | 0.153 | 0.577 | 0.708 | 0.126 |
| mnist | missing | 10% | 0.418 | 0.323 | 0.149 | 0.567 | 0.702 | 0.125 |
| mnist | missing | 30% | 0.368 | 0.294 | 0.138 | 0.541 | 0.687 | 0.122 |
| mnist | missing | 50% | 0.303 | 0.253 | 0.123 | 0.502 | 0.665 | 0.118 |
| mnist | missing | 70% | 0.216 | 0.190 | 0.102 | 0.436 | 0.629 | 0.112 |
| mnist | gaussian | 0 | 0.438 | 0.336 | 0.153 | 0.578 | 0.690 | 0.127 |
| mnist | gaussian | 0.25 | 0.430 | 0.332 | 0.152 | 0.574 | 0.688 | 0.127 |
| mnist | gaussian | 0.5 | 0.409 | 0.320 | 0.147 | 0.564 | 0.682 | 0.126 |
| mnist | gaussian | 0.75 | 0.381 | 0.304 | 0.140 | 0.550 | 0.674 | 0.125 |
| mnist | gaussian | 1 | 0.351 | 0.285 | 0.133 | 0.533 | 0.665 | 0.124 |
| mnist | gaussian | 1.5 | 0.298 | 0.250 | 0.119 | 0.499 | 0.646 | 0.121 |
| fashion_mnist | missing | 0% | 0.527 | 0.384 | 0.170 | 0.618 | 0.695 | 0.094 |
| fashion_mnist | missing | 10% | 0.497 | 0.369 | 0.164 | 0.606 | 0.688 | 0.094 |
| fashion_mnist | missing | 30% | 0.427 | 0.332 | 0.150 | 0.574 | 0.671 | 0.093 |
| fashion_mnist | missing | 50% | 0.342 | 0.280 | 0.130 | 0.528 | 0.645 | 0.091 |
| fashion_mnist | missing | 70% | 0.234 | 0.205 | 0.104 | 0.453 | 0.603 | 0.090 |
| fashion_mnist | gaussian | 0 | 0.526 | 0.386 | 0.170 | 0.619 | 0.684 | 0.096 |
| fashion_mnist | gaussian | 0.25 | 0.515 | 0.380 | 0.168 | 0.614 | 0.681 | 0.095 |
| fashion_mnist | gaussian | 0.5 | 0.485 | 0.365 | 0.162 | 0.602 | 0.674 | 0.095 |
| fashion_mnist | gaussian | 0.75 | 0.446 | 0.344 | 0.154 | 0.584 | 0.665 | 0.094 |
| fashion_mnist | gaussian | 1 | 0.406 | 0.321 | 0.145 | 0.565 | 0.654 | 0.094 |
| fashion_mnist | gaussian | 1.5 | 0.336 | 0.277 | 0.128 | 0.525 | 0.631 | 0.092 |
| digits | missing | 0% | 0.561 | 0.452 | 0.151 | 0.669 | 0.429 | 0.200 |
| digits | missing | 10% | 0.529 | 0.432 | 0.147 | 0.654 | 0.425 | 0.197 |
| digits | missing | 30% | 0.454 | 0.382 | 0.139 | 0.616 | 0.417 | 0.191 |
| digits | missing | 50% | 0.362 | 0.315 | 0.142 | 0.560 | 0.407 | 0.183 |
| digits | missing | 70% | 0.245 | 0.223 | 0.178 | 0.470 | 0.393 | 0.174 |
| digits | gaussian | 0 | 0.558 | 0.454 | 0.147 | 0.671 | 0.409 | 0.225 |
| digits | gaussian | 0.25 | 0.546 | 0.446 | 0.146 | 0.665 | 0.406 | 0.224 |
| digits | gaussian | 0.5 | 0.514 | 0.426 | 0.144 | 0.650 | 0.400 | 0.220 |
| digits | gaussian | 0.75 | 0.472 | 0.398 | 0.141 | 0.629 | 0.392 | 0.216 |
| digits | gaussian | 1 | 0.429 | 0.368 | 0.139 | 0.605 | 0.384 | 0.211 |
| digits | gaussian | 1.5 | 0.354 | 0.313 | 0.138 | 0.557 | 0.368 | 0.202 |
| california_housing | missing | 0% | 0.722 | 0.641 | 0.177 | 0.797 | 0.174 | 0.039 |
| california_housing | missing | 10% | 0.670 | 0.598 | 0.198 | 0.769 | 0.177 | 0.039 |
| california_housing | missing | 30% | 0.554 | 0.502 | 0.260 | 0.702 | 0.183 | 0.040 |
| california_housing | missing | 50% | 0.422 | 0.389 | 0.365 | 0.611 | 0.187 | 0.041 |
| california_housing | missing | 70% | 0.270 | 0.255 | 0.560 | 0.475 | 0.185 | 0.044 |
| california_housing | gaussian | 0 | 0.679 | 0.602 | 0.195 | 0.771 | 0.179 | 0.042 |
| california_housing | gaussian | 0.25 | 0.664 | 0.588 | 0.195 | 0.762 | 0.185 | 0.044 |
| california_housing | gaussian | 0.5 | 0.624 | 0.554 | 0.198 | 0.740 | 0.192 | 0.045 |
| california_housing | gaussian | 0.75 | 0.572 | 0.511 | 0.203 | 0.711 | 0.199 | 0.046 |
| california_housing | gaussian | 1 | 0.519 | 0.467 | 0.212 | 0.679 | 0.204 | 0.047 |
| california_housing | gaussian | 1.5 | 0.426 | 0.389 | 0.238 | 0.619 | 0.212 | 0.048 |

### Reliability-response (Sec 14) — does mean hidden π fall as corruption rises?

| Dataset | Corruption | L1 π (clean → worst) | L2 π (clean → worst) | Read |
|---|---|---|---|---|
| mnist | missing | 0.439 → 0.216 | 0.335 → 0.190 | π falls with severity |
| mnist | gaussian | 0.438 → 0.298 | 0.336 → 0.250 | π falls with severity |
| fashion_mnist | missing | 0.527 → 0.234 | 0.384 → 0.205 | π falls with severity |
| fashion_mnist | gaussian | 0.526 → 0.336 | 0.386 → 0.277 | π falls with severity |
| digits | missing | 0.561 → 0.245 | 0.452 → 0.223 | π falls with severity |
| digits | gaussian | 0.558 → 0.354 | 0.454 → 0.313 | π falls with severity |
| california_housing | missing | 0.722 → 0.270 | 0.641 → 0.255 | π falls with severity |
| california_housing | gaussian | 0.679 → 0.426 | 0.602 → 0.389 | π falls with severity |

_A sensible mechanism generally shows lower effective precision as observation reliability deteriorates; where it does not, the row is flagged and left unrepaired (Sec 14)._

## G. Confidence interventions (Sec 15) — evaluation only, no retraining

Trained CellV0.3 on Fashion-MNIST and California Housing, corrupted observations held fixed, reliability map swapped: `true` (actual `c`), `all_ones` (`c := 1`), `shuffled` (`c` permuted across features per example). Primary metric, mean over 3 seeds × 3 replicas.

| Dataset | Corruption | Regime | true | all-ones (Δ) | shuffled (Δ) |
|---|---|---|---|---|---|
| fashion_mnist | missing | in_dist_max (sev 0.3) | 0.8636 | -0.0023 | -0.0335 |
| fashion_mnist | missing | ood_max (sev 0.7) | 0.7679 | -0.1341 | -0.4243 |
| fashion_mnist | gaussian | in_dist_max (sev 0.75) | 0.8723 | -0.0015 | -0.0030 |
| fashion_mnist | gaussian | ood_max (sev 1.5) | 0.8471 | -0.0156 | -0.0301 |
| california_housing | missing | in_dist_max (sev 0.3) | 0.6169 | -0.2415 | -0.3653 |
| california_housing | missing | ood_max (sev 0.7) | 0.2519 | -0.6051 | -0.5019 |
| california_housing | gaussian | in_dist_max (sev 0.75) | 0.4998 | -0.0276 | -0.0249 |
| california_housing | gaussian | ood_max (sev 1.5) | 0.3306 | -0.1232 | -0.0997 |

_A negative `shuffled` Δ that is larger in magnitude than the `all_ones` Δ means correct observation↔reliability alignment (not just the marginal `c` distribution) matters to CellV0.3._

## Predeclared go/no-go read (Sec 19)

Sub-criteria the task named (Sec 19), scored **only** on the comparisons where the baseline trained stably: the Plain / Reliability-Gated MLP diverges on some California Housing seeds (Failures), and Reliability-Gated ≡ Plain under missingness (see the header note), so those cells carry no independent signal and are excluded from the counts below. |Δ AUC| < 0.005 counts as a tie.

| Sub-criterion | vs Confidence MLP | vs Reliability-Gated MLP (Gaussian only) |
|---|---|---|
| corruption-AUC Δ | ahead 7, tied 0, behind 1 of 8 (median +0.0278) | ahead 2, tied 1, behind 0 of 3 (median +0.0250) |
| smaller OOD degradation | 5 of 8 | 2 of 3 |
| correct confidence beats all-ones **and** shuffled | 24/24 intervention rows | — |
| mean hidden π falls with severity | 8/8 cells (both layers — see Table F) | — |

**Positive (not uniform)** — CellV0.3 has the larger corruption-AUC than the Confidence MLP in 7/8 stable comparisons (median +0.0278), decisively and on every seed on MNIST and Fashion-MNIST under both corruptions (+0.02 to +0.08 AUC); it **loses** to the Confidence MLP on Digits/missing at every severity; Digits/Gaussian is a tie. Against the Reliability-Gated MLP (Gaussian only — it is identical to the Plain MLP under missingness) CellV0.3 is ahead on MNIST and Fashion-MNIST and tied on Digits. On California Housing the Plain and Reliability-Gated MLPs diverge on some seeds; CellV0.3 and the Confidence MLP are the only stable models and CellV0.3 edges it. Correct confidence beats all-ones **and** shuffled in every intervention row, and mean hidden π falls with severity in every cell. This is not calibration and not merely a plain-MLP win — it is a real, dataset-dependent robustness advantage over baselines given the identical reliability signal, strongest on the image tasks and absent on Digits/missing.

_This is the evidence as recorded. Do not write the paper yet; do not alter V0.3 after these results (Sec 18). `e`/`u`/`π` are internal computational reliability variables, not calibrated predictive probabilities._

## Failures and caveats

**Divergence / NaNs:** none.

**Step-cap hits (early stopping did not terminate):**

- california_housing / gaussian / CellV0.3 (1/3 seeds)

**High seed variance at the clean severity (primary-metric seed-std > 0.05 accuracy / 0.15 R²):** california_housing/missing/Plain MLP (r2 std 40.060); california_housing/missing/Reliability-Gated MLP (r2 std 40.060); california_housing/gaussian/Plain MLP (r2 std 37.522); california_housing/gaussian/Reliability-Gated MLP (r2 std 0.831)

**Parameter-match failures (>2% off the CellV0.3 target):** none.

**Cells where CellV0.3's corruption-AUC clearly loses (< −0.005) to a reliability-aware baseline:** digits/missing: -0.0106 vs Confidence MLP

**CellV0.3 hidden precision does not respond to corruption (|Δ mean π| ≤ 0.005 across the grid, both layers):** none.

**Numeric-range flag** (a single test example × cell with output precision `< 1e-6` — large local conflict; all finite, no NaN/divergence, no protocol impact):

- digits/missing s=0% layer1 (min π=4.2e-11)
- digits/missing s=0% layer2 (min π=1.0e-08)
- digits/missing s=10% layer1 (min π=3.7e-11)
- digits/missing s=10% layer2 (min π=1.0e-08)
- digits/missing s=30% layer1 (min π=3.1e-11)
- digits/missing s=30% layer2 (min π=1.0e-08)
- digits/gaussian s=0 layer1 (min π=3.8e-11)
- digits/gaussian s=0 layer2 (min π=1.0e-08)
- digits/gaussian s=0.25 layer1 (min π=3.8e-11)
- digits/gaussian s=0.25 layer2 (min π=1.0e-08)
- digits/gaussian s=0.5 layer1 (min π=3.6e-11)
- digits/gaussian s=0.5 layer2 (min π=1.0e-08)
- digits/gaussian s=0.75 layer1 (min π=3.3e-11)
- digits/gaussian s=0.75 layer2 (min π=1.0e-08)
- digits/gaussian s=1 layer1 (min π=3.1e-11)
- digits/gaussian s=1 layer2 (min π=1.0e-08)
- digits/gaussian s=1.5 layer1 (min π=2.7e-11)
- digits/gaussian s=1.5 layer2 (min π=1.0e-08)

