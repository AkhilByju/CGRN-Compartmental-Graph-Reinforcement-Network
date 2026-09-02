# Experiment 004A/004B — `scale_stable_precision` results (raw)

Run 2026-09-01. `python experiments/004_cellv0_scaling/run_004a.py --aggregation scale_stable_precision`
and `python experiments/004_cellv0_scaling/run_004b.py --aggregation scale_stable_precision`.
Source data: `results/processed/004a_scale_stable_precision_summary.json`,
`results/processed/004b_scale_stable_precision_summary.json`. 5 seeds (0–4) per
cell throughout. No commentary below — just the numbers.

## 004A — model-size scaling

`mlp` vs. `scale_stable_precision` (parameter-matched), 100,000 training
examples, 6000 steps, per dataset/scale, mean ± std across seeds.

### c2_interaction (headline = accuracy)

| Scale | Target params | `scale_stable_precision` params | `mlp` params | Param match | `mlp` | `scale_stable_precision` |
|---|---|---|---|---|---|---|
| S0 | 600 | ~632 | ~611 | -3.32% | 0.9432 ± 0.0064 | 0.9369 ± 0.0093 |
| S1 | 2,500 | ~2,434 | ~2,387 | -1.93% | 0.9413 ± 0.0067 | 0.9379 ± 0.0100 |
| S2 | 10,000 | ~10,066 | ~9,986 | -0.79% | 0.9384 ± 0.0064 | 0.9335 ± 0.0099 |
| S3 | 40,000 | ~39,746 | ~39,587 | -0.40% | 0.9329 ± 0.0090 | 0.9323 ± 0.0121 |
| S4 | 150,000 | ~150,136 | ~149,755 | -0.25% | 0.9202 ± 0.0160 | 0.9337 ± 0.0097 |

### r2_interaction (headline = R²)

| Scale | Target params | `scale_stable_precision` params | `mlp` params | Param match | `mlp` | `scale_stable_precision` |
|---|---|---|---|---|---|---|
| S0 | 600 | ~616 | ~639 | +3.73% | 0.9731 ± 0.0023 | 0.9696 ± 0.0032 |
| S1 | 2,500 | ~2,542 | ~2,539 | -0.12% | 0.9716 ± 0.0021 | 0.9720 ± 0.0033 |
| S2 | 10,000 | ~9,997 | ~10,089 | +0.92% | 0.9709 ± 0.0028 | 0.9717 ± 0.0012 |
| S3 | 40,000 | ~40,172 | ~40,189 | +0.04% | 0.9678 ± 0.0030 | 0.9727 ± 0.0019 |
| S4 | 150,000 | ~149,864 | ~150,145 | +0.19% | 0.9610 ± 0.0081 | 0.9696 ± 0.0055 |

### u2_heteroscedastic_interaction (headline = R²)

| Scale | Target params | `scale_stable_precision` params | `mlp` params | Param match | `mlp` | `scale_stable_precision` |
|---|---|---|---|---|---|---|
| S0 | 600 | ~616 | ~639 | +3.73% | 0.8969 ± 0.0055 | 0.8925 ± 0.0074 |
| S1 | 2,500 | ~2,542 | ~2,539 | -0.12% | 0.8953 ± 0.0071 | 0.8971 ± 0.0069 |
| S2 | 10,000 | ~9,997 | ~10,089 | +0.92% | 0.8882 ± 0.0033 | 0.8976 ± 0.0092 |
| S3 | 40,000 | ~40,172 | ~40,189 | +0.04% | 0.8862 ± 0.0049 | 0.8934 ± 0.0113 |
| S4 | 150,000 | ~149,864 | ~150,145 | +0.19% | 0.8778 ± 0.0072 | 0.8924 ± 0.0112 |

### Per-layer internal state (`scale_stable_precision` only, mean ± std across seeds)

**c2_interaction**

| Scale | Params | layer1 evidence | layer1 uncertainty | layer2 evidence | layer2 uncertainty |
|---|---|---|---|---|---|
| S0 | ~632 | 0.4076 ± 0.0441 | 2.0112 ± 0.1140 | 0.2226 ± 0.0208 | 3.6078 ± 0.3051 |
| S1 | ~2,434 | 0.3957 ± 0.0067 | 2.0359 ± 0.0310 | 0.2083 ± 0.0083 | 3.7943 ± 0.1362 |
| S2 | ~10,066 | 0.3739 ± 0.0198 | 2.0704 ± 0.0455 | 0.1828 ± 0.0062 | 4.1532 ± 0.0630 |
| S3 | ~39,746 | 0.3820 ± 0.0135 | 2.0461 ± 0.0377 | 0.1815 ± 0.0050 | 4.2497 ± 0.0752 |
| S4 | ~150,136 | 0.3725 ± 0.0203 | 2.0459 ± 0.0571 | 0.1713 ± 0.0106 | 4.4733 ± 0.1701 |

**r2_interaction**

| Scale | Params | layer1 evidence | layer1 uncertainty | layer2 evidence | layer2 uncertainty |
|---|---|---|---|---|---|
| S0 | ~616 | 0.3510 ± 0.0276 | 1.9831 ± 0.1625 | 0.1872 ± 0.0084 | 3.9761 ± 0.2643 |
| S1 | ~2,542 | 0.3786 ± 0.0261 | 1.8866 ± 0.0622 | 0.1948 ± 0.0186 | 3.7245 ± 0.2969 |
| S2 | ~9,997 | 0.3936 ± 0.0107 | 1.8231 ± 0.0279 | 0.1902 ± 0.0094 | 3.7242 ± 0.1271 |
| S3 | ~40,172 | 0.3983 ± 0.0157 | 1.7803 ± 0.0281 | 0.1849 ± 0.0115 | 3.7865 ± 0.1523 |
| S4 | ~149,864 | 0.4093 ± 0.0133 | 1.7503 ± 0.0310 | 0.1883 ± 0.0103 | 3.7604 ± 0.1500 |

**u2_heteroscedastic_interaction**

| Scale | Params | layer1 evidence | layer1 uncertainty | layer2 evidence | layer2 uncertainty |
|---|---|---|---|---|---|
| S0 | ~616 | 0.3558 ± 0.0331 | 2.0388 ± 0.1789 | 0.1953 ± 0.0109 | 3.8753 ± 0.3129 |
| S1 | ~2,542 | 0.3745 ± 0.0306 | 1.9231 ± 0.0763 | 0.1939 ± 0.0184 | 3.7773 ± 0.3252 |
| S2 | ~9,997 | 0.3919 ± 0.0092 | 1.8420 ± 0.0298 | 0.1915 ± 0.0097 | 3.7276 ± 0.1231 |
| S3 | ~40,172 | 0.3923 ± 0.0130 | 1.8287 ± 0.0217 | 0.1827 ± 0.0107 | 3.8597 ± 0.1340 |
| S4 | ~149,864 | 0.4036 ± 0.0151 | 1.7943 ± 0.0276 | 0.1864 ± 0.0119 | 3.8394 ± 0.1726 |

### Ablation deltas at S0/S2/S4 (`scale_stable_precision` only, delta = perturbed − baseline, mean ± std across seeds)

**c2_interaction** (delta_accuracy)

| Scale | baseline | evidence_ones | uncertainty_ones | shuffle_evidence_uncertainty | uncertainty_random |
|---|---|---|---|---|---|
| S0 | 0.9369 ± 0.0093 | -0.1180 ± 0.1366 | -0.1628 ± 0.0570 | -0.3420 ± 0.0140 | -0.3256 ± 0.0360 |
| S2 | 0.9335 ± 0.0099 | -0.2610 ± 0.0884 | -0.3360 ± 0.0713 | -0.4280 ± 0.0196 | -0.3394 ± 0.0365 |
| S4 | 0.9337 ± 0.0097 | -0.3405 ± 0.0775 | -0.4092 ± 0.0552 | -0.4485 ± 0.0161 | -0.3776 ± 0.0418 |

**r2_interaction** (delta_r2)

| Scale | baseline | evidence_ones | uncertainty_ones | shuffle_evidence_uncertainty | uncertainty_random |
|---|---|---|---|---|---|
| S0 | 0.9696 ± 0.0032 | -5.0230 ± 0.8805 | -3.0826 ± 1.1465 | -20.2364 ± 7.3089 | -11.1898 ± 4.0412 |
| S2 | 0.9717 ± 0.0012 | -4.4851 ± 1.0993 | -4.9137 ± 0.9694 | -18.3107 ± 3.3515 | -6.6179 ± 1.2858 |
| S4 | 0.9696 ± 0.0055 | -4.6689 ± 0.6111 | -5.2082 ± 0.6055 | -16.6229 ± 2.0810 | -5.6131 ± 0.6240 |

**u2_heteroscedastic_interaction** (delta_r2)

| Scale | baseline | evidence_ones | uncertainty_ones | shuffle_evidence_uncertainty | uncertainty_random |
|---|---|---|---|---|---|
| S0 | 0.8925 ± 0.0074 | -2.9504 ± 1.2626 | -1.7954 ± 0.5879 | -12.8391 ± 3.6415 | -8.3175 ± 2.9694 |
| S2 | 0.8976 ± 0.0092 | -3.3944 ± 0.8839 | -3.6170 ± 0.6800 | -13.5483 ± 2.2177 | -4.8209 ± 0.8681 |
| S4 | 0.8924 ± 0.0112 | -3.3926 ± 0.6289 | -3.7438 ± 0.5567 | -11.8571 ± 1.6669 | -4.0272 ± 0.6738 |

## 004B — data scaling

Fixed scale `S3` (~40K parameters, ~40,172 `scale_stable_precision` /
~40,189 `mlp`), 6000 steps, `n_train` swept. Mean ± std across seeds.

### c2_interaction (headline = accuracy)

| Data scale | n_train | `mlp` | `scale_stable_precision` | delta |
|---|---|---|---|---|
| D0 | 1,000 | 0.9153 ± 0.0069 | 0.9368 ± 0.0040 | +0.0215 |
| D1 | 3,000 | 0.9297 ± 0.0050 | 0.9396 ± 0.0073 | +0.0099 |
| D2 | 10,000 | 0.9350 ± 0.0062 | 0.9456 ± 0.0054 | +0.0106 |
| D3 | 30,000 | 0.9400 ± 0.0097 | 0.9476 ± 0.0052 | +0.0076 |
| D4 | 100,000 | 0.9329 ± 0.0090 | 0.9323 ± 0.0121 | -0.0006 |
| D5 | 300,000 | 0.9287 ± 0.0067 | 0.9383 ± 0.0075 | +0.0096 |

### r2_interaction (headline = R²)

| Data scale | n_train | `mlp` | `scale_stable_precision` | delta |
|---|---|---|---|---|
| D0 | 1,000 | 0.9459 ± 0.0023 | 0.9725 ± 0.0012 | +0.0266 |
| D1 | 3,000 | 0.9538 ± 0.0080 | 0.9681 ± 0.0045 | +0.0143 |
| D2 | 10,000 | 0.9659 ± 0.0026 | 0.9736 ± 0.0012 | +0.0077 |
| D3 | 30,000 | 0.9636 ± 0.0042 | 0.9683 ± 0.0061 | +0.0048 |
| D4 | 100,000 | 0.9678 ± 0.0030 | 0.9727 ± 0.0019 | +0.0050 |
| D5 | 300,000 | 0.9692 ± 0.0033 | 0.9704 ± 0.0034 | +0.0012 |

### u2_heteroscedastic_interaction (headline = R²)

| Data scale | n_train | `mlp` | `scale_stable_precision` | delta |
|---|---|---|---|---|
| D0 | 1,000 | 0.8195 ± 0.0100 | 0.8873 ± 0.0110 | +0.0678 |
| D1 | 3,000 | 0.8552 ± 0.0038 | 0.8982 ± 0.0076 | +0.0430 |
| D2 | 10,000 | 0.8880 ± 0.0049 | 0.8976 ± 0.0036 | +0.0095 |
| D3 | 30,000 | 0.8826 ± 0.0061 | 0.8930 ± 0.0108 | +0.0104 |
| D4 | 100,000 | 0.8862 ± 0.0049 | 0.8934 ± 0.0113 | +0.0072 |
| D5 | 300,000 | 0.8883 ± 0.0051 | 0.8979 ± 0.0041 | +0.0095 |

## Raw data

Full per-seed records (75 rows for 004A, 90 for 004B, including MAE/RMSE,
wall-clock times, layer-level g-gate stats, and all four ablation conditions'
raw metrics not just deltas) are in
`results/processed/004a_scale_stable_precision_summary.json` and
`results/processed/004b_scale_stable_precision_summary.json`.
