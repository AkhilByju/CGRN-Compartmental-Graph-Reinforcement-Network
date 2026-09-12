# Architecture V2 -- Strong CNN + Transformer Baseline Benchmark (CIFAR-10)

7 run records loaded from `/Users/byjukollon/projects/CGRN/experiments/belief_dendrite/strong_baselines/results/raw`. Families present: ['belief_dendrite', 'confidence_cnn', 'confidence_tiny_vit', 'reliability_gated_tiny_vit', 'scalar_dendrite', 'small_cnn', 'tiny_vit'].

BeliefDendriteNetwork and its equations are frozen and reused unmodified from `src/models/architecture_v2/belief_dendrite.py` -- see `docs/architecture_v2.md`. MNIST/Fashion-MNIST were not rerun for this new model roster (spec Sec 2 permits skipping them when not cheap; CIFAR-10 is the primary and, here, only result). The optional dense CellV0.3 reference (spec Sec 5) was not run, to avoid delaying the seven required arms, per that section's own instruction.

## Main table -- CIFAR-10 primary benchmark

Mean +/- population-std over seeds. `params` is the sized parameter count against a 250,000-parameter target (spec Sec 6).

| model | params | clean | patch12 | patch16 | patch20 | patch24 | corruption_AUC | OOD_drop | ms/step |
|---|---|---|---|---|---|---|---|---|---|
| small_cnn | 253,396 | 0.8360+/-0.0000 | 0.7824+/-0.0000 | 0.7257+/-0.0000 | 0.6238+/-0.0000 | 0.4532+/-0.0000 | 17.684+/-0.000 | 0.3292+/-0.0000 | 49.48+/-0.00 |
| confidence_cnn | 253,963 | 0.8382+/-0.0000 | 0.7872+/-0.0000 | 0.7379+/-0.0000 | 0.6414+/-0.0000 | 0.4888+/-0.0000 | 17.904+/-0.000 | 0.2984+/-0.0000 | 48.51+/-0.00 |
| tiny_vit | 243,690 | 0.7234+/-0.0000 | 0.6811+/-0.0000 | 0.6292+/-0.0000 | 0.5425+/-0.0000 | 0.4166+/-0.0000 | 15.395+/-0.000 | 0.2645+/-0.0000 | 27.13+/-0.00 |
| confidence_tiny_vit | 243,770 | 0.7043+/-0.0000 | 0.6685+/-0.0000 | 0.6224+/-0.0000 | 0.5507+/-0.0000 | 0.4221+/-0.0000 | 15.188+/-0.000 | 0.2464+/-0.0000 | 27.17+/-0.00 |
| reliability_gated_tiny_vit | 243,690 | 0.7197+/-0.0000 | 0.6747+/-0.0000 | 0.6249+/-0.0000 | 0.5372+/-0.0000 | 0.4098+/-0.0000 | 15.282+/-0.000 | 0.2649+/-0.0000 | 26.48+/-0.00 |
| scalar_dendrite | 249,388 | 0.6017+/-0.0000 | 0.5511+/-0.0000 | 0.4837+/-0.0000 | 0.3735+/-0.0000 | 0.2480+/-0.0000 | 12.051+/-0.000 | 0.3031+/-0.0000 | 22.23+/-0.00 |
| belief_dendrite | 249,388 | 0.6040+/-0.0000 | 0.5547+/-0.0000 | 0.5051+/-0.0000 | 0.4082+/-0.0000 | 0.2704+/-0.0000 | 12.345+/-0.000 | 0.2843+/-0.0000 | 74.06+/-0.00 |

## MPS speed preflight

Device: `mps`. Batch size used for the sweep: `256`.

| family | params | ms/step | examples/sec | MPS current alloc (MB) |
|---|---|---|---|---|
| small_cnn | 253,396 | 35.68 | 7175.3 | 15.97 |
| confidence_cnn | 253,963 | 41.73 | 6134.5 | 15.98 |
| tiny_vit | 243,690 | 23.72 | 10794.4 | 18.79 |
| confidence_tiny_vit | 243,770 | 23.98 | 10675.6 | 20.08 |
| reliability_gated_tiny_vit | 243,690 | 23.83 | 10742.1 | 19.05 |
| scalar_dendrite | 249,388 | 18.77 | 13641.2 | 16.79 |
| belief_dendrite | 249,388 | 69.09 | 3705.4 | 16.88 |

## Reliability interventions (spec Sec 12)

`delta_X = accuracy(correct reliability) - accuracy(X)`. A large positive delta means the model's use of reliability depends on correct spatial alignment, not merely its presence.

| family | severity | acc_correct | acc_all_ones | acc_shuffled | delta_all_ones | delta_shuffled |
|---|---|---|---|---|---|---|
| confidence_cnn | 12.0 | 0.7891 | 0.7350 | 0.7283 | +0.0541 | +0.0608 |
| confidence_cnn | 24.0 | 0.4870 | 0.3168 | 0.2425 | +0.1702 | +0.2445 |
| confidence_tiny_vit | 12.0 | 0.6694 | 0.6003 | 0.5802 | +0.0691 | +0.0892 |
| confidence_tiny_vit | 24.0 | 0.4182 | 0.2384 | 0.1587 | +0.1798 | +0.2595 |
| reliability_gated_tiny_vit | 12.0 | 0.6739 | 0.6204 | 0.6224 | +0.0535 | +0.0515 |
| reliability_gated_tiny_vit | 24.0 | 0.4104 | 0.2347 | 0.2210 | +0.1757 | +0.1894 |
| belief_dendrite | 12.0 | 0.5542 | 0.4904 | 0.4720 | +0.0638 | +0.0822 |
| belief_dendrite | 24.0 | 0.2658 | 0.1710 | 0.1628 | +0.0948 | +0.1030 |

## V2 mechanism diagnostics (BeliefDendrite, spec Sec 10)

Per-severity mean over seeds (layer 1 = the local_2d receptive-field layer). `corr(overlap, branch_pi)` and `corr(overlap, somatic_contribution)` test the causal chain overlap -> branch precision -> somatic routing (Sec 13 Q6).

| severity | branch_pi (L1) | soma_pi (L1) | soma_conflict u (L1) | eff._branches (L1) | corr(overlap,branch_pi) | corr(overlap,somatic_contrib) |
|---|---|---|---|---|---|---|
| 0 | 0.5789 | 0.5038 | 0.2727 | 3.483 | nan | nan |
| 4 | 0.5723 | 0.4984 | 0.2743 | 3.482 | -0.1217 | -0.0181 |
| 8 | 0.5437 | 0.4754 | 0.2780 | 3.454 | -0.4571 | -0.0765 |
| 12 | 0.4848 | 0.4263 | 0.2715 | 3.328 | -0.7165 | -0.1165 |
| 16 | 0.3966 | 0.3514 | 0.2531 | 3.198 | -0.8351 | -0.1342 |
| 20 | 0.2880 | 0.2587 | 0.2229 | 3.079 | -0.8878 | -0.1670 |
| 24 | 0.1707 | 0.1585 | 0.1815 | 2.991 | -0.9129 | -0.2312 |

## Transformer diagnostics (spec Sec 11)

`ConfidenceTinyViT`: `||reliability_embedding|| / ||image_patch_embedding||`, mean over tokens/examples. `ReliabilityGatedTinyViT`: mean patch reliability. Both per severity, mean over seeds.

| severity | ConfidenceTinyViT norm ratio | ReliabilityGatedTinyViT mean patch reliability |
|---|---|---|
| 0 | 1.0418 | 1.0000 |
| 4 | 1.0263 | 0.9844 |
| 8 | 0.9813 | 0.9376 |
| 12 | 0.9055 | 0.8595 |
| 16 | 0.7980 | 0.7502 |
| 20 | 0.6572 | 0.6098 |
| 24 | 0.4808 | 0.4381 |

## Sec 13 comparison questions

### Q1: is V2 merely rediscovering convolution?
`belief_dendrite` vs `small_cnn`

- clean accuracy: belief_dendrite=0.6040+/-0.0000  small_cnn=0.8360+/-0.0000  delta=-0.2320
- corruption_AUC: belief_dendrite=12.3452+/-0.0000  small_cnn=17.6843+/-0.0000  delta=-5.3391
- acc@patch24 (max OOD): belief_dendrite=0.2704+/-0.0000  small_cnn=0.4532+/-0.0000  delta=-0.1828

### Q2: does a CNN with the exact reliability mask eliminate the advantage?
`belief_dendrite` vs `confidence_cnn`

- clean accuracy: belief_dendrite=0.6040+/-0.0000  confidence_cnn=0.8382+/-0.0000  delta=-0.2342
- corruption_AUC: belief_dendrite=12.3452+/-0.0000  confidence_cnn=17.9039+/-0.0000  delta=-5.5587
- acc@patch24 (max OOD): belief_dendrite=0.2704+/-0.0000  confidence_cnn=0.4888+/-0.0000  delta=-0.2184

### Q3: can a Transformer learn the same behavior?
`belief_dendrite` vs `tiny_vit`

- clean accuracy: belief_dendrite=0.6040+/-0.0000  tiny_vit=0.7234+/-0.0000  delta=-0.1194
- corruption_AUC: belief_dendrite=12.3452+/-0.0000  tiny_vit=15.3955+/-0.0000  delta=-3.0503
- acc@patch24 (max OOD): belief_dendrite=0.2704+/-0.0000  tiny_vit=0.4166+/-0.0000  delta=-0.1462

### Q4a: does explicit reliability eliminate the advantage (additive)?
`belief_dendrite` vs `confidence_tiny_vit`

- clean accuracy: belief_dendrite=0.6040+/-0.0000  confidence_tiny_vit=0.7043+/-0.0000  delta=-0.1003
- corruption_AUC: belief_dendrite=12.3452+/-0.0000  confidence_tiny_vit=15.1882+/-0.0000  delta=-2.8430
- acc@patch24 (max OOD): belief_dendrite=0.2704+/-0.0000  confidence_tiny_vit=0.4221+/-0.0000  delta=-0.1517

### Q4b: does explicit reliability eliminate the advantage (gated)?
`belief_dendrite` vs `reliability_gated_tiny_vit`

- clean accuracy: belief_dendrite=0.6040+/-0.0000  reliability_gated_tiny_vit=0.7197+/-0.0000  delta=-0.1157
- corruption_AUC: belief_dendrite=12.3452+/-0.0000  reliability_gated_tiny_vit=15.2823+/-0.0000  delta=-2.9371
- acc@patch24 (max OOD): belief_dendrite=0.2704+/-0.0000  reliability_gated_tiny_vit=0.4098+/-0.0000  delta=-0.1394

### Q5: is the gain dendritic locality rather than belief propagation?
`belief_dendrite` vs `scalar_dendrite`

- clean accuracy: belief_dendrite=0.6040+/-0.0000  scalar_dendrite=0.6017+/-0.0000  delta=+0.0023
- corruption_AUC: belief_dendrite=12.3452+/-0.0000  scalar_dendrite=12.0513+/-0.0000  delta=+0.2939
- acc@patch24 (max OOD): belief_dendrite=0.2704+/-0.0000  scalar_dendrite=0.2480+/-0.0000  delta=+0.0224

### Q6: does V2 continue to show the intended mechanism on a harder visual dataset?
`patch overlap -> branch pi -> somatic contribution` -- see the BeliefDendrite diagnostics table above; both correlations at the max-OOD severity (24.0) are the headline numbers (expected direction for both: negative).

- corr(overlap, branch_pi) @ patch=24.0: -0.9129+/-0.0000
- corr(overlap, somatic_contribution) @ patch=24.0: -0.2312+/-0.0000
