# Architecture V2 BeliefDendrite backend optimization — results

Measured on this machine (MPS), exact CIFAR-10 strong-baselines shapes:
`H=267` somas/layer, `B=4` branches/soma, layer 1 `K=192` (local 8x8 RGB
patches, `input_dim=3072`), layer 2 `K=32` (balanced random,
`input_dim=267`), `batch=256`. 30 warmup + 100 measured steps per backend
(`experiments/belief_dendrite/backend_benchmark.py`).

## Speed

| backend | forward (ms) | forward+backward (ms) | speedup vs frozen |
|---|---|---|---|
| `frozen_reference` (unmodified) | 27.87 | 60.37 | 1.00x |
| `sparse_fast` (raw-moment, gather-based) | 29.12 | 56.47 | 1.07x |
| `dense_fast` (raw-moment, scatter + 2 GEMMs) | 4.21 | 11.25 | **5.36x** |

`dense_fast` at 11.25ms/step is faster than the frozen `scalar_dendrite`
control measured in the same session (18.77ms/step) — despite computing
strictly more information (`mu` + evidence + uncertainty + precision, not
just `mu`) — purely from trading a gather-heavy access pattern for two
regular GEMMs.

`sparse_fast`'s small win (not the ~2-3x the raw-moment algebra alone
would predict from eliminating ~half the elementwise multiplies) confirms
the user's diagnosis: this workload is memory-traffic/dispatch-bound on
MPS, not arithmetic-bound, so reducing elementwise-op count without
reducing the actual tensor footprint barely moves the needle.

## Why: largest intermediate tensor (layer 1, the expensive layer)

| backend | largest intermediate | elements | approx size (float32) |
|---|---|---|---|
| frozen / sparse_fast | gathered `mu_g`, `pi_g`: `[256, 1068, 192]` (one each) | 52.4M each | ~210MB each, ~420MB combined |
| dense_fast | scattered `V_dense`: `[1068, 3072]`; GEMM input `[512, 3072]` | 3.3M / 1.6M | ~13MB / ~6MB |

The dense backend's per-forward-call tensors are roughly **30x smaller**
than the sparse backend's gathered tensors, despite the GEMM doing ~16x
more nominal multiply-adds (`3072/192` ≈ the inverse of layer 1's ~6.25%
density) — exactly the "GEMM throughput beats gather memory traffic" bet
the user proposed, confirmed rather than assumed.

## Correctness (parity)

`tests/test_architecture_v2_belief_dendrite_fast.py`, 12/12 passing:

- Forward outputs (`mu`, `evidence`, `uncertainty`) and every
  `BranchSomaDiagnostics` field (`mu_branch`, `e_branch`, `u_branch`,
  `pi_branch`, `A_cable`, `e_soma`, `branch_contribution()`) match the
  frozen reference within `atol=1e-5, rtol=1e-4`, on:
  - a generic random topology,
  - the exact CIFAR `local_2d` shape,
  - a near-total-precision-loss edge case (one branch's entire receptive
    field driven to `reliability=1e-3`).
- Parameter gradients match within the same tolerance (both backends).
- One full `AdamW` step from identical initialization produces matching
  parameters within the same tolerance (both backends).
- Both backends expose exactly the frozen layer's parameter set — no new
  trainable parameters (dense_fast's dense weight matrix is a
  differentiable *view* of the sparse parameters, scattered fresh every
  forward call, not a parameter itself).

Typical observed max absolute difference against the frozen reference in
these tests: ~1e-8 to ~3e-8 (float32 rounding-level, not a systematic
deviation).

## Recommendation

`dense_fast` is the clear winner at this benchmark's density (~6.25%
layer 1, ~12% layer 2) and shapes. It is implemented and parity-verified
but **not** wired into `experiments/belief_dendrite/strong_baselines/
models.py` — the just-completed CIFAR sweep used the frozen reference
throughout, as instructed ("let the current run finish... those numbers
should stay the frozen reference implementation"). Swapping the
strong-baselines `BeliefDendriteModel` to build on `BeliefDendriteNetworkDenseFast`
for a future re-run is a one-line change once that's explicitly wanted;
not done here.

Not attempted, per the user's own prioritization ("the second is the
serious optimization *if* we decide V2 is worth engineering further"):
a custom fused Metal kernel (optimization 2), and exploiting shared
`local_2d` receptive-field overlap to deduplicate repeated patch pixels
across branches (optimization 3). `torch.compile` on MPS was not
benchmarked either; given `dense_fast` already achieves 5.36x with plain
eager GEMMs, chasing compiler fusion on top did not seem worth the risk
this session flagged (MPS compile correctness/perf reports are mixed).
