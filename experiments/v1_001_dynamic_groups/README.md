# v1_001 — Does self-organizing computation actually help?

**Status: written, not run at full scale.** Smoke-tested at tiny scale
(see below) — the pipeline runs end-to-end, trains, evaluates, and saves
graph-evolution artifacts without error. No result at the scale that would
answer the actual question yet.

The first real CellV1 experiment (`docs/architecture_v1.md`), specified by
the user on 2026-09-02. Compares four architectures on four tasks in one
run, deciding `docs/architecture_v1.md` §9's staging question (local-only
vs. local+global) directly rather than as a separate follow-up.

**Naming note:** the user called this "Experiment 005" in conversation,
but `005` already means `experiments/005_recurrence/` ("Recurrent Reuse")
in `docs/experiment_protocol.md`'s numbered V0/baseline sequence — a
different hypothesis (H2), unrelated to CellV1. Rather than overwrite that
meaning or renumber the whole 005-013 sequence again, CellV1 experiments
get their own `v1_NNN` namespace, independent of the V0 line's numbering
(`CLAUDE.md` §2's CellV1 exception; `docs/experiment_protocol.md`'s new
"CellV1 experiment track" note).

## The four architectures

| Architecture | What it is |
|---|---|
| `mlp` | `MLPBaseline`, parameter-matched to `cellv1_full` at whichever scale is running |
| `cellv0.1` | `BeliefNetwork(aggregation="scale_stable_precision")`, parameter-matched to `cellv1_full` |
| `cellv1_local` | `SparseDynamicBeliefGraph(use_global_routing=False)` — dynamic local graph only |
| `cellv1_full` | `SparseDynamicBeliefGraph(use_global_routing=True)` — dynamic local + dynamic global routing |

**Wired to CellV1.1 (sparse), not dense.** Dense `DynamicBeliefGraph`
doesn't scale (`O(n_cells^2)`) and stays in the repo only as
`docs/architecture_v1.md` §11's correctness reference — this harness
always builds `SparseDynamicBeliefGraph` (`sparse_model.py`).

`cellv1_local` is a genuine ablation (no global-routing/need/offer modules
at all), not a full model with its global output suppressed — it
therefore has *fewer* parameters than `cellv1_full`, reported as-is rather
than matched to it (the user: "we need to expose it rather than hide
it"). `mlp` and `cellv0.1` are matched to `cellv1_full`'s actual parameter
count, **recomputed at every scale** (their exact sizes differ between
V1-S0/S1/S2 since `cellv1_full`'s own count does).

**CellV1 is fixed across every arm/task/scale**
(`experiments/v1_001_dynamic_groups/harness.py`): `association_dim=8`,
`num_steps=6`, the same reused `scale_stable_precision` fusion, the same
`sparsemax` routing, the same shared update functions, the same LSH
hyperparameters (`num_hashes=2`, `bits=6`, `chunk_size_local=10`,
`chunk_size_global=4`). Only `n_cells` (population size) varies, over the
three scales the user asked to sweep:

| Scale | `n_cells` | `T` |
|---|---:|---:|
| `V1-S0` | 128 | 6 |
| `V1-S1` | 256 | 6 |
| `V1-S2` | 512 | 6 |

No hyperparameter hunting beyond that yet. `--n-objects`/`--k-min`/
`--k-max` are also wired all the way through the harness now (they were
already accepted by `src.data.synthetic.dynamic_groups.dynamic_groups`,
but the harness used to hardcode 24 objects in a few places) — not used
for anything but the default 24-object/`K∈{2..5}` setting yet, but a
higher-complexity run (48/96 objects, more groups) needs no code changes
when that's next.

## The four tasks

- `r2_interaction`, `c2_interaction`, `u2_heteroscedastic_interaction` —
  existing CellV0 datasets (`src/data/synthetic/`), reused as-is. Useful
  for a `cellv0.1` -> `cellv1_full` sanity check (large regressions here
  would mean something is broken), but **not** designed to require
  grouping, so not the basis for judging CellV1's core hypothesis.
- `dynamic_groups` — the important one, designed specifically around
  CellV1's hypothesis (`src/data/synthetic/dynamic_groups.py`). 24 objects
  per example, each `(k1, k2, v)`; a random number of hidden groups
  `K in {2..5}` per example, never labeled; target requires both a local
  per-group summary and a global cross-group interaction term (see that
  module's docstring for the exact formula). Unsolvable by independent
  per-object processing or by fixed compartments — the correct grouping
  changes every example.

### Input adapter

`r2`/`c2`/`u2` use CellV1's default `PopulationEncoder` (dense projection
to every cell). `dynamic_groups` uses
`src/models/architecture_v1/object_encoder.py::ObjectSeededEncoder`
instead: each of the first 24 cells is seeded one-to-one from one object
(`mu = v`, `z = F_seed(k1, k2)`, `e = u = 1`); the remaining `n_cells - 24`
cells start neutral (`mu = 0`, low evidence, high uncertainty) and are
available to be recruited during refinement — no special recruitment
machinery, just this initial condition. Per the user's note: a dense
global-broadcast adapter lets every cell see the whole input immediately,
which would partially bypass the local-then-global organization this
experiment is meant to test.

## What's measured

Standard prediction metrics (MAE/RMSE/R² or accuracy) for all four
architectures, parameter counts, and wall-clock (train + inference) for
all four — reported honestly even though CellV1 is expected to be
considerably more expensive per parameter than the baselines.

For `cellv1_local`/`cellv1_full` only, graph-analysis metrics
(`harness.py::_graph_metrics`): local-graph sparsity at `t=0` and `t=T`,
mean local-graph change per step, and — only for `dynamic_groups`, where
the true grouping is known — AUROC of local-association strength
predicting "same true group" (at `t=0` and `t=T`), and (for `cellv1_full`
only) the fraction of global edges that cross between different true
groups. Nothing beyond what the user asked for.

**Graph evolution:** for up to 100 held-out `dynamic_groups` examples, the
full per-step `(a_local, a_global)` sequence is saved to
`results/raw/graph_evolution/v1_001_<arch>_dynamic_groups_seed<N>.npz` —
raw material for the qualitative check the user wants most: does the local
graph visibly sharpen from a messy `t=0` into clear groups by `t=T`, and
do global edges concentrate on genuine cross-group links? Not summarized
by any script here — inspect directly (e.g. `np.load(...)`, then compare
`a_local[0]` vs `a_local[-1]` against `group_id` for a few examples).

## Commands

```bash
# Quick correctness check (~10-15s, all 4 models, one task -- NOT a real result):
# --n-cells overrides --scales with one small custom value.
python experiments/v1_001_dynamic_groups/run_v1_001.py \
    --tasks dynamic_groups --n-cells 30 \
    --steps 20 --batch-size 8 --n-train 60 --n-val 20 --n-test 20 --graph-eval-n 10 \
    --results-dir /tmp/v1_001_quicktest --summary-out /tmp/v1_001_quicktest/summary.json

# The real run: dynamic_groups, all 3 scales (V1-S0/S1/S2 = 128/256/512 cells, T=6), 3 seeds:
python experiments/v1_001_dynamic_groups/run_v1_001.py \
    --tasks dynamic_groups --seeds 0 1 2 --steps 1500 --n-train 3000

# One scale only (e.g. just V1-S0, for a faster look before committing to all three):
python experiments/v1_001_dynamic_groups/run_v1_001.py \
    --tasks dynamic_groups --scales V1-S0 --seeds 0 1 2 --steps 1500 --n-train 3000

# All four tasks (R2/C2/U2 too), all scales, three seeds:
python experiments/v1_001_dynamic_groups/run_v1_001.py --seeds 0 1 2 --steps 1500 --n-train 3000
```

**On "can we skip `mlp`/`cellv0.1` to save time":** no need to -- they're
plain feedforward nets, always fast regardless of scale (measured: ~5-8ms/
step vs. `cellv1_full`'s 168-861ms/step across the three scales), so
keeping them in every run costs almost nothing extra. Reusing the old
Experiment 004A/004B `mlp`/`cellv0.1` numbers instead isn't an option --
they're parameter-matched to fixed S0-S4 targets, not to `cellv1_full`'s
actual size at each of `V1-S0`/`S1`/`S2`, which is the whole point of the
comparison.

**Expected wall-clock, measured, not estimated** (CPU, `batch_size=64`,
`dynamic_groups`, `association_dim=8`, `hidden_dim=32`):

| `n_cells` | `cellv1_full` ms/step | `cellv1_local` ms/step |
|---:|---:|---:|
| 128 (V1-S0) | 168 | 127 |
| 256 (V1-S1) | 336 | 262 |
| 512 (V1-S2) | 861 | 680 |

At `--steps 1500`, summed over all three scales and both CellV1 arms, one
seed's `dynamic_groups` run is roughly `1500 * (168+127+336+262+861+680)
ms ≈ 3660s ≈ 61 minutes`; three seeds ≈ **3 hours**. `mlp`/`cellv0.1`
add negligible time. MPS (preferred by `src.utilities.device.get_device`,
this project's primary dev hardware) hasn't been benchmarked for this op
mix (`sparsemax`'s sort/cumsum/`searchsorted`/gather) -- may be faster or
slower than this CPU measurement. Reduce `--steps`/`--n-train`/`--scales`/
`--seeds` for a faster, lower-fidelity first look.

## What would make this a genuinely exciting result

Not a headline-metric edge alone. In combination:
`cellv1_full`/`cellv1_local` beat `mlp`/`cellv0.1` in low-data settings;
`cellv1_local` discovers the hidden groups (`local_group_agreement_auroc`
well above 0.5) despite never being told them; `cellv1_full` beats
`cellv1_local` specifically where the target needs the cross-group term;
the discovered graph visibly differs input-to-input and step-to-step
(the saved `graph_evolution` files); and the same `T`-step machinery is
doing this without `T` separate layers. See `docs/architecture_v1.md`'s
research-log entry for this experiment once it's run.

## After this experiment: don't immediately modify CellV1

Per the user's spec, this experiment should decide *what part* needs
work, not trigger an immediate rewrite:

- CellV1 doesn't form useful groups -> work on association / `z`.
- Groups form but global routing doesn't help -> work on global
  communication.
- Graph behavior looks right but predictions are poor -> work on cell
  update / readout / training.
- Everything works but compute is bad -> work on a sparse implementation.
