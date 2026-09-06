# v1_002 — Does CellV1.5's structural plasticity earn its keep?

CellV1.5's first evaluation (`docs/architecture_v1.md` §16). Filed inside
`v1_001_dynamic_groups/` because it reuses that experiment's task, splits,
convergence protocol, and parameter-matching helpers wholesale — only the
arms and the substrate measurements are new. Entry points:
`structural_harness.py` (the comparison), `run_structural_comparison.py`
(the runner), `tests/test_structural_harness.py` (what makes the
comparison attributable).

**CellV1.5 itself is frozen for this evaluation.** Nothing under
`src/models/architecture_v1/structural*.py` is modified by any file here.

## The question, and why there are three arms and not two

§16's claim is not "a sparse persistent substrate works." It is that
*slow structural plasticity* — repeated useful interactions becoming
structural connections, over optimizer-step timescales — is what makes
such a substrate worth having. A two-arm `cellv0.1` vs. CellV1.5
comparison cannot see that: whatever CellV1.5 scores would be the joint
product of its cell math, its `w_ij * a_ij(t)` decomposition, its sparse
fusion, *and* its rewiring, with no way to say which part did the work.

So the plasticity is the isolated variable:

| Arm | What it is |
|---|---|
| `cellv0.1` | `BeliefNetwork(aggregation="scale_stable_precision")`, parameter-matched to CellV1.5 |
| `cellv1.5_frozen` | `StructuralBeliefGraph`; the §16.8 bootstrap topology held fixed for **all** of training |
| `cellv1.5_plastic` | The identical model, with §16.9's prune/grow schedule executing |

Reading the outcome:

- Plastic > frozen ≈ or > `cellv0.1` → structural plasticity is doing
  real work; §16's core hypothesis survives its first test.
- Plastic ≈ frozen → the substrate may still be fine, but the *rewiring*
  is not what makes it so. A near-random bootstrap topology with a budget
  of `k_bar * n_cells` edges would then be doing everything the schedule
  was supposed to add.
- Both ≈ `cellv0.1` → CellV1.5 is, on this task, an expensive way to be
  a BeliefCell network, the same finding CellV1.4 landed on.

No `mlp`, no `field_*`, no `association_*`, no `cellv1_local`/`cellv1_full`
— §16.1 supersedes those for this variant and they stay in the repo as
frozen historical baselines. Their numbers are already in
`docs/research_log.md`.

## Enforcing "identical except whether rewiring runs"

The comparison only means something if the two CellV1.5 arms differ in
exactly one respect. Four things make that true, each with a test:

1. **Same initialization.** Each arm is constructed from a freshly reset
   RNG state, so their parameters are bit-identical at step 0 — not
   "same distribution," identical.
2. **Same bootstrap topology.** Bootstrap draws from a dedicated
   generator seeded identically per arm, so both start from the same
   substrate rather than two samples of a near-random one.
3. **Same data order.** Plasticity's LSH candidate retrieval draws from
   its *own* generator. Sharing the global stream would have shifted the
   plastic arm's minibatch order, so a difference in R² would partly have
   been a difference in which batches each arm saw.
4. **Same bookkeeping.** Both arms call `update_edge_utility()` every
   step. It touches no parameter and no gradient — withholding it from
   the frozen arm would have changed nothing except making the two runs
   less comparable.

Consequence, asserted directly in
`test_frozen_and_plastic_are_identical_before_first_event`: through
§16.9's warm-up the two arms are the *same run*, parameter for parameter.
They diverge at the first plasticity event and only there.

## Protocol

`dynamic_groups_global` (the argmax/argmin aggregate-value pairing task,
not the spatial-pairing one), two conditions matching
`run_complexity_scaling_association.py`'s definitions exactly so the
numbers sit next to CellV1.4's already-recorded sweep:

| Condition | `n_objects` = `n_cells` | `K` |
|---|---:|---|
| `hard` | 96 | 8..14 |
| `very_hard` | 192 | 12..20 |

3 seeds each. Convergence-based training with best-validation
checkpointing (validate every 100 steps, restore the best checkpoint
before test evaluation), AdamW at `lr=1e-2`, `batch_size=64`,
`n_train=3000`.

### Patience, and the delayed-breakout artifact

`--patience-steps 2000 --max-steps 15000`, against the `500`/`5000` the
CellV1.3/1.4 sweeps used. That older setting produced a documented
artifact (`docs/research_log.md`, 2026-09-03 CellV1.3.1 entry, finding 3):
`field_t2` at `very_hard`/seed 0 early-stopped at R²=0.0667 while the same
architecture reached 0.79/0.82 on the other two seeds — a model killed on
a plateau before it broke out, and reported as an architectural result.

Nothing here can prove a plateau was never cut short, so the run reports
the evidence instead: every arm records `total_steps_run` and
`hit_step_cap`, so a run that stopped because it ran out of budget is
distinguishable from one that stopped because it stopped improving.

## What is recorded

Per arm: test R², MAE, RMSE, parameter count, steps and wall-clock to the
best checkpoint, total steps and wall-clock actually spent, and inference
wall-clock.

CellV1.5 arms additionally, measured **on the restored best-validation
checkpoint** — the same weights and topology that produced the reported R²,
not wherever training happened to stop:

- **`structural_edges_changed_fraction`** — the fraction of edges not
  present in the bootstrap topology, plus
  `bootstrap_edges_retained_fraction` and the raw added/removed counts.
  §16.8 calls the bootstrap topology "disposable"; this is whether it
  actually got disposed of. Exactly 0 for the frozen arm, by construction.

  Reported twice, because "final" is genuinely ambiguous under early
  stopping: the plain key is measured on the **best checkpoint** (the
  substrate that produced the reported R²), and
  `..._at_end` on the topology training **ended** on. The best checkpoint
  often predates most plasticity events, so quoting only it understates
  how far rewiring got, and quoting only the end describes a substrate
  the reported R² never used. `edges_changed_between_best_and_end` is the
  gap, and `plasticity_events_at_best_checkpoint` vs.
  `plasticity_events_total` says how many events fall in it.
- **Functional edge-use rate across inputs** — see below.
- **Final degree distribution** — in- and out-degree mean/std/min/max/
  median/p90/p99/zero-fraction/Gini. §16.8 explicitly declines to
  constrain per-cell degree ("a cell may end up with 2 edges or 30"), so
  how unequal the distribution ends up is a result, not a diagnostic.
- **`plasticity_events_total`**, `plasticity_events_at_best_checkpoint`,
  `entered_freeze_window`, `min_in_degree` (§16.8's one safety constraint).

Raw arrays — degrees, final and bootstrap edge indices, per-edge utility,
age, and per-edge use rates — go to
`results/raw/structural_substrate/*.npz`, so the substrate can be
re-analyzed without re-running anything.

### How "functionally in use" is defined

§16 defines no such measure, so this is a measurement choice and is
stated rather than assumed. §16.5 fuses cell `j`'s incoming edges weighted
by precision `p_ij = a_ij(t) * e_i / (u_i² + eps)`. The share of `j`'s
incoming precision that edge `(i, j)` supplies is therefore the model's
*own* statement of how much that edge counts right now — no new mechanism
is introduced to measure it. An edge is **in use** for an input when

```text
share_ij  >=  tau / in_degree(j)
```

i.e. at least `tau` times what a uniform contribution would be, at any of
the `T` refinement steps.

`tau = 1.0` is the headline and the least arbitrary value: "this edge
carries at least its even share." Everything is also reported at
`tau in {0.5, 2.0, 4.0}` so no conclusion rests on the cutoff — a smoke
run showed `tau <= 0.5` saturating (essentially every edge in use for
essentially every input), which is exactly why the cutoff is reported as
a profile. `mean_effective_in_degree` (the perplexity `exp(H(share))` of
each target's share distribution) is threshold-free: near the mean
in-degree means the substrate is used near-uniformly, well below it means
the fusion genuinely selects a subset.

The three buckets that answer §16.0's claim directly, and always sum to 1:

- `edges_never_used_fraction` — dead weight the budget is paying for.
- `edges_always_used_fraction` — a static graph wearing a dynamic one's
  clothes.
- `edges_input_dependent_fraction` — used for some inputs and not
  others. This is the number §16.0's "different inputs activate different
  subsets" actually predicts, together with `mean_pairwise_jaccard`
  (overlap of two random inputs' active edge sets; 1.0 means every input
  uses the same subset).

## Implementation choices this harness had to make

Flagged rather than silent, the same way §16.10 flags CellV1.5's own:

- **`T = 2`.** §16 never pins the refinement-step count. 2 matches
  CellV1.4's `association_local_global_t2`, the immediately-preceding
  variant evaluated on this task under this protocol. Identical across
  both CellV1.5 arms by construction; `--num-steps` overrides it.
- **`ObjectSeededEncoder`, not `PopulationEncoder`.** §16.10 proposes the
  latter, but this task is object-structured and every prior CellV1 arm
  evaluated on it uses the former — a dense global-broadcast adapter lets
  every cell see the whole input immediately, partially bypassing the
  organization being measured. Same substitution `harness.py` already
  applies to every other CellV1 variant here.
- **`total_steps = max_steps` for §16.9's freeze window.** The freeze is
  "the last `freeze_fraction` of total training," which a convergence
  protocol does not know in advance. The training *budget* is used, so
  the freeze window is the last 20% of the cap and is entered only by
  runs that get that far — visible per run via `entered_freeze_window`.
- **Checkpointing through a resized `EdgeRegistry`.** `structural.py`
  flags that its `state_dict()` round-trips only when the edge count
  matches at load time; best-validation checkpointing hits exactly that
  case. `_restore_model_state` re-points the three registry buffers to
  the checkpoint's shapes before loading. Harness-level workaround, not a
  change to CellV1.5.
- **Parameter matching is approximate.** CellV1.5 is the size-defining
  target and `cellv0.1` is matched to it, following this folder's
  existing convention — but `BeliefNetwork`'s size grid is coarse at
  these `in_features`, so the match is not exact.
  `param_match_error_fraction` is recorded per run rather than the match
  being implied to be exact.

## A device bug found, and deliberately not fixed

CellV1.5's LSH candidate retrieval
(`structural_plasticity.py::_candidate_pairs`) builds its random
hyperplanes on the default device rather than on `phi`'s, so bootstrap and
growth raise `RuntimeError: Placeholder storage has not been allocated on
MPS device!` on any non-CPU device. CellV1.1 avoids this by registering
its hyperplanes as a buffer (`sparse_routing.py`), which moves with the
model; CellV1.5 creates them per call.

It is a one-line device-placement fix and nothing to do with §16's math —
but CellV1.5 is frozen for this evaluation, so this experiment runs on CPU
(`--device cpu`, the default here) and the bug is reported rather than
patched. Worth fixing before any run large enough to want a GPU.

## Commands

```bash
# The real run: both conditions, 3 seeds. CPU, roughly 1-2 hours.
python experiments/v1_001_dynamic_groups/run_structural_comparison.py

# One condition:
python experiments/v1_001_dynamic_groups/run_structural_comparison.py --levels hard

# Fast correctness check (~15s) -- NOT a result. Small enough to be
# nonsense numerically, big enough to pass warm-up so plasticity fires.
python experiments/v1_001_dynamic_groups/run_structural_comparison.py \
    --levels hard --seeds 0 --n-objects-override 24 --n-train 400 \
    --n-val 100 --n-test 100 --val-every 25 --patience-steps 200 --max-steps 600 \
    --results-dir /tmp/v1_002_smoke --graph-dir /tmp/v1_002_smoke/graphs \
    --summary-out /tmp/v1_002_smoke/summary.json
```

Summary JSON: `results/processed/v1_002_structural_substrate.json`.
Per-run records: `results/raw/` (`RunRecord`, the standard metadata
contract). Substrate arrays: `results/raw/structural_substrate/`.

## After this experiment

Per the user's instruction, no CellV1.5 math changes and no new routing
mechanisms before these results are in. What the outcome should decide,
not what it should trigger:

- Plastic beats frozen → the schedule is load-bearing; the next questions
  are its knobs (`k_bar`, prune fraction, interval) and whether the
  effect grows with `n_cells`.
- Plastic ≈ frozen, both strong → the substrate's *sparsity and
  persistence* are what matter, not the rewiring. That would make the
  bootstrap topology, not the plasticity schedule, the thing to study.
- Plastic ≈ frozen, both weak → look upstream of plasticity (the
  `w_ij`/`a_ij` split, the 2-source fuse, `T`) before touching §16.9.
