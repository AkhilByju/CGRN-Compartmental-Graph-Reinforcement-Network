# Experiment 004 — CellV0 Scaling

**Status:** 004A/C/D **done** — Result A (see `docs/research_log.md`
"Experiment 004A/C/D results"): `precision` and `mlp` are within seed noise
at ~600 params, and `precision`'s lead widens monotonically through ~150K
params on all three datasets. 004C found ablation damage does *not* shrink
with scale (hypothesis-favorable). 004D caught a real scale-instability in
`precision`'s `evidence` formula (unnormalized sum -> grows linearly with
`hidden_cells`), recorded, not fixed. Full write-up + interactive graph:
https://claude.ai/code/artifact/8e792a74-b856-4f44-a7f5-f4d35286b9cc

**All of 004A/B/C/D plus the cell-count-matched comparison are now done**
— see each section below and `docs/research_log.md` for full numbers.
`reliability` (optional secondary) was not run at any scale in any part of
this experiment.

**004E/F done** — the duplication test confirms `precision`'s scale
instability is a formal property of the formula, and `normalized_precision`
(a new, user-specified 4th aggregation candidate) fixes it exactly as
intended (`docs/research_log.md` "Experiment 004E/004F"). **004G/H written,
not run** — see their sections below for commands.

**004I done** — `"scale_stable_precision"` ("CellV0.1", a 5th aggregation
candidate, refining 004F's fix) implemented and unit-verified
(`docs/research_log.md` "Experiment 004I"). `run_004a.py`/`run_004b.py` now
take `--aggregation` so 004A/004B can be rerun against it — commands in
their sections below; not yet run.

## What's frozen

Identical to Experiments 002/003, unchanged here: `BeliefCell`'s state
(`cell.py`), `precision`'s aggregation formula (`integration.py`), network
depth (2 `BeliefLayer`s + linear readout, `belief_network.py`), AdamW, `tanh`
activations, and a plain MSE/cross-entropy training procedure. No calibration
loss, no recurrence, no compartment changes. The *only* thing that varies
across 004A/B is model size (`hidden_cells` / MLP `hidden_dim`); 004C/D
inspect already-trained 004A models, they don't train anything new.

**Main comparison:** `mlp_baseline` vs. `belief_network[precision]`
(`support_conflict` dropped per Experiment 002/003's evidence collapse
finding; `reliability` optional, not run here).

**Datasets:** `r2_interaction`, `c2_interaction`,
`u2_heteroscedastic_interaction` — the three where Experiment 003D found the
evidence/uncertainty intervention actually mattered (`docs/research_log.md`
"Experiment 003D results"). `r0_linear`/`c0_linear` are no longer
interesting (003D found the belief mechanism barely used there).

## 004A — Model-size scaling

Holds training-set size fixed at 100,000 examples and sweeps target
parameter count:

| Scale | Target params | Actual (varies slightly by dataset's `out_features`) |
|---|---|---|
| S0 | ~600 | `hidden_cells` search lands within a few % — exact count logged per run |
| S1 | ~2,500 | |
| S2 | ~10,000 | |
| S3 | ~40,000 | |
| S4 | ~150,000 | |
| ~~S5~~ | ~~~500,000~~~ | **dropped** — see below |

**S5 dropped.** A single S5 `precision` run (100K examples, 6000 steps)
measured ~285s locally; the full S5 sweep (3 datasets x 5 seeds) alone would
cost ~71 minutes, more than every other scale combined. Per the experiment
spec's own contingency ("if 500K becomes annoying on the M4, stop at 150K"),
S0–S4 is the run grid. Several orders of magnitude above 600 (600 -> 150,000
is ~250x) is preserved.

**Parameter matching:** for each scale, `hidden_cells` is chosen (closed-form
search, `scaling_harness.py::_match_hidden_cells`, using
`docs/architecture_v0.md` §7's `out_cells * (2*in_cells + 1)` per-`BeliefLayer`
formula) to land near the target; the `precision` model's *actual* resulting
parameter count then becomes the target for `MLPBaseline`, matched via the
existing `src.models.baselines.mlp.match_hidden_dim` search (same approach
`experiments/002_cell_v0/harness.py` already uses). Typical match is within
~1–4% (coarser at S0, where the search grid is sparse — reported per run as
`param_match_pct_off`).

**Seeds:** 5 (0–4). Per the spec, any scale/dataset combination showing an
interesting effect gets re-run with 10 seeds as a follow-up, not by default.

**Metrics per run:** prediction (R² for `r2_interaction`/
`u2_heteroscedastic_interaction`, accuracy for `c2_interaction`), exact
parameter count, train/inference wall-clock, mean/std across seeds.

**Code:** `scaling_harness.py::run_scale` (one (dataset, scale, seed) run,
both architectures), `run_004a.py` (the grid + summary table + a
`results/processed/004a_summary.json` dump for plotting).

```bash
python experiments/004_cellv0_scaling/run_004a.py
python experiments/004_cellv0_scaling/run_004a.py --scales S0 S1 S2 S3 S4 --seeds 0 1 2 3 4 --steps 6000
python experiments/004_cellv0_scaling/run_004a.py --aggregation scale_stable_precision
```

`--aggregation` (added for Experiment 004I) reruns this exact grid against
any `BeliefLayer` method (default `precision`, matching the run below;
`scale_stable_precision` is "CellV0.1" — see 004I) — not yet run for
anything but the default.

**The graph:** performance vs. log(parameter count), `mlp` vs. `precision`,
one line per dataset — **Result A**: parity at S0, `precision`'s lead
widens monotonically to S4 on all three datasets. Rendered graph + full
tables: https://claude.ai/code/artifact/8e792a74-b856-4f44-a7f5-f4d35286b9cc.
Numbers and caveats: `docs/research_log.md` "Experiment 004A/C/D results".

## 004B — Data scaling

**Status: done** — real sample-efficiency edge on `r2_interaction`/
`u2_heteroscedastic_interaction` (largest at 1,000 examples, e.g. U2
+5.89pt R² at D0 shrinking to a stable ~+0.8-1.0pt plateau by D2+), flatter
and smaller on `c2_interaction` (`docs/research_log.md` "Experiment 004B
results"). Fixed model size at `scale=S3` (~40K parameters — a scale from
004A's results where both architectures are clearly past the steep part of
their own learning curve), swept training-set size:

| Scale | Training examples |
|---|---|
| D0 | 1,000 |
| D1 | 3,000 |
| D2 | 10,000 |
| D3 | 30,000 |
| D4 | 100,000 |
| D5 | 300,000 |

Same two architectures, same three datasets, same frozen hyperparameters as
004A — only `n_train` varies.

```bash
python experiments/004_cellv0_scaling/run_004b.py
python experiments/004_cellv0_scaling/run_004b.py --aggregation scale_stable_precision
```

`--aggregation` (added for Experiment 004I): same as 004A — not yet run for
anything but the default `precision`.

## 004C — Evidence/uncertainty intervention at scale

Folded into 004A rather than run separately: at `scaling_harness.SCALES`
labels `S0`/`S2`/`S4` (~600/~10K/~150K), each just-trained `precision` model
is also run through Experiment 003D's four perturbation conditions
(`src.evaluation.intervention.perturb_belief`) with no retraining
(`scaling_harness.py::_ablation_metrics`, `run_ablation=True` by default at
those scales). Answers whether "intervention damage" (baseline minus
perturbed performance) grows, shrinks, or stays flat with model size —
**it grows or holds steady, not shrinks**: the larger model is not
learning to route around evidence/uncertainty (one partial exception:
`uncertainty_random` on `r2_interaction` does shrink with scale — see
`docs/research_log.md` for the full breakdown and a candidate mechanism).

## 004D — Internal state vs. scale

Also folded into every 004A run (`scaling_harness.py::_internal_state_metrics`),
logged unconditionally for `precision` at every scale: per layer (`layer1`,
`layer2`) mean/std of `evidence`, mean/std of `uncertainty`, mean/std/min/max
of the relevance gate `g = sigmoid(relevance_logit)`, and the
evidence-uncertainty correlation; plus, on `u2_heteroscedastic_interaction`
only, the correlation between the final layer's uncertainty and the dataset's
known noise level. Purpose: catch scale-instability in the `precision`
formula (e.g. evidence exploding as `hidden_cells` grows) *without* fixing
it — per the spec, a pathology found here is recorded as a scaling failure
of the current formula, not patched. **Found:** `layer2`'s `evidence` grows
~linearly with `hidden_cells` (21x from S0 to S4) because the aggregation
formula sums evidence over incoming cells with no normalization
(`integration.py::_precision_fusion`); `uncertainty` correspondingly
collapses toward zero. `layer1` (fixed `in_features=4` incoming cells at
every scale) doesn't show the effect, confirming the mechanism. See
`docs/research_log.md` for the full numbers.

## Parameter-matched vs. cell-count-matched

**Status: done** (`docs/research_log.md` "Cell-count-matched results").
Two different fairness regimes:

- **Parameter-matched (main comparison, 004A/B):** same trainable parameter
  count. Answers: which architecture uses parameters more efficiently?
- **Cell-count-matched (N0/N1/N2 = 15/68/271 hidden units, reusing 004A's
  own S0/S2/S4 widths):** same number of MLP hidden units and `BeliefCell`s
  (so `precision` has 82–99% *more* parameters at the same width, since
  each connection carries both a content weight `w` and a relevance
  parameter `g`). Answers: is one `BeliefCell` computationally richer than
  one ordinary neuron, independent of what it costs in parameters?

```bash
python experiments/004_cellv0_scaling/run_004_cell_count_matched.py
```

**Result is weaker and more mixed than the parameter-matched comparison**,
not stronger: despite the 82–99% parameter handicap in `mlp`'s favor,
`precision` is *behind* `mlp` at N0 on two of three datasets, and only pulls
clearly ahead at N2 on `c2_interaction`/`u2_heteroscedastic_interaction`
(not `r2_interaction`, which stays flat even with ~2x the parameters).
Counterintuitively, at the closest matched point (N2 ≈ S4), 004A's
parameter-matched comparison shows a *larger* `precision` advantage than
this cell-count-matched one does — the opposite of the naive expectation,
since this regime hands `precision` bonus parameters for free on top of
matched width. Flagged as an open observation, not yet explained (see the
research log entry for a candidate confound: `mlp`'s *width*, not just its
parameter count, may matter at this fixed step budget, independent of the
architecture question).

## 004E — The duplication test

**Status: done** (`docs/research_log.md` "Experiment 004E/004F"). Before
touching topology/recurrence/compartments, checks whether 004A's scaling
advantage is real by testing `precision`'s aggregation formula directly, no
training required: feed a `BeliefLayer` N identical copies of the exact
same belief content through N "equivalent connections" (every incoming
connection given the identical weight), for N in `{1, 2, 4, 8, 16}`
(`duplication_test.py`). Confirmed both analytically and via
`tests/test_duplication_invariance.py` (11 tests): `precision`'s `evidence`
grows exactly `N`x and `uncertainty` shrinks exactly `1/sqrt(N)` for
duplicated (i.e. informationless) copies — the width problem is a formal
property of the formula, not an artifact of any particular training run.

```bash
python experiments/004_cellv0_scaling/duplication_test.py
```

## 004F — Normalized Precision

**Status: done.** A new, 4th `BeliefLayer` aggregation candidate,
user-specified (not agent-invented — CLAUDE.md Sec 2), added alongside
`"precision"` rather than replacing it:
`src/models/architecture_v0/integration.py::_normalized_precision_fusion`.
Identical to `"precision"` except `evidence` and the base-uncertainty term
are divided by total incoming relevance `G = sum(g)` instead of left as raw
sums; the content-weighting (`alpha`) and disagreement term — what lets
cells compete for influence over `mu` — are untouched. Verified via 004E's
duplication test (analytically and in
`tests/test_duplication_invariance.py`) to keep `evidence`/`uncertainty`
exactly invariant to duplicate count, unlike `"precision"`. Documented as a
candidate in `docs/architecture_v0.md` §1.

## 004G — Compact 3-way scale rerun

**Status: written, not run.** `mlp` / `precision` / `normalized_precision`
at S0 (~600) / S2 (~10K) / S4 (~150K), on R2/C2/U2, 5 seeds — same frozen
hyperparameters as 004A. Always logs per-layer internal state (evidence/
uncertainty mean, by layer) for both belief variants, to check directly
whether `normalized_precision` keeps evidence/uncertainty scale-stable
under real training (not just in 004E's static test) and whether it
performs comparably to `precision` on actual data. No ablation (kept
compact — 004C already covers `precision`'s ablation-at-scale).

```bash
python experiments/004_cellv0_scaling/run_004g.py
```

## 004H — Fair MLP optimization check

**Status: written, not run.** At S0/S2/S4, trains `mlp` and `precision`
(only — this is specifically about *why* `mlp` declines relative to
`precision` in 004A, not about `normalized_precision`) at three learning
rates (0.3x/1x/3x the usual `1e-2`, same tuning budget for both
architectures) and records train/val/test **loss** (not just the headline
metric) for each. If `mlp`'s training loss itself gets worse with scale,
that's an optimization-difficulty story; if training loss is fine but
test loss/metric is worse, that's generalization/overfitting — a different
conclusion, and 004A's caveat about MLP's decline stays unresolved until
this runs. 3 seeds by default (not 5) — a targeted diagnostic, not a
result meant to stand alone statistically.

```bash
python experiments/004_cellv0_scaling/run_004h.py
```

## 004I — CellV0.1 (Scale-Stable Precision)

**Status: implemented and unit-tested; not yet run through 004A/004B.** A
5th `BeliefLayer` aggregation candidate, `"scale_stable_precision"`
(`src/models/architecture_v0/integration.py::_scale_stable_precision_fusion`),
user-specified as a refinement of 004F's `"normalized_precision"`: instead
of dividing `evidence`/base-uncertainty by raw total relevance `G =
sum(g)`, divide by an **effective source count** — a participation-ratio
(Kish effective-sample-size) statistic `N_eff = (sum(g))^2 / (sum(g^2) +
eps)`. Kept alongside both `"precision"` and `"normalized_precision"`, not
replacing either. Verified in `tests/test_scale_stable_precision.py` (plus
extensions to `tests/test_duplication_invariance.py`) to do what it's
supposed to: pass 004E's duplication test the same way `"normalized_precision"`
does, *and* — the actual point of the refinement — correctly discount
evidence for a growing number of weakly-relevant connections in a way
`"normalized_precision"` provably cannot (its raw-sum normalizer is
insensitive to how many sources contribute when their evidence is uniform).
See `docs/research_log.md` ("Experiment 004I") for a mistake this test
suite caught in the first draft of the reasoning, corrected before it
reached any real numbers.

`experiments/004_cellv0_scaling/scaling_harness.py::run_scale` (used by
both `run_004a.py` and `run_004b.py`) now takes an `aggregation` parameter
(default `"precision"`, fully backward-compatible) so the existing 004A/B
grids can be rerun against this new cell type with no new code — see the
`--aggregation` flag documented in the 004A/004B sections above. Not run —
commands there.

## Hypotheses tested

H1 (does the richer cell's relative standing change with scale — Result
A/B/C in `docs/research_log.md`), H5 (sample efficiency, via 004B). Answers
Q3–Q4 in `docs/hypotheses.md`.
