# Research Log

Append-only research diary, newest entry at the bottom. Don't rewrite past
entries — if an earlier decision turns out wrong, add a new entry noting the
correction and why; the history of "what we believed when" is itself useful.

## Entry format

```markdown
## YYYY-MM-DD — Short title

**Context:** what prompted this entry.
**Decision / finding:** what was decided or observed.
**Why:** the reasoning.
**Follow-up:** what this implies for next steps, if anything.
```

## Entries

## 2026-08-31 — Repository initialized

**Context:** Starting the CGRN project from a research blueprint covering
motivation, architecture design space, hypotheses, benchmark plan, and
experiment sequence.

**Decision / finding:** Initialized the repository as reproducibility
scaffolding only — directory structure, docs, config/run/logging
conventions, generic (architecture-agnostic) training and evaluation
infrastructure, and stub files for baselines and for `architecture_v0`. No
architecture math was invented and no compartment/graph/plasticity logic was
implemented.

**Why:** The blueprint is explicit (see `docs/architecture_v0.md` and
`CLAUDE.md`) that `CellV0` has not been mathematically specified yet, and
that specifying it is a research decision for the user, not something a
coding agent should infer. Isolating this decision keeps the eventual
architecture experiments interpretable.

**Follow-up:** The next research task is **Architecture Specification
V0.1** — mathematically defining `CellV0` per `docs/architecture_v0.md` §7.
Only after that should `src/models/architecture_v0/cell.py` be implemented,
followed by Experiment 001 (baseline validation) and Experiment 002 (CellV0
alone).

## 2026-08-31 — CellV0's state decided: BeliefCell

**Context:** Working through what "CellV0" concretely replaces in an
ordinary network. Earlier framing (multiple intra-cell compartments,
`h_i(t)` state) was drifting toward treating CellV0 as a recurrent module
sitting *inside* a network, rather than a direct replacement for a single
neuron.

**Decision / finding:** CellV0 replaces the ordinary neuron's scalar
activation `a_i = phi(sum_j w_ij a_j + b_i)` with a structured belief state
`B_i = (mu_i, u_i, e_i)`:

- `mu_i in R` — content: what the cell currently believes.
- `e_i in R+` — evidence: how much supporting information produced it.
- `u_i in R+` — uncertainty: how (un)confident the cell is in `mu_i`.

Wherever an ordinary network has one neuron, CellV0 has one such
`BeliefCell` (implemented in `src/models/architecture_v0/cell.py`). This
supersedes the earlier intra-cell "compartments" framing (`D_i1..D_iB`) —
compartments are not part of CellV0; a belief cell holds one `(mu, u, e)`
state, not several.

`tau` (persistence, `tau in [0,1]`) stays out of the `BeliefCell` for V0.0.
A feedforward cell fires once, so there's nothing for persistence to
persist across yet; it's reintroduced when V0.1 adds recurrence. (Decision
between "keep `tau` as a dormant field now" vs. "add it only when
recurrence arrives": chose the latter — a feedforward-only V0.0
experiment is cleaner without a field that does nothing yet.)

**What's still open (not decided by this entry):** how a cell aggregates N
incoming `BeliefCell`s (each via a learned weight `w_ij`) into its own
outgoing `BeliefCell`. A candidate direction was sketched but explicitly
**not frozen**:

- Proposed content from sender `j`: `m_ij = w_ij * mu_j`.
- Support weight: `s_ij = |w_ij| * e_j / (u_j + eps)` — large evidence
  strengthens a message, large uncertainty weakens it, a weak connection
  weakens it.
- Receiver's content: `z_i = (sum_j s_ij * m_ij) / (sum_j s_ij + eps) + b_i`,
  then `mu_i = phi(z_i)`.
- Receiver's evidence: `e_i = sum_j s_ij`.
- Receiver's uncertainty (simple version): `u_i = 1 / sqrt(e_i + eps)`.
- Receiver's uncertainty (more sophisticated version, accounting for
  disagreement among senders, not just their count): weighted variance
  `V_i = (sum_j s_ij * (m_ij - z_i)^2) / (sum_j s_ij + eps)`, then
  `u_i = f(V_i, e_i)` for some `f` — so that many strong-but-disagreeing
  inputs don't produce spuriously low uncertainty the way plain evidence
  accumulation would.

**Why:** Isolating "what is CellV0's state" from "how do cells combine"
lets the state be frozen and implemented now (§7 item 1 of
`docs/architecture_v0.md`) while the harder aggregation question is worked
out separately, without blocking all progress on it.

**Follow-up:** `src/models/architecture_v0/cell.py` implements
`BeliefCell` (validated: non-negative evidence/uncertainty, shape/dtype/
device agreement across fields). `integration.py` remains a blocked stub —
implement it only once one of the above formulas (or a replacement) is
confirmed, not inferred from this log entry. `docs/architecture_v0.md` §1
and §7 were updated to match this decision.

## 2026-09-01 — CellV0 aggregation candidates: three methods implemented, none chosen

**Context:** Following up on the previous entry's open question (how N
incoming `BeliefCell`s combine into one outgoing `BeliefCell`). Rather than
committing to one formula, the decision was to fully specify three
candidate aggregation methods, implement all three behind one switchable
interface, and let an empirical comparison (Experiment 002/003) decide
whether any of them is worth keeping.

**Decision / finding:** All three methods share the same learned-connection
structure: a content weight `w_ij` (how sender `j`'s content affects
receiver `i`) producing a message `m_ij = w_ij * mu_j`, and a relevance
gate `g_ij = sigmoid(a_ij)` (how relevant `j` is to `i`). They differ only
in how they turn incoming `(m_ij, g_ij, e_j, u_j)` into the receiver's
`(mu_i, e_i, u_i)`:

**Method A — Reliability-Weighted Consensus (`"reliability"`)**

```text
r_ij     = g_ij * e_j / (1 + u_j)                       # reliability
alpha_ij = r_ij / (sum_k r_ik + eps)                     # normalized influence
c_i      = sum_j alpha_ij * m_ij                         # consensus content
mu_i     = tanh(c_i + b_i)
e_i      = sum_j r_ij                                    # total reliable support
d_i      = sum_j alpha_ij * (m_ij - c_i)^2               # weighted disagreement
u_i      = sqrt(d_i + 1 / (e_i + eps))
```

More evidence and less uncertainty in a sender increase its influence;
uncertainty in the receiver grows from disagreement among reliable senders
*and* from a lack of total evidence -- either alone can drive it up.

**Method B — Support/Conflict Integration (`"support_conflict"`)**

```text
r_ij = g_ij * e_j / (1 + u_j)                            # same reliability as A
q_ij = tanh(m_ij)                                        # bounded claim, [-1, 1]
P_i  = sum_j r_ij * max(q_ij, 0)                         # positive support
N_i  = sum_j r_ij * max(-q_ij, 0)                        # negative support
e_i  = P_i + N_i
c_i  = (P_i - N_i) / (e_i + eps)                         # net support
mu_i = tanh(c_i + b_i)
conflict_i = 2 * min(P_i, N_i) / (e_i + eps)
u_i  = conflict_i + 1 / sqrt(e_i + eps)
```

Makes internal disagreement explicit rather than implicit: `P=10,N=0` and
`P=5,N=5` both have `e_i=10`, but only the second has high conflict.
"Positive"/"negative" are learned latent directions, not a hand-given
semantic meaning.

**Method C — Precision-Based Belief Fusion (`"precision"`)**

```text
pi_ij = g_ij * e_j / (u_j^2 + eps)                       # precision
alpha_ij = pi_ij / (sum_k pi_ik + eps)
c_i   = sum_j alpha_ij * m_ij
mu_i  = tanh(c_i + b_i)
e_i   = sum_j g_ij * e_j                                 # evidence kept separate from precision
u_base_i = sqrt(1 / (sum_j pi_ij + eps))
d_i   = sum_j alpha_ij * (m_ij - c_i)^2
u_i   = sqrt(u_base_i^2 + d_i)
```

The most probabilistically motivated: precision (inverse-variance-style)
combination, penalizing uncertainty quadratically rather than linearly --
deliberately harsher than A/B on unreliable senders.

**Shared decisions across all three:**

- `phi = tanh` for content, for this first experiment (keeps `mu` in
  `[-1, 1]`, making agreement/conflict comparisons cleaner across methods).
  A final regression output head would use no activation, so predictions
  aren't artificially bounded -- that's a decoder-level choice, out of
  scope for `BeliefLayer` itself.
- Input initialization for directly observed raw features (first
  experiments only): `mu_j = x_j`, `e_j = 1`, `u_j = 1` for every feature
  -- equal evidence/uncertainty everywhere so the network has to learn any
  useful structure itself, not receive it for free
  (`BeliefCell.from_observed_features`).
- `eps = 1e-8` throughout, to avoid division by zero without materially
  changing behavior at realistic evidence/uncertainty scales.
- All three guarantee `mu_i = f(mu_j, e_j, u_j)`: backprop from the final
  loss reaches incoming evidence and uncertainty, not just content -- they
  are not merely reported, they influence what the network predicts.

**Why:** Committing to one formula without evidence would be exactly the
kind of premature architectural claim this project's philosophy warns
against (`docs/research_thesis.md` §2). Implementing all three behind an
identical interface makes the choice an experimental question with a
controlled comparison (same weights/gates/bias structure, same `phi`, same
initialization), rather than a design debate.

**Follow-up:** `src/models/architecture_v0/integration.py` implements
`BeliefLayer(in_cells, out_cells, aggregation="reliability" |
"support_conflict" | "precision")`. `src/models/architecture_v0/cell.py`
gained `BeliefCell.from_observed_features`. Tests
(`tests/test_belief_layer.py`) cover: output shape, that outputs are
finite and satisfy `BeliefCell`'s own non-negativity invariants (a
correctness check on the formulas themselves), that gradients reach
incoming `mu`/`evidence`/`uncertainty` for all three methods, and one
hand-verified numeric case for Method A. Parameter count is now closed-form
(`docs/architecture_v0.md` §7 item 6: `out_cells * (2*in_cells + 1)`) and
identical across methods, so a parameter-matched plain-MLP baseline is
straightforward once Experiment 001 exists. Next: Experiment 002 runs all
three plus an MLP baseline on parameter-matched, otherwise-identical
networks (`docs/architecture_v0.md` §1 "Test the cell before the graph") --
this has not been done yet; nothing here should be read as "Method X
performed better."

## 2026-09-01 — Experiment 002 initial results: mixed-to-negative

**Context:** Ran the comparison Experiment 002 was designed for: an MLP
baseline vs. `BeliefNetwork` under each of the three aggregation methods,
on R0/R1/R2 (regression) and C0/C1/C2 (classification), 3 seeds each,
~parameter-matched (570-720 params depending on dataset), 2000 steps,
hidden_cells=16. Full setup in `experiments/002_cell_v0/README.md`; raw
per-run JSON in `results/raw/` (gitignored, regenerate with
`python experiments/002_cell_v0/run_all.py --seeds 0 1 2 --steps 2000`).

**Finding 1 — predictive performance: no consistent advantage.**

```text
                 mlp      reliability   support_conflict   precision
R0 linear        0.9927   0.9924        0.9918             0.9925    (R^2)
R1 nonlinear     0.9241   0.9267        0.9182 (+/-0.007)  0.9267    (R^2)
R2 interaction   0.9691   0.9617        0.9051 (+/-0.014)  0.9686    (R^2)
C0 linear        0.9433   0.9411        0.9422             0.9411    (accuracy)
C1 XOR           0.8978   0.9000        0.8911             0.8989    (accuracy)
C2 interaction   0.9389   0.9356        0.9289             0.9500    (accuracy)
```

`reliability` and `precision` track the MLP baseline closely everywhere
(within ~1%, i.e. noise at 3 seeds). `support_conflict` is the one
consistent standout, and it's a negative one: clearly worse on R2
(0.905 vs. 0.969, and the highest variance of any cell, +/-0.014) and
mildly worse elsewhere. No method showed the hoped-for edge on the
interaction-heavy tasks (R2, C2) over a parameter-matched MLP -- the
motivating question in `docs/architecture_v0.md` §1 ("does carrying
evidence/uncertainty help, for which method, and for which function
classes?") gets a "not yet demonstrated" answer at this scale.

**Finding 2 — uncertainty calibration: direction flips with task
difficulty, and is not seed noise.** The "ambiguous vs. clear" test
(mean uncertainty on near-decision-boundary points minus mean uncertainty
on far-from-boundary points; positive = intended behavior) gave, with
signs consistent across all 3 seeds individually (not just in the mean):

```text
                 reliability          precision            support_conflict
C0 linear        +0.056 (3/3 +)       +0.130 (3/3 +)       +3.03 (3/3 +, huge)
C1 XOR           -0.146 (3/3 -)       -0.194 (3/3 -)       +3.04 (3/3 +, huge)
C2 interaction   -0.046 (3/3 -)       -0.048 (3/3 -)       +0.88 (3/3 +, huge)
```

`reliability` and `precision` show the intended direction (ambiguous >
clear uncertainty) *only* on the easiest, linearly separable task, and
consistently invert on the two harder, nonlinear tasks. That is close to
the opposite of what would make evidence/uncertainty useful -- if
anything, uncertainty is least trustworthy exactly where it would matter
most (XOR, interaction).

**Finding 3 — `support_conflict` shows evidence collapse.** Its `evidence`
values (`P_i + N_i`) were ~0.02-0.1 across every dataset, roughly 10-500x
smaller than `reliability`'s (~0.5-1.5) or `precision`'s (~3-11) evidence
on the same data, while its `uncertainty` was correspondingly 4-16 (vs.
0.3-1.6 for the other two) -- an order of magnitude larger and far noisier
across seeds. Its "correct-direction" ambiguous/clear gaps above are not
evidence it calibrates well; they're a symptom of `1/sqrt(evidence + eps)`
dominating `uncertainty` when evidence is near-collapsed, which explains
both the huge magnitudes and the huge per-seed variance (e.g. C1 XOR:
2.02, 3.28, 3.82 across seeds -- a large spread even though the sign is
stable).

**Why (mechanistic hypothesis, not yet verified further):** nothing in the
training loss directly supervises `evidence` or `uncertainty` -- only
`mu` (via the readout layer) reaches the loss. Evidence/uncertainty do
receive gradient (confirmed in `tests/test_belief_layer.py`), but only
*instrumentally*, insofar as they shape `mu` through `alpha`/`precision`
weighting. There is no term in the loss rewarding "uncertainty should be
higher on ambiguous inputs" -- so a configuration where the optimizer sets
evidence/uncertainty however is most convenient for fitting `mu`, with no
resemblance to genuine epistemic uncertainty, is not just possible but
arguably the default outcome. That would explain both the direction-flip
(nothing forces a consistent relationship between margin and uncertainty)
and `support_conflict`'s evidence collapse (nothing forces evidence to stay
away from a degenerate near-zero solution). This has not been tested
further -- e.g. by adding an explicit calibration term to the loss and
seeing whether the flip disappears -- so it remains a hypothesis, not a
conclusion.

**Relation to `docs/hypotheses.md`:** this is a mixed-to-negative result in
the sense §2 of `research_thesis.md` explicitly asks for: performance is
roughly neutral (not a clean loss, not a clean win), but the specific
*qualitative* property motivating the whole design (H1: structured
evidence/uncertainty should provide a useful representational bias) is not
supported at this scale -- and the failure has a plausible, checkable
mechanism (evidence/uncertainty are unsupervised side-channels), which is
exactly the kind of "rigorously analyzed failure" `docs/hypotheses.md`
"What would constitute a useful negative result" calls for.

**Follow-up (not yet done, listed as candidates, not started):**
- Experiment 003 (cell ablations) could isolate whether the direction-flip
  is intrinsic to these formulas or an artifact of this scale/hidden width
  -- e.g. sweep `hidden_cells`, steps, and learning rate before drawing a
  final conclusion.
- Test the "unsupervised side-channel" hypothesis directly: add an
  explicit calibration signal (e.g. penalize low uncertainty on
  known-ambiguous training examples) as a deliberate, isolated change and
  see whether calibration improves -- this would itself be a new,
  documented variable, not folded into "the" aggregation formula.
- Investigate `support_conflict`'s evidence collapse specifically (e.g.
  does it correlate with `relevance_logit` being driven very negative?)
  before deciding whether it's a fixable initialization/scaling issue or
  a structural problem with the method.
- None of this yet justifies dropping any of the three methods, or
  concluding CellV0 "doesn't work" -- see `docs/research_thesis.md` §2 on
  not over-concluding from a first pass at one scale.

## 2026-09-01 — Experiment 003D results: evidence/uncertainty are not decorative, but aren't calibrated either

**Context:** Follow-up to Experiment 002's open question ("is `(evidence,
uncertainty)` actually used, or has the network learned to route around
it?"). Trained a `precision` `BeliefNetwork` on Experiment 002's six
datasets (5 seeds, 2000 steps, `hidden_cells=16`, `lr=1e-2` -- identical
hyperparameters to Experiment 002 for comparability), then at inference
time perturbed the evidence/uncertainty that the trained `layer1` hands to
`layer2` under four conditions (`src/evaluation/intervention.py`) and
measured the change in test R²/accuracy relative to the unperturbed
baseline. Full setup: `experiments/003_cell_ablation/README.md` "003D";
code: `evidence_uncertainty_harness.py`, `run_003d.py`; raw per-run JSON in
`results/raw/003d_evidence_uncertainty_intervention_*` (gitignored,
regenerate with `python experiments/003_cell_ablation/run_003d.py --seeds 0
1 2 3 4 --steps 2000`).

**Finding 1 -- performance is NOT invariant to evidence/uncertainty
perturbation; `(e, u)` are load-bearing, not decorative, on every dataset
except the most linear ones.**

```text
                 baseline    evidence=1   uncertainty=1   shuffle(e,u)   uncertainty=random
r0_linear        0.9924 R2   -0.008       -0.008          -0.049         -0.018
r1_nonlinear     0.9288 R2   -0.327       -0.327          -1.988         -0.636
r2_interaction   0.9699 R2   -1.145       -1.536          -10.924        -12.284
c0_linear        0.9473 acc  +0.000       -0.001          +0.001         -0.003
c1_xor           0.9033 acc  -0.044       -0.159          -0.186         -0.257
c2_interaction   0.9467 acc  -0.177       -0.310          -0.379         -0.423
```

(deltas are mean-across-5-seeds change from baseline; R² can go arbitrarily
negative -- it is not bounded below like accuracy -- so e.g. r2's -12.284
means predictions become far worse than always predicting the training
mean, not a small regression.)

Every perturbation on every dataset except `r0_linear`/`c0_linear` causes a
substantial-to-catastrophic performance drop. That directly answers 003's
motivating question: for `precision`, the network has **not** learned to
route around its evidence/uncertainty channels -- `layer2`'s aggregation
depends on them in a way that matters a great deal for the final
prediction, most dramatically on the interaction-heavy tasks (`r2`, `c2`)
and XOR (`c1`).

**Finding 2 -- the "decorative state" pattern does hold, but only on the
two most linear datasets.** `r0_linear` and `c0_linear` -- the one dataset
per track with no interaction/nonlinearity -- show deltas within the
seed-to-seed noise band (std ~0.004-0.024, comparable to the delta itself).
On every other dataset the effect is large and consistent in sign across
all 5 seeds individually (not just the mean). So "is CellV0 basically an
MLP with decorative state" gets a dataset-dependent answer: yes-ish on
trivial linear problems (where there is little for a weighting mechanism
to do), clearly no on anything with interactions or XOR-like structure.

**Finding 3 (structural, not a bug -- verified directly) -- for
single-input-feature datasets (`r0_linear`, `r1_nonlinear`), `layer1`'s
output evidence and uncertainty are exactly input-independent constants
satisfying `evidence * uncertainty^2 == 1` (checked numerically: e.g.
`(out.evidence * out.uncertainty**2)` is `1.0000` for every hidden cell to
float32 precision). This follows directly from `_precision_fusion`'s
formula with exactly one incoming cell: `BeliefCell.from_observed_features`
fixes the raw input's `evidence=1, uncertainty=1`, so with one incoming
cell `alpha=1` trivially, disagreement is exactly 0, and
`base_uncertainty_sq = 1/precision = 1/(g*e_j) = 1/g` while
`evidence_out = g*e_j = g` -- giving `evidence_out * uncertainty_out^2 ==
1` exactly, independent of `x`. That is *why* `evidence_ones` and
`uncertainty_ones` produce numerically identical deltas on `r0_linear`/
`r1_nonlinear` (setting either channel to 1 collapses `layer2`'s precision
weighting to the same value, `g_layer2 * g_layer1`) but clearly different
deltas everywhere else (`r2_interaction`, `c1_xor`, `c2_interaction` all
have >1 input feature, so `layer1`'s evidence/uncertainty do vary with
`x`). Not itself evidence for or against "decorative state" -- it's a
reminder that with a single incoming cell, `precision`'s evidence/
uncertainty carry no *per-example* information no matter what, only a
fixed per-hidden-cell learned weighting.

**Relation to Experiment 002:** this result does not contradict Experiment
002's calibration finding (uncertainty doesn't track true ambiguity,
direction flips on harder tasks). "The network relies heavily on this
channel" and "this channel represents genuine epistemic uncertainty" are
different properties -- 003D shows the former is true (at least for
`precision`, on non-trivial datasets); Experiment 002 already showed the
latter is not. Together: `evidence`/`uncertainty` behave like a real,
load-bearing internal precision-weighting/gating mechanism the network
learned to lean on for combining cell content -- just not one that
happens to track the ambiguity/noise structure a human would call
"uncertainty."

**Not yet done:** `reliability`/`support_conflict` (only `precision` was
tested here); whether the same catastrophic-on-interaction pattern holds
at other `hidden_cells`/scales; and whether the effect is specific to the
`layer1`->`layer2` boundary or would also show up in a deeper stack once
one exists (Experiment 005+).

## 2026-09-01 — Experiment 004A/C/D results: Result A (better parameter scaling), ablation damage does not shrink with scale, and a scale-instability in `precision`'s evidence formula

**Context:** Model-size scaling of the frozen `precision` `BeliefNetwork`
(`BeliefCell`, aggregation math, depth, AdamW, `tanh`, plain-loss training
procedure all unchanged from Experiments 002/003) vs. a parameter-matched
`mlp_baseline`, at 100,000 fixed training examples, 6,000 steps, 5 seeds,
on `r2_interaction`/`c2_interaction`/`u2_heteroscedastic_interaction` --
the three datasets Experiment 003D found the belief mechanism actually
mattered for. Five target scales, S0 (~600 params) to S4 (~150,000);
**S5 (~500,000) was dropped** before running anything -- a single
uncontended S5 `precision` run measured ~285s, so the full S5 sweep alone
would have cost more wall-clock than S0-S4 combined, matching the
experiment spec's own "if 500K becomes annoying, stop at 150K" contingency.
Parameter matching (closed-form `hidden_cells` search for `precision`,
`match_hidden_dim` for `mlp`, both against the *other* model's actual
count) landed within 0.04-3.6% mean absolute error per scale. Full
interactive results, the primary performance-vs-log(params) graph, and the
tables below: https://claude.ai/code/artifact/8e792a74-b856-4f44-a7f5-f4d35286b9cc
(also `results/raw/004a_model_size_scaling_*.json`,
`results/processed/004a_summary.json`). Code:
`experiments/004_cellv0_scaling/`.

**Finding 1 (004A) -- Result A: `precision` scales better than `mlp` on
every dataset tested.** Per the experiment spec's own taxonomy of possible
outcomes, this is "Result A": MLP and Precision are within seed noise at
S0, and the gap opens *monotonically* in Precision's favor from S1 onward:

```text
                              S0 (~600)   S1 (~2.5K)  S2 (~10K)   S3 (~40K)   S4 (~150K)
r2_interaction    (R2, delta) -0.0018     +0.0008     +0.0009     +0.0040     +0.0070
c2_interaction   (acc, delta) -0.0057     -0.0064     -0.0003     +0.0035     +0.0141
u2_heterosced.    (R2, delta) -0.0010     +0.0012     +0.0069     +0.0095     +0.0181
```

(delta = precision_mean - mlp_mean, mean of 5 seeds each; positive =
precision ahead.) Every one of the three datasets shows the identical
qualitative shape: parity or a small precision deficit at S0, a sign flip
around S1-S2, and the *largest* precision advantage at the *largest* scale
tested. This is the specific pattern the experiment plan flagged in advance
as the interesting one -- not a fixed gap, but a widening one, suggesting
the belief mechanism's relative value grows with capacity rather than
staying flat (Result B) or reversing (Result C).

**Caveat on Finding 1 -- MLP's own score declines with scale, not just
matched by precision's gain.** Most of the widening gap is `mlp`'s own
performance *dropping* as it gets wider at this fixed 100K-example/
6,000-step budget: e.g. `c2_interaction` accuracy goes 94.3% (S0) ->
92.0% (S4) for `mlp`, while `precision` stays roughly flat (93.8% ->
93.4%) to slightly recovering. A wider MLP may simply need more optimizer
steps to fit the same amount of data at this learning rate (AdamW,
`lr=1e-2`, unchanged across scales per the "frozen hyperparameters" rule)
-- this has *not* been isolated from "MLP is worse at large scale" as a
genuine architectural property. Before treating Result A as conclusive,
the natural check is training the largest `mlp` for more steps (or sweeping
LR) to see whether its decline is an optimization artifact of the fixed
budget or persists.

**Finding 2 (004C) -- ablation damage does not shrink with scale; if
anything it holds steady or grows.** Experiment 003D's evidence/uncertainty
intervention, re-run on the just-trained `precision` model at S0/S2/S4 with
no retraining. `c2_interaction` (accuracy units, the cleanest to read since
R2 is unbounded below):

```text
              S0        S2        S4
e=1        -9.5pt    -36.7pt   -41.0pt
u=1        -31.1pt   -32.1pt   -38.3pt
shuffle    -36.4pt   -44.0pt   -45.0pt
u=random   -38.2pt   -40.3pt   -39.0pt
```

Damage *grows* from S0 to S4 on every one of the four perturbations for
`c2_interaction`, and stays severe (R2/U2 deltas remain in the
single-to-double-digit-R2 range, i.e. catastrophic) across all three
scales on `r2_interaction`/`u2_heteroscedastic_interaction` too -- with one
partial exception: `uncertainty_random` on `r2_interaction` *shrinks* from
-13.3 (S0) to -4.6 (S4), the one condition/dataset pair that doesn't fit
the pattern (plausible mechanism, not yet verified: layer2's `uncertainty`
values shrink toward 0 as scale grows -- see Finding 3 -- so a substitute
drawn from that same shrinking observed range becomes a smaller absolute
perturbation at large scale, specific to this one mode's construction).
Overall this directly answers 004C's motivating question in the
hypothesis-favorable direction: **the larger model is not learning to
route around evidence/uncertainty** -- exactly the "genuinely interesting"
outcome the experiment spec called out in advance, as opposed to damage
shrinking toward zero (which would have meant large models bypass the
mechanism).

**Finding 3 (004D) -- a real, mechanistically-explained scale-instability
in `precision`'s `evidence` formula.** Per-layer internal state, logged at
every scale for `precision` (shown for `r2_interaction`; `c2_interaction`/
`u2_heteroscedastic_interaction` match the same pattern):

```text
                        S0      S1      S2      S3      S4
layer2 evidence mean   10.6    23.3    49.1    105.8   220.3   (21x growth)
layer2 uncertainty mean 0.91    0.61    0.43    0.32    0.24   (shrinks toward 0)
layer1 evidence mean    1.67    1.65    1.69    1.74    1.79   (~flat)
```

`layer2`'s `evidence` grows *linearly* with `hidden_cells`
(evidence/hidden_cells is ~0.71-0.81 at every scale) while `layer1`'s stays
flat -- exactly the difference the formula predicts:
`_precision_fusion`'s `evidence = (g * e_j).sum(dim=-1)`
(`integration.py`) is an **unnormalized sum over incoming cells**, and
`layer1`'s incoming cell count is fixed at `in_features=4` regardless of
scale, while `layer2`'s incoming cell count *is* `hidden_cells` -- so only
`layer2`'s evidence inherits the width-dependent growth. `uncertainty`
falls correspondingly, since `base_uncertainty_sq = 1/(precision.sum() +
eps)` and `precision` is dominated by the same growing sum. Net effect: a
`precision` model becomes more "confident" (near-zero `uncertainty`) purely
as a function of width, with no relation to actual predictive difficulty --
this is a **scaling failure of the current formula**, recorded here and
deliberately **not fixed** (CLAUDE.md Sec 2 -- changing the aggregation
math is a user research decision). One positive counterpoint: on
`u2_heteroscedastic_interaction`, the correlation between `layer2`'s final
`uncertainty` and the dataset's *known* noise level stays stable
(+0.37 to +0.46) across all five scales -- the raw magnitude of
`uncertainty` is scale-unstable, but its *relative ordering* across
examples (which is what a correlation measures) is not visibly degraded by
this pathology, at least at these scales.

**Update:** 004B and the cell-count-matched comparison were run by the user
immediately after this sweep (see "Experiment 004B results" and
"Cell-count-matched results" below). `reliability` (optional secondary per
the spec) -- still not run at any scale.

**Methodology note:** wall-clock timing from this sweep is contaminated --
004B ran concurrently on the same machine and contended for the same
GPU/MPS device, inflating both runs' `train_wall_clock_seconds` (observed
~1.5-2.9x versus an isolated calibration measurement at S2/S4). Prediction
metrics, parameter counts, and internal-state statistics are unaffected by
GPU contention; only the timing columns in `results/raw/` for this sweep
should be treated as unreliable.

## 2026-09-01 — Experiment 004B results: a real sample-efficiency edge on 2 of 3 datasets, weaker and flatter on the third

**Context:** Data scaling at one fixed model size (`scale=S3`, ~40,000
parameters -- `precision`: 40,172/39,746 params for r2+u2/c2 respectively;
parameter-matched `mlp`: 40,189/39,587, within 0.04-0.40%), sweeping
training-set size D0-D5 (1,000 to 300,000 examples), 5 seeds, otherwise
identical frozen hyperparameters to 004A (6,000 steps, `lr=1e-2`, AdamW).
Run by the user (`experiments/004_cellv0_scaling/run_004b.py`), concurrently
with the 004A sweep above. Full data:
`results/raw/004b_data_scaling_*.json`, `results/processed/004b_summary.json`.

```text
                              D0 (1K)   D1 (3K)   D2 (10K)  D3 (30K)  D4 (100K) D5 (300K)
r2_interaction    (R2, delta) +0.0229   +0.0177   +0.0046   +0.0030   +0.0040   +0.0012
c2_interaction   (acc, delta) +0.0050   +0.0088   +0.0107   +0.0048   +0.0035   +0.0081
u2_heterosced.    (R2, delta) +0.0589   +0.0426   +0.0090   +0.0079   +0.0095   +0.0079
```

(delta = precision_mean - mlp_mean, 5 seeds; per-seed std for every cell is
in `results/processed/004b_summary.json` -- typical std is 0.002-0.015,
smaller than the D0/D1 deltas on `r2_interaction`/`u2_heteroscedastic_interaction`.)

**Finding -- `precision` is more sample-efficient than `mlp` on
`r2_interaction` and (especially) `u2_heteroscedastic_interaction`, in
exactly the shape the experiment spec's own illustrative example
predicted.** Both datasets show their *largest* `precision` advantage at
the *smallest* training-set size, shrinking (though not vanishing) as data
grows: `u2` goes from a +5.89pt R2 edge at 1,000 examples to a stable
~+0.8-1.0pt plateau from 10,000 examples onward; `r2` goes from +2.29pt at
1,000 examples down to +0.12pt at 300,000. This is a genuine
"BeliefCells learn the structure with less data" result, independent of
whether the two architectures eventually tie at high data -- the kind of
finding the spec called out as potentially "more interesting than final
accuracy."

**`c2_interaction` doesn't show the same shape.** Its delta stays small
(+0.35pt to +1.07pt) and non-monotonic across every data size, with no
pronounced low-data spike -- `precision` is ahead of `mlp` at all six data
sizes on this dataset, but not in a way that reads as "wins big when data
is scarce." Whether this is a real difference between the regression tasks
(r2/u2) and the classification task (c2), or just higher variance in an
accuracy metric at these sample sizes, is not yet distinguished.

**Consistency check:** D4 (n_train=100,000) uses the exact same
config as Experiment 004A's S3 point (same scale, same data size, same
seeds 0-4) and reproduces it exactly -- e.g. `r2_interaction`
`mlp=0.9678, precision=0.9718` in both. Confirms the harness's seeding is
deterministic and the two sweeps agree where they overlap.

**Not yet done:** only one model size (~40K params) tested; whether the
sample-efficiency edge on `r2`/`u2` holds, grows, or shrinks at other
scales (a joint model-size x data-size grid) is untested. `reliability` not
run.

## 2026-09-01 — Cell-count-matched results: a weaker, more mixed advantage than parameter-matched, and a surprising cross-comparison

**Context:** The second fairness regime (docs/experiment_protocol.md
Experiment 004, "Parameter-matched vs. cell-count-matched"): `mlp`
(`hidden_dim=N`) vs. `precision` (`hidden_cells=N`) at the *same* `N`,
rather than the same parameter count. Run by the user
(`run_004_cell_count_matched.py`) at N0=15/N1=68/N2=271 -- the same widths
Experiment 004A's own parameter-matching search landed on for S0/S2/S4 --
100,000 training examples, 5 seeds, otherwise identical frozen
hyperparameters. Because every connection in a `BeliefLayer` carries both a
content weight `w` and a relevance parameter (`integration.py`), matching
`N` does *not* match parameters: `precision` has **82-99% more parameters**
than `mlp` at the same `N` (the gap widens toward ~2x as `N` grows, since
`BeliefLayer`'s `hidden_cells^2` term dominates). Full data:
`results/raw/004_cell_count_matched_*.json`,
`results/processed/004_cell_count_matched_summary.json`.

```text
                    N0 (15 units, +82-86% params)  N1 (68 units, +95-96%)  N2 (271 units, +99%)
r2_interaction                     -0.0017                +0.0012                +0.0003
c2_interaction                     -0.0043                -0.0012                +0.0026
u2_heterosced.                     +0.0003                +0.0017                +0.0155
```

(delta = precision_mean - mlp_mean, 5 seeds.)

**Finding -- despite an 82-99% parameter handicap in `mlp`'s favor,
`precision` mostly does not win by much, and loses outright at the
smallest size on two of three datasets.** At N0, `precision` is *behind*
`mlp` on `r2_interaction` (-0.17pt R2) and `c2_interaction` (-0.43pt
accuracy) even though it has 82-86% more parameters to work with there. Only
at N2 (271 units, ~150K vs ~75K params) does `precision` pull clearly
ahead, and only on `c2_interaction` (+0.26pt) and `u2_heteroscedastic_interaction`
(+1.55pt) -- `r2_interaction` stays essentially flat (+0.03pt) even with
`precision` given nearly double the parameters. Taken at face value, this
is a weak result for "one `BeliefCell` is intrinsically richer than one
ordinary neuron": if it were unambiguously true, the *free* extra
parameters here should have produced a clearer, more consistent edge than
this.

**A surprising cross-comparison with 004A's parameter-matched result.** At
the closest matched point (N2 = 271 units ~ Experiment 004A's S4, ~150K
`precision` params), parameter-matched (004A) shows a *larger* `precision`
advantage than cell-count-matched does here, on 2 of 3 datasets:

```text
                    004A parameter-matched, S4    cell-count-matched, N2
r2_interaction              +0.0070                       +0.0003
c2_interaction               +0.0141                       +0.0026
u2_heterosced.               +0.0181                       +0.0155
```

This is the opposite of the naive expectation -- cell-count-matched hands
`precision` extra parameters *for free* on top of matched width, so if
"richer cell, costs more parameters" were the whole explanation for 004A's
S4 result, cell-count-matched should show an equal or *larger* gap, not a
smaller one. Instead, forcing `mlp` to spend an *equal total budget* as
extra width (004A's regime) produced a bigger `precision` advantage than
giving `precision` bonus parameters at matched width (this regime). This
doesn't fit any of the three outcomes the experiment spec sketched in
advance cleanly, and is flagged here as an open observation, not a
conclusion -- five seeds at three widths is not enough to rule out noise,
and no significance test has been run on this specific comparison. A
plausible non-mechanistic explanation: `mlp`'s *width* itself (not just its
parameter count) may matter for how well it fits this data at a fixed
6,000-step budget, independent of the architecture question -- 004A's wider
matched `mlp` (e.g. ~384 hidden units at S4 for `c2_interaction`, vs. this
section's `mlp` capped at 271) may fit better for reasons unrelated to
`precision` vs. `mlp` at all. Worth isolating before drawing a real
conclusion from this cross-comparison.

**Not yet done:** more than 3 widths; seed-level significance testing on
either fairness regime or the cross-comparison; `reliability`.

## 2026-09-01 — Experiment 004E/004F: the duplication test confirms the scaling failure formally, and the fix works

**Context:** Before touching topology/recurrence/compartments, the user
asked to first resolve whether 004A's scaling advantage is real. 004E is a
training-free diagnostic: feed a `BeliefLayer` N identical copies of the
exact same belief content (same `mu`, `evidence`, `uncertainty`) through N
"equivalent connections" (every incoming connection given the identical
content weight and relevance gate, via
`experiments/004_cellv0_scaling/duplication_test.py`), for N in
`{1, 2, 4, 8, 16}`. Since the N copies carry zero new information beyond
N=1, a principled aggregation rule should leave `evidence`/`uncertainty`
approximately unchanged as N grows. 004F is the user-specified fix:
`"normalized_precision"` (Method D,
`src/models/architecture_v0/integration.py::_normalized_precision_fusion`),
kept alongside `"precision"`, not replacing it. Both the closed-form
prediction below and `tests/test_duplication_invariance.py` (11 tests, all
passing) verify these findings; the human-readable CLI report
(`duplication_test.py`) has not been run by the agent -- see the command at
the end of this entry.

**Finding -- `"precision"` fails the duplication test exactly as
predicted, and the failure is provably a property of the formula, not an
artifact of any particular trained weights.** With canonical "equivalent
connection" values (`mu=0.5`, `evidence=1`, `uncertainty=1`,
`content_weight=1`, `relevance_logit=0` so `g=0.5`), Method C's closed-form
behavior as a function of N:

```text
              N=1     N=16    ratio
evidence      0.500   8.000   16.00x   (== N, exactly)
uncertainty   1.414   0.354   0.25x    (== 1/sqrt(N), exactly)
mu            0.462   0.462   1.00x    (content estimate correctly unaffected)
```

`evidence = sum(g*e_j)` and `uncertainty ~ 1/sqrt(sum(precision))` are both
unnormalized sums over incoming cells, so duplicating the same content N
times inflates "how much evidence" and deflates "how uncertain" purely as a
function of width -- exactly what Experiment 004D's evidence-explosion
finding predicted, now demonstrated as a formal property independent of
training. `mu` (the content estimate) is correctly invariant in both
formulas -- duplication only breaks evidence/uncertainty, not the belief
network's actual prediction machinery.

**004F's fix works exactly as intended.** `"normalized_precision"` divides
both `evidence` and the base-uncertainty term by `G = sum(g)`, the total
incoming relevance, while leaving the content-weighting (`alpha`) and
disagreement term byte-for-byte identical to Method C -- cells still
compete for influence over `mu` exactly as before. Closed-form (same
canonical values): `evidence = 1.000` and `uncertainty = 1.000` at
*every* N from 1 to 16 (the `N` in numerator and denominator cancel
exactly for identical duplicates). `tests/test_duplication_invariance.py`
confirms both the failure and the fix numerically (within 2-5% tolerance,
accounting for `eps`), and additionally checks every N in `{1,2,4,8,16}`
individually for `normalized_precision` (not just the N=1-vs-16 endpoints).

**Not yet done:** 004G (compact 3-way scale rerun -- does
`normalized_precision` also perform comparably to `precision` on real
data, and does its scale-stability advantage hold under training, not just
this static test?) and 004H (fair MLP optimization check) are written
(`run_004g.py`, `run_004h.py`) but not run -- commands below. The
duplication test itself was only run through canonical, hand-chosen
"equivalent connection" weights, not weights drawn from the network's
actual `kaiming_uniform_` initialization distribution -- the qualitative
N-dependence (linear evidence growth, 1/sqrt(N) uncertainty shrinkage for
Method C; invariance for Method D) is a property of the formula and doesn't
depend on the specific weight value chosen, but this hasn't been swept
across a distribution of realistic weights to confirm the *quantitative*
match holds beyond the one canonical case.

**Commands to run (not yet run by the agent):**

```bash
python experiments/004_cellv0_scaling/duplication_test.py
python experiments/004_cellv0_scaling/run_004g.py
python experiments/004_cellv0_scaling/run_004h.py
```

## 2026-09-01 — Experiment 004I: "CellV0.1" (Scale-Stable Precision), and a correction to 004F's own reasoning

**Context:** User-specified refinement of Method D (`"normalized_precision"`,
004F): instead of normalizing `evidence`/base-uncertainty by raw total
relevance `G = sum(g)`, normalize by an **effective source count** -- a
participation-ratio (Kish effective-sample-size) statistic over the
relevance gates, `N_eff = (sum(g))^2 / (sum(g^2) + eps)`. Implemented as a
5th `BeliefLayer` aggregation method, `"scale_stable_precision"`
(`src/models/architecture_v0/integration.py::_scale_stable_precision_fusion`),
kept alongside `"precision"` and `"normalized_precision"`, not replacing
either. `run_scale` (`experiments/004_cellv0_scaling/scaling_harness.py`,
used by both `run_004a.py` and `run_004b.py`) now takes an `aggregation`
parameter (default `"precision"`, unchanged behavior) so the existing
004A/004B grids can be rerun against any of the five methods without new
code -- commands at the end of this entry. Not run by the agent (per the
user's "don't run them" instruction earlier this session).

**A mistake caught by writing the tests first, corrected before it reached
any real numbers.** The first draft of this method's docstring claimed
"`N_eff` reduces to exactly `G` for N identical duplicates" -- this is
wrong. The correct identity is `N_eff = N` (the duplicate *count*,
independent of the shared relevance value `g0`), while Method D's
`G = N*g0`. These coincide only if `g0 = 1`, which `sigmoid` never reaches.
`tests/test_duplication_invariance.py`'s cross-comparison test
(`test_normalized_and_scale_stable_settle_at_a_fixed_ratio_for_equal_relevance`)
caught this immediately (asserted numeric equality, got `0.5` vs `1.0` at
the canonical `g0=0.5`) before any research-log claim was written from the
wrong premise. Corrected finding: for N *equal-relevance* duplicates, both
Method D and Method E are each individually duplicate-invariant (neither's
`evidence`/`uncertainty` changes with `N`) -- but they settle at different
absolute levels (`e0` for Method D, `g0*e0` for Method E), not different
`N`-dependence. Both formulas' docstrings and the test suite now state this
correctly (23 tests across `test_duplication_invariance.py` and the new
`tests/test_scale_stable_precision.py`, all passing).

**Where Method E actually earns its keep -- unequal relevance, not the
duplication test.** `tests/test_scale_stable_precision.py` demonstrates the
real distinguishing behavior directly: with one dominant, highly-relevant
connection (`g=0.9`) and a growing number of genuinely weak ones
(`g=0.005`, chosen so even 50 of them sum to less than the dominant
source), Method E's evidence stays within 2x of the single-source baseline
out to 50 junk connections, while Method D's evidence is **provably
insensitive to junk count entirely** when incoming evidence is uniform
across cells -- the raw-sum normalizer cancels identically against the
numerator regardless of how relevance is distributed, so Method D cannot
tell "one relevant source" from "one relevant source plus fifty weak ones."
An earlier attempt at this same test used `g_junk=0.05` (not 0.005) and
failed at 50 junk connections (ratio 0.31, outside the intended 2x bound)
-- 50 connections at `g=0.05` sum to 2.5, which is *not* negligible next to
the dominant source's `g=0.9`. Left in as a cautionary note in the test
file: "weak individually" is not the same as "negligible in aggregate,"
exactly the distinction `N_eff` is designed to get right that a naive
"ignore small g" heuristic would not.

**Not yet done:** actually running 004A/004B (or 004G) with
`"scale_stable_precision"` to see whether it reproduces 004A's Result A
while additionally fixing 004D's evidence-explosion pathology (the
motivating question) -- commands below. `reliability`/`support_conflict`
remain untested at any scale.

**Commands to run (not yet run by the agent):**

```bash
python experiments/004_cellv0_scaling/run_004a.py --aggregation scale_stable_precision
python experiments/004_cellv0_scaling/run_004b.py --aggregation scale_stable_precision
```

## 2026-09-01 — Architecture V1 proposed: a self-organizing, input-dependent belief graph (not finalized)

**Context:** Discussion of what "graph/cluster organization" (§3) and
"fast dynamic association" (§4) in `docs/architecture_v0.md` — both marked
`[OPEN]` since that document's first draft — should actually look like,
beyond a fixed cluster topology.

**Proposal (not a decision):** Replace the fixed layer/cluster structure
entirely with a single persistent pool of `BeliefCell`s whose local
groupings and long-range communication are recomputed every refinement
step from the cells' current states, rather than fixed at design time.
Local "regions" are emergent dense areas of an association graph, not a
`num_clusters` hyperparameter; a small, input-dependent subset of cells
dynamically mediates communication between regions based on salience
(evidence/uncertainty/disagreement), not a fixed set of "global" units.
Also identifies a gap in the current primitive: `(mu, e, u)` alone can't
support meaningful dynamic grouping (similar `mu` doesn't imply related
content), motivating a proposed fourth cell field `k` (association/key
vector) — `CellV1 = (mu, e, u, k)`.

**Why:** Everything tested in Experiments 002–004 uses the same fixed
wiring for every input. This direction makes the wiring itself a function
of the input and the evolving cell states — closer to the "persistent
computational substrate" framing in `docs/research_thesis.md` §8 than the
fixed-graph approach `docs/architecture_v0.md` §3 deliberately started
with.

**Status:** Written up in full as `docs/architecture_v1.md`, marked
**PROPOSED, NOT FINALIZED**. Eleven open questions remain (dimensionality
of `k`, the association/similarity function, neighbor-selection rule,
global-routing salience criterion and mechanism, `N`/`T`, whether `k`
itself evolves, the decode step, initialization, and how to stage this
relative to "isolation of variables" given it's a larger jump than any
single CellV0 change so far). Per `CLAUDE.md` §2, no dynamic-graph routing,
association, or `BeliefCell` state changes are implemented from this —
awaiting further specifics from the user.

**Follow-up:** User will provide more detail on the open questions in
`docs/architecture_v1.md` §6 before any of this moves toward
implementation.

## 2026-09-02 — CellV1 fully specified and implemented: Dynamic Belief Graph

**Context:** Follow-up to 2026-09-01's Architecture V1 proposal. User did a
literature search (DGCNN, Routing Transformer, Slot Attention, capsule
routing, Growing Neural Gas, RIMs, Global Workspace models -- full list and
citations in `docs/architecture_v1.md` §0) and returned with a fully
specified CellV1 synthesis, then explicitly asked for it to be
implemented.

**Decision / finding:** Implemented in `src/models/architecture_v1/`
exactly as specified: `BeliefCellV1 = (mu, e, u, z)` (`z` renamed from the
prior turn's `k`); `LocalAssociation` (learned semantic-distance metric +
`sparsemax`-with-null-option + mutual `sqrt(a_ij * a_ji)`); `GlobalRouting`
(directed query/key compatibility gated by learned `need`/`offer`
functions and penalized by existing local edges); belief fusion at every
stage (local, global, and the final self/local/global fuse) reuses
Method E / "CellV0.1"'s exact formula
(`src.models.architecture_v0.integration._scale_stable_precision_fusion`),
reproduced in a new `fusion.py::precision_fusion` rather than imported
because V1 additionally needs the returned content-weights `alpha` for the
semantic-address update; the semantic address itself updates via a small
shared MLP and stays L2-normalized. One shared step module is applied `T`
times (default 6), not `T` independently-parameterized layers.

Two pieces were never specified by the user because they're glue code any
implementation needs regardless of the belief mechanism -- how raw
features become the initial 128-cell population, and how a
variable-topology population reads out to a fixed-size prediction. Chosen
defaults (documented as implementation choices, not specified formulas, in
`docs/architecture_v1.md` §6): a learned linear projection to
`(mu, z)` with `e = u = 1` for initialization (mirroring CellV0's
`from_observed_features`), and precision-weighted pooling of `(mu, z)`
across all cells into one linear head for decoding. `sparsemax` (the
alpha=2 member of the entmax family the user referenced) was chosen over
general alpha-entmax for a dependency-free closed form.

**A real bug caught before it reached any experiment.** `sqrt(a_ij *
a_ji)` (the mutual-local-association step) has an infinite gradient
exactly at `a_ij * a_ji == 0` -- and `sparsemax` produces exact zeros *by
design*, so on the very first backward pass every parameter in the model
got a `nan`/`inf` gradient (confirmed via a smoke test before the test
suite was written: `pred, cells = model.forward_with_cells(x);
loss.backward()` produced `nan` grads on all 28 named parameters). Fixed
with a shifted, gradient-safe sqrt (`routing.py::_safe_sqrt`, `sqrt(x +
eps) - sqrt(eps)`) that keeps exact-zero *values* (true sparsity is
preserved, unlike plain `sqrt(x + eps)`, which leaves a small nonzero
floor on every "zero" entry) while keeping the gradient finite everywhere.
Verified post-fix: full forward+backward on the default-scale
configuration (`n_cells=128`, `association_dim=8`, `num_steps=6`,
`batch=32`) produces finite gradients on every parameter, and 20 AdamW
steps monotonically decrease loss with no divergence. Separately verified
`LocalAssociation` produces a genuinely sparse (86.5% exact zeros at
`n=20`, random init), symmetric, zero-diagonal graph, and `GlobalRouting`
produces a sparse, zero-diagonal, *directed* (not symmetric) graph -- both
match the specified formulas' intended qualitative behavior.

**Why:** The user's spec resolved essentially every open question
`docs/architecture_v1.md` originally listed with concrete math grounded in
a literature search, and explicitly asked for implementation -- this is
squarely a user-specified research decision being implemented, not an
agent-invented one (`CLAUDE.md` §2's carve-out for CellV1, added to
`CLAUDE.md` §2 itself alongside this entry).

**Status:** Implemented and unit-tested (`tests/test_belief_cell_v1.py`,
`tests/test_sparsemax.py`, `tests/test_routing_v1.py`,
`tests/test_fusion_v1.py`, `tests/test_dynamics_v1.py`,
`tests/test_model_v1.py` -- 37 tests, all passing; full suite 173/173).
Tests check shape/invariant/gradient-finiteness properties, not
performance -- **no experiment has been run.** `docs/architecture_v1.md`
§9 flags an open staging question (isolating the local-only hypothesis
from the global-routing hypothesis) that should be decided before the
first CellV1 experiment is scoped, since this is a larger simultaneous
change than any single CellV0 iteration so far.

**Follow-up:** No experiment number/config exists yet for CellV1 --
next step is scoping one (docs/experiment_protocol.md doesn't yet have a
CellV1 entry) and deciding the §9 staging question first.

## 2026-09-02 — v1_001_dynamic_groups scoped and implemented: one experiment, four architectures, four tasks

**Context:** User specified the first real CellV1 experiment directly,
answering §9's staging question by putting `cellv1_local` (dynamic local
graph, no global routing) and `cellv1_full` (local + global) in the same
run alongside `mlp` and `cellv0.1`, rather than as separate follow-ups.
Also flagged an architectural concern before spending compute: the dense
`PopulationEncoder` lets every cell see the whole input immediately, which
can bypass the local-then-global organization CellV1 is meant to study.

**Decision / finding:** Implemented as `experiments/v1_001_dynamic_groups/`.
New pieces:

- `src/data/synthetic/dynamic_groups.py` -- a synthetic task designed
  specifically around the hypothesis: 24 objects per example, each
  `(k1, k2, v)`, clustered around `K in {2..5}` hidden centers never
  labeled to the model; target requires both a per-group local summary
  (`h_k = tanh(sum v_j)`) and a cross-group global term
  (`h_left * h_right`, the groups with min/max center x-coordinate) --
  unsolvable by independent per-object processing or fixed compartments,
  since the correct grouping changes every example.
- `src/models/architecture_v1/object_encoder.py::ObjectSeededEncoder` --
  the structured input adapter the user asked for: each of the first 24
  cells is seeded one-to-one from one object (`mu = v`, `z = F_seed(k1,
  k2)`), the rest start neutral (low evidence, high uncertainty, a
  learned shared "empty" `z`) and are available to be recruited during
  refinement, with no special recruitment machinery. `PopulationEncoder`
  (the dense adapter) is untouched and stays the default for R2/C2/U2.
- `DynamicBeliefGraphStep`/`SemanticUpdateFunction`
  (`src/models/architecture_v1/dynamics.py`,
  `shared_functions.py`) gained `use_global_routing: bool`. `False`
  constructs no global-routing/need/offer modules at all -- a real
  ablation (fewer parameters), not a full model with its global output
  suppressed. `DynamicBeliefGraphStep`/`Core`/`DynamicBeliefGraph` gained
  `forward_with_graphs`, returning the per-step `(a_local, a_global)`
  association matrices for the graph-analysis metrics and the
  graph-evolution artifact (`harness.py::_save_graph_evolution`) the user
  specifically asked to save (up to 100 held-out examples, every step) for
  later qualitative inspection.
- `experiments/v1_001_dynamic_groups/harness.py` -- parameter-matches
  `mlp`/`cellv0.1` to `cellv1_full`'s actual parameter count (`cellv1_local`
  reported unmatched, honestly smaller); trains all four with identical
  optimizer/steps/batch size; computes the graph-analysis metrics the user
  asked for (local sparsity, local-graph change over `T`, AUROC of
  local-association strength vs. true group membership, fraction of
  global edges crossing true groups) and nothing beyond that.

**Naming collision, resolved without asking:** the user called this
"Experiment 005," but `005` already means `experiments/005_recurrence/`
in `docs/experiment_protocol.md`'s numbered sequence -- an unrelated,
CellV0-line hypothesis (H2). Filed as `v1_001_dynamic_groups` instead,
under a new "CellV1 experiment track" section in
`docs/experiment_protocol.md` (its own `v1_NNN` numbering, independent of
001-013) and `CLAUDE.md` §2's CellV1 exception -- flagged clearly to the
user rather than silently overloading `005`'s meaning.

**Why:** Every mechanism-level choice here was the user's -- the four-arm
comparison, the dataset's local/global target structure, the object-seeded
initialization, exactly which graph metrics to record. The ablation-flag
and graph-introspection plumbing needed to run that comparison are
implementation of the user's specified experiment, not new architecture
math (`CLAUDE.md` §2's CellV1 exception).

**Status:** Implemented and smoke-tested at tiny scale (`n_cells=30`,
`num_steps=2`, a few training steps) -- runs end to end, trains all four
architectures, writes `RunRecord`s, and produces a loadable graph-evolution
`.npz` with sane shapes/values (e.g. `local_group_agreement_auroc` above
0.5 even at this untrained toy scale, since the seeded `z`'s already carry
the raw semantic coordinates). Full test suite 192/192 (19 new tests:
`tests/test_dynamic_groups_data.py`, `tests/test_object_encoder.py`,
`tests/test_local_global_ablation.py`). **Not run at the real
`n_cells=128`/`num_steps=6` scale** -- a CPU timing check put `cellv1_full`
training at roughly 1s/step there, so the default single-seed, four-task
run is a multi-hour CPU commitment (documented in the experiment README);
deferred pending the user's go-ahead on scale/seeds rather than launched
unilaterally.

**Follow-up:** Decide run scale (steps/n_train/seeds) and launch
`run_v1_001.py`, or reduce for a faster first low-fidelity pass. Per the
user's spec, the result should determine what to work on next (association/
`z`, global communication, cell update/readout, or a sparse
implementation) -- not trigger an immediate CellV1 rewrite regardless of
outcome.

## 2026-09-02 (later same day) — CellV1.1: LSH-based sparse routing, dense kept as reference

**Context:** Before spending compute on `v1_001`, user identified that
dense CellV1's `O(n_cells^2)` local/global association is a scaling dead
end for an architecture whose whole point is eventually supporting large
cell populations -- "even if it wins at N=30 or N=128, we've built
ourselves into a scaling dead end." Specified CellV1.1: keep the cell
state and every piece of math exactly as-is, replace only *how a cell
finds candidates worth scoring*, grounded in Reformer (LSH attention,
`O(n log n)`) and Routing Transformer (online-clustering attention,
`O(n^1.5)`) as precedent -- explicitly preferring the Reformer end of that
spectrum over accepting `O(n^1.5)`.

**Decision / finding:** Implemented in
`src/models/architecture_v1/{lsh,sparse_routing,sparse_dynamics,sparse_model}.py`.
Fixed (not learned) random-hyperplane hashes turn `z` (local) or `q`/`k`
(global) into bucket ids; cells are sorted by bucket id once per hash
round (`O(n_cells log n_cells)`), and `torch.searchsorted` finds where
each query's own bucket id would insert into that sorted array -- this
works for the *asymmetric* global case (`q != k`) as well as the
symmetric local case, unlike Reformer's original chunk-position trick,
which assumes query=key. A small window of the sorted order becomes the
candidate pool; the *same* semantic-distance/query-key formulas from the
dense version then run only on that `O(pool_size)` pool
(`sparsemax`, mutual `sqrt(a_ij * a_ji)` included -- mutuality is checked
via `lsh.gather_rows`, looking inside candidate `j`'s own small candidate
row for `i`, rather than a dense lookup). No `(n_cells, n_cells)` tensor
is ever constructed -- a masked dense tensor was explicitly rejected as
not actually saving anything.

**Correctness, not just "looks reasonable":** with the candidate pool
configured to cover every cell (one hash round, `chunk_size == n_cells`),
`SparseDynamicBeliefGraphStep` and the dense `DynamicBeliefGraphStep`,
given identical weights, produce numerically identical output up to
float32 rounding (`tests/test_sparse_dense_consistency.py`; `~6e-8` max
`mu` difference, exactly `0` on evidence/uncertainty/z, confirmed both
single-step and chained over 3 steps, both ablation arms). This is the
purpose of keeping dense in the repo unmodified as "CellV1 Dense
Reference" (`docs/architecture_v1.md` §11) -- a correctness anchor, not
something to run the real experiment with.

**Measured speedup, not just asserted complexity:** CPU, `batch=16`,
`T=6`, `hidden_dim=32`. Dense roughly quadruples per doubling of
`n_cells` (matching `O(n^2)`); sparse grows far more slowly:

    n_cells=128: dense 351.0 ms/step, sparse 163.5 ms/step (2.15x)
    n_cells=256: dense 1307.7 ms/step, sparse 287.9 ms/step (4.54x)
    n_cells=512: dense 6903.3 ms/step, sparse 575.2 ms/step (12.00x)

The speedup widens with scale, not a fixed constant factor -- the
complexity argument, not just the mechanism, is empirically validated.

**Why:** Every piece of this was the user's specification (which prior
architecture to draw on, which end of the complexity spectrum to target,
that the cell/update math must not change) -- implementing the indexing
plumbing to realize that spec is not a new architectural decision
(`CLAUDE.md` §2's CellV1 exception).

**Status:** Implemented and tested (`tests/test_lsh.py`,
`tests/test_sparse_routing.py`, `tests/test_sparse_dynamics.py`,
`tests/test_sparse_model.py`, `tests/test_sparse_dense_consistency.py` --
31 tests; full suite 223/223). Hash/chunk-size/window defaults
(`num_hashes=2`, `bits=6`, `chunk_size_local=10`, `chunk_size_global=4`,
`window=0`) are reasonable, matching the user's suggested orders of
magnitude -- not tuned, no experiment run yet.

**Follow-up:** `experiments/v1_001_dynamic_groups/harness.py` still builds
the dense `DynamicBeliefGraph`. Switching it to
`SparseDynamicBeliefGraph` -- and whether/how far to push `n_cells` up
(user suggested 128 -> 256 -> 512 -> 1,024) now that it's actually
tractable -- is an open decision, not made here.

## 2026-09-02 (later still) — v1_001 rewired to CellV1.1; scale sweep added

**Context:** User confirmed CellV1.1 (sparse) is now the version worth
experimenting with, dense staying only as the correctness reference, and
asked for `v1_001_dynamic_groups` rewired to it with a 128/256/512-cell
scale sweep (`T=6` fixed), `dynamic_groups` run first.

**Decision / finding:** `harness.py` now builds `SparseDynamicBeliefGraph`
exclusively; `mlp`/`cellv0.1` are re-parameter-matched to `cellv1_full`'s
actual count at whichever `n_cells` is running (`V1_SCALES = {"V1-S0":
128, "V1-S1": 256, "V1-S2": 512}`). Also threaded `n_objects`/`k_min`/
`k_max` all the way through (`_build_splits`, the `ObjectSeededEncoder`
construction, both graph-metric functions) rather than the hardcoded
`N_OBJECTS` import the original version used -- `dynamic_groups()` already
accepted these, but the harness didn't expose them, so a higher-complexity
run later (48/96 objects, more groups, per the user's suggestion) needs no
code changes now.

Graph-analysis metrics needed a real rework, not just a model swap:
CellV1.1's routing returns gathered `(candidate_idx, weights)` pairs, and
a receiver's pool membership can differ *entirely* between steps (the LSH
hash depends on `z`, which changes every step) -- so "how much did the
local graph change" and "does this edge agree with the true group" can't
be computed positionally the way the dense `(n_cells, n_cells)` version
could. Rewritten using `lsh.lookup_value`/`gather_scalar` for
identity-aware comparison (`harness.py::_sparse_change`,
`_sparse_group_agreement_auroc`, `_sparse_global_cross_group_fraction`).
`_save_graph_evolution`'s `.npz` payload now includes `local_candidate_idx`/
`global_candidate_idx` alongside the weights -- without them the saved
sparse graphs are uninterpretable.

**Why:** Model choice, scale sweep, and metric set were all specified by
the user; the graph-metric rework is implementation necessity (CellV1.1's
own output shape), not a new research decision.

**Status:** Rewired and smoke-tested (2 tasks, 2 seeds, `n_cells=30`,
tiny steps) end to end via the real CLI -- trains, evaluates, saves a
correctly-shaped graph-evolution `.npz`. Full test suite still 223/223
(no `src/` files changed, only the experiment harness). Measured real
per-step cost at each scale (CPU, `batch_size=64`, `dynamic_groups`):
128 cells 168/127 ms (full/local), 256 cells 336/262 ms, 512 cells
861/680 ms -- used to give the user an honest wall-clock estimate
(~3 hours for the full `dynamic_groups`, 3-scale, 3-seed run at
`--steps 1500 --n-train 3000`) rather than a guess. **Not run at real
scale** -- deferred to the user's own command, per their request.

**Follow-up:** Run the real `dynamic_groups` sweep; if V1-S2 (512 cells)
shows something useful, consider 1,024 next (user: not before). The
easy/medium/hard `n_objects`/group-count variants stay unused until after
this first pass, per the user's explicit sequencing.

## 2026-09-02 (later still) — CellV1 flatlines; diagnosed to a state-collapse issue, not routing; fixed with a learned write gate (CellV1.2)

**Context:** User ran `v1_001_dynamic_groups` (`dynamic_groups`, V1-S0/S1,
2 seeds, 1500 steps) before this log entry existed to record it. Result:
`mlp` R² 0.61-0.68, `cellv0.1` R² 0.74-0.77, `cellv1_local`/`cellv1_full`
R² 0.00-0.03 -- at a matched parameter count, so not simply "V1 has fewer
parameters." User called it decisive on its own and asked for one targeted
isolation diagnostic rather than more scale sweeping: dense CellV1
(zero LSH) vs. sparse, same task/scale/seed, recording *train* R² as well
as test R², to fork between (a) LSH routing destroying the signal, (b)
the architecture/training setup itself, (c) generalization, or (d)
optimization/signal-propagation.

**Decision / finding:** `experiments/v1_001_dynamic_groups/diagnose_v1_learning.py`
(new, standalone -- not wired into the RunRecord machinery, a quick
diagnostic not a tracked experiment). `dense` (CellV1 Dense Reference,
`docs/architecture_v1.md` §7) on `dynamic_groups`/V1-S0/seed 0/400 steps:
train R² = 0.0164, test R² = 0.0079 -- essentially identical failure to
sparse, and failing on the *training set itself*. This ruled out both
LSH-specific explanations (dense uses none) and generalization/overfitting
(train fails too) in one run, pointing straight at the core recurrent
update.

Caught and fixed a real bug in the diagnostic script itself while getting
this result: running evaluation unbatched (3000 training examples in one
forward pass) crashed with `RuntimeError: Invalid buffer size: 46.88 GiB`
on the `full_pool_sparse` variant -- `sparse_routing.py`'s `gather_rows`
intermediate is `O(batch * n_cells * pool^2)`, and with the diagnostic's
"pool covers everyone" config (`pool == n_cells`) that's effectively
`O(batch * n_cells^3)`, fine at training's `batch_size=64`, not at 3000
examples in one shot. Fixed by chunking the eval forward pass
(`_batched_predict`); not a bug in the sparse routing itself, a bug in
how the diagnostic evaluated it.

**Diagnosis (user, before rerunning anything further):** CellV1's
recurrent update fully overwrites every cell's `(mu, e, u)` with the
fused self+local+global proposal every step, with no mechanism for a
cell to partially resist that -- six rounds of message-passing +
consensus fusion under a *shared* update rule is the recurrent/message-
passing analogue of GNN oversmoothing (state collapse/over-mixing). `z`
already had partial inertia (a small fixed-`eta` residual step);
`mu`/`e`/`u` had none at all.

**Fix -- CellV1.2, a learned per-channel write gate**
(`src/models/architecture_v1/shared_functions.py::WriteGateFunction`,
wired into both `dynamics.py` and `sparse_dynamics.py` identically, since
this changes the shared recurrent-update math, not routing): the
self+local+global fusion now produces a *proposal*
(`mu_hat`/`e_hat`/`u_hat`), and `beta = sigmoid(F_gate(mu, e, u, mu_local,
e_local, u_local[, mu_global, e_global, u_global]))` (one gate per
channel, `beta_mu`/`beta_e`/`beta_u`/`beta_z`, shared trunk) decides how
much is accepted: `next = (1 - beta) * old + beta * proposal`. `z`'s fixed
`eta` scalar is retired -- `beta_z` takes over its exact residual-update
role, now learned and per-cell/per-example instead of a constant. Gate
bias initialized to `-2.0` (`sigmoid(-2.0) ~= 0.12`, the user's suggested
0.1-0.2 range) -- cells mostly preserve themselves early in training.

A shape bug caught immediately by the test suite: `beta_z` is `(batch,
n_cells)` but `z_delta` is `(batch, n_cells, association_dim)` --
`beta_z * z_delta` doesn't broadcast correctly without
`beta_z.unsqueeze(-1)` first (28 test failures until fixed, all in the
`RuntimeError: size mismatch at dimension 2` pattern). Fixed; full suite
back to 223/223, including `tests/test_sparse_dense_consistency.py`
(`write_gate`'s weights are now also copied between the dense/sparse pair
that test builds, confirming the gated update stays an exact dense/sparse
match too, not just the routing).

**Why:** The diagnostic protocol (which variant, what to record, the
train-vs-test fork) and the architectural fix (gated write, per-channel,
bias-initialized small) were both the user's specification -- implementing
them is not a new architecture decision.

**Status:** Implemented, tested (223/223, including a new dense/sparse
consistency check covering the gate), and confirmed fixed. Confirming run
-- `normal_sparse` (real LSH config, not the full-pool diagnostic config),
same task/scale/seed, 500 steps: train R² `0.0164` -> **`0.7193`**, test
R² `0.0079` -> `0.7299`. Within the user's "0.5-0.8" bar for "the
bottleneck is found," and test tracks train closely (the fix didn't open
a new overfitting gap) -- roughly matching `cellv0.1`'s original-run level
(`0.74-0.77`). The gated write is confirmed as the fix.

**Follow-up:** Resume the `V1-S1`/`V1-S2` (256/512-cell) scale sweep on
`dynamic_groups` with the gate in place -- the question the original
`run_v1_001.py` command was trying to answer before this flatline was
found.

## 2026-09-03 — CellV1.3: Self-Organizing Refinement Field implemented, after a real debugging chain

**Context:** User identified CellV1.1's LSH candidate search as an
engineering choice imposed on the architecture (fixed hash/chunk-size
parameters deciding neighbor counts) rather than something learned, and
specified a replacement: local structure as density modes of a
continuous, learned kernel field (differentiable mean-shift), routing
position `r` and semantic identity `z` given separate, distinct update
mechanisms instead of both being folded into one graph-routing
abstraction. Full design, the six-bug debugging chain (wrong kernel
family for the positive-random-feature approach; a numerical stabilizer
that didn't actually cancel; a hidden `O(n*R^2)` term inside a claimed
`O(n*R)` reduction; signed-kernel division blowups even after switching
to the correct Gaussian-kernel RFF and to orthogonal features), and the
fix (self-anchoring the local density on the exactly-known `K(i,i)=1`
rather than further epsilon patching) are recorded in full in
`docs/architecture_v1.md` §13 -- not duplicated here.

**Decision / finding:** Implemented at the user's stated defaults
(`R=256`, `T=1`, self-anchored orthogonal RFF, "do not tune
evidence/uncertainty further before end-to-end evaluation"):
`src/models/architecture_v1/{random_features,field_functions,field_fusion,
field_dynamics,field_model}.py`. `BeliefCellV1`'s `(mu, e, u, z)` and the
scale-stable-precision fusion formula are unchanged throughout -- only
neighbor-finding changed, twice now (discrete LSH candidates ->
continuous kernel field).

**Status:** Implemented and unit-tested (`tests/test_random_features.py`,
`tests/test_field_functions.py`, `tests/test_field_fusion.py`,
`tests/test_field_dynamics.py`, `tests/test_field_model.py` -- 30 tests;
full suite 253/253) plus one training smoke test (15 AdamW steps, loss
decreasing monotonically, no divergence). Measured (not estimated):
`n_cells=128`, `R=256`, `T=1`, `batch=32`, CPU -- 24 ms/training-step,
versus CellV1.1 sparse's ~168 ms/step at a comparable scale (and that was
`T=6`; this is `T=1`). Confirmed `RoutingGateFunction` correctly gets
zero gradient at the default `T=1` (`r`'s movement only matters for a
future step that doesn't exist yet) and a real one at `T=2` -- a
deliberate regression test.

**Why:** Every design decision (kernel type, role separation of `r`/`z`,
which literature to draw on, the self-anchoring fix, "stop tuning e/u,
move to evaluation") was the user's; implementing and debugging the
approximation math to actually realize that spec is not a new
architectural decision (`CLAUDE.md` §2's CellV1 exception).

**Not yet done:** No experiment has been run with the field variant --
unit-tested for "matches the specified math, produces valid bounded
output," not for task performance. `evidence`/`uncertainty`'s noisier
(vs. `mu`/`r_bar`) convergence in the R-sweep diagnostic is flagged, not
resolved, per explicit instruction to defer it. Global (cross-region)
field communication is designed in conversation but not implemented --
this pass is local-field-only, one step.

**Follow-up:** Decide whether to run `v1_001_dynamic_groups`-style
end-to-end evaluation with this field variant (would need a new encoder/
harness wiring, since the field's `FieldCellState` isn't a drop-in
`BeliefCellV1`), or scale `T` up first, or investigate `evidence`/
`uncertainty`'s convergence now that the rest is stable.

## 2026-09-03 (later same day) — CellV1.3.1: global communication added to the field; a new non-local task exposes a real, seed-dependent instability at scale

**Context:** CellV1.3 shipped local-field-only, one step -- global
(cross-region) communication was designed in the original conversation
but explicitly deferred. The user asked for it, reusing the same send/
need/query-key mechanism dense/sparse CellV1 already validated (§5 Part
III of `docs/architecture_v1.md`), not a new design:
`global_field.py`'s `GlobalSendFunction`/`GlobalNeedFunction`/
`GlobalQueryKey` plus exact linear attention (`linear_global_belief_field`,
Katharopoulos et al. 2020's positive-feature-map trick -- `phi(x) =
ELU(x) + 1 > 0`, so `phi(q_i)^T phi(k_j)` *is* the compatibility kernel,
not an approximation; no random projection, none of the "does this
converge as R grows" debugging the local field needed). Wired into
`field_dynamics.py`/`field_model.py` via an additive `use_global` flag,
local-only path unchanged when off (per the module's own docstring; see
the test-coverage gap noted under Status).

Before this could be tested meaningfully, a new task was needed:
`dynamic_groups`'s existing cross-group term pairs groups by their
*spatial* center (`k1`), a property close enough to each object's own
encoded features that a wide local field could plausibly shortcut it
without any real non-local channel. `dynamic_groups_global`
(`src/data/synthetic/dynamic_groups.py`) changes the pairing to
`argmax`/`argmin` of the groups' *aggregate values* `h_k = tanh(sum
v_j)` -- content only knowable after comparing every group's sum against
every other's, which a similarity-based local field structurally cannot
shortcut. 5 new tests (`tests/test_dynamic_groups_data.py`).

**Findings, in order:**

1. *n_cells=128, `dynamic_groups` (spatial-pairing task), R=256, fixed
   1500 steps, 3 seeds* -- the field's original comparison scale.
   `field_local_global_t2` (R²=0.7872±0.0146) slightly ahead of
   `field_t2` local-only (0.7784±0.0043), `field_t1` (0.7787±0.0094),
   and `cellv0.1` (0.7662±0.0145). A real, if modest, edge for the global
   channel at this scale -- no collapse in this final recorded run.
2. *Complexity-level sweep, `dynamic_groups_global`, R=256, fixed 1500
   steps, 3 seeds x 4 levels (easy/medium/hard/very_hard =
   24/48/96/192 objects)* -- `run_complexity_scaling.py`. Confirmed the
   modest global edge holds through `hard` (e.g. `hard`:
   field_local_global_t2 0.8230±0.0099 vs. cellv0.1 0.8294±0.0084 vs.
   field_t2 0.8215±0.0092 -- all close). Earlier iterations of this sweep
   hit real single-seed collapses attributed (per
   `global_field.py::GlobalSendFunction`'s docstring, written
   contemporaneously) to an unlearned, loud global channel getting a
   full vote from step 0 -- fixed by initializing send/need's bias
   near-off (`init_bias=-2.0`, matching `WriteGateFunction`'s existing
   convention) rather than the unbiased default. `diagnose_global_collapse.py`
   was written to isolate undertraining vs. RFF-resolution vs. a deeper
   redesign as the explanation for a specific very_hard/seed=0 collapse
   (test R²=0.224 at the time, vs. local-only 0.781 and CellV0.1 0.816)
   -- no saved output from that script was found in this working tree,
   so its three-way conclusion isn't independently confirmed here; what
   *is* confirmed by the fixed-step sweep's final recorded numbers is
   that the collapse is gone at that specific point.
3. *Fixed-step protocol itself flagged as confounded* -- per
   `run_complexity_scaling_convergence.py`'s own stated rationale: a
   24-object and a 192-object problem don't converge on the same
   optimization timescale, so comparing both at one arbitrary step count
   isn't fair. Re-run under early stopping
   (`harness.py::_train_until_convergence`, validate every 100 steps,
   patience 500, cap 5000) with R raised to 512 (found to let the global
   pathway converge in roughly a third of the R=256 steps at
   n_objects=192). Under this corrected protocol, R² stays close across
   cellv0.1/field_t2/field_local_global_t2 at every level (e.g.
   `very_hard`: cellv0.1 0.8212±0.0058, field_local_global_t2
   0.8165±0.0125, both stable) **except one case: `field_t2` (local-only,
   no global) at `very_hard`/seed=0 converged to R²=0.0667** against
   seed 1/2's 0.79/0.82 -- i.e. the instability recurred even under the
   corrected protocol and higher R, and this time in the *local-only*
   variant, not the global one. This is a real, currently unresolved
   reliability question about the ORFF-kernel field at this scale/seed
   combination -- not fixed by the send/need init change (a global-only
   mechanism) and not explained by the "global amplifies it" framing
   that motivated `diagnose_global_collapse.py`.
4. As part of the convergence-protocol comparison, `mlp` was also run at
   `easy` (`run_mlp_convergence_only.py`'s protocol): converges in far
   fewer steps (80 vs. 2400+ for every belief-cell variant) but to lower
   accuracy (R²=0.6978 vs. ~0.76-0.80 for the others) -- consistent with
   every prior CellV0/CellV1 baseline comparison in this log.

**Status:** Implemented, unit-tested (`tests/test_global_field.py`, 8
tests; full suite 274/274). One gap: `field_dynamics.py`'s module
docstring claims a "`tests/test_field_local_global_ablation.py`
zero-regression check" verifying the `use_global=False` path is
byte-identical to pre-change behavior -- no file or test by that name
exists in this working tree. Flagged, not fixed (out of scope for this
logging pass); the claim should either be backed by an actual test or
removed from the docstring.

**Why:** The mechanism (reuse §5 Part III's send/need/query-key design,
exact linear attention per Katharopoulos et al.) and the new task
(argmax/argmin pairing to force genuine non-locality) were the user's
specification; implementing, wiring, and diagnosing them is not a new
architecture decision (`CLAUDE.md` §2's CellV1 exception).

**Follow-up:** The `very_hard`/local-only collapse (finding 3) is open --
worth its own isolation diagnostic (does it reproduce with a different
seed offset? is it R-dependent the way the global collapse was
hypothesized to be?) before trusting any single-seed field result at
this scale. The fixed-step `field_global` sweep's numbers (findings 1-2)
should be read as history, not as the current benchmark -- superseded by
the convergence-based protocol.

## 2026-09-03 (later still) — CellV1.4: Learned Association Field replaces the ORFF kernel with an exact, learned one; matches CellV0.1 with much less seed variance

**Context:** CellV1.3's local field spent R=256-512 random Fourier
features approximating a Gaussian kernel chosen ahead of time over a
routing coordinate `r` -- and the entry above shows that even at R=512,
the RFF approximation is not fully scale-stable (one seed still
collapsed at `very_hard`). The user's redesign, in their own words: "We
shouldn't spend hundreds of dimensions accurately approximating a
similarity function that *we chose*. The network should learn the
similarity function itself." `learned_association.py`: `phi_i =
softplus(F_assoc(mu_i, e_i, u_i, z_i))`, `K_ij = phi_i . phi_j` -- one
small shared function, dot product *is* the kernel by construction, not
an estimate of one. No routing coordinate `r`, no bandwidth/mass
functions, no mean-shift -- state is plain `BeliefCellV1 (mu, e, u, z)`,
`assoc_dim` default 32 (an order of magnitude below the field's R=512).
Global communication (CellV1.3.1, above) reused completely unmodified,
per the user's explicit instruction to keep it and feed it the
association fusion's local output instead of the field's.

**Findings:**

1. *Single-seed sanity check* (`check_association_field.py`, seed 0,
   easy/very_hard, convergence protocol) -- the user's explicit "narrow
   check... not another huge sweep" before committing to a fuller run.
   Against `field_local_global_t2` (R=512) at matched params: `easy` --
   field R²=0.8152/600 steps/11.9s vs. association R²=0.8069/200
   steps/3.24s; `very_hard` -- field R²=0.8136/2300 steps/80.7s vs.
   association R²=0.8205/900 steps/16.1s. At both levels the association
   field reached comparable-or-better accuracy in roughly a third the
   steps and a fifth the wall-clock, on this one seed.
2. *3-seed, 4-level convergence sweep*
   (`run_complexity_scaling_association.py`, `cellv0.1` vs.
   `association_local_global_t2` only -- field variants "retired from
   active development" per this script's own docstring, their numbers
   already on record above) -- confirms the pattern is not a
   single-seed artifact. R² tracks `cellv0.1` tightly at every level:
   easy 0.7896±0.0124 vs. 0.7880±0.0219, medium 0.8405±0.0071 vs.
   0.8448±0.0073, hard 0.8287±0.0092 vs. 0.8292±0.0096, very_hard
   0.8211±0.0004 vs. 0.8191±0.0052 -- differences at or below one
   architecture's own seed noise at every level, and (unlike the field
   variants, finding 3 above) **no collapse on any of the 12 (level,
   seed) combinations**, and very_hard's seed variance is the tightest
   of any model/level combination recorded in this experiment. Steps-
   to-convergence is mixed, not uniformly faster: association converges
   in fewer steps than cellv0.1 at `easy` (967 vs. 2300) but comparable-
   to-slightly-more at `hard`/`very_hard` (1267 vs. 967; 1367 vs. 1233)
   -- the "3-9x fewer steps" pattern from the single-seed check (finding
   1) does not hold up unchanged in the fuller sweep and should not be
   quoted as the headline number. What *does* hold up: per-step
   wall-clock is consistently lower than the ORFF field it replaced
   (e.g. `very_hard`: ~18ms/step for association vs. ~37ms/step for
   field_local_global_t2 at R=512), consistent with `assoc_dim=32` vs.
   `R=512`'s expected `O(n*D)` cost difference, though the ~2x wall-clock
   gap is smaller than the ~16x feature-count ratio would suggest --
   other fixed costs (global communication, write-gate, semantic update)
   are shared between the two and don't shrink.

**Status:** Implemented (`learned_association.py`,
`association_dynamics.py`, `association_model.py`), unit-tested
(`tests/test_learned_association.py`, 8 tests; full suite 274/274), and
now run through the same 3-seed/4-level convergence protocol as the
field variant it replaces. Not yet done: the adaptive per-input
continue/refine gate (`T1` vs. `T2` decided from task loss rather than
fixed `num_steps`) mentioned in the module's own docstring as
deliberately deferred ("I would not do another huge sweep") --
`AssociationRefinementCore` still takes a fixed `num_steps`, a known gap
against the user's full spec, not a bug.

**Why:** The redesign rationale (stop approximating a chosen kernel,
learn one directly), which state fields survive (`(mu,e,u,z)`, no `r`),
and which mechanisms are reused unmodified (global communication,
`precision_fusion`, `WriteGateFunction`) were the user's explicit
instructions across this session; implementing and evaluating them is
not a new architecture decision.

**Follow-up:** This is now the more reliable of the two CellV1
field-style variants (no collapse across 12 level/seed combinations, vs.
the ORFF field's one). Given the session's later pivot toward a
persistent structural substrate (`w_ij`) gating a dynamic association
kernel (see this same date's later research-direction discussion), the
existing `AssociationFunction`'s `phi_i = softplus(F_assoc(...))` is a
natural starting point for that dynamic term `a_ij(t)` -- what's missing
for that next step is the slow, structurally-sparse `w_ij` component
itself, not a new kernel.

## 2026-09-03 (later still) — CellV1.5 proposed: persistent structural substrate, not finalized

**Context:** After CellV1.4 landed cleanly (previous entry), the user
relayed a design conversation (conducted outside this session, pasted in
in full) proposing a pivot for Hypothesis B itself, not just another
CellV0.1-adjacent tweak. The diagnosis: every CellV1 variant tried so far
(§1-§15 of `docs/architecture_v1.md`) treats "dynamic functional
organization" as "rebuild the candidate graph from scratch every
refinement step" -- dense pairwise scoring, LSH pools, an RFF field, now
an exact learned kernel. Grounded in cell-assembly theory (Buzsáki 2010,
"Neural syntax"; the "synapsemble" framing), the proposal is that this
doesn't match cortical organization: a relatively persistent synaptic
substrate, with functional assemblies emerging from *which parts of it
are currently active*, not from redrawing the substrate itself.

**Decision:** Captured, not implemented. Before writing any code, I
flagged that this is a new architecture decision under `CLAUDE.md` §2's
CellV1 exception (dynamic-graph routing and fast contextual association
are only exempted from the "MUST NOT invent architecture math" rule
*because* the user specifies them explicitly, turn by turn -- the same
now applies to structural plasticity, which the MUST-NOT list names
directly) and asked two targeted questions rather than guessing:

1. What to do about a backlog of real, undocumented results from this
   same session (the CellV1.3.1/CellV1.4 work just logged above) before
   scoping a pivot -- user chose to log/commit that work first (done;
   see the two entries above and `docs/architecture_v1.md` §14-§15).
2. How new synapse candidates should be proposed, given the user's own
   constraint that this can't be an `O(n_cells^2)` scan or purely random
   assignment -- user specified: reuse `AssociationFunction`'s already-
   validated learned `phi` representation as the retrieval space, but
   search it via an approximate nearest-neighbor / maximum-inner-product
   index rather than exhaustive pairwise scoring, explicitly warning that
   LSH/ANN must be *only* the indexing algorithm over that learned space,
   never mistaken for the growth criterion itself (the way §11's LSH
   *was* the whole routing mechanism, on a fixed/unlearned hash).

**Status:** Proposal captured in `docs/architecture_v1.md` §16 as
CellV1.5, explicitly marked PROPOSED, NOT FINALIZED, with seven open
questions listed (exact utility-statistic and growth-score formulas; how
`w_ij` is parameterized as a PyTorch parameter given edges are added and
pruned at runtime -- a real engineering question, not just a research
one; the aggregation formula's exact domain once edges are sparse; edge
budget and bootstrap topology; the plasticity schedule; and the
relationship to existing local/global routing, §5/§14). No code written.
Full suite still 274/274 (unchanged -- no implementation touched).

**Why:** The core hypothesis and every mechanism decision so far (the
`w_ij * a_ij(t)` decomposition, reusing `AssociationFunction`, occasional
not per-step plasticity, ANN-as-indexing-not-criterion) came from the
user across two turns; this is the same "capture the proposal, list what
remains open" step CellV1 itself went through on 2026-09-01 before being
fully specified the next day (§17's revision history) -- not a
regression in process, the established one.

**Follow-up:** Whenever the user is ready to resolve the open questions
(most consequentially #3, `w_ij`'s parameterization, since it determines
whether this needs genuinely new PyTorch parameter-management machinery
or can reuse the existing shared-function convention via persistent
per-cell structural addresses), this becomes CellV1.5 proper the same
way CellV1.1-CellV1.4 each did: a fully specified doc section, then
implementation.

## 2026-09-03 (later still) — CellV1.5's structural-weight representation resolved: address-derived, not a stored parameter

**Context:** Immediately following the previous entry, the user resolved
the single open question flagged as having real engineering stakes: how
`w_ij` is represented, given edges are added and pruned at runtime and
this codebase has no existing precedent for a parameter tensor that
changes shape during training.

**Decision:** Every persistent cell gets a learned **structural address**
`s_i in R^{d_s}` — a fixed, input-independent per-cell identity (an
`(n_cells, d_s)` parameter table, trained by ordinary backprop, constant
across every example/batch, explicitly *not* derived from the current
belief state the way `z` is). For an existing directed edge `(i, j)`:
`q_i = W_out s_i`, `k_j = W_in s_j` (shared learned linear maps, no
bias), `w_ij = tanh(q_i . k_j / sqrt(d_s))`. Nothing about `w_ij` is
stored — only the sparse edge-index list plus non-parameter structural
metadata (utility, age) persists; `w_ij` is recomputed from `s`/`W_out`/
`W_in` for whichever edges currently exist, every forward pass.

**Why this resolves the flagged engineering problem:** A literal
per-edge `nn.Parameter` would need the parameter tensor itself to
grow/shrink as edges are added/pruned, colliding with both this
codebase's fixed-shape-parameter convention and the optimizer's
per-parameter state (Adam moments for a parameter that no longer exists;
initialization for one that didn't exist a moment ago). Address-derived
weights sidestep this entirely: growth needs no weight initialization
scheme (a new edge's `w_ij` is simply the formula evaluated on that
pair, immediately well-defined by the two cells' existing addresses),
and pruning is a plain row deletion from the edge-index list (there was
never a per-edge optimizer slot to reconcile). This is also the same
"shared function evaluated over the current structure, not a per-pair
free parameter" convention every other CellV1 mechanism already follows
(`docs/architecture_v1.md` §2's original rule against `BeliefLayer`-style
permanent per-pair parameters, now extended from routing to the
structural graph itself).

**Compute:** project every cell's address once per forward pass
(`q_all = S W_out^T`, `k_all = S W_in^T`, `O(n_cells * d_s^2)`), then
gather and dot only for the `E` existing edges (`O(E * d_s)`) — no
`(n_cells, n_cells)` tensor at any point, the same budget every prior
CellV1 sparsity mechanism (§11's LSH pools, §15's association kernel)
already holds itself to.

**Status:** Recorded in `docs/architecture_v1.md` §16 (moved from "open
questions" into "what's specified"; the revision history's previous
entry is updated to point here). Six open questions remain: the
utility-statistic formula, the growth-score formula, the aggregation
formula's exact domain once neighbors are sparse, edge budget/bootstrap
topology, the plasticity schedule, and the relationship to existing
local/global routing (§5, §14). No code written — this is still a
documentation-only resolution, not an implementation.

**Why:** The address/projection formula, the constraint that structural
addresses must not depend on current input, and the storage-only-topology
decision were all the user's explicit specification; recording them is
not a new architecture decision.

**Follow-up:** Of the six remaining open questions, none has the same
"blocks even starting to write code" character #3 did — the rest are
tunable formulas/schedules more in the spirit of CellV1's own §6
"implementation-choice defaults," except question 6 (relationship to
existing local/global routing), which is a real scope decision about
whether CellV1.5 replaces or extends §5/§14. Worth resolving that one
specifically before implementation starts, since it determines how much
of the existing dense/sparse/field machinery this variant actually reuses
versus supersedes.

## 2026-09-03 (later still) — CellV1.5 fully specified: all six open questions resolved in one turn

**Context:** Following the previous entry's resolution of `w_ij`'s
representation, the user resolved the remaining six open questions from
`docs/architecture_v1.md` §16 in a single turn, explicitly to avoid
another round of back-and-forth and let implementation proceed directly:
"I'd resolve all six now and let the coding agent implement a single
clean CellV1.5, rather than opening another round of architecture
debate."

**Decisions, in full:**

- **Utility (pruning signal):** for edge `i -> j`, full effective
  transmitted content `m_ij = w_ij * a_ij(t) * mu_i`, loss-sensitivity
  score `q_ij = |m_ij * dL/dm_ij|`, EMA'd as `U_ij <- 0.99 U_ij + 0.01
  q_ij`. Prune by low `U`, explicitly not by small `w` (a small-weight
  edge can still be structurally important under some functional
  states). Edge metadata is exactly `{endpoints, U, age}`.
- **Growth score:** `G_ij = mean_batch(phi_i . phi_j * A_i * A_j)`,
  where `A_i` is an EMA'd per-cell activity measure, `A_i <- 0.99 A_i +
  0.01 (|mu_i| * e_i / (1 + u_i))`. No new learned function ("we already
  have enough learned machinery") — candidate retrieval stays ANN/MIPS
  over the already-validated `phi`-space from §15's `AssociationFunction`.
- **Aggregation:** content `m_ij = w_ij * mu_i` and precision `p_ij =
  a_ij(t) * e_i/(u_i^2+eps)` — the `w`/`a` split mirrors CellV0.1's
  successful content/gate separation — fused over structural
  in-neighbors via the same scale-stable precision formula every CellV1
  variant already uses. Self is always included, via the established
  two-source `precision_fusion` + learned-gate pattern (not injected
  into the neighbor sum), then the existing residual write gate.
- **Edge budget/bootstrap:** one *global* synapse budget `E_max = k_bar
  * n_cells`, default `k_bar = 8`, per-cell degree otherwise
  unconstrained (a cell might end up with 2 edges or 30 — "that's what
  we want"). One candidate-growth pass before training fills the budget
  from a near-random initial topology, explicitly disposable. One safety
  constraint, no more: every cell needs at least one incoming structural
  edge.
- **Plasticity schedule:** keyed to optimizer steps, never refinement
  steps. 200-step warm-up (topology fixed, everything else trains
  normally), then every 100 steps: prune the bottom 5% of edges by `U`
  (subject to the in-degree-1 protection), grow the same number of
  highest-`G` candidates. Rewiring freezes for the last ~20% of training
  so the substrate can settle before evaluation. All four numbers
  (200/100/5%/20%) are explicitly config, not theoretical constants.
- **Scope:** CellV1.5 replaces dense CellV1's local/global routing (§5),
  the RFF field (§13), and the global field (§14) *entirely* for this
  variant — "do not run CellV1.5 alongside" any of them. One substrate;
  local-vs-global is never labeled or measured, only emergent from which
  structural edges happen to be short/long-range for a given input.
  Those sections remain in the repository, frozen, as historical
  baselines — the same convention already used for CellV1 Dense
  Reference (§7) after §11.

**Status:** `docs/architecture_v1.md` §16 rewritten top-to-bottom from
"PROPOSED, NOT FINALIZED" to "SPECIFIED," incorporating every formula
above precisely, plus a new §16.10 "implementation-choice defaults"
subsection for the handful of genuine glue-level gaps the spec didn't
need to cover (candidate retrieval reuses `lsh.py`'s existing
random-hyperplane LSH machinery, applied to `phi`-space instead of
routing vectors, rather than a new ANN dependency; `A_i`'s EMA decay
reuses `U_ij`'s `0.99/0.01`; `d_s=16` as a default, matching the scale of
`association_dim`/`global_dim` elsewhere; the plasticity-event trigger
is an explicit model method the training loop calls after
`optimizer.step()`, not folded into `forward`, since it's
architecture-specific rather than generic training infrastructure). No
code written yet — this entry and the doc update are the specification
step; implementation is next.

**Why:** Every formula, constraint, and scope decision above is the
user's; the implementation-choice defaults are glue-level (which ANN
mechanism to reuse, an unspecified EMA decay, a dimension default,
where a training-loop hook lives) in the same spirit as §6's
encoder/decoder defaults — not architecture decisions.

**Follow-up:** Implementation is the next step: a new
`src/models/architecture_v1/structural_*.py` module family (structural
addresses/projections, the sparse edge registry as plain tensors/buffers
— never `nn.Parameter` — utility/activity EMA tracking, LSH-based growth
candidate retrieval reusing `lsh.py`, the sparse-neighbor aggregation
step, and the plasticity-event method), tests mirroring this codebase's
existing shape/invariant/gradient-finiteness convention, and training-
loop wiring for the optimizer-step-keyed plasticity schedule. No
experiment run yet.

## 2026-09-03 (later still) — CellV1.5 implemented

**Context:** Following the previous entry's full specification (all six
open questions resolved, `docs/architecture_v1.md` §16 rewritten as
SPECIFIED), this entry records the implementation itself --
`src/models/architecture_v1/structural.py`,
`structural_fusion.py`, `structural_dynamics.py`,
`structural_plasticity.py`, `structural_model.py` (§16.12 has the full
module map).

**What was built, following the spec exactly:**

- `StructuralAddress`: persistent, input-independent `s_i in R^{d_s}`
  plus `W_out`/`W_in`, `w_ij = tanh(q_i . k_j / sqrt(d_s))` computed only
  for existing edges, never stored.
- `EdgeRegistry`: the sparse topology plus utility/age, as plain
  `register_buffer` tensors reassigned (not `nn.Parameter`) whenever
  edges are added/removed -- genuinely resizable at runtime, unlike
  every other buffer in this codebase. `state_dict()` round-trips only
  when the edge count matches at load time; resuming into a
  differently-sized registry isn't handled (flagged, not solved).
- `CellActivity`: `A_i`'s EMA (§16.7), detached before reduction so it
  doesn't leak the live autograd graph across calls.
- `sparse_structural_fusion`: the scale-stable precision-fusion formula
  (Method E / CellV0.1, reused by every CellV1 variant) restricted to
  each cell's structural in-neighbors, via `index_add_`-based scatter
  reduction rather than a padded dense-per-target layout. The padding
  approach was rejected specifically because CellV1.5's graph has no
  degree bound (§16.8: "a hub cell might have 30 edges") -- padding
  every target row to the population's max degree could cost `O(n_cells
  * E)` in the pathological case where most edges converge on one cell,
  which is quadratic-ish and defeats the point of a sparse substrate.
  Scatter reduction is `O(E)` regardless of degree distribution.
- `StructuralRefinementStep`/`Core`: dynamic gate (`AssociationFunction`,
  reused unmodified from §15) -> sparse structural fusion -> self+
  structural 2-source fuse (`precision_fusion`, learned gate) -> the
  existing `WriteGateFunction` -> a `z` update analogous to §15's. `w`
  is computed once per forward pass (before the `T`-step loop), not
  recomputed per step, since structural addresses don't change within a
  call -- only the functional gate and belief state do.
- `structural_plasticity.py`: growth (LSH candidate retrieval over
  `phi`-space, reusing `lsh.py` unmodified, exact `G_ij` scoring only on
  the retrieved pool), pruning (bottom-`U` fraction, in-degree-1
  protected), bootstrap, and the 200-warmup/100-interval/5%-prune/
  20%-freeze schedule, all as config with the spec's stated defaults.

**A real bug found and fixed during implementation (not just a numerical
one -- a structural gradient-flow bug):** §16.6 defines the utility
signal from `m_ij = w_ij * a_ij(t) * mu_i`, distinct from §16.5's own
`m_ij = w_ij * mu_i` (content only) -- the spec reuses the name `m_ij`
for two different quantities across two subsections. The first
implementation attempt computed the full-message quantity as a plain
side artifact (`w * a * mu`, retained via `.retain_grad()`) without it
actually appearing in the computation that produces the fused output.
Since nothing in the forward graph depended on it, its `.grad` after
`backward()` was structurally guaranteed to be `None` -- not a
numerical-precision issue, a "this tensor was never part of the
computation" issue, caught by
`tests/test_structural_fusion.py::test_edge_message_full_gradient_is_real_and_matches_algebra`
before it could silently ship as "utility always stays zero." Fixed by
re-associating the precision-fusion arithmetic: since `p_ij * m_ij =
(a_ij * precision_no_a_ij) * m_ij = (a_ij * m_ij) * precision_no_a_ij =
edge_message_full * precision_no_a_ij`, routing the fusion's `pm_sum`
through `edge_message_full * precision_no_a` (provably the same value,
verified in the same test file's dense-consistency check) instead of
`p * m` directly gives `edge_message_full` a genuine place in the graph
-- its retained gradient is now the real `dL/d(edge_message_full)`.

**Status:** Implemented and unit-tested -- 45 new tests
(`test_structural.py` 10, `test_structural_fusion.py` 9 including the
dense/sparse consistency check on hand-built chain/star/hub/isolated-
cell graphs (the same bar `test_sparse_dense_consistency.py` holds
CellV1.1 to) and an `O(E)`-not-`O(n_cells^2)` empirical scaling check,
`test_structural_plasticity.py` 11 including the in-degree-1 safety
constraint tested under adversarial pruning (fraction=1.0, i.e. "try to
prune everything"), `test_structural_dynamics.py` 10,
`test_structural_model.py` 5 including a 15-step AdamW smoke test
through the full training-loop integration contract (forward ->
backward -> `update_edge_utility` -> `optimizer.step` ->
`maybe_run_structural_plasticity`) -- loss decreased monotonically
end-to-start, no divergence, edge count and in-degree invariants held
throughout). Full suite 274 -> 319, all passing. **No experiment has
been run** -- this is "matches the specified formulas, produces valid
finite output, scales linearly in edge count" validation, the same bar
every other CellV1 variant was held to before its first real run, not a
task-performance claim.

**Also found while touching `docs/architecture_v1.md` for this entry:**
an earlier edit to §16 (the "finalize spec" pass) left the old, now-
superseded "Motivation"/"Core decomposition"/"What's specified"/"Open
questions"/"Relationship to prior variants" subsections from the
PROPOSED-status draft still in the file, duplicated after the new
§16.0-§16.11 content and before §17's revision history -- a real
leftover-content bug, not a design issue. Removed as part of this
entry's doc update; §16.12 (the implementation map) now sits where that
dead block was.

**Why:** Every formula implemented is the user's, specified in the
previous two log entries; the implementation choices made here
(scatter-reduction over padding, the `edge_message_full` re-association,
module/file split) are engineering realizations of that spec, not new
architecture decisions -- flagged inline in each module's docstring the
same way §16.10's implementation-choice defaults are.

**Follow-up:** No experiment has been run with CellV1.5 yet. Natural
next steps, not yet decided: wiring it into `v1_001_dynamic_groups`'s
harness alongside `cellv0.1`/`association_local_global_t2` for a
matched-parameter comparison, and separately, `EdgeRegistry`'s
`state_dict()`/`load_state_dict()` limitation (round-trips only when
edge count matches) would need addressing before checkpoint-resume
support is needed for a real training run.

## 2026-09-06 — CellV1.6 implemented: Precision-Regulated Assembly (unevaluated)

**Context:** The user specified CellV1.6 as an *implementation task
only* ("Do not redesign the architecture, add alternative mechanisms, or
perform literature research"). Unlike CellV1.1–1.5, it is not part of
the `(mu, e, u, z)` dynamic-graph line — it goes back to the frozen
CellV0.1 `BeliefNetwork` (`src/models/architecture_v0/belief_network.py`:
plain `(mu, e, u)`, `scale_stable_precision` fusion, linear readout) and
adds exactly one input-dependent population-competition gate between its
two layers. Full spec now in `docs/architecture_v1.md` §17.

**What was built, following the spec exactly:**

- `assembly_gate.py::PrecisionRegulatedAssemblyGate`. Per cell:
  `log_precision_i = log(e_i + eps) - log(u_i^2 + eps)`; `drive_i =
  F_part([mu_i, log_precision_i])` where `F_part` is ONE shared
  `2 -> 8 -> 1` SiLU MLP (no per-cell network); per-example
  standardization `z_i = (drive_i - mean)/sqrt(var + eps)`; population
  confidence `C = mean(log_precision)`; window `delta = 0.05 +
  softplus(width_bias - softplus(kappa_raw) * C)` (two learned scalars);
  `participation_i = relu(1 - (z_max - z_i)/(delta + eps))`, naturally in
  `[0, 1]`, max-drive cell exactly 1, cells > `delta` below the max
  exact 0, never renormalized. No detach anywhere.
- `assembly_model.py::PrecisionRegulatedAssemblyNetwork`. `BeliefNetwork`
  with the gate spliced between `layer1` and `layer2`; `use_assembly_gate`
  flag; `last_assembly_diagnostics` (detached: `assembly_active_fraction`,
  `assembly_mean_participation`, `assembly_delta`,
  `assembly_population_log_precision`).
- The **only** edit to existing code: `integration.BeliefLayer.forward`
  gained an optional `source_participation` arg. `None` (every existing
  call site) → byte-for-byte identical to before. Given →
  `g_effective[b, j, i] = g[j, i] * participation[b, i]`, then the
  unchanged scale-stable precision equations. Verified: full suite was
  334/334 before, still 334 of those passing after.

**Two implementation-choice defaults, flagged in §17.3 and the module
docstring:**

- `F_part`'s output `Linear` has `bias=False`. Because the drive is
  immediately per-example standardized (invariant to a uniform additive
  constant), an output bias is a permanently zero-gradient parameter —
  the same reason `bias=False` precedes BatchNorm. Caught by the
  "gradients reach every F_part parameter" test failing on
  `participation_drive.2.bias` with an *exact* zero; removed rather than
  kept as dead weight. `F_part` is 32 params; total added over CellV0.1
  is **34, constant** (`+ kappa_raw + width_bias`), independent of every
  dimension.
- `width_bias_init = 3.0` / `kappa_raw_init = 0.0` → initial `delta` ~2.5–3
  in standardized-drive units, so most cells participate at init and the
  network learns to sparsify (mirrors `WriteGateFunction`'s near-off
  bias-init convention, §12). Not a tuned assembly-size target — and per
  the spec there is no auxiliary sparsity loss, target support %, L0
  penalty, or fixed winner count anywhere; "activates almost every cell"
  is an allowed empirical result.

**Status:** Implemented and unit-tested — 26 new tests
(`test_assembly_gate.py` 18, `test_assembly_model.py` 8), full suite
334 → 360, all passing. Covered: shape/dtype/device incl. MPS (the
first MPS-gated tests in the repo — they run, not skip, on this
hardware); participation finite and in `[0, 1]`; max-drive cell = 1;
exact zero outside the learned window; no fixed winner count (two
populations, different active counts); `delta` non-increasing when
evidence rises or uncertainty falls uniformly; standardized `z`
invariant to positive affine changes of the drive logits; gradients
reach `mu`/`e`/`u`/`F_part`/`kappa_raw`/`width_bias`; gate-off and
participation-ones both reproduce CellV0.1 to `torch.equal` /
`atol=1e-6`; no `[N, N]` tensor (`TorchDispatchMode` max-numel check)
+ linear memory scaling (8×-cells timing); 30-step AdamW smoke run
(loss 1.05 → 0.54, no NaNs). Measured gate-only forward overhead:
0.14–0.24 ms (MPS) / 0.23–1.32 ms (CPU) across batch×n_cells
64²–256². **No experiment has been run** — correctness-tested only,
the same bar every CellV1 variant was held to before its first run.

**Why:** Every formula is the user's, specified in one turn as an
implementation task. The two flagged defaults are engineering
realizations (a dead-parameter removal, a bias-init) not architecture
decisions.

**Follow-up:** Not yet decided — a matched-parameter comparison against
`cellv0.1` (and the MLP baseline) on a task where input-dependent
assembly selection could plausibly matter would be the natural first
experiment. Until then CellV1.6 is proposed-and-implemented, unevaluated;
do not describe it as working.

## 2026-09-06 (same day) — CellV1.6 first validation experiment: no improvement over CellV0.1

**Scope (user's spec):** the first validation experiment for CellV1.6.
Implementation only — no architecture change, no gate tuning, no
alternative mechanism. Compare **only** `cellv0.1` and `cellv1_6`, on the
existing `dynamic_groups_global` benchmark at `hard` (96 objects) and
`very_hard` (192 objects), seeds 0/1/2. Reuse the CellV0.1 complexity
experiment's exact protocol
(`experiments/v1_001_dynamic_groups/run_complexity_scaling_association.py`
/ `harness.run_convergence_association_comparison`): same dataset
generation and train/val/test splits (3000/500/500), same optimizer
family (AdamW, lr 1e-2, batch 64), same convergence-based evaluation
(`_train_until_convergence`: validate every 100 steps, stop after 500
steps without val-R² improvement, cap 5000, restore best-val checkpoint),
same parameter budget. No special LR/schedule for `cellv1_6`.

**Setup.** `cellv1_6` = `cellv0.1`'s exact 2-`BeliefLayer` backbone plus
one 34-param `PrecisionRegulatedAssemblyGate` between the layers. Both
arms use the same `hidden_cells`, sized to the same per-level budget the
complexity experiment used (the param count of
`association_local_global_t2` there — 3547 at `hard`, 3739 at
`very_hard`), via the identical `_match_belief_hidden_cells` call. That
budget, against a 288-/576-dim flattened input, forces a **tiny hidden
population: 6 cells at `hard`, 3 at `very_hard`** (each hidden `BeliefCell`
costs ≈2·in_features params in layer 1). So the "population participation
competition" runs over 3–6 cells, with the max-drive cell always pinned
to participation 1. This is what the frozen matched-parameter protocol
dictates; recorded as an obvious structural fact, not interpreted.

Harness additions (`experiments/v1_001_dynamic_groups/`): `harness.py`
gained `run_convergence_assembly_comparison` /
`_build_assembly_comparison_models` and `_train_until_convergence` an
opt-in `capture_final_state` (returns end-of-training params before the
best-checkpoint restore, so the gate's end-state regime is inspectable);
`run_assembly_comparison.py` is the entry script. `test_assembly_comparison_harness.py`
checks `capture_final_state` leaves the trajectory unchanged and the two
arms are matched +34.

**Protocol-fidelity check.** The `cellv0.1` arm here reproduces the
complexity experiment's recorded `cellv0.1` numbers to ≤0.001 R²
(`hard` mean 0.8287 vs 0.8293 there; `very_hard` 0.8193 vs 0.8191) —
confirms the protocol is faithfully re-run, differences are float/MPS
nondeterminism.

**The 12 runs** (2 models × 2 levels × 3 seeds), best-val checkpoint step
/ test R² / total steps / wall-clock (best-val checkpoint restored before
test eval; param counts constant per level):

| level | seed | model | best step | test R² | total steps | wall (s) | params |
|---|---|---|---|---|---|---|---|
| hard | 0 | cellv0.1 | 300 | 0.8393 | 800 | 2.50 | 3547 |
| hard | 0 | cellv1_6 | 500 | 0.8222 | 1000 | 3.79 | 3581 |
| hard | 1 | cellv0.1 | 700 | 0.8163 | 1200 | 4.05 | 3547 |
| hard | 1 | cellv1_6 | 400 | 0.8106 | 900 | 3.39 | 3581 |
| hard | 2 | cellv0.1 | 400 | 0.8304 | 900 | 2.69 | 3547 |
| hard | 2 | cellv1_6 | 300 | 0.8180 | 800 | 2.97 | 3581 |
| very_hard | 0 | cellv0.1 | 700 | 0.8165 | 1200 | 3.59 | 3484 |
| very_hard | 0 | cellv1_6 | 700 | 0.8172 | 1200 | 4.53 | 3518 |
| very_hard | 1 | cellv0.1 | 700 | 0.8154 | 1200 | 3.57 | 3484 |
| very_hard | 1 | cellv1_6 | 1100 | 0.7963 | 1600 | 6.68 | 3518 |
| very_hard | 2 | cellv0.1 | 800 | 0.8259 | 1300 | 3.98 | 3484 |
| very_hard | 2 | cellv1_6 | 600 | 0.8101 | 1100 | 5.69 | 3518 |

**Mean ± std (population std, n=3):**

| level | model | test R² | best step | total steps | wall (s) | params |
|---|---|---|---|---|---|---|
| hard | cellv0.1 | 0.8287 ± 0.0095 | 467 ± 170 | 967 | 3.08 | 3547 |
| hard | cellv1_6 | 0.8169 ± 0.0048 | 400 ± 82 | 900 | 3.39 | 3581 (+34) |
| very_hard | cellv0.1 | 0.8193 ± 0.0047 | 733 ± 47 | 1233 | 3.71 | 3484 |
| very_hard | cellv1_6 | 0.8079 ± 0.0087 | 800 ± 216 | 1300 | 5.63 | 3518 (+34) |

- **Test R²:** CellV1.6 is **lower at both levels** — `hard` Δ = −0.0118
  (≈ −1.2 × cellv0.1's seed std), `very_hard` Δ = −0.0114 (≈ −2.4 ×).
  Per seed, `cellv0.1` wins 5 of 6 pairs; the exception is
  `very_hard`/seed 0, a near-tie (`cellv1_6` +0.0007). Small, but
  consistently negative — not a collapse, not a large regression.
- **Parameter count:** +34 (+0.96% `hard`, +0.98% `very_hard`). Not
  meaningful. (Constraint satisfied — the gate does not meaningfully
  grow the model.)
- **Convergence-step ratio** (`cellv1_6` / `cellv0.1`, best-val
  checkpoint): 0.86 (`hard`), 1.09 (`very_hard`). No consistent
  direction; `cellv1_6`'s best-step seed variance is large (±82, ±216).
  Total-steps comparable (900 vs 967; 1300 vs 1233).
- **Runtime ratio** (total wall-clock): 1.10× (`hard`), 1.52×
  (`very_hard`). `cellv1_6` is slower per run. The `very_hard` 1.52×
  overstates the mechanism cost — these are 3–6-hidden-cell models where
  the gate's fixed overhead (measured 0.14–0.24 ms/forward on MPS in the
  implementation writeup) is a large fraction of a ~3500-param backbone,
  and single wall-clock measurements over 3 seeds are noisy
  (`cellv1_6` `very_hard`: 4.53 / 6.68 / 5.69 s).

**Assembly diagnostics** (mean over seeds, measured on the full test set
at three points; `init` = before any training, `best` = at the best-val
checkpoint, `end` = at the point training stopped):

| level | phase | active_fraction | mean_participation | delta | population_log_precision |
|---|---|---|---|---|---|
| hard | init | 1.0000 | 0.7959 | 4.0299 | −1.3865 |
| hard | best | 0.5510 | 0.3425 | 1.7237 | −1.5331 |
| hard | end | 0.5626 | 0.3365 | 1.7272 | −1.5398 |
| very_hard | init | 1.0000 | 0.8515 | 4.0298 | −1.3864 |
| very_hard | best | 0.5476 | 0.4235 | 1.5139 | −1.5641 |
| very_hard | end | 0.5500 | 0.4288 | 1.5700 | −1.5697 |

- **Active fraction does NOT stay ~1.0** and does **NOT** collapse toward
  zero — it settles at ~0.55 at both levels (with 3–6 cells, ~0.55 means
  the max-drive cell plus, on average, roughly one more). Reported
  plainly, per the important-checks instruction; no sparsity penalty
  added, nothing repaired.
- **Both `delta` and active fraction move:** `delta` narrows from 4.03
  to ~1.5–1.7 (~2.5×), active fraction 1.00 → 0.55, mean participation
  ~0.80 → ~0.34–0.42. So the gate learned a real, non-trivial competition
  regime — not the "delta changes but participation doesn't" case, and
  not a no-op.
- **`best` ≈ `end`** on every diagnostic (active fraction 0.551 vs 0.563;
  delta 1.72 vs 1.73) — the operating regime is stable through the
  patience tail, not still drifting when training stops.
- `population_log_precision` (C) drifts slightly more negative
  (−1.39 → −1.53 / −1.56) — the belief states' evidence/uncertainty
  balance shifted modestly under end-to-end training; a small effect.

**Verdict on the frozen hypothesis** ("does letting CellV0.1's internal
evidence/uncertainty dynamically regulate population participation improve
computation over CellV0.1 alone, without meaningfully increasing parameter
count or optimization cost?"): **negative on this benchmark.** The gate
adds ~1% params, costs modestly more wall-clock, converges in a
comparable number of steps, and learns a genuine sparse operating regime
(active fraction 1.0 → 0.55, window 4.0 → 1.6) — but test R² is
consistently ~0.011–0.012 lower than plain CellV0.1 at both complexity
levels. Not tuned, not redesigned; the result stands as tested.

**Obvious caveat, not an interpretation:** the matched-parameter budget
forces a 3–6-cell hidden population, a degenerate regime for a
"population competition" mechanism (at `very_hard` the gate chooses among
3 cells, one always pinned on). Whether the mechanism behaves differently
with a wider hidden population is a separate question this experiment
does not touch (it would break the parameter match and the frozen
protocol).

**Why:** The experiment protocol, models compared, and parameter budget
are all the user's explicit spec; the two harness additions
(`capture_final_state`, the `_build_assembly_comparison_models` /
`run_convergence_assembly_comparison` pair) are experiment-local
composition of existing pieces following the established
`harness.py` pattern, not new architecture or protocol.

## 2026-09-08 — Paper A Phase 1: CellV0.1 standalone public-benchmark screen

**Context:** CellV1 work frozen. Question: does CellV0.1 — the plain
`(mu, e, u)` belief cell with `scale_stable_precision` fusion
(`src/models/architecture_v0/`, `BeliefNetwork`, unchanged) — deserve a
paper as a standalone artificial-neuron primitive? Phase 1 is
implementation + a screening run on public data, nothing tuned, CellV0.1
untouched. New code is all under `experiments/paper_a/` (composition only).

**Setup (frozen before looking at any number):** 7 datasets (sklearn
Breast Cancer / Wine / Digits / Diabetes / California Housing; openml
MNIST / Fashion-MNIST, flattened, no CNN) × {25%, 100%} training pool ×
seeds 0/1/2 × 3 models: (A) CellV0.1, largest hidden-cell count within a
fixed per-class parameter budget (~10k / ~25k / ~150k); (B) a
parameter-matched 1-hidden-layer SiLU MLP (≤0.12% off A's count on every
dataset); (C) a state-count MLP, hidden width 3×A's hidden cells, **not**
parameter-matched. Shared protocol: AdamW, lr=1e-2, wd=0, best-val
restore, ~1500-step early-stop patience, 15000-step cap, CPU. 126 main +
9 fixed-confidence ablation runs + 1 deterministic duplication experiment.
Full tables: `experiments/paper_a/phase1_results.md`.

**A. Where CellV0.1 lands vs the parameter-matched MLP (headline, mean of
3 seeds, +ve = CellV0.1 better):**

| dataset | 25% | 100% |
|---|---|---|
| MNIST (acc) | +0.007 | +0.015 |
| Fashion-MNIST (acc) | +0.010 | +0.014 |
| California Housing (R²) | +0.017 | +0.080 |
| Diabetes (R²) | +0.359 | +0.081 |
| Breast Cancer (acc) | −0.009 | +0.003 |
| Wine (acc) | −0.019 | −0.009 |
| Digits (acc) | −0.053 | −0.040 |

Ahead by >0.01 in 7 of 14 (dataset, fraction) cells, behind by >0.01 in 3
(all of Digits + Wine-25%), within ±0.01 in 4.

- **Images:** small but fully consistent edge — every one of the 12
  image seed-comparisons is positive.
- **Regression:** the large Diabetes/California margins are **partly a
  baseline artifact**: under the shared lr=1e-2 the matched MLP is
  unstable on these small regression sets (Diabetes-25% MLP
  R²=0.017±0.302 — one seed to negative R²; California-100% MLP
  R²=0.697±0.079, *worse* than its 0.732 at 25%). No NaN/divergence, so
  the protocol was not changed (per the task's instruction); reported as
  a caveat, not a clean win.
- **Digits:** CellV0.1 clearly loses, at both fractions, and the wider
  state-count MLP beats it too — a genuine handicap on this task.
- The advantage is **not** reliably larger at 25% data (mixed; on
  Diabetes CellV0.1 is flat 0.376→0.381 while the MLP catches up).

**B. State-count control:** the substantially wider (not
parameter-matched) MLP does **not** close CellV0.1's edge on
MNIST/Fashion-MNIST/California/Diabetes — on the image sets it is in fact
*worse* than the parameter-matched MLP (overfits / optimizes worse at
lr=1e-2). "Give an ordinary MLP as many hidden activations as CellV0.1 has
state values" does not reproduce the image-dataset edge.

**C. Fixed-confidence ablation (Sec 11) — the paper's actual hypothesis:**
force every belief entering a fusion to `e=u=1`, same content weights,
same parameter count. Digits −0.001, Diabetes −0.001, Fashion-MNIST
+0.007. So the **dynamically propagated evidence/uncertainty state is
inert** on the tabular datasets where CellV0.1 does well — its behaviour
there comes from the richer per-connection content parameterisation (a
content weight *and* a relevance gate, two params/connection), not the
belief state. Only on Fashion-MNIST does the propagated `e`/`u`
contribute measurably (~0.7 pt).

**D. Duplication-invariance (Sec 10, deterministic):** a fixed
heterogeneous 6-source belief set, whole multiset duplicated ×1/2/4/8/16.
`scale_stable_precision` (and `normalized_precision`): output (mu, e, u)
invariant to float64 tolerance (max |Δ| ≈ 3e-9). The pre-scale-stable
`precision` rule (kept unmodified as the historical control): evidence
grows exactly ×m (5.26 → 84.16 at ×16), uncertainty shrinks. The central
scale-stability property holds exactly.

**E. Cost & stability:** CellV0.1 trains ~100–250× slower in wall-clock
than the matched MLP under the same step regime (MNIST/Fashion-MNIST
CellV0.1 runs converge and early-stop at ~4k–12k steps — not
cap-censored). Internal `e`/`u` finite and positive everywhere, but
Digits layer-1 max uncertainty spikes to ~2×10⁴ and California to ~5×10²
— a numerical-range flag, not a failure. Diabetes-100% CellV0.1 has ~3×
the matched MLP's seed variance.

**Verdict on whether CellV0.1 deserves a paper — neutral, leaning weak.**
The advantage does not disappear on all public data (consistent ~1% edge
on both flattened-image sets, every seed) and is not reproduced by
widening the MLP — so this is not the "bad" outcome. But it is not
"strong" either: the edge is small where it is clean, reversed on Digits,
confounded by baseline instability on the regression sets, and not
reliably data-efficiency-driven. Most importantly, the fixed-confidence
ablation shows the mechanism the paper is *about* — recursively
propagating evidence/uncertainty — contributes only on Fashion-MNIST;
elsewhere CellV0.1's numbers are carried by its content-pathway
parameterisation. Reported as evidence, no redesign (Sec 12/14).

**Why:** dataset list, model families, budgets, regimes, protocol are the
Paper-A task spec verbatim; `experiments/paper_a/` is experiment-local
composition of existing `src/` infrastructure following the
`experiments/*/harness.py` pattern — no new architecture, math, or
training procedure, and CellV0.1's equations are unchanged.

## 2026-09-09 — CellV0.2 (Conservative Precision-Gain Cell): implemented, unevaluated

**Context:** User specified a second CellV0-line cell in full mathematical
detail and asked for an implementation + evaluation on the *existing frozen*
Paper-A Phase-1 protocol (not a new benchmark suite). This entry records the
implementation; the frozen benchmark results are the next entry, written
after the 42 CellV0.2 runs complete.

**What CellV0.2 is.** A separate belief-aggregation operator, **not** a sixth
`BeliefLayer` method — it removes CellV0.1's relevance-gate matrix
(`a_ij`/`g_ij`) entirely. Implemented in
`src/models/architecture_v0/precision_gain.py` (`PrecisionGainLayer`,
`BeliefNetworkV02`); documented in `docs/architecture_v0.md` §9. Same
`BeliefCell` state, but the *input* belief is `e = 1, u = 0`.

Per layer (`V` `[out, in]`, `gamma = softplus(gain_raw)` `[out]`, `bias`
`[out]` — the only three parameter objects):

```text
pi_j   = e_j / (1 + e_j u_j)                 effective precision
r_j    = 2 pi_j / (pi_j + mean_k pi_k)       relative gain, in (0, 2), mean NOT detached
x_j    = r_j mu_j
c_i    = (sum_j x_j V_ij) / sum_j|V_ij|      signed consensus            (GEMM)
mu_i   = tanh(gamma_i c_i + bias_i)
e_i    = (sum_j pi_j |V_ij|) / sum_j|V_ij|   inherited support (convex)   (GEMM)
u_i    = max(0, (sum_j x_j^2 |V_ij|)/sum_j|V_ij| - c_i^2)                 (GEMM)
```

`gain_raw` is inverse-softplus-initialized so `gamma_i = ||V_i||_1` at
init → the content path reduces *exactly* to `tanh(F.linear(mu, V, b))` on a
neutral-confidence input, and `(V, gamma)` can represent any linear weight
row. `BeliefNetworkV02` = two such layers + a readout applied to
`relative_gain(final_precision) * final_mu` (the readout consumes
confidence; it does not emit a belief state).

**Properties verified** (`tests/test_precision_gain.py`, 22 tests; full
suite 494 pass): shape/dtype/device incl. MPS; `e > 0`, `u >= 0`, all
finite; `0 < relative_gain < 2` and `= 1` for equal precisions;
neutral-confidence reduction to `tanh(F.linear(mu, V, b))` to ~1e-5;
linear-row expressivity; uniform replication invariance ×2/×4/×8/×16 to
~1e-9 (float64); `e_out` a convex combination of source precisions
(`e_out <= max pi`, `pi_out <= e_out`); conflicting messages → higher
`u_out` → lower effective precision at fixed inherited support; gradients
reach source `mu`/`e`/`u`, `V`, `gain_raw`, `bias`; positive row-scaling of
`V` leaves the normalized consensus / `e_out` / `u_out` unchanged;
torch-dispatch check confirming **no rank-3 `(B, out, in)` edge tensor** and
no batched matmul; AdamW smoke train reduces loss with no NaNs.

**Isolated-layer cost** (`experiments/paper_a/bench_cellv02_layer.py`, CPU,
1 thread, forward+backward ms/iter over 200 iters, approximate — single
wall-clock timings are noisy):

| in→out (batch) | CellV0.2 | CellV0.1 (`scale_stable_precision`) | Linear+Tanh | V0.2 / V0.1 | V0.2 / Linear |
|---|---|---|---|---|---|
| 30→83 (128) | 0.20 | 1.73 | 0.08 | 0.11× | 2.4× |
| 64→123 (128) | 0.27 | 4.68 | 0.12 | 0.06× | 2.3× |
| 784→157 (128) | 0.98 | 36.9 | 0.27 | 0.03× | 3.6× |
| 157→157 (128) | 0.56 | 8.13 | 0.17 | 0.07× | 3.2× |
| 256→256 (256) | 1.30 | 41.3 | 0.34 | 0.03× | 3.8× |

CellV0.2's three GEMMs are ~15–35× cheaper than CellV0.1's `(B, out, in)`
broadcast fusion, and ~2–4× a plain `Linear+Tanh`. Parameter-count
formulas: CellV0.2 per layer `out·(in+2)`; CellV0.1 per layer `out·(2·in+1)`.

**Why this is allowed under CLAUDE.md §2:** like CellV1 and `BeliefLayer`
Methods D/E, CellV0.2 is the user's own full specification, delivered with
an explicit "implement and evaluate" instruction — not an agent-invented
aggregation rule. It is kept strictly separate from CellV0.1 (own module,
own experiment id `paper_a_phase1_cellv02`, own summarizer); CellV0.1's
equations and recorded results are untouched. The README's Phase-1
anti-cherry-picking line ("do not create CellV0.2") governed the *screening
run itself* — this is a later, separately-commissioned architecture line
run through the identical frozen protocol, reusing the recorded CellV0.1 /
matched-MLP arms for comparison rather than re-running or altering them.

## 2026-09-09 — CellV0.2: frozen Paper-A Phase-1 results

**Context:** CellV0.2 implemented (previous entry). Run through the
**identical frozen** Phase-1 protocol — same 7 datasets, same seeded
splits, {25%, 100%}, seeds 0/1/2, AdamW (lr=1e-2, wd=0), best-val restore,
~1500-step patience, 15000-step cap, CPU. 42 CellV0.2 runs (experiment id
`paper_a_phase1_cellv02`). CellV0.1 and the parameter-matched MLP are **not**
re-run — their recorded `paper_a_phase1` records are the comparison arms.
Nothing tuned; equations frozen before the run. Full tables:
`experiments/paper_a/phase1_v02_results.md`.

**Sizing.** CellV0.2's hidden width fitted to the *same* per-dataset budget
as CellV0.1. No gate matrix → ~1.4-1.9× the hidden cells (e.g. digits
123 vs 82 @ ~24.8k params; MNIST 157 vs 85 @ ~149k). Reported, not
equalized — the lower per-connection cost is part of the architecture.

**A. Headline (mean of 3 seeds), CellV0.2 − recorded arm:**

| dataset | vs matched MLP (25 / 100) | vs CellV0.1 (25 / 100) |
|---|---|---|
| breast_cancer | −0.006 / +0.000 | +0.003 / −0.003 |
| wine | −0.009 / −0.009 | +0.009 / +0.000 |
| digits | **+0.005 / +0.000** | **+0.057 / +0.040** |
| diabetes | +0.213 / +0.038 | **−0.146 / −0.043** |
| california_housing | +0.042 / +0.100 | +0.025 / +0.020 |
| mnist (acc) | +0.017 / +0.017 | +0.010 / +0.003 |
| fashion_mnist (acc) | +0.020 / +0.019 | +0.010 / +0.005 |

Vs the parameter-matched MLP: CellV0.2 ahead by >0.01 in **8 of 14**
(dataset, fraction) cells, behind by >0.01 in **0**, within ±0.01 in 6.

- **Digits — CellV0.2 removes CellV0.1's only clean public-data loss.**
  CellV0.1 was −0.053 / −0.040 behind the matched MLP on Digits (and also
  behind the wider state-count MLP). CellV0.2 pulls to parity
  (+0.005 / +0.000) — a +0.057 / +0.040 swing over CellV0.1. Most of this
  is the extra hidden cells the cheaper parameterisation buys (123 vs 82).
- **Images — the small CellV0.1 edge holds and slightly widens.** Every one
  of the 12 MNIST/Fashion-MNIST seed comparisons vs the matched MLP is
  positive (+0.017 to +0.020 on the mean); vs CellV0.1, +0.003 to +0.010.
- **California — CellV0.2 beats both** CellV0.1 (+0.02-0.03) and the MLP
  (+0.04-0.10).
- **Diabetes — CellV0.2 is worse than CellV0.1** (−0.146 at 25%, −0.043 at
  100%), the one clear regression. It still sits above the matched MLP, but
  the Phase-1 caveat stands: the matched MLP is unstable on the small
  regression sets at lr=1e-2 (Diabetes-25% MLP R²=0.017±0.302), so
  "CellV0.2 > MLP on Diabetes/California" is not a clean win. Against the
  stable CellV0.1 baseline, CellV0.2 loses on Diabetes and wins on
  California.
- Breast Cancer / Wine: within ±0.01 of both arms.

**B. Cost.** CellV0.2 converges in **fewer** best-validation steps than
CellV0.1 on every dataset except Breast Cancer, and trains **3-48× faster**
in wall-clock (Digits ~16-19×, MNIST ~28-31×, Fashion-MNIST ~42-48×) — its
three GEMMs vs CellV0.1's `(batch, out_cells, in_cells)` broadcast fusion.
The whole 42-run sweep took ~6 min; the CellV0.1 Phase-1 image runs alone
took hours. No divergence, no NaNs, no step-cap hits.

**C. The confidence mechanism looks close to inert — same as Phase-1's
finding for CellV0.1.** Relative-gain diagnostics
`2π / (π + mean π)` sit at **mean ≈ 0.99-1.00** (per-cell std 0.01-0.09) in
every trained network, on every dataset — i.e. the precision-based
modulation of each source message is operating very close to the identity.
Layer 1's inherited precision `e` is moreover *exactly* 1 by construction
(all input features start at precision 1 and `e_out` is a convex
combination), so layer 1 only ever varies `u`. Effective precision by layer:
L1 mean 0.43-0.72, L2 mean 0.33-0.67 (the `u`-driven part is real), but the
gain it produces is near-1. So — as the Phase-1 fixed-confidence ablation
concluded for CellV0.1 — CellV0.2's advantage is carried by its
**content-pathway reparameterisation** (one normalised signed matrix `V`
plus a separate positive gain; exact `tanh(F.linear)` reduction at init;
~1.5× more hidden cells per parameter), **not** by propagating
evidence/uncertainty. No CellV0.2 fixed-confidence ablation was run (not in
scope; the relative-gain diagnostic is the observational proxy).

**D. Numeric-range flag.** Digits (all 3 seeds, both fractions) has a single
test-example × cell with layer-1 effective precision ~1e-11 (a large local
disagreement `u`) — the CellV0.2 analogue of the Digits layer-1 max-`u`
spike noted for CellV0.1 in Phase-1. All values finite; no NaN, no
divergence, no protocol impact.

**Verdict — CellV0.2 is a clear content-pathway improvement over CellV0.1 on
this frozen protocol, at a fraction of the compute.** It removes CellV0.1's
one clean public-data loss (Digits), holds the every-seed image edge, adds a
California edge, converges faster, and runs 3-48× faster — with one
regression (Diabetes vs CellV0.1) and the unchanged regression-baseline
caveat. But the mechanism the V0 line is *about* — recursively propagating
evidence/uncertainty — is, by the relative-gain diagnostics, close to inert
in the trained networks, exactly as Phase-1 found for CellV0.1. Reported as
evidence: no redesign, no calibration/sparsity losses, no learned gain
temperature, no new datasets, no CellV0.2.1.

**Why:** identical frozen protocol; CellV0.2 is the user's own full
specification with an explicit implement-and-evaluate instruction; CellV0.1
and its recorded results are untouched; the comparison reuses the recorded
arms rather than re-running them; nothing was tuned after seeing results.
