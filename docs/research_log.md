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
