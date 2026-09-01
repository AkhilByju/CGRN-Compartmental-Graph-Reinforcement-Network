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
