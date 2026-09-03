# Architecture V0 — Design Space and Open Questions

**Status: PARTIALLY SPECIFIED.** CellV0's *state* (§1) is decided and
implemented (`src/models/architecture_v0/cell.py`). Its *aggregation
operator* — how a cell combines incoming state into its own — now has
**five implemented candidates, not a final choice**
(`src/models/architecture_v0/integration.py`'s `BeliefLayer`). Everything
in §2 onward is still open. No code beyond `cell.py` and `integration.py`
should be implemented past a stub until the sections marked `OPEN` below
are resolved by the user and this document is updated.

The next concrete research task is **Experiment 002/003** — running the
`BeliefLayer` aggregation methods against each other and a parameter-matched
plain-MLP baseline (`docs/experiment_protocol.md`) to see whether any of
them, or none, is worth carrying forward. This is an experimentation task,
not a further design task — **do not add a further candidate or otherwise
keep redesigning §1 without the user explicitly specifying the change**, the
way `"normalized_precision"` (Method D, Experiment 004F) and
`"scale_stable_precision"` (Method E, Experiment 004I) were: both fix a
scale-instability the user found in Method C, both diagnosed/specified by
the user, neither invented by an agent.

**§3 and §4 below are superseded by a separate, now-implemented line:**
`docs/architecture_v1.md` — a self-organizing, input-dependent "Dynamic
Belief Graph" replacing the fixed layer/cluster structure entirely, with a
fourth cell field `z` (CellV1). Specified by the user on 2026-09-02 and
implemented in `src/models/architecture_v1/`. This document (V0's fixed
2-`BeliefLayer` stack) is not superseded as an experimental line — it
continues to stand on its own (Experiments 002-004) — but §3/§4's open
sketches are no longer the live design questions for graph structure;
`docs/architecture_v1.md` is.

## 1. CellV0 — the basic computational unit

**State: SPECIFIED. Aggregation: FIVE CANDIDATES IMPLEMENTED, none chosen.**

CellV0 replaces the ordinary artificial neuron's single scalar activation

```text
a_i = phi(sum_j w_ij * a_j + b_i)
```

with a structured **belief state** carried by a `BeliefCell`
(`src/models/architecture_v0/cell.py`):

```text
B_i = (mu_i, u_i, e_i)

mu_i in R    -- content: what the cell currently believes
e_i  in R+   -- evidence: how much supporting information produced it
u_i  in R+   -- uncertainty: how (un)confident the cell is in mu_i
```

Everywhere an ordinary network would put one neuron, CellV0 puts one
`BeliefCell`. Information flows as structured belief states rather than
bare scalar activations:

```text
Ordinary neuron                     CellV0 (Belief Cell)

x1 --w1--\                          B1=(mu1,u1,e1) --w1--\
x2 --w2--+-sum-phi->a_i             B2=(mu2,u2,e2) --w2--+-???->B_i=(mu_i,u_i,e_i)
x3 --w3--/                          B3=(mu3,u3,e3) --w3--/
```

`tau_i` (persistence, `tau in [0,1]`) is part of the eventual
`(mu, u, e, tau)` state but is deliberately **not** a `BeliefCell` field in
V0.0: a feedforward cell fires once, so there's nothing yet for persistence
to persist across. It is reintroduced when V0.1 adds recurrence (§2).

Earlier drafts of this document described CellV0 in terms of multiple
intra-cell "compartments" (`D_i1..D_iB`). That is **not** part of the
decided design — a `BeliefCell` holds one `(mu, u, e)` state, not several
sub-states — see `src/models/architecture_v0/compartment.py`, kept only as
a stub in case a later, richer cell revisits the idea.

### The belief-aggregation operator: five implemented candidates

The `BeliefCell` state is settled; **how a cell combines N incoming
`BeliefCell`s (each reaching it through a learned connection) into its own
outgoing `BeliefCell` has five implemented candidates, and no choice among
them has been made.** `BeliefLayer` (`src/models/architecture_v0/integration.py`)
implements all five behind one interface — `BeliefLayer(in_cells,
out_cells, aggregation=...)` — so that comparing them is a controlled
experiment over the aggregation rule alone, everything else held fixed.

All five share the same learned-connection structure: a content weight
`w_ij` (how cell `j`'s content affects cell `i`) producing a message
`m_ij = w_ij * mu_j`, and a relevance gate `g_ij = sigmoid(a_ij)` (how
relevant cell `j` is to cell `i`).

- **`"reliability"`** — reliability-weighted consensus. Reliability
  `r_ij = g_ij * e_j / (1 + u_j)` sets each sender's influence; content is
  the reliability-weighted average of incoming messages; evidence is total
  reliability; uncertainty grows with both weighted disagreement among
  senders and a lack of total evidence.
- **`"support_conflict"`** — explicit positive vs. negative support.
  Bounds each message to `[-1, 1]` (`q_ij = tanh(m_ij)`) and separately
  sums reliability-weighted positive and negative support (`P_i`, `N_i`);
  content is net support `(P_i - N_i) / (P_i + N_i)`; evidence is total
  support `P_i + N_i`; uncertainty grows with explicit conflict
  `2*min(P_i,N_i)/(P_i+N_i)` and with a lack of evidence.
- **`"precision"`** — probabilistic precision fusion. Precision
  `pi_ij = g_ij * e_j / (u_j^2 + eps)` (uncertainty penalized quadratically,
  more harshly than the other two methods) sets each sender's influence;
  content is the precision-weighted average; evidence sums `g_ij * e_j`;
  uncertainty combines a precision-derived base term with weighted
  disagreement. Experiment 004D found this sum is **not scale-stable**:
  `evidence` grows ~linearly with the number of incoming cells regardless of
  whether they carry new information, and 004E's duplication test confirmed
  this formally (identical duplicated inputs inflate `evidence` and deflate
  `uncertainty` with no new information present).
- **`"normalized_precision"`** — Method D, added in Experiment 004F
  specifically to fix that instability, keeping everything else about
  Method C unchanged. Same per-connection precision
  `pi_ij = g_ij * e_j / (u_j^2 + eps)` and the same content-weighting
  `alpha_ij = pi_ij / sum(pi)` (cells still compete for influence over
  `mu_i` exactly as in Method C) — only `evidence` and the base-uncertainty
  term are normalized by total incoming relevance `G_i = sum_j(g_ij)`
  instead of left as raw sums: `e_i = sum_j(g_ij * e_j) / G_i` and
  `u_base_i = sqrt(1 / (mean_j(pi_ij) + eps))`. Conceptual reframing behind
  the fix: a layer's incoming cells are different *representations* of
  information, not automatically independent *observations* of it, so
  duplicating identical content should not mechanically manufacture more
  evidence. See `src/models/architecture_v0/integration.py`'s
  `_normalized_precision_fusion` docstring for the full derivation and
  `docs/research_log.md` ("Experiment 004F") for the motivation.
- **`"scale_stable_precision"`** — Method E, "CellV0.1", added in
  Experiment 004I as a refinement of Method D, not a third independent fix.
  Identical to Method D except *what* `evidence`/the base-uncertainty term
  are normalized by: instead of raw total relevance `G_i = sum_j(g_ij)`,
  Method E uses an **effective source count** — a participation-ratio (Kish
  effective-sample-size) statistic `N_eff_i = (sum_j(g_ij))^2 /
  (sum_j(g_ij^2) + eps)`. Ten equally-relevant inputs give `N_eff ~ 10`; one
  dominant input among many near-irrelevant ones gives `N_eff ~ 1`. For N
  identical, equally-weighted duplicates, `N_eff` reduces to exactly `N`
  (same as Method D's `G`), so both methods are duplicate-invariant on
  004E's test — they settle at *different* constants unless the shared
  relevance happens to equal 1, not a different N-dependence. Where the two
  genuinely diverge is *unequal* relevance: with uniform evidence across
  incoming cells, Method D's `evidence` is always exactly the per-cell
  value no matter how many weakly-relevant connections are added (the raw
  sum cancels identically), while Method E's `N_eff` correctly discounts
  for having more effective sources even when individually weak. See
  `src/models/architecture_v0/integration.py`'s
  `_scale_stable_precision_fusion` docstring and
  `tests/test_scale_stable_precision.py` for the full derivation and the
  test that demonstrates this divergence directly.

All five guarantee `mu_i = f(mu_j, e_j, u_j)` for every incoming cell, so
gradients from the eventual loss reach not just incoming content but
incoming evidence and uncertainty too — evidence/uncertainty are not merely
reported, they influence what the network predicts (verified in
`tests/test_belief_layer.py`).

Exact formulas, worked examples, and the reasoning behind each are recorded
in `docs/research_log.md` ("CellV0 aggregation candidates", "Experiment
004F", "Experiment 004I"). **None of the five is chosen** — see Experiment
002/003 in `docs/experiment_protocol.md` for the comparison that started
deciding whether any of the first three, over a plain MLP unit, is worth
keeping. Initial Experiment 002 results (`docs/research_log.md`, "Experiment 002
initial results") are mixed-to-negative: no consistent performance
advantage over a parameter-matched MLP, and the intended "uncertainty rises
on ambiguous inputs" behavior only appears on the easiest task and inverts
on harder ones. Experiment 004 later found `"precision"` specifically *does*
scale better than a parameter-matched MLP as model size grows
(`docs/research_log.md` "Experiment 004A/C/D results", "Result A") — not
yet a final verdict on any method, but a reason to keep treating this as an
open experimental question, not a settled preference.

### Why belief cells might matter (hypothesis, not a claim)

Not simply "more computation per unit is better" — that isn't guaranteed.
The hypothesis is that carrying evidence/uncertainty alongside content may
let a network propagate information more usefully than an undifferentiated
scalar activation can — e.g. weighting a confident input more than an
uncertain one. Possible advantages: more informative message-passing,
better calibration, improved sample efficiency, useful inductive bias for
relational/uncertain tasks. Possible disadvantages: harder optimization,
unnecessary complexity, slower execution, or no meaningful advantage over
an ordinary MLP unit once compute/parameters are matched. **Experimentally
determined, not assumed** — see Experiment 002/003 in
`experiment_protocol.md`.

### Test the cell before the graph

Before any graph complexity, compare parameter-matched variants: a
conventional MLP-like unit (plain scalar activation) vs. a `BeliefLayer`
network run with each of `"reliability"`, `"support_conflict"`, and
`"precision"`. Input belief initialization for this comparison is decided
too: for a directly observed raw feature `x_j`, start with
`mu_j = x_j, e_j = 1, u_j = 1` (`BeliefCell.from_observed_features`) so
every feature begins equally uncertain and any useful evidence/uncertainty
structure has to be learned, not hand-given. Natural ablation axes once the
initial comparison is in: evidence-weighting on/off, uncertainty derived
from evidence alone vs. evidence+disagreement, and uncertainty used vs.
ignored by downstream cells. Does carrying evidence/uncertainty help, for
which method, and for which function classes? Does it just increase
compute? (Experiment 002/003.)

## 2. Layers → persistent computational substrate [OPEN, dependent on §1]

A conventional (Transformer-like) model: `input -> Layer_1 -> Layer_2 -> ...
-> Layer_N -> output`, each layer with its own parameters. The proposed
direction instead reuses the **same parameters** repeatedly across
iterations: `initialize substrate -> iteration_1 -> iteration_2 -> ... ->
iteration_T`. This decouples computational depth from parameter count.

> Hypothesis: capability may depend not only on parameter count but on how
> effectively those parameters can be repeatedly applied to an evolving
> internal state.

### Iterative computation — the evaluation protocol this implies

If trained with, say, 4 refinement steps, evaluate at 1, 2, 4, 8, 16, and
possibly more iterations at test time. Does additional inference compute
help? Does it saturate or degrade? Does the model generalize to more
reasoning iterations than seen in training? Does task difficulty correlate
with useful computation depth? (Experiment 005.) Initially the iteration
count is externally controlled — adaptive/learned halting is a later
extension, not part of V0.

## 3. Graph / cluster organization [OPEN, dependent on §1–2]

Cells should eventually be organized into circuits/clusters rather than
interacting via unconstrained global attention:

```text
Cluster A              Cluster B
o--o--o                o--o--o
|\/| /                 |\/| /
o--o--o                o--o--o
   |                       ^
   +----------->-----------+
```

Within a cluster: relatively rich communication, room for local
specialization, repeated local circulation. Between clusters: more
selective/limited-bandwidth communication, avoiding dense all-to-all
interaction. Motivated computationally (efficiency/inductive bias), not
biologically dogmatically:

> Rich local computation combined with selective global communication may be
> more efficient than repeatedly performing dense global interaction.

### Fixed graph before dynamic graph

**Do not** begin with cells that permanently create/delete connections. Start
with a predefined, fixed topology (e.g. 8 clusters x 16 cells = 128 cells
total, or smaller for early runs). Learned edge strengths, dynamic routing,
activity-dependent communication, and structural plasticity are later
extensions only, after the fixed-topology case is understood (Experiment
005; dynamic topology is explicitly out of scope until then).

## 4. Fast dynamic association (later extension, NOT part of CellV0) [OPEN]

Distinguish **slow learned connectivity** (persistent trained parameters,
e.g. `W_ij`) from **fast contextual association** — a temporary relation that
changes while processing the current input, e.g. `A_ij(t)`, with effective
interaction conceptually `W_eff(t) = W + alpha * A(t)`. This would let the
system temporarily associate concept-representations relevant to the current
input (e.g. binding "Bob" <-> "red key" <-> "laboratory" while reading a
passage) without permanently rewriting parameters. **Not required for
CellV0** — tracked here only so it isn't lost; do not implement until the
project reaches this stage explicitly.

## 5. Latent world state [OPEN, dependent on §1–3]

The full system state at reasoning iteration `t`: `Z_t = {h_1(t), ...,
h_N(t)}` — the model's current internal belief/representation. Input
produces `Z_0`; the recurrent graph performs `Z_0 -> Z_1 -> ... -> Z_T`; the
final state decodes to language, classification, regression, actions, etc.
This separates *internal computation* from *output representation*.

### Language as interface, not necessarily the internal thought medium

```text
language / observations -> encoder -> internal distributed state
  -> iterative recurrent reasoning -> updated latent/world state
  -> decoder -> language / action / prediction
```

Different from requiring the entire internal reasoning process to operate
directly through generated text tokens. See `benchmark_plan.md` §"World
representation tests" for how this is operationalized and measured (never
via unfalsifiable claims like "the model understands" — see
`hypotheses.md` §"Operationalizing understanding").

## 6. Diffusion-like iterative refinement (conceptual framing, not a
   commitment to a formal diffusion model) [OPEN]

Instead of `token -> next token -> next token`, the system could perform
`incomplete/noisy latent state -> refinement -> refinement -> ... ->
coherent latent representation`. Input-corruption tasks (e.g. masking facts
in a synthetic-world description and having the model reconstruct the
correct latent/output) could test whether repeated latent refinement is
useful. This is a candidate *task design* for later experiments, not an
architectural commitment.

## 7. What Architecture Specification V0.1 must still define

1. ~~Cell state — shape, dtype, initialization.~~ **SPECIFIED** — see §1
   (`BeliefCell`: `mu`, `evidence`, `uncertainty`; `tau` dormant until V0.1).
2. ~~Aggregation inputs — exactly what enters the belief-aggregation
   operator.~~ **SPECIFIED (x5)** — all five `BeliefLayer` methods take the
   same inputs: incoming `BeliefCell`s, a per-edge content weight `w_ij`, a
   per-edge relevance gate `g_ij`, and a per-output-cell bias. See §1.
3. ~~Aggregation computation.~~ **FIVE CANDIDATES IMPLEMENTED, none
   chosen** — `"reliability"`, `"support_conflict"`, `"precision"`,
   `"normalized_precision"`, and `"scale_stable_precision"` in
   `integration.py`. Experiment 002/003 began selecting among the first
   three (or none); Experiments 004F and 004I added the fourth and fifth to
   address a scaling failure found in `"precision"`.
4. ~~Aggregation parameterization.~~ **SPECIFIED** — per-edge weights
   `w_ij` and `g_ij` (an ordinary dense layer's worth of parameters,
   duplicated for the gate), same for all five methods.
5. Output/message representation — what a cell sends downstream (its full
   `BeliefCell`, or a derived subset?). **SPECIFIED for V0.0**: the full
   `BeliefCell`, unchanged, per the implemented `BeliefLayer.forward`.
6. ~~Parameter count.~~ **SPECIFIED** — `out_cells * (2 * in_cells + 1)`
   per `BeliefLayer` (content weight + relevance logit + bias), identical
   across all three methods, so parameter-matching a baseline is direct.
7. Computational complexity — FLOPs per cell per layer. Same asymptotic
   order as a `Linear` layer (`O(in_cells * out_cells)`), with a small
   constant-factor overhead per method; exact profiling is part of
   Experiment 002/003, not yet measured. [OPEN]
8. Limiting cases — can any aggregation method reduce to an ordinary
   weighted-sum-plus-activation (recovering a plain MLP neuron) under some
   setting? Not yet checked for any of the three. Makes ablations
   tractable — see Experiment 003. [OPEN]
9. Expected advantages (hypotheses to test — draft in §1 above). [OPEN]
10. Expected failure modes (hypotheses to test). [OPEN]
11. Persistence (`tau`) reintroduction rule for V0.1's recurrence — deferred,
    not required for V0.0. [OPEN, deferred]

## 8. Explicitly deferred (do not implement as part of V0)

Everything in this list is a legitimate later research direction, not a
rejected idea — it is deferred to preserve experimental interpretability
(§6 of `research_thesis.md`):

- STDP / biologically realistic spikes / dopamine simulation
- Local-only learning, replay, metaplasticity, structural plasticity
- Weight changes during inference; continual/adaptive learning; multi-timescale
  memory (fast state / working state / episodic state / slow knowledge)
- Dynamic/learned graph topology (before the fixed-topology case is understood)
- Fast contextual association `A_ij(t)` (§4 above)
- Monte Carlo particles / multiple latent hypotheses (Experiment 012 — only
  after the single-trajectory case is well understood)
- Energy-based / MCMC inference (Experiment 013 — only if Experiment 012
  strongly justifies it)
- Novel optimizers or novel training objectives

## 9. Revision history

- Repository initialization: document created as a design-space placeholder;
  no architectural decisions made yet.
- CellV0 state decided: replaced the compartment-based framing of §1 with
  the `BeliefCell` state `(mu, evidence, uncertainty)`, implemented in
  `src/models/architecture_v0/cell.py`. The belief-aggregation operator
  (how N incoming `BeliefCell`s combine into one) remains open — see
  `docs/research_log.md` for the candidate formulas under consideration.
- Belief-aggregation operator: three candidates implemented (not chosen).
  `BeliefLayer` (`src/models/architecture_v0/integration.py`) implements
  `"reliability"`, `"support_conflict"`, and `"precision"` behind one
  switchable interface, plus `BeliefCell.from_observed_features` for
  initializing beliefs from raw input features. Experiment 002/003 will
  compare them against each other and a plain-MLP baseline before any one
  is adopted.
- Fourth aggregation candidate added: `"normalized_precision"` (Experiment
  004F), user-specified to fix a scale-instability Experiment 004D found in
  `"precision"` (`evidence` grows unboundedly with `hidden_cells`). Divides
  `evidence` and the base-uncertainty term by total incoming relevance
  instead of leaving them as raw sums; content-weighting (`alpha`) and the
  disagreement term are untouched. Kept alongside `"precision"`, not a
  replacement for it — see §1 and `docs/research_log.md` ("Experiment
  004F").
- Fifth aggregation candidate added: `"scale_stable_precision"` /
  "CellV0.1" (Experiment 004I), user-specified refinement of
  `"normalized_precision"`. Normalizes `evidence`/base-uncertainty by an
  effective source count (a participation-ratio statistic over incoming
  relevance) instead of raw total relevance, so a long tail of weakly-
  relevant connections can't inflate evidence the way summing raw relevance
  still can under `"normalized_precision"`. Kept alongside both
  `"precision"` and `"normalized_precision"` — see §1 and
  `docs/research_log.md` ("Experiment 004I").
