# Architecture V0 — Design Space and Open Questions

**Status: PARTIALLY SPECIFIED.** CellV0's *state* (§1) is decided and
implemented (`src/models/architecture_v0/cell.py`). Its *aggregation
operator* — how a cell combines incoming state into its own — is still
open, along with everything in §2 onward. No code beyond `cell.py` should
be implemented past a stub until the sections marked `OPEN` below are
resolved by the user and this document is updated.

The next concrete research task (not a coding task) is to finalize the
belief-aggregation operator described in §1 — see `docs/research_log.md`
for the candidate formulas currently under consideration.

## 1. CellV0 — the basic computational unit

**State: SPECIFIED. Aggregation: OPEN.**

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

### What's still open: the belief-aggregation operator (the `???` above)

The `BeliefCell` state is settled; **how a cell combines N incoming
`BeliefCell`s (each reaching it through a learned weight `w_ij`) into its
own outgoing `BeliefCell` is not.** This is the actual remaining content of
"CellV0's math" — state shape, dtype, and what's dormant are now fixed.

Candidate direction under consideration (recorded so it isn't lost —
**not frozen**, do not implement until confirmed): each incoming cell
contributes a proposed content `m_ij = w_ij * mu_j` and a support weight
`s_ij` that grows with the sender's evidence and shrinks with its
uncertainty (e.g. `s_ij = |w_ij| * e_j / (u_j + eps)`); the receiving
cell's content is an evidence-weighted combination of the `m_ij` (plus
bias, plus nonlinearity); its evidence accumulates total incoming support;
and its uncertainty is derived from evidence and
agreement/disagreement among incoming messages rather than learned as an
arbitrary extra output. See `docs/research_log.md` for the specific
formulas under consideration and why. **Implementation belongs in
`src/models/architecture_v0/integration.py` and is blocked until a
formulation is confirmed.**

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
conventional MLP-like unit (plain scalar activation) vs. a `BeliefCell`
layer using whichever aggregation operator is confirmed. Once that operator
exists, natural ablation axes include: evidence-weighting on/off,
uncertainty derived from evidence alone vs. evidence+disagreement, and
uncertainty used vs. ignored by downstream cells. Does carrying
evidence/uncertainty help, and for which function classes? Does it just
increase compute? (Experiment 002/003.)

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
with useful computation depth? (Experiment 004.) Initially the iteration
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

Before `integration.py` is implemented, a revision of this file must
specify, precisely enough to implement, the remaining open items:

1. ~~Cell state — shape, dtype, initialization.~~ **SPECIFIED** — see §1
   (`BeliefCell`: `mu`, `evidence`, `uncertainty`; `tau` dormant until V0.1).
2. Aggregation inputs — exactly what enters the belief-aggregation
   operator (which neighboring `BeliefCell`s, which connection parameters,
   whether a bias/nonlinearity is part of the operator). [OPEN]
3. Aggregation computation — the exact function producing the outgoing
   `(mu_i, evidence_i, uncertainty_i)` from incoming `BeliefCell`s. [OPEN]
4. Aggregation parameterization — per-edge weights `w_ij` as in an
   ordinary layer, or something richer? [OPEN]
5. Output/message representation — what a cell sends downstream (its full
   `BeliefCell`, or a derived subset?). [OPEN]
6. Parameter count — closed-form or computed formula per cell/layer. [OPEN]
7. Computational complexity — FLOPs per cell per layer. [OPEN]
8. Limiting cases — can the aggregation operator reduce to an ordinary
   weighted-sum-plus-activation (recovering a plain MLP neuron) under some
   setting? Makes ablations tractable — see Experiment 003. [OPEN]
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
- Monte Carlo particles / multiple latent hypotheses (Experiment 011 — only
  after the single-trajectory case is well understood)
- Energy-based / MCMC inference (Experiment 012 — only if Experiment 011
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
