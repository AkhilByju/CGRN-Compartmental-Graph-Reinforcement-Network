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

**§9 (below) is a second, separate CellV0-line cell — CellV0.2, the
Conservative Precision-Gain Cell.** User-specified in full mathematical
detail (2026-09-09) and implemented in
`src/models/architecture_v0/precision_gain.py` (`PrecisionGainLayer`,
`BeliefNetworkV02`). It is **not** a sixth `BeliefLayer` aggregation method:
it removes the relevance-gate matrix entirely. It reuses the `BeliefCell`
state but with input belief `e = 1, u = 0`. Like CellV1, this is the user's
own specification, not an agent-invented aggregation rule — the "do not add
a further candidate" rule for §1's `BeliefLayer` is unchanged and still
applies to `BeliefLayer`. CellV0.2 has been evaluated on the frozen Paper-A
Phase-1 protocol (`docs/research_log.md`, 2026-09-09).

**§10 (below) is a third, separate CellV0-line cell — CellV0.3, the
Conflict-Normalized Belief Cell.** User-specified in full mathematical
detail (2026-09-09) and implemented in
`src/models/architecture_v0/conflict_normalized.py`
(`ConflictNormalizedLayer`, `BeliefNetworkV03`). Same one-signed-matrix
parameterization as CellV0.2, but CellV0.2's population-relative precision
gain `2 pi / (pi + mean pi)` is removed entirely and each cell's
`sqrt(precision)` is folded into its *own* activation. Also the user's own
specification, delivered with an explicit implement-and-evaluate
instruction; CellV0.1's and CellV0.2's equations and recorded results are
untouched.

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

## 9. CellV0.2 — Conservative Precision-Gain Cell (separate line, IMPLEMENTED)

**Status: SPECIFIED (user, 2026-09-09) and IMPLEMENTED**
(`src/models/architecture_v0/precision_gain.py`). Evaluated on the frozen
Paper-A Phase-1 protocol — see `docs/research_log.md` (2026-09-09) and
`experiments/paper_a/phase1_v02_results.md`.

CellV0.2 is a second CellV0-line aggregation operator, specified in full by
the user across one conversation turn. It is **not** a sixth `BeliefLayer`
method (§1) — it removes the separate relevance-gate matrix (`a_ij` / `g_ij`)
entirely — so it lives in its own module (`PrecisionGainLayer`,
`BeliefNetworkV02`), not behind `BeliefLayer(aggregation=...)`. It reuses the
`BeliefCell` state `(mu, e, u)` but initializes the *input* belief
`e = 1, u = 0` (not `e = u = 1`).

### 9.1 Per-layer computation

Source population `(mu_j, e_j, u_j)`, output cell `i`. `eps` is used only as a
denominator floor to prevent division by zero (`e, u >= 0` already makes
`1 + e u >= 1`, so effective precision needs none).

```text
effective precision    pi_j     = e_j / (1 + e_j u_j)
population mean         pi_bar   = mean_j pi_j                    (NOT detached)
relative gain          r_j      = 2 pi_j / (pi_j + pi_bar)        in (0, 2); = 1 iff all pi equal
gain-modulated message x_j      = r_j * mu_j

abs_V   = |V|                              V has shape [out_cells, in_cells]
row_l1_i = sum_j |V_ij|
gamma_i  = softplus(gain_raw_i)            > 0

signed consensus       c_i      = (sum_j x_j V_ij) / row_l1_i           # GEMM 1
content                mu_i     = tanh(gamma_i c_i + b_i)
inherited support      e_i      = (sum_j pi_j |V_ij|) / row_l1_i        # GEMM 2   (convex comb -> <= max_j pi_j)
second moment          s_i      = (sum_j x_j^2 |V_ij|) / row_l1_i       # GEMM 3
disagreement           u_i      = max(0, s_i - c_i^2)                   # |V|-weighted variance of sign(V_ij) x_j

next-layer precision   pi_i     = e_i / (1 + e_i u_i)                   # <= e_i
```

Three trainable parameter objects per layer, and no gate matrix:

| Parameter | Shape | Role |
|---|---|---|
| `V` | `[out_cells, in_cells]` | signed connection directions |
| `gain_raw` | `[out_cells]` | pre-softplus output amplitude `gamma` |
| `bias` | `[out_cells]` | |

Per-layer parameter count: **`out_cells * (in_cells + 2)`** — vs CellV0.1's
`out_cells * (2 * in_cells + 1)`.

### 9.2 Initialization

`V` uses the project's standard linear-layer init (Kaiming-uniform,
`a = sqrt(5)`). `gain_raw` is set by **inverse-softplus so that
`gamma_i = ||V_i||_1` at init**. Two consequences the tests pin
(`tests/test_precision_gain.py`):

1. **Neutral-confidence reduction.** On an input with `e = 1, u = 0`
   (`pi = 1`, `r = 1`, `x = mu`), the layer's content path reduces
   *exactly* to `tanh(F.linear(mu, V, b))`.
2. **Linear-row expressivity.** Setting `gamma_i = ||V_i||_1` makes the
   normalized `(V, gamma)` pair represent any conventional linear weight
   row.

### 9.3 Properties (all in `tests/test_precision_gain.py`)

- `e_i` is a convex combination of source precisions → `e_i <= max_j pi_j`
  and `pi_i <= e_i` ("conservative").
- Uniform replication invariance: duplicating the source population and the
  matching columns of `V` (×2, ×4, ×8, ×16) leaves `(mu_i, e_i, u_i)`
  unchanged.
- Row-scaling invariance: multiplying a whole row of `V` by a positive
  constant (with `gamma` fixed) changes neither the normalized consensus,
  `e_i`, nor `u_i`.
- Conflicting gain-modulated messages raise `u_i` and therefore lower the
  effective output precision, at fixed inherited support.
- Every layer forward is three GEMMs — **no `(batch, out_cells, in_cells)`
  edge tensor is materialized** (unlike `BeliefLayer`'s broadcast).

### 9.4 `BeliefNetworkV02`

`input → PrecisionGainLayer → PrecisionGainLayer → confidence-scaled linear
readout`. Mirrors CellV0.1's `BeliefNetwork` except the readout **consumes
confidence**: it is applied to `relative_gain(final_precision) * final_mu`,
not `final_mu`. The readout does not emit a belief state. Hidden width is
fitted to the same Phase-1 parameter budget as CellV0.1; because CellV0.2
spends one parameter per connection instead of two, that budget buys
~1.5× the hidden cells (reported, not equalized).

### 9.5 What is NOT changed

No learned gates, no auxiliary losses, no learned temperature/coefficient/
exponent, no CellV1 mechanisms, no tuning of the formulas. CellV0.1's
equations, parameters, and recorded Phase-1 results are untouched.

## 10. CellV0.3 — Conflict-Normalized Belief Cell (separate line, IMPLEMENTED)

**Status: SPECIFIED (user, 2026-09-09), IMPLEMENTED and EVALUATED**
(`src/models/architecture_v0/conflict_normalized.py`). The frozen Paper-A
Phase-1 result is in `docs/research_log.md` (2026-09-09) and
`experiments/paper_a/phase1_v03_results.md`: CellV0.3 removes CellV0.2's
absolute-confidence cancellation and its output precision genuinely varies
example/cell-wise (CoV 0.15–0.28, vs CellV0.2's near-constant relative
gain), and on the image tasks training drives the confidence scale
`sqrt(pi_out)` down to ~0.57 to attenuate layer 2 — but the headline
accuracy is **indistinguishable from CellV0.2's** on all 7 datasets. Making
confidence causally load-bearing this way bought no accuracy on the frozen
screen. No redesign (Sec 10.6).

CellV0.3 is a **third** CellV0-line aggregation operator, specified in full
by the user. It keeps CellV0.2's parameterization exactly — one signed `V`
`[out, in]` plus per-output `gain_raw` / `bias`, no relevance-gate matrix —
so it lives in its own module (`ConflictNormalizedLayer`,
`BeliefNetworkV03`), not behind `BeliefLayer(aggregation=...)`. It reuses the
`BeliefCell` state `(mu, e, u)` with input belief `e = 1, u = 0`, but reads
`e` as *inherited support* and `u` as an *internal conflict/disagreement*
state — **not** calibrated predictive uncertainty.

### 10.1 Motivation

CellV0.2's relative precision gain `2 pi / (pi + mean_k pi_k)` removes
*absolute* confidence: if every source's precision is scaled down by the
same factor, the gain stays exactly `1` and nothing downstream can tell the
population became less reliable. CellV0.3 removes that population-relative
normalization. Each cell forms a precision-weighted consensus, measures the
conflict among its sources, derives its own usable precision, and uses that
precision *inside its own activation in the same forward step* — so
evidence/conflict is causally relevant, not passive metadata.

### 10.2 Per-layer computation

Source population `(mu_j, e_j, u_j)`, output cell `i`. `eps` is a
denominator/clamp floor for numerical safety only.

```text
usable precision       pi_j    = e_j / (1 + e_j u_j)                     (no population mean; no relative gain)

abs_V   = |V|                                    V has shape [out_cells, in_cells]
row_l1_i = max(sum_j |V_ij|, eps)
A_ij    = |V_ij| / row_l1_i                      unsigned structural weights, sum_j A_ij ~ 1
S_ij    = V_ij  / row_l1_i                       signed content weights

inherited support      e_i     = sum_j A_ij pi_j                  # GEMM  (convex comb -> <= max_j pi_j)
signed consensus       c_i     = (sum_j S_ij pi_j mu_j) / e_i     # GEMM, safe denominator
second moment          s_i     = (sum_j A_ij pi_j mu_j^2) / e_i   # GEMM, safe denominator
conflict               u_i     = max(0, s_i - c_i^2)              # A-weighted variance of sign(V_ij) mu_j
output precision       pi_i    = e_i / (1 + e_i u_i)              # <= e_i <= max_j pi_j
confidence scale       k_i     = sqrt(pi_i)
content                mu_i    = tanh(gamma_i c_i k_i + b_i),  gamma_i = softplus(gain_raw_i)
```

Three trainable parameter objects per layer, and no gate matrix:

| Parameter | Shape | Role |
|---|---|---|
| `V` | `[out_cells, in_cells]` | signed connection directions |
| `gain_raw` | `[out_cells]` | pre-softplus output amplitude `gamma` |
| `bias` | `[out_cells]` | |

Per-layer parameter count: **`out_cells * (in_cells + 2)`** — identical to
CellV0.2's.

The dense forward is three GEMMs — `(pi*mu) @ S.T`, `pi @ A.T`,
`(pi*mu^2) @ A.T` — and **no `(batch, out_cells, in_cells)` edge tensor** is
materialized. Gradients from the task loss flow through
`mu_out -> pi_out -> e_out/u_out -> input e/u` as well as through the content
path; `pi_out` is not detached and not normalized against the rest of the
population.

### 10.3 Initialization

`V` uses the project's standard linear-layer init (Kaiming-uniform,
`a = sqrt(5)`). `gain_raw` is set by inverse-softplus so `gamma_i = ||V_i||_1`
at init — the same amplitude convention as CellV0.2. **Unlike CellV0.2**, a
neutral-confidence input does *not* generically reduce the layer to
`tanh(F.linear(mu, V, b))`: `sqrt(pi_i) = 1` requires `pi_i = 1`, i.e.
`e_i = 1` *and* `u_i = 0` (zero conflict). The zero-conflict, unit-precision
case does reduce exactly (`tests/test_conflict_normalized.py`).

### 10.4 Properties (all in `tests/test_conflict_normalized.py`)

- `e_i` is a convex combination of source precisions → `e_i <= max_j pi_j`
  and `pi_i <= e_i` ("conservative"); a layer cannot manufacture confidence
  beyond its strongest source.
- Perfectly-agreeing signed messages → `u_i ≈ 0` → `pi_i ≈ e_i`.
- With consensus held fixed, more disagreement raises `u_i`, lowers `pi_i`,
  and shrinks the confidence-scaled activation.
- **Absolute-confidence sensitivity** (the property CellV0.2 lost):
  uniformly lowering every source's precision — same `mu`, same relative
  precision pattern, same weight structure — produces a strictly lower
  `sqrt(pi_out)` and a different activation. `pi = [1,1,1]` and
  `pi = [0.1,0.1,0.1]` do not give the same output.
- Uniform replication invariance ×2/×4/×8/×16 (float64), positive
  V-row-scaling invariance of the normalized quantities.

### 10.5 `BeliefNetworkV03`

`input → ConflictNormalizedLayer → ConflictNormalizedLayer → linear readout`.
The readout consumes `final_mu` **directly** — it is *not* re-scaled by
precision (contrast CellV0.2), because each CellV0.3 cell has already folded
its own output precision into its activation. Hidden width is fitted to the
same Phase-1 parameter budget as CellV0.1/CellV0.2; the resulting (larger)
hidden-cell count matches CellV0.2's and is reported, not equalized.

### 10.6 What is NOT changed

No learned gates, no auxiliary/calibration losses, no learned
temperature/exponent, no precision floor beyond numerical safety, no
residual paths, no CellV1 mechanisms, no per-dataset tuning. The `sqrt`
exponent is fixed. CellV0.1's and CellV0.2's equations, parameters, and
recorded results are untouched.

## 11. Revision history

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
- CellV0.2 — Conservative Precision-Gain Cell (2026-09-09): a **separate**
  CellV0-line cell (new §9), user-specified in full and implemented in
  `src/models/architecture_v0/precision_gain.py`. Drops the relevance-gate
  matrix; carries one signed `V` plus per-output `gain_raw`/`bias`;
  effective precision `e/(1+e u)` drives a relative-gain modulation of each
  source message; `e` propagates as a `|V|`-weighted convex combination
  (conservative) and `u` as the `|V|`-weighted variance of the signed
  messages. `gain_raw` inverse-softplus-initialized so the layer reduces to
  `tanh(F.linear(mu, V, b))` at neutral confidence. Not a `BeliefLayer`
  method — it has its own `PrecisionGainLayer` / `BeliefNetworkV02`.
  Evaluated on the frozen Paper-A Phase-1 protocol (`docs/research_log.md`,
  2026-09-09). CellV0.1 and the `BeliefLayer` methods are unchanged.
- CellV0.3 — Conflict-Normalized Belief Cell (2026-09-09): a **third**
  separate CellV0-line cell (new §10), user-specified in full and
  implemented in `src/models/architecture_v0/conflict_normalized.py`
  (`ConflictNormalizedLayer` / `BeliefNetworkV03`). Same
  one-signed-matrix parameterization as CellV0.2, but CellV0.2's
  population-relative gain `2 pi / (pi + mean pi)` is removed entirely:
  each cell forms a precision-weighted signed consensus, takes the
  A-weighted variance of the signed messages as its conflict `u`,
  propagates `e` conservatively (`e_out = sum_j A_ij pi_j`), derives
  `pi_out = e_out / (1 + e_out u_out)`, and folds `sqrt(pi_out)` into its
  own `tanh` activation — so *absolute* confidence is causally
  load-bearing. Readout consumes `mu` directly (not re-scaled). Evaluated
  on the frozen Paper-A Phase-1 protocol (`docs/research_log.md`,
  2026-09-09). CellV0.1, CellV0.2, and the `BeliefLayer` methods are
  unchanged.
