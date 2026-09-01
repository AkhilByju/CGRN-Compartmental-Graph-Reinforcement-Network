# Architecture V0 — Design Space and Open Questions

**Status: UNSPECIFIED. This document frames the design space; it is not a
finalized specification.** No code in `src/models/architecture_v0/` should
be implemented beyond stubs until the sections marked `OPEN` below are
resolved by the user and this document is updated to `SPECIFIED`.

The next concrete research task (not a coding task) is **Architecture
Specification V0.1**: a mathematical definition of `CellV0` alone. See §7
for exactly what that spec must contain.

## 1. CellV0 — the basic computational unit [OPEN]

Not intended to literally simulate a biological neuron. Inspired by the idea
that biological neurons contain multiple computational compartments
(dendritic branches), a conventional artificial unit approximately performs:

```text
input -> weighted transformation -> nonlinearity -> output
```

`CellV0` should instead conceptually resemble:

```text
                incoming information
          /          |          \
     Compartment A
     Compartment B
     Compartment C
     Compartment D
          \          |          /
                integration
                     |
              central cell state
              /              \
      outgoing message    persistent state
```

Sketch of the (not-yet-finalized) recurrence, using cell `i`, compartment
`b`, time/iteration `t`:

- Persistent state: `h_i(t)`
- Compartments: `D_i1, D_i2, ..., D_iB`
- A generic compartment: `d_i,b(t) = f_b(local input, current cell state,
  incoming messages, possibly compartment state)`
- Integration: `u_i(t) = S(d_i,1, d_i,2, ..., d_i,B)`
- State update: `h_i(t+1) = Update(h_i(t), u_i(t))`

**None of `f_b`, `S`, or `Update` are chosen yet.** Choosing them is the
Architecture Specification V0.1 task.

### Why richer cells might matter (hypothesis, not a claim)

Not simply "more computation per unit is better" — that isn't guaranteed.
The hypothesis is that structured internal computation may provide useful
representational biases that let fewer units express more complex
interactions. Possible advantages: representation efficiency, specialized
local processing, richer feature interactions, stronger nonlinear
composition, parameter efficiency, more useful recurrent state behavior.
Possible disadvantages: harder optimization, unnecessary complexity, slower
execution, redundant computation, poor hardware utilization, or no
meaningful advantage over ordinary MLP blocks. **Experimentally determined,
not assumed** — see Experiment 002/003 in `experiment_protocol.md`.

### Test the cell before the graph

Before any graph complexity, compare parameter-matched variants: conventional
MLP-like unit, 1-compartment, 2-compartment, 4-compartment, 8-compartment
cells. Does compartment structure help, and if so up to what point? Does it
just increase compute? Does it help specific function classes, sample
efficiency, or optimization stability? (Experiment 002/003.)

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

## 7. What Architecture Specification V0.1 must define

Before `cell.py` is implemented, a document (a revision of this file) must
specify, precisely enough to implement:

1. Cell state — shape, dtype, initialization.
2. Compartment inputs — what information enters each compartment.
3. Compartment computation — the exact function each compartment computes.
4. Compartment parameterization — do compartments share transformations?
5. Integration mechanism — exactly how compartment outputs combine (`S`).
6. State update rule — exactly how `h_i(t+1)` is computed (`Update`).
7. Output/message representation — what a cell can send to others.
8. Parameter count — closed-form or computed formula per cell.
9. Computational complexity — FLOPs per cell per iteration.
10. Limiting cases — can CellV0 reduce to an ordinary MLP or GRU/LSTM cell
    under some parameter setting? (This makes ablations tractable — see
    Experiment 003.)
11. Expected advantages (hypotheses to test).
12. Expected failure modes (hypotheses to test).

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
