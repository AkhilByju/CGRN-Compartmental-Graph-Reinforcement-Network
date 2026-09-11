# Architecture V2 — Belief Dendritic Network (implemented, unevaluated)

Status as of this writing: **implemented and tested, not yet benchmarked.**
Per `CLAUDE.md` Sec 2's `docs/architecture_v0.md`/`docs/architecture_v1.md`
carve-out, this line was specified by the user in full mathematical detail,
in one message, together with an explicit implementation request — the same
bar `docs/architecture_v1.md` was held to. Nothing here is agent-invented
math.

## 0. Relationship to Architecture V0 and V1

Independent of both. It reuses exactly two things from `architecture_v0`:
`BeliefCell` (`src/models/architecture_v0/cell.py`) as the `(mu, e, u)`
state container, and `effective_precision` (`src/models/architecture_v0/
precision_gain.py`) — `pi = e / (1 + e u)`. It does not import, wrap, or
modify `architecture_v1` at all. Where V1 organizes many identical cells
into a dynamic, input-dependent, learned topology, V2 keeps the topology
**fixed** and instead makes each individual neuron structurally richer — see
Sec 6.

## 1. Core hypothesis and the gap it targets

CellV0.3 (`docs/architecture_v0.md` Sec 10) gave one flat belief-fusion
operator: `N` sources in, one belief out. Paper A (Phase 1–3,
`docs/research_log.md`) tested that operator directly against MLP/CellV0.3
baselines under missingness and got a real-data near-tie with a stratum
advantage under high missingness — worth recording, not worth chasing
further as a flat operator alone (`experiments/paper_a/real_reliability/
README.md`).

The research question V2 asks is architectural, not "tune V0.3 harder":
**what if a single neuron is not one belief-fusion computation, but several,
composed hierarchically?** The literature survey behind this (user's
research pass, not independently re-verified by the agent) found:

- Uncertainty-aware state-space models (KOSS, Kalman-optimal gains
  controlling selective state propagation) already occupy the "latent
  uncertainty gates a sequence model" space.
- Predictive-coding architectures already propagate confidence alongside
  prediction error through a cortical hierarchy.
- Dendritic ANNs (2025, restricted-input branches + scalar somatic
  aggregation) show parameter efficiency and robustness gains from
  structure alone — but their dendrites are ordinary scalar point neurons.
- Active-dendrite / contextual-disinhibition networks gate a neuron's
  activation using **external** context or inhibition signals, not a
  belief the branch computed about its own reliability.

The identified gap: no architecture where each dendritic compartment is
itself belief-valued (carries content, support, conflict, and precision)
and the soma routes among dendrites purely by the reliability those
dendrites computed internally — no router, no external context, no top-k.

## 2. The mechanism

One neuron is `B` sparse belief dendrites feeding one belief soma. Both
levels run the *same* CellV0.3-style conservative belief fusion — recursion,
not a new primitive.

### 2.1 Input belief

`mu = x`; `e = reliability` if given, else `e = 1`; `u = 0`
(`BeliefDendriteNetwork.input_belief`).

### 2.2 Dendritic (branch) belief fusion

Source population `(mu_j, e_j, u_j)`, usable precision
`pi_j = e_j / (1 + e_j u_j)`. Branch `b` reads a fixed set of `K` sources,
signed synaptic weights `V_branch[b, :]`:

```
A_bj = |V_bj| / (sum_k |V_bk| + eps)      unsigned, row sums to ~1
S_bj =  V_bj  / (sum_k |V_bk| + eps)      signed

e_branch_b   = sum_j A_bj pi_j                                # convex comb -> <= max_j pi_j
consensus_b  = (sum_j S_bj pi_j mu_j) / e_branch_b
second_b     = (sum_j A_bj pi_j mu_j^2) / e_branch_b
u_branch_b   = max(0, second_b - consensus_b^2)
pi_branch_b  = e_branch_b / (1 + e_branch_b u_branch_b)       # <= e_branch_b
mu_branch_b  = tanh(gamma_branch_b consensus_b sqrt(pi_branch_b) + bias_branch_b)
```

`gamma_branch = softplus(gain_raw_branch)`, inverse-softplus-initialized so
`gamma_branch_b = sum_k |V_bk|` at init (the CellV0.2/0.3 amplitude
convention, unmodified).

### 2.3 Somatic belief fusion over dendrites

Identical fusion, one level up: soma `i`'s `B` dendrites are now the
"sources", cable weights `V_cable[i, :]` in place of `V_branch`:

```
A_cable_ib = |V_cable_ib| / (sum_b |V_cable_ib| + eps)
S_cable_ib =  V_cable_ib  / (sum_b |V_cable_ib| + eps)

e_i          = sum_b A_cable_ib pi_branch_ib
consensus_i  = (sum_b S_cable_ib pi_branch_ib mu_branch_ib) / e_i
second_i     = (sum_b A_cable_ib pi_branch_ib mu_branch_ib^2) / e_i
u_i          = max(0, second_i - consensus_i^2)
pi_i         = e_i / (1 + e_i u_i)
mu_i         = tanh(gamma_i consensus_i sqrt(pi_i) + bias_i)
```

Output belief: `(mu_i, e_i, u_i)`, ready to feed the next
`BeliefDendriteLayer` or a linear readout (`mu` only — precision is already
folded into the activation, same convention as `BeliefNetworkV03`).

### 2.4 Routing is not a separate mechanism

Normalized somatic branch contribution:

```
r_b = A_cable_ib * pi_branch_ib / e_i,      sum_b r_b ~= 1
```

There is no router parameter anywhere (`layer.named_parameters()` is
exactly `{V_branch, gain_raw_branch, bias_branch, V_cable, gain_raw_soma,
bias_soma}` — checked directly in
`tests/test_architecture_v2_belief_dendrite.py`). A branch's influence is
a pure consequence of its own computed precision, tested directly by
degrading only one branch's sources and checking `r_b` for that branch (and
only that branch) falls (`test_degrading_one_branchs_sources_only_lowers_
that_branchs_precision_and_influence`).

## 3. Two conflict scales

- **Within-dendrite conflict**: disagreement among a branch's `K` sources
  raises `u_branch`, lowers `pi_branch`.
- **Between-dendrite conflict**: disagreement among a soma's `B` branches
  raises `u_soma` (`u_i` above), lowers `pi_i`.

Both directions are tested directly
(`test_within_branch_disagreement_raises_u_branch_and_lowers_pi_branch`,
`test_between_branch_disagreement_raises_u_soma_and_lowers_pi_soma`).

## 4. Properties that hold by construction (tested, not assumed)

- **Hierarchical conservative precision**:
  `pi_branch <= e_branch <= max_j pi_j` (convex combination of source
  precisions) and `pi_i <= e_i <= max_b pi_branch_b`, therefore
  `pi_i <= max_j pi_j` through both levels — confidence cannot spontaneously
  increase past the most reliable original source, even through two fusion
  steps.
- **Replication invariance**, at both levels: duplicating a branch's sources
  and their weights leaves that branch's belief unchanged; duplicating an
  equivalent dendrite (same sources, same `V_branch` row) and its cable
  weight leaves the soma's belief unchanged. Redundant information cannot
  manufacture confidence at either level.
- **Automatic branch suppression**: as `pi_branch -> 0`, that branch's `r_b`
  contribution to the somatic consensus approaches zero — no explicit gate.

## 5. Parameter count and complexity (Sec J, `param_count.py`)

One `BeliefDendriteLayer(H, B, K)` (identical for `ScalarDendriteLayer`):

```
params(H, B, K) = H*B*K (V_branch) + 2*H*B (branch gain/bias)
                + H*B (V_cable) + 2*H (soma gain/bias)
                = H * (B*(K+3) + 2)
```

Checked against the real `nn.Module` count (not assumed) in
`tests/test_architecture_v2_param_count.py`; `solve_hidden_width_for_budget`
inverts the (affine-in-`H`, for fixed `B`/`K1`/`K2`) two-layer formula to
find the largest uniform hidden width fitting a parameter budget.

Compute scales `O(batch * H * B * K)`, not `O(batch * H * input_dim)`: the
only per-batch tensor materialized is the gathered `[batch, H*B, K]`
activation (`DendriticConnectivity.gather` — one `index_select` + reshape).
If `B*K << input_dim`, a dendritic layer is *cheaper* than a dense one, and
because parameter count no longer scales with a dense `H x input_dim`
connection matrix, a network can afford far more somas than a
parameter-matched dense layer could — the problem noted in earlier
population-scale attempts, where a dense input projection consumed the
whole parameter budget before the population could grow past a handful of
cells.

## 6. Why a fixed dendritic topology is not the fixed-graph failure

This project already ruled out a fixed *cell population graph*
(`docs/architecture_v0.md` "Fixed graph before dynamic graph") — cells as
emergent, ambiguous units whose connectivity should plausibly be learned or
grown. A dendrite is different: it is a **structural component of one
neuron**, not a claim about which neurons should talk to which. Real
dendritic compartments have fixed physical receptive fields; which synapses
land on which branch is developmental, not the thing under test here. So
`DendriticConnectivity`'s indices are frozen, non-trainable buffers by
design (Sec 2), and *only* each branch's computed reliability — never its
wiring — is allowed to vary per input. Everything dynamic about this
architecture is confined to belief propagation, not topology.

## 7. Distinction from CellV0.3

Mechanically identical fusion formula, applied twice (Sec 2.2 = Sec 2.3 =
`ConflictNormalizedLayer`'s equations, `src/models/architecture_v0/
conflict_normalized.py`) — CellV0.3 is the base case of this recursion with
`B = 1` dendrite per soma reading all `input_dim` sources densely. What V2
adds is structural: many small sparse branches per soma instead of one
dense fan-in, so a corrupted or unreliable subset of the input is
*localized* to the branches that actually read it, rather than diluted
across one flat consensus over every source at once (Sec 8's localization
diagnostic tests this directly).

## 8. Distinction from the scalar dendritic controls (Sec G)

Both controls share the **identical** sparse topology and parameter shapes
as `BeliefDendriteLayer` (`DendriticConnectivity` objects are passed in, not
rebuilt, so the comparison is apples-to-apples — see
`test_belief_and_scalar_networks_can_share_identical_topology`):

- `ScalarDendriteNetwork`: no `(e, u)` at all — ordinary sparse tanh MLP with
  the same branch/soma structure. Isolates *dendritic structure* from
  *belief propagation*: if this alone beats a dense MLP under corruption,
  sparsity/locality is doing the work, not belief.
- `ScalarDendriteReliabilityGated`: reliability is multiplied into `x`
  **once**, at the input (`mu_input = reliability * x`), then fed through
  the same scalar dendritic network — proven algebraically identical to
  `ScalarDendriteNetwork(c * x)` on shared parameters
  (`test_reliability_gated_matches_scalar_network_on_pre_gated_input`).
  Isolates *hierarchical* belief propagation from a single input-side gate:
  if `BeliefDendriteNetwork` doesn't beat this, propagating belief through
  every branch/soma boundary isn't earning its cost over gating once.

## 9. Distinction from the dynamic-graph / structural-plasticity ideas this project ruled out

No mechanism here does any of: dynamic/learned topology, structural
plasticity, top-k or softmax routing, mixture-of-experts gating, attention,
ORFF/RBF fields, external context vectors, k-WTA, or an auxiliary
reliability loss (`CLAUDE.md`'s MUST-NOT list, `docs/architecture_v1.md`
Sec 11–17's ruled-out and superseded variants). The only thing that varies
per input is precision flowing through a **fixed** computation graph — see
Sec 6.

## 10. Implementation map

| Piece | File |
|---|---|
| `DendriticConnectivity` (`balanced_random`, `local_2d`, `gather`) | `src/models/architecture_v2/belief_dendrite.py` |
| `BeliefDendriteLayer`, `BranchSomaDiagnostics` | same |
| `BeliefDendriteNetwork` | same |
| `ScalarDendriteLayer`, `ScalarDendriteNetwork`, `ScalarDendriteReliabilityGated` | same |
| `make_connectivity_pair` (default two-stage topology) | same |
| Parameter-count formula + budget solver | `src/models/architecture_v2/param_count.py` |
| Tests (connectivity, mechanism, scalar controls, param count) | `tests/test_architecture_v2_*.py` |

Frozen benchmark (MNIST/Fashion-MNIST, two structured-corruption families,
six models, three seeds) lives under `experiments/belief_dendrite/` — see
that directory's `README.md` for the protocol and, once run, the results.
This document will get a Sec 11 "Results" appended after that frozen run
completes, per Sec Q's "do not redesign after the first frozen result" rule
— no V2.1 spun up automatically off this evidence, same discipline Paper A
closed with.

## 11. Revision history

- **2026-09-11**: Specified by the user in full mathematical detail (one
  message) and implemented: `DendriticConnectivity` (both connectivity
  modes), `BeliefDendriteLayer`/`Network`, the scalar dendritic controls,
  the parameter-count formula and solver, and the full Sec H test suite (50
  tests). Not yet benchmarked — `experiments/belief_dendrite/` is next.
