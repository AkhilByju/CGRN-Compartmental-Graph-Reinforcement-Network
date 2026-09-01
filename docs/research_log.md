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
