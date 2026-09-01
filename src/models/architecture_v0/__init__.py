"""ArchitectureV0 (working name CGRN) -- MOSTLY UNIMPLEMENTED BY DESIGN.

CellV0's state representation is now decided and implemented: `cell.py`
defines `BeliefCell`, the belief-state primitive `(mu, evidence,
uncertainty)` that replaces an ordinary neuron's scalar activation (see
docs/architecture_v0.md Sec 1 and docs/research_log.md).

Every other module in this package still raises `NotImplementedError`. In
particular, how a layer combines N incoming `BeliefCell`s into one outgoing
`BeliefCell` -- the belief-aggregation operator that will live in
`integration.py` and become `BeliefLayer` -- is still being decided. Do not
implement it speculatively; see CLAUDE.md Sec 2.
"""
