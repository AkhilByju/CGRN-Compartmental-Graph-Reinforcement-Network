"""ArchitectureV0 (working name CGRN) -- MOSTLY UNIMPLEMENTED BY DESIGN.

CellV0's state representation is decided and implemented: `cell.py` defines
`BeliefCell`, the belief-state primitive `(mu, evidence, uncertainty)` that
replaces an ordinary neuron's scalar activation.

CellV0's aggregation operator has three competing candidate
implementations, not a final choice: `integration.py` defines
`BeliefLayer(aggregation=...)` for `"reliability"`, `"support_conflict"`,
and `"precision"`. Which (if any) is adopted is an empirical question for
Experiment 002/003 (docs/experiment_protocol.md) -- do not treat any one of
them as settled.

Every other module in this package (`cluster.py`, `graph.py`,
`dynamics.py`, `encoder.py`, `decoder.py`, `model.py`) still raises
`NotImplementedError` and remains unspecified. See docs/architecture_v0.md
and CLAUDE.md Sec 2 before implementing any of them.
"""
