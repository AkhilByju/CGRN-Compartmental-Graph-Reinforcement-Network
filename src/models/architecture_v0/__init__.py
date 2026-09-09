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

`belief_network.py` implements `BeliefNetwork`, a minimal two-`BeliefLayer`
feedforward stack used ONLY by Experiment 002/003 to test CellV0 in
isolation -- it is NOT `ArchitectureV0` and does not imply `model.py` is
unblocked.

`precision_gain.py` implements **CellV0.2** -- the Conservative Precision-Gain
Cell (`PrecisionGainLayer`, `BeliefNetworkV02`). A SEPARATE CellV0-line
aggregation operator, specified in full by the user and distinct from the
`BeliefLayer` methods: it drops the relevance-gate matrix entirely, carries
one signed connection matrix `V` plus a per-output gain/bias, and its
readout consumes confidence. Same `BeliefCell` state; input belief
`e = 1, u = 0`. See docs/architecture_v0.md Sec 10 and
docs/research_log.md. This does not unblock `model.py` or any other stub.

Every other module in this package (`cluster.py`, `graph.py`,
`dynamics.py`, `encoder.py`, `decoder.py`, `model.py`) still raises
`NotImplementedError` and remains unspecified. See docs/architecture_v0.md
and CLAUDE.md Sec 2 before implementing any of them.
"""
