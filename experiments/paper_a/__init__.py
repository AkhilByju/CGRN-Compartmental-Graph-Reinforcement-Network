"""Paper A -- CellV0.1 standalone validation, Phase 1.

A self-contained public-benchmark harness for evaluating the frozen CellV0.1
primitive (`BeliefNetwork` with `scale_stable_precision` aggregation --
`src/models/architecture_v0/`) against a parameter-matched MLP and a wider
state-count-matched MLP control, plus two mechanistic checks (a deterministic
duplication-invariance experiment and a train-time fixed-confidence ablation).

This package only *composes* already-existing generic infrastructure
(`src/training`, `src/evaluation`, `src/utilities`) with public-dataset
loaders; it does not touch CellV0.1's equations (CLAUDE.md Sec 2, and the
Paper-A task's "CellV0.1 implementation is FROZEN" rule). See `README.md`.
"""
