"""Paper A -- Phase 2: reliability / corruption benchmark for CellV0.3.

Phase 1 asked whether the frozen CellV0.3 primitive deserves a paper on clean
public benchmarks. Phase 2 asks the one question that motivates the CellV0.3
line at all:

    When input information has heterogeneous and *known* reliability, does
    explicitly propagating reliability/conflict through CellV0.3 give a useful
    inductive bias beyond conventional networks that receive the same
    corrupted observations and the same reliability information?

This package only *composes* the frozen Phase-1 infrastructure
(`experiments/paper_a/datasets.py`, `src/training`, `src/evaluation`,
`src/utilities`) with a deterministic corruption layer and four
parameter-matched model families. It does **not** modify CellV0.1 / CellV0.2 /
CellV0.3 or any Phase-1 result. For Phase 2, CellV0.3's *input belief* is
initialized `mu = x_corrupted, e = reliability, u = 0` -- that is input data,
not an architecture change (`docs/architecture_v0.md` Sec 10 leaves the input
belief `e = 1, u = 0` only because Phase 1 had no reliability signal to pass).

See `README.md` and the committed report `reliability_results.md`.
"""
