"""Parameter counting and wall-clock timing -- the cross-cutting efficiency
measurements every experiment must report (docs/benchmark_plan.md "Compute
efficiency", docs/experiment_protocol.md "What every run must record").

A proper FLOP estimator is deliberately not included yet: a meaningful
FLOPs-per-refinement-iteration formula depends on CellV0's computational
complexity, which is not yet specified (docs/architecture_v0.md Sec 7 item
9). Add it once that's defined.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


@dataclass
class _Timer:
    elapsed_seconds: float = 0.0


@contextmanager
def wall_clock() -> Iterator[_Timer]:
    """Usage: `with wall_clock() as timer: ...` then read
    `timer.elapsed_seconds` after the block exits."""
    timer = _Timer()
    start = time.perf_counter()
    try:
        yield timer
    finally:
        timer.elapsed_seconds = time.perf_counter() - start
