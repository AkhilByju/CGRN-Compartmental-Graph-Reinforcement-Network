"""MLP baseline -- PLACEHOLDER, not yet implemented.

Implementation begins with Experiment 001 (docs/experiment_protocol.md).
Intended interface: a standard `torch.nn.Module` MLP with a configurable
number of layers and hidden width, used as a Track A/B/C baseline and as
the parameter-matched comparison point for CellV0 (Q1/Q3 in
docs/hypotheses.md).
"""

from __future__ import annotations

import torch


class MLPBaseline(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "MLPBaseline is a placeholder. Implement as part of Experiment 001 "
            "(docs/experiment_protocol.md)."
        )
