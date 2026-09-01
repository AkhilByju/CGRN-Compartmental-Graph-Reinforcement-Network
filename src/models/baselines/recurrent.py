"""GRU/LSTM recurrent baseline -- PLACEHOLDER, not yet implemented.

Implementation begins with Experiment 001 (docs/experiment_protocol.md).
Intended interface: a standard `torch.nn.Module` wrapping `nn.GRU` or
`nn.LSTM`, used as a Track C baseline and as a comparison point for
ArchitectureV0's iterative-refinement dynamics (H2 in docs/hypotheses.md).
"""

from __future__ import annotations

import torch


class RecurrentBaseline(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "RecurrentBaseline is a placeholder. Implement as part of Experiment 001 "
            "(docs/experiment_protocol.md)."
        )
