"""Tiny Transformer baseline -- PLACEHOLDER, not yet implemented.

Implementation begins with Experiment 001 (docs/experiment_protocol.md).
Intended interface: a standard `torch.nn.Module` encoder- or decoder-only
Transformer at debug/small-research scale (docs/experiment_protocol.md
"Model scales"), used as the primary Track C baseline and as the
parameter-/compute-matched comparison point throughout
docs/hypotheses.md.
"""

from __future__ import annotations

import torch


class TransformerBaseline(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "TransformerBaseline is a placeholder. Implement as part of Experiment 001 "
            "(docs/experiment_protocol.md)."
        )
