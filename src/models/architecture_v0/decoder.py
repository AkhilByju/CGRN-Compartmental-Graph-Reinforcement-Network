"""Decoder: final latent state `Z_T` -> task output (regression,
classification, language, actions) -- NOT YET IMPLEMENTED.

See docs/architecture_v0.md Sec 5. Depends on CellV0's state representation
being specified first.
"""

from __future__ import annotations

import torch


class Decoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Decoder is not yet specified -- depends on CellV0's state "
            "representation. See docs/architecture_v0.md Sec 5."
        )
