"""Encoder: input/observations -> initial distributed state `Z_0` -- NOT YET
IMPLEMENTED.

See docs/architecture_v0.md Sec 5 ("Latent world state" / "Language as
interface"). Depends on CellV0's state representation being specified
first.
"""

from __future__ import annotations

import torch


class Encoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Encoder is not yet specified -- depends on CellV0's state "
            "representation. See docs/architecture_v0.md Sec 5."
        )
