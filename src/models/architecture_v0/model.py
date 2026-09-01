"""ArchitectureV0: top-level model assembling encoder -> persistent
computational substrate (cells + clusters + graph + iterative dynamics) ->
decoder -- NOT YET IMPLEMENTED.

Do not assemble this until every submodule in this package is specified per
docs/architecture_v0.md Sec 7 ("What Architecture Specification V0.1 must
define"). See CLAUDE.md Sec 2.
"""

from __future__ import annotations

import torch


class ArchitectureV0(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "ArchitectureV0 cannot be assembled until CellV0 and its surrounding "
            "components are specified. See docs/architecture_v0.md and CLAUDE.md Sec 2."
        )
