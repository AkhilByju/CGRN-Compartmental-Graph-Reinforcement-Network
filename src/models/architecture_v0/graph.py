"""Inter-cell/inter-cluster graph communication -- NOT YET IMPLEMENTED.

Selective, bandwidth-limited communication between clusters over a **fixed**
topology (docs/architecture_v0.md Sec 3). Dynamic/learned topology is
explicitly deferred (docs/architecture_v0.md Sec 8) -- do not implement
edge learning, routing, or structural plasticity here.
"""

from __future__ import annotations

import torch


class FixedTopologyGraph(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError(
            "Graph communication is not yet specified. See docs/architecture_v0.md "
            "Sec 3. Only a fixed topology is in scope -- dynamic topology is "
            "deferred, see Sec 8."
        )
