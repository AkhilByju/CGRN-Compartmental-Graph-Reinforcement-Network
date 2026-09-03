"""Persistent structural substrate for CellV1.5 (docs/architecture_v1.md
§16): a sparse, slowly-learned synaptic graph, separate from the fast
per-input functional gate `a_ij(t)` (reused unmodified from
`learned_association.py::AssociationFunction`, wired in `structural_dynamics.py`).

Three pieces live here, none of them the classic "one big learned tensor"
shape this codebase otherwise uses:

- `StructuralAddress` -- the *learned* part: a persistent, input-
  independent per-cell "structural identity" `s_i` (contrast with `z`,
  which changes with the current belief state every refinement step),
  plus the shared `W_out`/`W_in` projections that turn a pair of
  addresses into an edge weight `w_ij = tanh(q_i . k_j / sqrt(d_s))`
  (§16.3). `w_ij` is never stored -- recomputed every forward pass for
  whichever edges currently exist.
- `EdgeRegistry` -- the *topology*: which directed pairs `(i, j)`
  currently exist, plus their non-parameter metadata (utility `U`, age).
  Deliberately not an `nn.Parameter` anywhere (§16.3's whole point): a
  literal per-edge weight parameter would need the parameter tensor
  itself to grow/shrink as edges are added/pruned, colliding with both
  this codebase's fixed-shape-parameter convention and the optimizer's
  per-parameter state. Buffers here are genuinely *resized* at runtime
  (`add_edges`/`remove_edges` reassign them) -- unlike every other buffer
  in this codebase, which is fixed-shape. `state_dict()`/`load_state_dict()`
  only round-trip correctly when the edge count at load time matches the
  saved one; resuming a checkpoint into a differently-sized registry
  isn't handled here (flagged, not solved -- out of scope for this pass).
- `CellActivity` -- the running per-cell activity `A_i` (§16.7) that
  feeds growth scoring, EMA'd at the same `0.99`/`0.01` decay as edge
  utility (§16.10 -- the user's spec didn't give `A_i` its own decay
  value, so this reuses `U_ij`'s for consistency, not a new invention).
"""

from __future__ import annotations

import torch
from torch import nn


class StructuralAddress(nn.Module):
    """Persistent per-cell structural identity `s_i in R^{d_s}` (§16.2/
    §16.3). Unlike every other learned tensor in this architecture, `s`
    does not depend on the current input or belief state -- it is a
    fixed `(n_cells, d_s)` parameter table, the same for every
    example/batch, trained by ordinary backprop like any other
    parameter ("who this cell structurally is," not "what it currently
    believes")."""

    def __init__(self, n_cells: int, d_s: int) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.d_s = d_s
        self.s = nn.Parameter(torch.randn(n_cells, d_s) * (d_s**-0.5))
        self.w_out = nn.Linear(d_s, d_s, bias=False)
        self.w_in = nn.Linear(d_s, d_s, bias=False)

    def project(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(q_all, k_all)`, each `(n_cells, d_s)` -- every
        cell's address projected once. No batch dimension: `s` doesn't
        depend on the input. `O(n_cells * d_s^2)`."""
        return self.w_out(self.s), self.w_in(self.s)

    def edge_weights(self, edge_index: torch.Tensor) -> torch.Tensor:
        """`edge_index`: `(E, 2)` long, `(source, target)` pairs.
        Returns `w`: `(E,)` -- `w_ij = tanh(q_i . k_j / sqrt(d_s))` for
        each existing edge, gathered from the once-per-forward-pass
        projection rather than recomputed per edge. `O(E * d_s)`, no
        `(n_cells, n_cells)` tensor at any point."""
        q_all, k_all = self.project()
        if edge_index.shape[0] == 0:
            return torch.zeros(0, dtype=q_all.dtype, device=q_all.device)
        q = q_all[edge_index[:, 0]]
        k = k_all[edge_index[:, 1]]
        return torch.tanh((q * k).sum(dim=-1) / (self.d_s**0.5))


class EdgeRegistry(nn.Module):
    """The sparse structural topology: which directed `(source, target)`
    pairs currently exist, plus non-parameter metadata (§16.2/§16.6/
    §16.8) -- utility `U` (the pruning signal) and age. Everything here
    is a plain buffer, never an `nn.Parameter`: `w_ij` itself is computed
    on demand from `StructuralAddress`, not stored (§16.3), so there is
    nothing here for the optimizer to own.

    `n_cells` is fixed at construction; `n_edges` changes as structural
    plasticity runs (`add_edges`/`remove_edges` reassign the buffers).
    """

    def __init__(self, n_cells: int) -> None:
        super().__init__()
        self.n_cells = n_cells
        self.register_buffer("edge_index", torch.zeros(0, 2, dtype=torch.long))
        self.register_buffer("utility", torch.zeros(0))
        self.register_buffer("age", torch.zeros(0, dtype=torch.long))

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[0]

    def in_degree(self) -> torch.Tensor:
        """`(n_cells,)` long -- how many existing edges currently target
        each cell. §16.8's safety constraint (every cell has in-degree
        >= 1) is enforced by callers of this, not by the registry
        itself."""
        degree = torch.zeros(self.n_cells, dtype=torch.long, device=self.edge_index.device)
        if self.n_edges > 0:
            degree.index_add_(
                0, self.edge_index[:, 1], torch.ones(self.n_edges, dtype=torch.long, device=self.edge_index.device)
            )
        return degree

    def add_edges(self, new_edges: torch.Tensor) -> None:
        """`new_edges`: `(k, 2)` long, `(source, target)`. Appends with
        fresh `utility=0`, `age=0`. Does not deduplicate against existing
        edges or against itself -- callers (`structural_plasticity.py`)
        are responsible for only proposing genuinely new pairs."""
        if new_edges.shape[0] == 0:
            return
        device = self.edge_index.device
        self.edge_index = torch.cat([self.edge_index, new_edges.to(device=device, dtype=torch.long)], dim=0)
        self.utility = torch.cat([self.utility, torch.zeros(new_edges.shape[0], device=device)], dim=0)
        self.age = torch.cat([self.age, torch.zeros(new_edges.shape[0], dtype=torch.long, device=device)], dim=0)

    def remove_edges(self, keep_mask: torch.Tensor) -> None:
        """`keep_mask`: `(n_edges,)` bool. Drops every edge where
        `False`."""
        self.edge_index = self.edge_index[keep_mask]
        self.utility = self.utility[keep_mask]
        self.age = self.age[keep_mask]

    def increment_age(self) -> None:
        if self.n_edges > 0:
            self.age = self.age + 1


class CellActivity(nn.Module):
    """Running per-cell activity `A_i` (§16.7): `A_i <- decay*A_i +
    (1-decay)*(|mu_i|*e_i/(1+u_i))`. Feeds growth scoring only -- a
    candidate edge is favored when both endpoints are currently
    meaningfully active/reliable, not just functionally compatible."""

    def __init__(self, n_cells: int, decay: float = 0.99) -> None:
        super().__init__()
        self.decay = decay
        self.register_buffer("value", torch.zeros(n_cells))

    def update(self, mu: torch.Tensor, evidence: torch.Tensor, uncertainty: torch.Tensor) -> None:
        """`mu`/`evidence`/`uncertainty`: `(batch, n_cells)`. Detaches
        before reducing -- this is bookkeeping state, not something the
        loss should backprop through, and keeping it attached to the
        live graph would leak memory across calls."""
        raw = (mu.detach().abs() * evidence.detach() / (1.0 + uncertainty.detach())).mean(dim=0)
        self.value = self.decay * self.value + (1.0 - self.decay) * raw
