"""Structural plasticity for CellV1.5: growth, pruning, the edge budget,
bootstrap, and the optimizer-step-keyed schedule (docs/architecture_v1.md
§16.7-§16.9). Everything here is occasional bookkeeping (default: once
every 100 optimizer steps, §16.9), not a per-forward-pass computation --
so unlike `structural_fusion.py`, which has a hard `O(E)` budget because
it runs on *every* call, a few of the operations below (existing-edge
deduplication, the in-degree-1 safety fallback) use plain Python
set/loop logic rather than fully vectorized tensor ops. This is a
deliberate scope choice: the spec's "never `O(n_cells^2)`" constraint is
about the growth *candidate search* (§16.7/§16.10's ANN/MIPS requirement,
satisfied below by reusing `lsh.py`), not about every line of the
periodic maintenance code touching it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.architecture_v1.lsh import lsh_candidates, random_hyperplanes
from src.models.architecture_v1.structural import CellActivity, EdgeRegistry, StructuralAddress


@dataclass
class StructuralPlasticityConfig:
    """§16.8/§16.9's schedule and budget, all explicitly config per the
    user ("not theoretical constants") -- these are the stated defaults,
    not tuned."""

    k_bar: int = 8  # E_max = k_bar * n_cells, a global budget (§16.8)
    warmup_steps: int = 200  # §16.9
    update_interval: int = 100  # §16.9
    prune_fraction: float = 0.05  # §16.9
    freeze_fraction: float = 0.2  # §16.9 -- rewiring stops in the last freeze_fraction of training
    # ANN/MIPS candidate retrieval over phi-space (§16.10 -- reuses lsh.py,
    # defaults matching CellV1.1's §11 orders of magnitude, not tuned).
    num_hashes: int = 2
    bits: int = 6
    chunk_size: int = 16


def should_run_plasticity(step: int, total_steps: int, config: StructuralPlasticityConfig) -> bool:
    """§16.9: fixed at its bootstrap topology during `warmup_steps`, then
    an event every `update_interval` steps, frozen again for the last
    `freeze_fraction` of `total_steps`."""
    if step < config.warmup_steps:
        return False
    if total_steps > 0 and step >= (1.0 - config.freeze_fraction) * total_steps:
        return False
    return (step - config.warmup_steps) % config.update_interval == 0


def _candidate_pairs(
    phi_mean: torch.Tensor, config: StructuralPlasticityConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    """§16.7's ANN/MIPS candidate retrieval: `lsh.py`'s random-hyperplane
    LSH + sort + `searchsorted` machinery (§11), reused unmodified,
    applied to `phi_mean` instead of routing vectors (§16.10 -- the LSH
    mechanism is only the search algorithm; it never ranks candidates
    itself, `_score_candidates` below does that from the exact `phi`
    dot product). `phi_mean`: `(n_cells, assoc_dim)` -- already
    batch-averaged (structural topology is shared across the batch, so
    candidate search operates on one representative vector per cell, not
    per example). Returns `(k, 2)` long, deduplicated, self-loops
    removed -- no `(n_cells, n_cells)` tensor at any point."""
    n_cells = phi_mean.shape[0]
    chunk_size = max(1, min(config.chunk_size, n_cells))
    hyperplanes = random_hyperplanes(dim=phi_mean.shape[-1], num_hashes=config.num_hashes, bits=config.bits, generator=generator)
    vectors = phi_mean.unsqueeze(0)  # (1, n_cells, assoc_dim) -- one "batch item"
    candidates = lsh_candidates(vectors, vectors, hyperplanes, chunk_size=chunk_size).squeeze(0)  # (n_cells, pool)
    sources = torch.arange(n_cells, device=phi_mean.device).unsqueeze(-1).expand_as(candidates)
    pairs = torch.stack([sources.reshape(-1), candidates.reshape(-1)], dim=-1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # drop self-loops
    if pairs.shape[0] == 0:
        return pairs
    return torch.unique(pairs, dim=0)


def _score_candidates(phi_mean: torch.Tensor, activity: CellActivity, pairs: torch.Tensor) -> torch.Tensor:
    """§16.7: `G_ij = phi_i . phi_j * A_i * A_j`, exact (not via the LSH
    hash) -- `phi_mean` is already `mean_batch(phi)`, and `A_i` is
    already an EMA (batch-independent), so this equals
    `mean_batch(phi_i . phi_j * A_i * A_j)` (the two terms don't need to
    be inside the same batch-mean since `A` doesn't vary over the
    current batch)."""
    phi_i = phi_mean[pairs[:, 0]]
    phi_j = phi_mean[pairs[:, 1]]
    compat = (phi_i * phi_j).sum(dim=-1)
    return compat * activity.value[pairs[:, 0]] * activity.value[pairs[:, 1]]


def _exclude_existing(edges: EdgeRegistry, pairs: torch.Tensor) -> torch.Tensor:
    if pairs.shape[0] == 0 or edges.n_edges == 0:
        return pairs
    existing = {(int(i), int(j)) for i, j in edges.edge_index.tolist()}
    keep = torch.tensor([(int(i), int(j)) not in existing for i, j in pairs.tolist()], device=pairs.device)
    return pairs[keep]


def run_growth(
    structural_address: StructuralAddress,
    edges: EdgeRegistry,
    activity: CellActivity,
    phi_mean: torch.Tensor,
    num_new: int,
    config: StructuralPlasticityConfig,
    generator: torch.Generator | None = None,
) -> None:
    """§16.7's growth step: retrieve candidates, score them exactly, add
    the top `num_new` not already existing. `structural_address` is
    accepted for interface symmetry with `bootstrap`/callers but isn't
    needed directly -- a new edge's weight is defined by the formula in
    `StructuralAddress.edge_weights`, not by anything computed here
    (§16.3: growth needs no weight initialization)."""
    del structural_address  # unused -- see docstring
    if num_new <= 0:
        return
    pairs = _candidate_pairs(phi_mean, config, generator=generator)
    pairs = _exclude_existing(edges, pairs)
    if pairs.shape[0] == 0:
        return
    scores = _score_candidates(phi_mean, activity, pairs)
    k = min(num_new, pairs.shape[0])
    top = torch.topk(scores, k).indices
    edges.add_edges(pairs[top])


def run_prune(edges: EdgeRegistry, fraction: float) -> None:
    """§16.8/§16.9: remove the lowest-`U` `fraction` of all existing
    edges, skipping (protecting) any edge whose removal would leave its
    target with in-degree 0 -- the one safety constraint (§16.8), even
    if that edge is in the bottom-utility bracket."""
    n = edges.n_edges
    if n == 0:
        return
    num_prune = int(n * fraction)
    if num_prune == 0:
        return
    in_deg = edges.in_degree().clone()
    order = torch.argsort(edges.utility)  # ascending: lowest utility first
    keep = torch.ones(n, dtype=torch.bool, device=edges.edge_index.device)
    pruned = 0
    for idx in order.tolist():
        if pruned >= num_prune:
            break
        target = int(edges.edge_index[idx, 1])
        if in_deg[target].item() <= 1:
            continue  # protected: would zero out this cell's in-degree
        keep[idx] = False
        in_deg[target] -= 1
        pruned += 1
    edges.remove_edges(keep)


def _enforce_min_in_degree(
    edges: EdgeRegistry, phi_mean: torch.Tensor, generator: torch.Generator | None = None
) -> None:
    """§16.8's safety constraint, the fallback path: any cell left with
    in-degree 0 (after bootstrap, or in principle after a growth pass
    that didn't happen to cover it) gets its single best-scoring
    candidate edge force-added. Uses an exact (not ANN) search over all
    `n_cells` sources for the (expected-rare) orphaned cell -- acceptable
    here specifically because this is a one-off safety fallback, not the
    primary growth mechanism, and runs only for cells the LSH-based
    growth pass missed."""
    n_cells = phi_mean.shape[0]
    in_deg = edges.in_degree()
    missing = (in_deg == 0).nonzero(as_tuple=True)[0]
    if missing.numel() == 0:
        return
    del generator  # exact search below; no randomness needed
    new_edges = []
    all_sources = torch.arange(n_cells, device=phi_mean.device)
    for j in missing.tolist():
        candidates_i = all_sources[all_sources != j]
        if candidates_i.numel() == 0:
            continue
        phi_j = phi_mean[j]
        phi_i = phi_mean[candidates_i]
        scores = (phi_i * phi_j.unsqueeze(0)).sum(dim=-1)
        best = int(candidates_i[torch.argmax(scores)])
        new_edges.append((best, j))
    if new_edges:
        edges.add_edges(torch.tensor(new_edges, dtype=torch.long, device=phi_mean.device))


def bootstrap(
    structural_address: StructuralAddress,
    edges: EdgeRegistry,
    activity: CellActivity,
    phi_mean: torch.Tensor,
    e_max: int,
    generator: torch.Generator | None = None,
) -> None:
    """§16.8: one candidate-growth pass filling the edge budget to
    `e_max` from the (meaningless-at-init, disposable) initial addresses,
    then enforcing in-degree >= 1 for any cell the growth pass missed."""
    run_growth(structural_address, edges, activity, phi_mean, num_new=e_max, config=StructuralPlasticityConfig(), generator=generator)
    _enforce_min_in_degree(edges, phi_mean, generator=generator)


def run_plasticity_event(
    structural_address: StructuralAddress,
    edges: EdgeRegistry,
    activity: CellActivity,
    phi_mean: torch.Tensor,
    config: StructuralPlasticityConfig,
    generator: torch.Generator | None = None,
) -> None:
    """§16.9: one structural-plasticity event -- prune the bottom
    `prune_fraction` by utility (respecting the in-degree-1 protection),
    then grow the same number of new highest-`G`-scoring candidates, so
    total edge count stays approximately fixed. Caller
    (`StructuralRefinementCore.maybe_run_structural_plasticity`) is
    responsible for only calling this when `should_run_plasticity` says
    it's due."""
    n_before = edges.n_edges
    run_prune(edges, config.prune_fraction)
    num_removed = n_before - edges.n_edges
    run_growth(structural_address, edges, activity, phi_mean, num_new=num_removed, config=config, generator=generator)
    _enforce_min_in_degree(edges, phi_mean, generator=generator)
    edges.increment_age()
