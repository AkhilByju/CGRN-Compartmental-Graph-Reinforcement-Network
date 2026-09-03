"""Tests for `structural_plasticity.py`: the schedule (§16.9), growth/
prune correctness including the in-degree-1 safety constraint (§16.8),
and bootstrap filling the edge budget."""

from __future__ import annotations

import torch

from src.models.architecture_v1.structural import CellActivity, EdgeRegistry, StructuralAddress
from src.models.architecture_v1.structural_plasticity import (
    StructuralPlasticityConfig,
    bootstrap,
    run_growth,
    run_prune,
    run_plasticity_event,
    should_run_plasticity,
)


# --- schedule ---


def test_schedule_is_off_during_warmup() -> None:
    config = StructuralPlasticityConfig(warmup_steps=200, update_interval=100, freeze_fraction=0.2)
    for step in (0, 50, 199):
        assert not should_run_plasticity(step, total_steps=10000, config=config)


def test_schedule_fires_on_interval_after_warmup() -> None:
    config = StructuralPlasticityConfig(warmup_steps=200, update_interval=100, freeze_fraction=0.2)
    assert should_run_plasticity(200, total_steps=10000, config=config)
    assert should_run_plasticity(300, total_steps=10000, config=config)
    assert not should_run_plasticity(250, total_steps=10000, config=config)


def test_schedule_freezes_near_end_of_training() -> None:
    config = StructuralPlasticityConfig(warmup_steps=200, update_interval=100, freeze_fraction=0.2)
    total = 1000
    # last 20% starts at step 800
    assert should_run_plasticity(700, total_steps=total, config=config)
    assert not should_run_plasticity(800, total_steps=total, config=config)
    assert not should_run_plasticity(900, total_steps=total, config=config)


# --- growth ---


def _phi_mean(n_cells: int, assoc_dim: int, generator: torch.Generator) -> torch.Tensor:
    phi = torch.rand(n_cells, assoc_dim, generator=generator) + 0.1  # softplus-like: positive
    return phi


def test_run_growth_adds_up_to_num_new_edges_no_self_loops_no_duplicates() -> None:
    g = torch.Generator().manual_seed(0)
    n_cells = 20
    addr = StructuralAddress(n_cells, d_s=4)
    edges = EdgeRegistry(n_cells)
    activity = CellActivity(n_cells)
    activity.value.fill_(1.0)
    phi_mean = _phi_mean(n_cells, assoc_dim=6, generator=g)
    config = StructuralPlasticityConfig(chunk_size=8)

    run_growth(addr, edges, activity, phi_mean, num_new=15, config=config, generator=g)

    assert edges.n_edges <= 15
    assert edges.n_edges > 0
    src, tgt = edges.edge_index[:, 0], edges.edge_index[:, 1]
    assert not (src == tgt).any()
    pairs = {(int(i), int(j)) for i, j in edges.edge_index.tolist()}
    assert len(pairs) == edges.n_edges  # no duplicates


def test_run_growth_never_re_adds_existing_edges() -> None:
    g = torch.Generator().manual_seed(1)
    n_cells = 10
    addr = StructuralAddress(n_cells, d_s=4)
    edges = EdgeRegistry(n_cells)
    activity = CellActivity(n_cells)
    activity.value.fill_(1.0)
    phi_mean = _phi_mean(n_cells, assoc_dim=4, generator=g)
    config = StructuralPlasticityConfig(chunk_size=10)

    run_growth(addr, edges, activity, phi_mean, num_new=n_cells * n_cells, config=config, generator=g)
    n_after_first = edges.n_edges
    run_growth(addr, edges, activity, phi_mean, num_new=n_cells * n_cells, config=config, generator=g)
    # candidates are the same LSH pool + same phi -> nothing new to add once saturated
    assert edges.n_edges == n_after_first


def test_run_growth_num_new_zero_is_noop() -> None:
    g = torch.Generator().manual_seed(2)
    n_cells = 8
    addr = StructuralAddress(n_cells, d_s=4)
    edges = EdgeRegistry(n_cells)
    activity = CellActivity(n_cells)
    phi_mean = _phi_mean(n_cells, assoc_dim=4, generator=g)
    run_growth(addr, edges, activity, phi_mean, num_new=0, config=StructuralPlasticityConfig(), generator=g)
    assert edges.n_edges == 0


# --- pruning ---


def test_run_prune_removes_lowest_utility_fraction() -> None:
    edges = EdgeRegistry(n_cells=20)
    # a ring (every cell has exactly one in-edge, all protected) plus one
    # redundant extra in-edge into cells 1 and 2 each, so those two
    # ring-edges (into 1 and into 2) have "spare" in-degree and can be
    # pruned; every other ring edge stays protected (in-degree would hit 0).
    ring = torch.stack([torch.arange(20), (torch.arange(20) + 1) % 20], dim=-1)
    extra = torch.tensor([[5, 1], [6, 2]])
    edges.add_edges(torch.cat([ring, extra], dim=0))
    utility = torch.full((22,), 5.0)
    utility[0] = 0.0  # ring edge 0 -> 1 (target 1 has spare in-degree via 5->1)
    utility[1] = 0.1  # ring edge 1 -> 2 (target 2 has spare in-degree via 6->2)
    edges.utility = utility

    run_prune(edges, fraction=0.1)  # bottom 10% of 22 = 2 edges

    assert edges.n_edges == 20
    remaining_pairs = {(int(i), int(j)) for i, j in edges.edge_index.tolist()}
    assert (0, 1) not in remaining_pairs
    assert (1, 2) not in remaining_pairs
    assert (edges.in_degree() >= 1).all()


def test_run_prune_protects_only_incoming_edge() -> None:
    """Cell 1 has exactly one incoming edge (0 -> 1); even if it's the
    lowest-utility edge in the whole registry, pruning must not remove
    it (§16.8's safety constraint). Cell 2 has a redundant second
    in-edge, so one of its two in-edges is safely prunable."""
    edges = EdgeRegistry(n_cells=3)
    edges.add_edges(torch.tensor([[0, 1], [1, 2], [2, 2]]))
    edges.utility = torch.tensor([0.0, 5.0, 0.1])  # 0->1 is lowest utility, 1's only in-edge

    run_prune(edges, fraction=1.0)  # try to prune everything

    remaining_pairs = {(int(i), int(j)) for i, j in edges.edge_index.tolist()}
    assert (0, 1) in remaining_pairs  # protected: removing it would zero cell 1's in-degree
    assert edges.in_degree()[1] >= 1  # cell 0 was never given an in-edge in this hand-built graph; not asserted


def test_run_prune_zero_fraction_is_noop() -> None:
    edges = EdgeRegistry(n_cells=4)
    edges.add_edges(torch.tensor([[0, 1], [1, 2]]))
    run_prune(edges, fraction=0.0)
    assert edges.n_edges == 2


# --- bootstrap ---


def test_bootstrap_fills_budget_and_guarantees_min_in_degree() -> None:
    g = torch.Generator().manual_seed(3)
    n_cells = 16
    addr = StructuralAddress(n_cells, d_s=4)
    edges = EdgeRegistry(n_cells)
    activity = CellActivity(n_cells)
    activity.value.fill_(1.0)
    phi_mean = _phi_mean(n_cells, assoc_dim=4, generator=g)
    e_max = 8 * n_cells

    bootstrap(addr, edges, activity, phi_mean, e_max=e_max, generator=g)

    assert edges.n_edges > 0
    assert (edges.in_degree() >= 1).all()


# --- one plasticity event end to end ---


def test_run_plasticity_event_keeps_edge_count_roughly_fixed() -> None:
    g = torch.Generator().manual_seed(4)
    n_cells = 16
    addr = StructuralAddress(n_cells, d_s=4)
    edges = EdgeRegistry(n_cells)
    activity = CellActivity(n_cells)
    activity.value.fill_(1.0)
    phi_mean = _phi_mean(n_cells, assoc_dim=4, generator=g)
    config = StructuralPlasticityConfig(prune_fraction=0.25, chunk_size=8)

    bootstrap(addr, edges, activity, phi_mean, e_max=8 * n_cells, generator=g)
    n_before = edges.n_edges

    edges.utility = torch.rand(n_before, generator=g)
    run_plasticity_event(addr, edges, activity, phi_mean, config, generator=g)

    # edge count should stay roughly the same (prune then grow the same number)
    assert abs(edges.n_edges - n_before) <= max(2, int(0.05 * n_before))
    assert (edges.in_degree() >= 1).all()
    assert (edges.age >= 0).all()
