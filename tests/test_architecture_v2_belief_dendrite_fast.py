"""Parity tests for the optimized Architecture V2 backends
(`src/models/architecture_v2/belief_dendrite_fast.py`) against the frozen
reference (`belief_dendrite.py`): identical parameters copied across,
identical input, and forward outputs / `(mu, e, u, pi)` diagnostics /
gradients / one optimizer step must all match within tight float32
tolerance. The frozen layer's own math/tests are untouched by any of this.
"""

from __future__ import annotations

import copy

import torch

from src.models.architecture_v0.cell import BeliefCell
from src.models.architecture_v2.belief_dendrite import (
    BeliefDendriteLayer,
    BeliefDendriteNetwork,
    DendriticConnectivity,
)
from src.models.architecture_v2.belief_dendrite_fast import (
    BeliefDendriteLayerDenseFast,
    BeliefDendriteLayerSparseFast,
    BeliefDendriteNetworkDenseFast,
    BeliefDendriteNetworkSparseFast,
)

ATOL = 1e-5
RTOL = 1e-4


def _copy_weights(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(copy.deepcopy(src.state_dict()))


def _random_connectivity(seed: int = 0) -> DendriticConnectivity:
    return DendriticConnectivity.balanced_random(
        input_dim=50, num_somas=6, branches_per_soma=3, sources_per_branch=8, seed=seed
    )


def _local2d_connectivity(seed: int = 0) -> DendriticConnectivity:
    """The exact CIFAR-benchmark shape family: RGB local patches."""
    return DendriticConnectivity.local_2d(
        channels=3, height=32, width=32, num_somas=12, branches_per_soma=4, patch=8, seed=seed
    )


def _random_belief(batch: int, input_dim: int, seed: int) -> BeliefCell:
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(batch, input_dim, generator=g)
    e = torch.rand(batch, input_dim, generator=g).clamp_min(1e-3)
    u = torch.rand(batch, input_dim, generator=g) * 0.5
    return BeliefCell(mu=mu, evidence=e, uncertainty=u)


def _assert_forward_parity(
    frozen: BeliefDendriteLayer, fast: torch.nn.Module, belief: BeliefCell
) -> None:
    _copy_weights(frozen, fast)
    (ref_out, ref_diag) = frozen.forward_verbose(belief)
    (fast_out, fast_diag) = fast.forward_verbose(belief)

    torch.testing.assert_close(fast_out.mu, ref_out.mu, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_out.evidence, ref_out.evidence, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_out.uncertainty, ref_out.uncertainty, atol=ATOL, rtol=RTOL)

    torch.testing.assert_close(fast_diag.mu_branch, ref_diag.mu_branch, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_diag.e_branch, ref_diag.e_branch, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_diag.u_branch, ref_diag.u_branch, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_diag.pi_branch, ref_diag.pi_branch, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_diag.A_cable, ref_diag.A_cable, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(fast_diag.e_soma, ref_diag.e_soma, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(
        fast_diag.branch_contribution(), ref_diag.branch_contribution(), atol=ATOL, rtol=RTOL
    )


def test_sparse_fast_matches_frozen_forward_on_random_topology() -> None:
    conn = _random_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerSparseFast(conn)
    belief = _random_belief(7, conn.input_dim, seed=1)
    _assert_forward_parity(frozen, fast, belief)


def test_dense_fast_matches_frozen_forward_on_random_topology() -> None:
    conn = _random_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerDenseFast(conn)
    belief = _random_belief(7, conn.input_dim, seed=1)
    _assert_forward_parity(frozen, fast, belief)


def test_sparse_fast_matches_frozen_forward_on_local2d_cifar_shape() -> None:
    conn = _local2d_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerSparseFast(conn)
    belief = _random_belief(5, conn.input_dim, seed=2)
    _assert_forward_parity(frozen, fast, belief)


def test_dense_fast_matches_frozen_forward_on_local2d_cifar_shape() -> None:
    conn = _local2d_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerDenseFast(conn)
    belief = _random_belief(5, conn.input_dim, seed=2)
    _assert_forward_parity(frozen, fast, belief)


def test_sparse_fast_matches_frozen_forward_with_a_fully_missing_branch() -> None:
    """Edge case: one branch's entire receptive field has ~zero precision
    (as `missing_patch`-style corruption produces) -- exercises the eps
    clamps in both the frozen and fast implementations identically."""
    conn = _local2d_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerSparseFast(conn)
    belief = _random_belief(4, conn.input_dim, seed=3)
    low_e = belief.evidence.clone()
    low_e[:, :64] = 1e-3  # first branch's 8x8x1-channel-ish region driven near-zero precision
    belief = BeliefCell(mu=belief.mu, evidence=low_e, uncertainty=belief.uncertainty)
    _assert_forward_parity(frozen, fast, belief)


def _grad_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def test_sparse_fast_matches_frozen_gradients() -> None:
    conn = _random_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerSparseFast(conn)
    _copy_weights(frozen, fast)
    belief = _random_belief(6, conn.input_dim, seed=4)

    frozen.forward(belief).mu.sum().backward()
    fast.forward(belief).mu.sum().backward()

    ref_grads, fast_grads = _grad_dict(frozen), _grad_dict(fast)
    assert set(ref_grads) == set(fast_grads)
    for name in ref_grads:
        torch.testing.assert_close(fast_grads[name], ref_grads[name], atol=ATOL, rtol=RTOL)


def test_dense_fast_matches_frozen_gradients() -> None:
    conn = _random_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerDenseFast(conn)
    _copy_weights(frozen, fast)
    belief = _random_belief(6, conn.input_dim, seed=4)

    frozen.forward(belief).mu.sum().backward()
    fast.forward(belief).mu.sum().backward()

    ref_grads, fast_grads = _grad_dict(frozen), _grad_dict(fast)
    assert set(ref_grads) == set(fast_grads)
    for name in ref_grads:
        torch.testing.assert_close(fast_grads[name], ref_grads[name], atol=ATOL, rtol=RTOL)


def _one_optimizer_step(model: torch.nn.Module, belief: BeliefCell) -> dict[str, torch.Tensor]:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    opt.zero_grad()
    model.forward(belief).mu.sum().backward()
    opt.step()
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def test_sparse_fast_matches_frozen_after_one_optimizer_step() -> None:
    conn = _random_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerSparseFast(conn)
    _copy_weights(frozen, fast)
    belief = _random_belief(6, conn.input_dim, seed=5)

    ref_params = _one_optimizer_step(frozen, belief)
    fast_params = _one_optimizer_step(fast, belief)
    for name in ref_params:
        torch.testing.assert_close(fast_params[name], ref_params[name], atol=ATOL, rtol=RTOL)


def test_dense_fast_matches_frozen_after_one_optimizer_step() -> None:
    conn = _random_connectivity()
    frozen = BeliefDendriteLayer(conn)
    fast = BeliefDendriteLayerDenseFast(conn)
    _copy_weights(frozen, fast)
    belief = _random_belief(6, conn.input_dim, seed=5)

    ref_params = _one_optimizer_step(frozen, belief)
    fast_params = _one_optimizer_step(fast, belief)
    for name in ref_params:
        torch.testing.assert_close(fast_params[name], ref_params[name], atol=ATOL, rtol=RTOL)


def test_sparse_fast_network_matches_frozen_network_end_to_end() -> None:
    ref = BeliefDendriteNetwork.build(
        in_features=50, out_features=4, hidden1=6, hidden2=5, branches_per_soma=3, seed=0
    )
    fast = BeliefDendriteNetworkSparseFast(ref.layer1.connectivity, ref.layer2.connectivity, 4)
    _copy_weights(ref, fast)

    x = torch.randn(9, 50)
    c = torch.rand(9, 50).clamp_min(1e-3)
    torch.testing.assert_close(fast(x, c), ref(x, c), atol=ATOL, rtol=RTOL)


def test_dense_fast_network_matches_frozen_network_end_to_end() -> None:
    ref = BeliefDendriteNetwork.build(
        in_features=50, out_features=4, hidden1=6, hidden2=5, branches_per_soma=3, seed=0
    )
    fast = BeliefDendriteNetworkDenseFast(ref.layer1.connectivity, ref.layer2.connectivity, 4)
    _copy_weights(ref, fast)

    x = torch.randn(9, 50)
    c = torch.rand(9, 50).clamp_min(1e-3)
    torch.testing.assert_close(fast(x, c), ref(x, c), atol=ATOL, rtol=RTOL)


def test_fast_backends_add_no_new_trainable_parameters() -> None:
    conn = _random_connectivity()
    frozen_names = {n for n, _ in BeliefDendriteLayer(conn).named_parameters()}
    sparse_names = {n for n, _ in BeliefDendriteLayerSparseFast(conn).named_parameters()}
    dense_names = {n for n, _ in BeliefDendriteLayerDenseFast(conn).named_parameters()}
    assert frozen_names == sparse_names == dense_names
