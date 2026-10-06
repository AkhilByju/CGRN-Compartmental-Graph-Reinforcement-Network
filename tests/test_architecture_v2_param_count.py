"""Architecture V2 -- exact parameter-count formula and budget solver
(docs/architecture_v2.md Sec J). Predictions are checked against the real
`nn.Module` parameter count, never assumed.
"""

from __future__ import annotations

import pytest
import torch

from src.evaluation.efficiency import count_parameters
from src.models.architecture_v2.belief_dendrite import (
    BeliefDendriteLayer,
    BeliefDendriteNetwork,
    DendriticConnectivity,
    ScalarDendriteLayer,
)
from src.models.architecture_v2.param_count import (
    belief_dendrite_layer_param_count,
    readout_param_count,
    solve_hidden_width_for_budget,
    two_layer_param_count,
)


@pytest.mark.parametrize(("h", "b", "k"), [(5, 4, 6), (1, 1, 1), (17, 3, 9), (128, 4, 32)])
def test_formula_matches_actual_belief_layer_parameter_count(h: int, b: int, k: int) -> None:
    conn = DendriticConnectivity.balanced_random(
        input_dim=max(k, 40), num_somas=h, branches_per_soma=b, sources_per_branch=k, seed=0
    )
    layer = BeliefDendriteLayer(conn)
    predicted = belief_dendrite_layer_param_count(h, b, k)
    assert predicted == count_parameters(layer)


@pytest.mark.parametrize(("h", "b", "k"), [(5, 4, 6), (17, 3, 9)])
def test_formula_matches_actual_scalar_layer_parameter_count(h: int, b: int, k: int) -> None:
    conn = DendriticConnectivity.balanced_random(
        input_dim=max(k, 40), num_somas=h, branches_per_soma=b, sources_per_branch=k, seed=0
    )
    layer = ScalarDendriteLayer(conn)
    assert belief_dendrite_layer_param_count(h, b, k) == count_parameters(layer)


def test_two_layer_formula_matches_actual_network() -> None:
    torch.manual_seed(0)
    branches, k1, k2, hidden, out_features = 4, 20, 9, 9, 5
    # two_layer_param_count assumes a *uniform* hidden width and fixed,
    # given K1/K2 (not the `min(32, ...)` default coupling) -- match that
    # exactly when building the real network to compare against.
    net = BeliefDendriteNetwork.build(
        in_features=64,
        out_features=out_features,
        hidden1=hidden,
        hidden2=hidden,
        branches_per_soma=branches,
        seed=0,
        sources_per_branch_1=k1,
        sources_per_branch_2=k2,
    )
    predicted = two_layer_param_count(hidden, branches, k1, k2, out_features)
    assert predicted == count_parameters(net)


def test_readout_param_count_matches_nn_linear() -> None:
    linear = torch.nn.Linear(37, 5)
    assert readout_param_count(37, 5) == count_parameters(linear)


@pytest.mark.parametrize("budget", [500, 5_000, 50_000, 150_000, 1_000_000])
def test_solver_finds_the_largest_hidden_width_within_budget(budget: int) -> None:
    branches, k1, k2, out_features = 4, 32, 32, 10
    sol = solve_hidden_width_for_budget(budget, branches, k1, k2, out_features)
    if sol.hidden > 0:
        assert two_layer_param_count(sol.hidden, branches, k1, k2, out_features) <= budget
    assert two_layer_param_count(sol.hidden + 1, branches, k1, k2, out_features) > budget


def test_solver_result_is_buildable_and_matches_actual_count() -> None:
    branches, k1, k2, out_features = 4, 16, 16, 10
    budget = 150_000
    sol = solve_hidden_width_for_budget(budget, branches, k1, k2, out_features)
    net = BeliefDendriteNetwork.build(
        in_features=200,
        out_features=out_features,
        hidden1=sol.hidden,
        hidden2=sol.hidden,
        branches_per_soma=branches,
        seed=0,
        sources_per_branch_1=k1,
        sources_per_branch_2=k2,
    )
    assert count_parameters(net) == sol.parameter_count
    assert count_parameters(net) <= budget
