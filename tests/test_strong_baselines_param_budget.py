"""Strong-baselines parameter-budget solver correctness
(experiments/belief_dendrite/strong_baselines/param_budget.py, spec Sec 6):
every closed-form formula is checked against the real constructed
`nn.Module`'s parameter count, not assumed.
"""

from __future__ import annotations

from experiments.belief_dendrite.strong_baselines.models import (
    ConfidenceCNN,
    ConfidenceTinyViT,
    ReliabilityGatedTinyViT,
    SmallCNN,
    TinyViT,
)
from experiments.belief_dendrite.strong_baselines.param_budget import (
    MAX_DEVIATION,
    TARGET_PARAM_BUDGET,
    cnn_param_count,
    solve_cnn_width,
    solve_vit_embed_dim,
    vit_param_count,
)
from src.evaluation.efficiency import count_parameters


def test_cnn_param_count_matches_real_module_small_cnn() -> None:
    for width in (8, 16, 32, 63, 100):
        model = SmallCNN(width)
        assert count_parameters(model) == cnn_param_count(width, in_channels=3)


def test_cnn_param_count_matches_real_module_confidence_cnn() -> None:
    for width in (8, 16, 32, 63, 100):
        model = ConfidenceCNN(width)
        assert count_parameters(model) == cnn_param_count(width, in_channels=4)


def test_vit_param_count_matches_real_module_tiny_vit() -> None:
    for d in (16, 32, 80, 120):
        model = TinyViT(d)
        assert count_parameters(model) == vit_param_count(d)


def test_vit_param_count_matches_real_module_confidence_tiny_vit() -> None:
    for d in (16, 32, 80, 120):
        model = ConfidenceTinyViT(d)
        assert count_parameters(model) == vit_param_count(d, reliability_linear=True)


def test_reliability_gated_tiny_vit_adds_no_parameters_over_plain() -> None:
    d = 80
    plain = count_parameters(TinyViT(d))
    gated = count_parameters(ReliabilityGatedTinyViT(d))
    assert plain == gated == vit_param_count(d)


def test_solve_cnn_width_lands_within_budget_tolerance() -> None:
    sol = solve_cnn_width(TARGET_PARAM_BUDGET, in_channels=3)
    assert sol.relative_deviation <= MAX_DEVIATION
    assert sol.parameter_count == cnn_param_count(sol.width, in_channels=3)


def test_solve_vit_embed_dim_is_divisible_by_num_heads_and_within_tolerance() -> None:
    sol = solve_vit_embed_dim(TARGET_PARAM_BUDGET, num_heads=4)
    assert sol.width % 4 == 0
    assert sol.relative_deviation <= MAX_DEVIATION
    assert sol.parameter_count == vit_param_count(sol.width)


def test_every_required_family_lands_within_five_percent_of_budget() -> None:
    from experiments.belief_dendrite.strong_baselines.models import MODEL_FAMILIES, build_model

    for fam in MODEL_FAMILIES:
        built = build_model(fam, seed=0)
        assert built.relative_deviation <= MAX_DEVIATION, (fam, built.parameter_count)
