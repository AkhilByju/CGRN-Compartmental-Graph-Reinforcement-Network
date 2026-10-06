"""Architecture V2 frozen-benchmark model builder
(experiments/belief_dendrite/models.py, docs/architecture_v2.md Sec N):
every family lands near the ~150k parameter budget, the three dendritic
families share identical topology at a fixed seed, and every family
exposes the uniform `(x, c) -> logits` interface.
"""

from __future__ import annotations

import torch

from experiments.belief_dendrite.models import (
    DENDRITE_FAMILIES,
    MODEL_FAMILIES,
    OUT_FEATURES,
    PARAM_BUDGET,
    build_model,
)


def test_every_family_lands_within_two_percent_of_the_budget() -> None:
    for fam in MODEL_FAMILIES:
        built = build_model(fam, seed=0)
        rel_diff = abs(built.parameter_count - PARAM_BUDGET) / PARAM_BUDGET
        assert rel_diff <= 0.02, (
            f"{fam}: {built.parameter_count} vs budget {PARAM_BUDGET} ({rel_diff:.3%})"
        )


def test_every_family_has_the_uniform_x_c_forward_interface() -> None:
    x = torch.randn(5, 784)
    c = torch.rand(5, 784)
    for fam in MODEL_FAMILIES:
        model = build_model(fam, seed=0).model
        out = model(x, c)
        assert out.shape == (5, OUT_FEATURES), fam
        assert torch.isfinite(out).all(), fam


def test_dendritic_families_share_identical_topology_at_a_fixed_seed() -> None:
    built = {fam: build_model(fam, seed=7) for fam in DENDRITE_FAMILIES}
    conns = {}
    for fam, b in built.items():
        model = b.model
        net = model if hasattr(model, "layer1") else model.net
        conns[fam] = net.layer1.connectivity.source_idx
    families = list(conns)
    for other in families[1:]:
        assert torch.equal(conns[families[0]], conns[other]), (families[0], other)


def test_dendritic_families_get_different_topology_for_different_seeds() -> None:
    from experiments.belief_dendrite.models import BeliefDendriteNetwork

    a = build_model("belief_dendrite", seed=0).model
    b = build_model("belief_dendrite", seed=1).model
    assert isinstance(a, BeliefDendriteNetwork)
    assert not torch.equal(a.layer1.connectivity.source_idx, b.layer1.connectivity.source_idx)


def test_confidence_mlp_and_belief_dendrite_are_reliability_aware() -> None:
    from experiments.belief_dendrite.models import RELIABILITY_AWARE

    x = torch.randn(4, 784)
    for fam in RELIABILITY_AWARE:
        model = build_model(fam, seed=0).model
        c_full = torch.ones(4, 784)
        c_low = torch.full((4, 784), 1e-3)
        out_full = model(x, c_full)
        out_low = model(x, c_low)
        assert not torch.allclose(out_full, out_low), fam
