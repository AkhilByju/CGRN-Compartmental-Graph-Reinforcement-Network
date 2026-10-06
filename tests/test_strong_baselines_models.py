"""Strong-baselines model families
(experiments/belief_dendrite/strong_baselines/models.py, spec Sec 4):
output shapes, reliability-channel wiring for each family, and MPS forward/
backward for all seven with no NaNs. `ScalarDendriteModel`/`BeliefDendriteModel`
are checked to be thin wrappers around the frozen, unmodified
`src/models/architecture_v2/belief_dendrite.py` classes -- spec Sec 4F/4G's
"No changes" / "Do not otherwise modify its equations."
"""

from __future__ import annotations

import torch

from experiments.belief_dendrite.strong_baselines.device_utils import resolve_device
from experiments.belief_dendrite.strong_baselines.models import (
    MODEL_FAMILIES,
    OUT_FEATURES,
    VIT_NUM_PATCHES,
    VIT_PATCH,
    BeliefDendriteModel,
    ConfidenceCNN,
    ConfidenceTinyViT,
    ReliabilityGatedTinyViT,
    ScalarDendriteModel,
    SmallCNN,
    TinyViT,
    build_model,
    patch_reliability_map,
)
from src.models.architecture_v2.belief_dendrite import BeliefDendriteNetwork, ScalarDendriteNetwork


def _batch(n: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 32, 32, generator=g)
    c = torch.ones(n, 3, 32, 32)
    c[:, :, :8, :8] = 1e-3  # a fake corrupted top-left patch, channel-constant
    return x, c


def test_small_cnn_output_shape() -> None:
    x, c = _batch()
    model = SmallCNN(width=16)
    out = model(x, c)
    assert out.shape == (4, OUT_FEATURES)


def test_confidence_cnn_uses_exactly_one_extra_reliability_channel() -> None:
    x, c = _batch()
    model = ConfidenceCNN(width=16)
    assert model.net.conv1.in_channels == 4
    out = model(x, c)
    assert out.shape == (4, OUT_FEATURES)
    # Changing the reliability channel (holding the image fixed) must change the output --
    # confirms the reliability channel is actually wired into the conv trunk, not dropped.
    out_ones = model(x, torch.ones_like(c))
    assert not torch.allclose(out, out_ones)


def test_tiny_vit_token_count_is_patches_plus_class_token() -> None:
    model = TinyViT(embed_dim=32)
    assert model.num_patches == VIT_NUM_PATCHES == 64
    x, c = _batch()
    tok = model.encode(model.patch_tokens(x))
    assert tok.shape == (4, VIT_NUM_PATCHES + 1, 32)


def test_patch_embedding_shape_before_cls_and_pos() -> None:
    model = TinyViT(embed_dim=32)
    x, _ = _batch()
    patch_tok = model.patch_tokens(x)
    assert patch_tok.shape == (4, VIT_NUM_PATCHES, 32)


def test_reliability_patch_pooling_matches_manual_mean() -> None:
    x, c = _batch()
    pooled = patch_reliability_map(c, patch=VIT_PATCH)
    assert pooled.shape == (4, VIT_NUM_PATCHES, 1)
    # The corrupted 8x8 top-left region is exactly two 4x4 patches wide/tall,
    # so patches (0,0), (0,1), (1,0), (1,1) (grid index 0,1,8,9) must equal 1e-3.
    grid = pooled.reshape(4, 8, 8, 1)
    assert torch.allclose(grid[:, 0, 0], torch.full((4, 1), 1e-3), atol=1e-6)
    assert torch.allclose(grid[:, 1, 1], torch.full((4, 1), 1e-3), atol=1e-6)
    assert torch.allclose(grid[:, 2, 2], torch.ones(4, 1), atol=1e-6)


def test_confidence_tiny_vit_adds_reliability_only_to_patch_tokens_not_cls() -> None:
    x, c = _batch()
    model = ConfidenceTinyViT(embed_dim=32)
    logits, image_tok, rel_emb = model.forward_verbose(x, c)
    assert logits.shape == (4, OUT_FEATURES)
    assert image_tok.shape == rel_emb.shape == (4, VIT_NUM_PATCHES, 32)
    # All-ones reliability collapses rel_emb to the same vector at every
    # patch/example (a single Linear(1, d, bias=False) applied to a constant).
    logits_ones, _, rel_emb_ones = model.forward_verbose(x, torch.ones_like(c))
    first = rel_emb_ones[0, 0]
    assert torch.allclose(rel_emb_ones, first.expand_as(rel_emb_ones), atol=1e-5)
    assert not torch.allclose(logits, logits_ones)


def test_reliability_gated_tiny_vit_multiplies_tokens_by_patch_reliability() -> None:
    x, c = _batch()
    model = ReliabilityGatedTinyViT(embed_dim=32)
    logits, rel = model.forward_verbose(x, c)
    assert logits.shape == (4, OUT_FEATURES)
    assert rel.shape == (4, VIT_NUM_PATCHES, 1)

    # Multiplicative gating (not additive): zero reliability must fully erase
    # the image's contribution to the patch tokens, so two *different* images
    # under all-zero reliability produce identical logits (only the class
    # token + positional embeddings remain). ConfidenceTinyViT's additive
    # combination has no equivalent property.
    x2 = torch.randn_like(x)  # a genuinely different image batch
    zeros = torch.zeros_like(c)
    logits_a, _ = model.forward_verbose(x, zeros)
    logits_b, _ = model.forward_verbose(x2, zeros)
    assert torch.allclose(logits_a, logits_b, atol=1e-5)


def test_dendrite_wrappers_are_unmodified_frozen_v2_classes() -> None:
    scalar = build_model("scalar_dendrite", seed=0).model
    belief = build_model("belief_dendrite", seed=0).model
    assert isinstance(scalar, ScalarDendriteModel)
    assert isinstance(belief, BeliefDendriteModel)
    assert isinstance(scalar.net, ScalarDendriteNetwork)
    assert isinstance(belief.net, BeliefDendriteNetwork)
    # Exactly the frozen parameter set (no router/gate/extra parameter added
    # by this benchmark's wrapper) -- same check as
    # tests/test_architecture_v2_belief_dendrite.py.
    belief_param_names = {name for name, _ in belief.net.named_parameters()}
    expected = {
        "layer1.V_branch", "layer1.gain_raw_branch", "layer1.bias_branch",
        "layer1.V_cable", "layer1.gain_raw_soma", "layer1.bias_soma",
        "layer2.V_branch", "layer2.gain_raw_branch", "layer2.bias_branch",
        "layer2.V_cable", "layer2.gain_raw_soma", "layer2.bias_soma",
        "readout.weight", "readout.bias",
    }
    assert belief_param_names == expected


def test_all_seven_families_forward_backward_no_nans_on_resolved_device() -> None:
    report = resolve_device()
    device = report.device
    x, c = _batch(6)
    x, c = x.to(device), c.to(device)
    # A dedicated seed, distinct from every other test's seed=0: `build_model`
    # caches (and `.to(device)` mutates in place) the dendrite connectivity
    # buffers per seed, so reusing seed=0 here would leave those buffers
    # stranded on `device` for any later CPU-only test in the same session.
    isolated_seed = 999_999
    try:
        for fam in MODEL_FAMILIES:
            model = build_model(fam, isolated_seed).model.to(device)
            out = model(x, c)
            assert out.shape == (6, OUT_FEATURES), fam
            assert torch.isfinite(out).all(), f"{fam} produced non-finite logits on {device}"
            loss = out.sum()
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), (
                        f"{fam}.{name} produced a non-finite gradient"
                    )
    finally:
        # Restore the cached dendrite connectivity for `isolated_seed` to CPU
        # so it can't leak a stranded MPS buffer into a later test.
        from experiments.belief_dendrite.strong_baselines.models import _dendrite_connectivity

        for conn in _dendrite_connectivity(isolated_seed):
            conn.to("cpu")
