"""`experiments/v1_001_dynamic_groups/structural_harness.py` -- CellV1.5's
first evaluation harness (`docs/architecture_v1.md` §16).

The experiment's whole claim rests on one thing being true: `cellv1.5_frozen`
and `cellv1.5_plastic` differ in *exactly* one respect, whether structural
rewiring executes. That is a property of the harness, not of CellV1.5, so it
is tested here rather than in `tests/test_structural_*.py` (which cover the
frozen model itself and are not touched by this experiment).

Also covered: best-validation checkpointing across a runtime-resized
`EdgeRegistry` (the limitation `structural.py`'s docstring flags), and the
substrate metrics, on hand-built graphs where the right answer is known by
inspection rather than by re-running the same code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "v1_001_dynamic_groups"
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

from structural_harness import (  # noqa: E402
    CELLV1_5_ARCHITECTURES,
    COMPLEXITY_LEVELS,
    STRUCTURAL_ARCHITECTURES,
    _build_structural_comparison_models,
    _clone_model_state,
    _degree_summary,
    _edge_set,
    _functional_edge_use,
    _restore_model_state,
    _structural_metrics,
    _train_until_convergence_structural,
    run_structural_comparison,
)

from src.models.architecture_v1.structural_model import StructuralBeliefGraph  # noqa: E402
from src.models.architecture_v1.structural_plasticity import (  # noqa: E402
    StructuralPlasticityConfig,
)

TASK = "dynamic_groups_global"
N_OBJECTS = 12
IN_FEATURES = N_OBJECTS * 3


def _config(**overrides) -> StructuralPlasticityConfig:
    """Small/fast plasticity schedule for tests -- the *shape* of §16.9,
    not its production 200/100/5%/20% values."""
    kwargs = {"k_bar": 4, "warmup_steps": 10, "update_interval": 5}
    return StructuralPlasticityConfig(**{**kwargs, **overrides})


def _models(seed: int = 0, config: StructuralPlasticityConfig | None = None):
    return _build_structural_comparison_models(
        TASK,
        in_features=IN_FEATURES,
        out_features=1,
        seed=seed,
        n_cells=N_OBJECTS,
        n_objects=N_OBJECTS,
        num_steps=2,
        plasticity_config=config or _config(),
    )


def _fake_data(n: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_FEATURES, generator=generator)
    y = x[:, 2::3].sum(dim=1, keepdim=True).tanh()
    return x, y


def _train(model, x, y, *, plasticity: bool, max_steps: int, seed: int = 0, config=None):
    return _train_until_convergence_structural(
        model,
        torch.nn.MSELoss(),
        x,
        y,
        x,
        y,
        torch.device("cpu"),
        seed,
        regression=True,
        batch_size=8,
        lr=1e-2,
        val_every=max(max_steps // 2, 1),
        patience_steps=10**9,  # never early-stop: these tests want a fixed step count
        max_steps=max_steps,
        plasticity_enabled=plasticity,
        is_structural=True,
    )


# ---------------------------------------------------------------------------
# The three arms, and the one-bit difference between the CellV1.5 two
# ---------------------------------------------------------------------------


def test_exactly_three_arms_and_no_retired_variants() -> None:
    """§16.1 supersedes the field/routing variants for CellV1.5; the user
    specified three arms, so a fourth silently appearing is a bug."""
    models, _ = _models()
    assert tuple(models) == STRUCTURAL_ARCHITECTURES
    assert STRUCTURAL_ARCHITECTURES == ("cellv0.1", "cellv1.5_frozen", "cellv1.5_plastic")


def test_both_cellv1_5_arms_start_bit_identical() -> None:
    models, sizing = _models()
    frozen, plastic = models["cellv1.5_frozen"], models["cellv1.5_plastic"]
    frozen_state, plastic_state = frozen.state_dict(), plastic.state_dict()
    assert set(frozen_state) == set(plastic_state)
    for key in frozen_state:
        assert torch.equal(frozen_state[key], plastic_state[key]), f"{key} differs at init"
    assert sizing["cellv1.5_frozen__params"] == sizing["cellv1.5_plastic__params"]


def test_bootstrap_topology_is_identical_across_arms() -> None:
    """Both arms must start from the same substrate, or "frozen vs.
    plastic" would also be "different bootstrap draw" (§16.8)."""
    models, _ = _models()
    x, y = _fake_data(32)
    _, frozen_boot = _train(models["cellv1.5_frozen"], x, y, plasticity=False, max_steps=1)
    _, plastic_boot = _train(models["cellv1.5_plastic"], x, y, plasticity=True, max_steps=1)
    assert frozen_boot is not None and plastic_boot is not None
    assert torch.equal(frozen_boot, plastic_boot)


def test_frozen_and_plastic_are_identical_before_first_event() -> None:
    """Up to the end of §16.9's warm-up, the two arms are the same run --
    same batches, same gradients, same parameters. If plasticity's LSH
    draws leaked into the global RNG stream, the minibatch order would
    diverge and this would fail."""
    config = StructuralPlasticityConfig(k_bar=4, warmup_steps=20, update_interval=5)
    models, _ = _models(config=config)
    x, y = _fake_data(64)
    frozen, plastic = models["cellv1.5_frozen"], models["cellv1.5_plastic"]
    _train(frozen, x, y, plasticity=False, max_steps=15, config=config)
    _train(plastic, x, y, plasticity=True, max_steps=15, config=config)

    pairs = zip(frozen.named_parameters(), plastic.named_parameters(), strict=True)
    for (name, a), (_, b) in pairs:
        assert torch.equal(a, b), f"parameter {name} diverged before the first plasticity event"
    assert torch.equal(frozen.core.edges.edge_index, plastic.core.edges.edge_index)


def test_frozen_arm_never_rewires_and_plastic_arm_does() -> None:
    config = _config(prune_fraction=0.2)
    models, _ = _models(config=config)
    x, y = _fake_data(64)

    frozen_stats, frozen_boot = _train(
        models["cellv1.5_frozen"], x, y, plasticity=False, max_steps=60, config=config
    )
    plastic_stats, plastic_boot = _train(
        models["cellv1.5_plastic"], x, y, plasticity=True, max_steps=60, config=config
    )

    assert frozen_stats["plasticity_events_total"] == 0
    assert plastic_stats["plasticity_events_total"] > 0
    # The frozen arm's topology is bit-identical to what bootstrap produced.
    assert torch.equal(models["cellv1.5_frozen"].core.edges.edge_index, frozen_boot)
    # The plastic arm's is not -- that is the independent variable working.
    assert _edge_set(models["cellv1.5_plastic"].core.edges.edge_index) != _edge_set(plastic_boot)


def test_plasticity_never_breaks_the_in_degree_one_guarantee() -> None:
    """§16.8's single safety constraint, checked at the point the
    experiment actually reads the model (the restored best checkpoint),
    not just inside `run_prune`."""
    config = _config(warmup_steps=5, prune_fraction=0.5)
    models, _ = _models(config=config)
    x, y = _fake_data(64)
    _train(models["cellv1.5_plastic"], x, y, plasticity=True, max_steps=60, config=config)
    assert int(models["cellv1.5_plastic"].core.edges.in_degree().min()) >= 1


# ---------------------------------------------------------------------------
# Checkpointing through a resized EdgeRegistry
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_when_the_edge_count_changed() -> None:
    """`structural.py`'s docstring flags that `state_dict()` only
    round-trips when the edge count matches at load time. Best-validation
    checkpointing hits exactly that case: the best checkpoint is usually
    from a step with a different edge count than wherever training
    stopped. Plain `load_state_dict` raises; `_restore_model_state` must
    not."""
    model = StructuralBeliefGraph(in_features=IN_FEATURES, out_features=1, n_cells=8, num_steps=1)
    x = torch.randn(4, IN_FEATURES)
    model.bootstrap_structural_graph(x, generator=torch.Generator().manual_seed(0))
    checkpoint = _clone_model_state(model)
    n_before = model.core.edges.n_edges

    # Simulate plasticity having changed the topology after the checkpoint.
    keep = torch.ones(n_before, dtype=torch.bool)
    keep[0] = False
    model.core.edges.remove_edges(keep)
    assert model.core.edges.n_edges != n_before
    with pytest.raises(RuntimeError):
        model.load_state_dict(checkpoint)

    _restore_model_state(model, checkpoint)
    assert model.core.edges.n_edges == n_before
    assert torch.equal(model.core.edges.edge_index, checkpoint["core.edges.edge_index"])
    assert torch.equal(model.core.edges.utility, checkpoint["core.edges.utility"])
    assert torch.equal(model.core.edges.age, checkpoint["core.edges.age"])
    # Still a working model, and the buffers are still registered buffers.
    assert torch.isfinite(model(x)).all()
    assert "core.edges.edge_index" in model.state_dict()


def test_restore_state_works_for_a_model_with_no_registry() -> None:
    """`cellv0.1` goes through the same code path and has no substrate."""
    models, _ = _models()
    cellv0_1 = models["cellv0.1"]
    checkpoint = _clone_model_state(cellv0_1)
    with torch.no_grad():
        for p in cellv0_1.parameters():
            p.add_(1.0)
    _restore_model_state(cellv0_1, checkpoint)
    for key, value in cellv0_1.state_dict().items():
        assert torch.equal(value, checkpoint[key])


# ---------------------------------------------------------------------------
# The substrate metrics, on graphs whose answers are known by inspection
# ---------------------------------------------------------------------------


def test_degree_summary_matches_hand_computed_values() -> None:
    summary = _degree_summary(np.array([0, 1, 1, 6]), "in_degree")
    assert summary["in_degree_mean"] == pytest.approx(2.0)
    assert summary["in_degree_min"] == 0
    assert summary["in_degree_max"] == 6
    assert summary["in_degree_median"] == pytest.approx(1.0)
    assert summary["in_degree_zero_fraction"] == pytest.approx(0.25)
    # Gini of a perfectly equal distribution is 0; this one is not equal.
    assert summary["in_degree_gini"] > 0.4
    assert _degree_summary(np.array([3, 3, 3, 3]), "d")["d_gini"] == pytest.approx(0.0)


def test_structural_change_fraction_is_computed_against_the_bootstrap_set() -> None:
    model = StructuralBeliefGraph(in_features=IN_FEATURES, out_features=1, n_cells=4, num_steps=1)
    bootstrap = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long)
    final = torch.tensor([[0, 1], [1, 2], [3, 1], [2, 0]], dtype=torch.long)  # 2 of 4 kept, 2 new
    model.core.edges.edge_index = final
    model.core.edges.utility = torch.zeros(4)
    model.core.edges.age = torch.zeros(4, dtype=torch.long)

    scalars, arrays = _structural_metrics(
        model, bootstrap, torch.randn(4, IN_FEATURES), torch.device("cpu"), seed=0
    )
    assert scalars["structural_edges_changed_fraction"] == pytest.approx(0.5)
    assert scalars["bootstrap_edges_retained_fraction"] == pytest.approx(0.5)
    assert scalars["n_edges_added_vs_bootstrap"] == 2
    assert scalars["n_edges_removed_vs_bootstrap"] == 2
    # in-degrees of `final`: cell0<-2, cell1<-0 and 3, cell2<-1, cell3 none.
    assert list(arrays["in_degree"]) == [1, 2, 1, 0]
    assert list(arrays["out_degree"]) == [1, 1, 1, 1]
    assert scalars["min_in_degree"] == 0.0


def test_a_frozen_topology_reports_zero_structural_change() -> None:
    models, _ = _models()
    x, y = _fake_data(64)
    frozen = models["cellv1.5_frozen"]
    _, bootstrap = _train(frozen, x, y, plasticity=False, max_steps=40)
    scalars, _ = _structural_metrics(frozen, bootstrap, x[:16], torch.device("cpu"), seed=0)
    assert scalars["structural_edges_changed_fraction"] == pytest.approx(0.0)
    assert scalars["bootstrap_edges_retained_fraction"] == pytest.approx(1.0)


def test_edge_use_rates_are_probabilities_and_the_buckets_partition() -> None:
    models, _ = _models()
    x, y = _fake_data(64)
    model = models["cellv1.5_plastic"]
    _train(model, x, y, plasticity=True, max_steps=40)
    use = _functional_edge_use(model, x[:24], torch.device("cpu"), batch_size=8)

    for tau in (0.5, 1.0, 2.0, 4.0):
        suffix = f"__tau{tau}"
        rate = use[f"_edge_use_rate_array{suffix}"]
        assert ((rate >= 0.0) & (rate <= 1.0)).all()
        assert use[f"edge_use_rate_mean{suffix}"] == pytest.approx(float(rate.mean()))
        buckets = (
            use[f"edges_never_used_fraction{suffix}"]
            + use[f"edges_always_used_fraction{suffix}"]
            + use[f"edges_input_dependent_fraction{suffix}"]
        )
        assert buckets == pytest.approx(1.0)
        assert 0.0 <= use[f"mean_pairwise_jaccard{suffix}"] <= 1.0
    # A stricter cutoff can never call more edges "in use" than a looser one.
    rates = [use[f"edge_use_rate_mean__tau{t}"] for t in (0.5, 1.0, 2.0, 4.0)]
    assert rates == sorted(rates, reverse=True)
    assert 1.0 <= use["mean_effective_in_degree"] <= float(model.core.edges.in_degree().max())


def test_edge_use_at_tau_one_is_the_uniform_share_bar() -> None:
    """The headline definition, checked against its own statement: on a
    graph where every target has exactly one in-edge, that edge supplies
    100% of the incoming precision, so it clears a `1.0/in_degree = 1.0`
    bar for every input and the use rate is exactly 1."""
    model = StructuralBeliefGraph(in_features=IN_FEATURES, out_features=1, n_cells=4, num_steps=1)
    model.core.edges.edge_index = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.long)
    model.core.edges.utility = torch.zeros(4)
    model.core.edges.age = torch.zeros(4, dtype=torch.long)
    x = torch.randn(8, IN_FEATURES)
    use = _functional_edge_use(model, x, torch.device("cpu"), batch_size=4)
    assert use["edge_use_rate_mean__tau1.0"] == pytest.approx(1.0)
    assert use["edges_always_used_fraction__tau1.0"] == pytest.approx(1.0)
    assert use["mean_effective_in_degree"] == pytest.approx(1.0, abs=1e-4)
    # And a bar strictly above the whole share is unreachable.
    assert use["edge_use_rate_mean__tau2.0"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end: every metric the experiment promises to record is recorded
# ---------------------------------------------------------------------------


def test_run_records_every_requested_metric(tmp_path) -> None:
    results = run_structural_comparison(
        TASK,
        seed=0,
        batch_size=8,
        n_train=64,
        n_val=32,
        n_test=32,
        val_every=10,
        patience_steps=20,
        max_steps=40,
        n_cells=N_OBJECTS,
        n_objects=N_OBJECTS,
        k_min=2,
        k_max=4,
        results_dir=tmp_path / "raw",
        graph_dir=tmp_path / "graphs",
        device=torch.device("cpu"),
        plasticity_config=StructuralPlasticityConfig(k_bar=4, warmup_steps=5, update_interval=5),
        edge_use_eval_n=16,
    )

    for arch in STRUCTURAL_ARCHITECTURES:
        for key in (
            "r2",  # test R^2
            "params",  # parameter count
            "steps_to_convergence",  # steps to convergence
            "wall_clock_to_convergence_seconds",  # wall-clock to convergence
            "total_steps_run",
            "hit_step_cap",
        ):
            assert f"{arch}__{key}" in results, f"missing {arch}__{key}"

    for arch in CELLV1_5_ARCHITECTURES:
        for key in (
            "structural_edges_changed_fraction",  # edges differing from bootstrap
            "edge_use_rate_mean__tau1.0",  # functional edge-use rate across inputs
            "edges_input_dependent_fraction__tau1.0",
            "mean_effective_in_degree",
            "in_degree_mean",  # final degree distribution
            "in_degree_max",
            "in_degree_gini",
            "out_degree_mean",
            "plasticity_events_total",
            "entered_freeze_window",
        ):
            assert f"{arch}__{key}" in results, f"missing {arch}__{key}"
        assert results[f"{arch}__params"] == results["cellv1.5_frozen__params"]

    assert results["cellv1.5_frozen__structural_edges_changed_fraction"] == pytest.approx(0.0)
    assert results["cellv1.5_frozen__plasticity_events_total"] == 0
    assert results["cellv1.5_plastic__plasticity_events_total"] > 0
    assert list((tmp_path / "graphs").glob("*.npz"))
    assert list((tmp_path / "raw").rglob("*.json"))


def test_complexity_levels_match_the_established_definitions() -> None:
    """`hard`/`very_hard` must be the same conditions
    `run_complexity_scaling_association.py` used, or the CellV1.4 numbers
    already on record aren't comparable."""
    assert COMPLEXITY_LEVELS == {
        "hard": {"n_objects": 96, "k_min": 8, "k_max": 14},
        "very_hard": {"n_objects": 192, "k_min": 12, "k_max": 20},
    }
