from pathlib import Path

from src.utilities.config import ExperimentConfig, config_hash, load_config, make_run_id


def test_load_config_round_trip(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment_id: "001_baseline_validation"
architecture: "mlp_baseline"
dataset: "synthetic_regression_linear"
seed: 0
learning_rate: 0.01
extra:
  hidden_dim: 64
compartment_count: 4
"""
    )
    config = load_config(config_path)
    assert config.experiment_id == "001_baseline_validation"
    assert config.learning_rate == 0.01
    assert config.extra["hidden_dim"] == 64
    # Unknown top-level keys fold into `extra` rather than being dropped or erroring.
    assert config.extra["compartment_count"] == 4


def test_config_hash_is_deterministic_and_order_independent() -> None:
    a = ExperimentConfig(
        experiment_id="e", architecture="m", dataset="d", seed=0, extra={"x": 1, "y": 2}
    )
    b = ExperimentConfig(
        experiment_id="e", architecture="m", dataset="d", seed=0, extra={"y": 2, "x": 1}
    )
    assert config_hash(a) == config_hash(b)


def test_config_hash_changes_with_content() -> None:
    a = ExperimentConfig(experiment_id="e", architecture="m", dataset="d", seed=0)
    b = ExperimentConfig(experiment_id="e", architecture="m", dataset="d", seed=1)
    assert config_hash(a) != config_hash(b)


def test_make_run_id_contains_experiment_id_and_hash() -> None:
    config = ExperimentConfig(
        experiment_id="002_cell_v0", architecture="cellv0", dataset="d", seed=0
    )
    run_id = make_run_id(config)
    assert run_id.startswith("002_cell_v0_")
    assert config_hash(config) in run_id
