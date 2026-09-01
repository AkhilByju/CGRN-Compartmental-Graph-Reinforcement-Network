import json
from pathlib import Path

from src.training.logging import RunRecord, write_run_record


def _make_record(run_id: str = "run_0000000000_20260101T000000Z") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        experiment_id="001_baseline_validation",
        architecture="mlp_baseline",
        config={"seed": 0},
        parameter_count=100,
        dataset="synthetic_regression_linear",
        seed=0,
        optimizer="adamw",
        learning_rate=1e-3,
        steps_completed=10,
        examples_or_tokens_seen=320,
        refinement_iterations=None,
        approximate_flops=None,
        train_wall_clock_seconds=1.23,
        inference_wall_clock_seconds=None,
        validation_metrics={"mae": 0.1},
        test_metrics={"mae": 0.12},
        git_commit=None,
        checkpoint_path=None,
    )


def test_write_run_record_creates_expected_json(tmp_path: Path) -> None:
    record = _make_record()
    out_path = write_run_record(record, tmp_path)
    assert out_path.exists()
    with out_path.open() as f:
        data = json.load(f)
    assert data["run_id"] == record.run_id
    assert data["validation_metrics"]["mae"] == 0.1
