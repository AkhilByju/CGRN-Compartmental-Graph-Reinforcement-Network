from pathlib import Path

import torch

from src.training.checkpointing import checkpoint_path, load_checkpoint, save_checkpoint


def test_save_and_load_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    run_id = "test_run_0000000000_20260101T000000Z"

    path = save_checkpoint(model, optimizer, run_id, tmp_path)
    assert path.exists()

    loaded = load_checkpoint(path)
    assert loaded["run_id"] == run_id
    assert "model_state_dict" in loaded
    assert "optimizer_state_dict" in loaded


def test_checkpoint_path_includes_step_when_given(tmp_path: Path) -> None:
    no_step = checkpoint_path("run", tmp_path)
    with_step = checkpoint_path("run", tmp_path, step=10)
    assert no_step != with_step
    assert "step10" in with_step.name
