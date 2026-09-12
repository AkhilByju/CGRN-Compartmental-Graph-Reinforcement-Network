import types

import numpy as np
import torch

from experiments.nano_transformer.data import DatasetBundle
from experiments.nano_transformer.training import (
    RunConfig,
    evaluate_nll,
    select_best_lr,
    tokens_per_step,
    total_optimizer_steps,
    train_one_run,
)


def _make_fake_dataset(tmp_path, vocab_size=64, n_tokens=4000):
    rng = np.random.default_rng(0)
    train_ids = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    val_ids = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    train_path = tmp_path / "train_ids.npy"
    val_path = tmp_path / "val_ids.npy"
    test_path = tmp_path / "test_ids.npy"
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    np.save(test_path, val_ids)
    fake_tokenizer = types.SimpleNamespace(vocab_size=vocab_size)
    return DatasetBundle(
        tokenizer=fake_tokenizer,
        train_ids_path=train_path,
        val_ids_path=val_path,
        test_ids_path=test_path,
        meta={},
    )


def test_total_optimizer_steps_matches_token_budget():
    steps = total_optimizer_steps(token_budget=30_000_000)
    assert steps * tokens_per_step() >= 30_000_000
    assert (steps - 1) * tokens_per_step() < 30_000_000


def test_train_one_run_smoke_for_each_model_kind(tmp_path):
    dataset = _make_fake_dataset(tmp_path)
    for model_kind in ("swiglu", "cellv0.3", "fixed_confidence"):
        batch_size, context_length = 4, 32
        config = RunConfig(
            model_kind=model_kind,
            seed=0,
            learning_rate=1e-3,
            token_budget=batch_size * context_length * 2,  # 2 optimizer steps
            context_length=context_length,
            batch_size=batch_size,
            device_override="cpu",
            checkpoint_dir=tmp_path / "checkpoints",
            run_record_dir=tmp_path / "run_records",
        )

        result = train_one_run(config, dataset, log_every_steps=0)
        assert result["record"].steps_completed == 2
        assert result["record"].parameter_count > 0
        assert "nll" in result["record"].validation_metrics
        assert torch.isfinite(torch.tensor(result["record"].validation_metrics["nll"]))
        if model_kind == "cellv0.3":
            assert len(result["diagnostics"]) >= 1
        else:
            assert result["diagnostics"] == []


def test_evaluate_nll_is_deterministic_and_finite(tmp_path):
    dataset = _make_fake_dataset(tmp_path)
    from experiments.nano_transformer.training import build_model_and_spec

    torch.manual_seed(0)
    model, _ = build_model_and_spec("swiglu", dataset.tokenizer.vocab_size)
    device = torch.device("cpu")
    model.to(device)

    from experiments.nano_transformer.data import load_token_ids

    val_ids = load_token_ids(dataset.val_ids_path)
    metrics_a = evaluate_nll(model, val_ids, batch_size=4, context_length=32, device=device)
    metrics_b = evaluate_nll(model, val_ids, batch_size=4, context_length=32, device=device)
    assert metrics_a == metrics_b
    assert metrics_a["nll"] > 0
    assert metrics_a["perplexity"] > 1


def test_select_best_lr_picks_lower_val_nll():
    class _FakeRecord:
        def __init__(self, nll):
            self.validation_metrics = {"nll": nll}

    results = [
        {"record": _FakeRecord(2.5), "tag": "high"},
        {"record": _FakeRecord(1.1), "tag": "low"},
    ]
    best = select_best_lr(results)
    assert best["tag"] == "low"
