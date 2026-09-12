import numpy as np

from experiments.nano_transformer.data import DatasetBundle
from experiments.nano_transformer.run_scaling import run_scaling_experiment


def _make_fake_dataset(tmp_path, vocab_size=48, n_tokens=4000):
    import types

    rng = np.random.default_rng(0)
    train_ids = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    val_ids = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    test_ids = rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)
    train_path = tmp_path / "train_ids.npy"
    val_path = tmp_path / "val_ids.npy"
    test_path = tmp_path / "test_ids.npy"
    np.save(train_path, train_ids)
    np.save(val_path, val_ids)
    np.save(test_path, test_ids)
    fake_tokenizer = types.SimpleNamespace(vocab_size=vocab_size)
    return DatasetBundle(
        tokenizer=fake_tokenizer,
        train_ids_path=train_path,
        val_ids_path=val_path,
        test_ids_path=test_path,
        meta={},
    )


def test_run_scaling_experiment_smoke(tmp_path):
    dataset = _make_fake_dataset(tmp_path)
    fake_sweep_summary = {
        "models": {
            "swiglu": {"seeds": {"0": {"selected_lr": 1e-3}}},
            "cellv0.3": {"seeds": {"0": {"selected_lr": 3e-4}}},
        }
    }
    batch_size, context_length = 4, 16
    token_budget = batch_size * context_length * 2

    results = run_scaling_experiment(
        fake_sweep_summary,
        targets=(2000, 5000),
        model_kinds=("swiglu", "cellv0.3"),
        seed=0,
        token_budget=token_budget,
        context_length=context_length,
        device_override="cpu",
        results_dir=tmp_path / "results",
        dataset=dataset,
        log_every_steps=0,
    )

    assert set(results["targets"].keys()) == {"2000", "5000"}
    for target_results in results["targets"].values():
        assert set(target_results.keys()) == {"swiglu", "cellv0.3"}
        for model_kind, r in target_results.items():
            assert r["reused_lr"] in (1e-3, 3e-4)
            assert "nll" in r["test_metrics"]

    out_path = tmp_path / "results" / "processed" / "scaling_summary.json"
    assert out_path.exists()
