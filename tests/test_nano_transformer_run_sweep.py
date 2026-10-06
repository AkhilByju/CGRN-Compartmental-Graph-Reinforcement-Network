import json

import numpy as np

from experiments.nano_transformer.data import DatasetBundle
from experiments.nano_transformer.run_sweep import run_full_sweep


def _make_fake_dataset(tmp_path, vocab_size=48, n_tokens=6000):
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

    class _FakeTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size

        def encode(self, text):
            return [abs(hash(c)) % self.vocab_size for c in text.split()][:8] or [0]

        def decode(self, ids):
            return " ".join(str(i) for i in ids)

    return DatasetBundle(
        tokenizer=_FakeTokenizer(vocab_size),
        train_ids_path=train_path,
        val_ids_path=val_path,
        test_ids_path=test_path,
        meta={},
    )


def test_run_full_sweep_smoke(tmp_path):
    dataset = _make_fake_dataset(tmp_path)
    batch_size, context_length = 4, 16
    token_budget = batch_size * context_length * 2  # 2 optimizer steps per LR run

    summary = run_full_sweep(
        model_kinds=("swiglu", "cellv0.3"),
        seeds=(0,),
        lrs=(1e-3, 3e-4),
        token_budget=token_budget,
        batch_size=batch_size,
        context_length=context_length,
        device_override="cpu",
        results_dir=tmp_path / "results",
        log_every_steps=0,
        dataset=dataset,
    )

    assert set(summary["models"].keys()) == {"swiglu", "cellv0.3"}
    swiglu_seed0 = summary["models"]["swiglu"]["seeds"]["0"]
    assert swiglu_seed0["selected_lr"] in (1e-3, 3e-4)
    assert "nll" in swiglu_seed0["test_metrics"]
    assert len(swiglu_seed0["all_lr_val_nll"]) == 2

    cellv03_seed0 = summary["models"]["cellv0.3"]["seeds"]["0"]
    assert cellv03_seed0["mechanism_test"] is not None
    assert "deltas" in cellv03_seed0["mechanism_test"]
    assert swiglu_seed0.get("mechanism_test") in (None,)

    summary_path = tmp_path / "results" / "processed" / "sweep_summary.json"
    assert summary_path.exists()
    on_disk = json.loads(summary_path.read_text())
    assert on_disk["models"]["swiglu"]["seeds"]["0"]["selected_lr"] == swiglu_seed0["selected_lr"]

    gen_path = tmp_path / "results" / "raw" / "generations" / "swiglu_seed0.json"
    assert gen_path.exists()
    generations = json.loads(gen_path.read_text())
    assert len(generations) > 0
