import numpy as np
import torch

from experiments.nano_transformer.interventions import (
    model_supports_belief_interventions,
    run_critical_mechanism_test,
)
from experiments.nano_transformer.training import build_model_and_spec


def _fake_ids(vocab_size=32, n_tokens=2000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab_size, size=n_tokens, dtype=np.uint16)


def test_non_belief_models_return_none():
    for kind in ("swiglu", "fixed_confidence"):
        model, _ = build_model_and_spec(kind, vocab_size=32, context_length=32)
        assert not model_supports_belief_interventions(model)
        ids = _fake_ids()
        assert run_critical_mechanism_test(model, ids, 4, 32, torch.device("cpu")) is None


def test_belief_model_intervention_report_shape_and_differs_from_normal():
    torch.manual_seed(0)
    model, _ = build_model_and_spec("cellv0.3", vocab_size=32, context_length=32)
    assert model_supports_belief_interventions(model)

    ids = _fake_ids()
    report = run_critical_mechanism_test(model, ids, 4, 32, torch.device("cpu"))
    assert report is not None
    for key in ("normal", "precision_neutralized", "conflict_neutralized", "deltas"):
        assert key in report
    for mode in ("normal", "precision_neutralized", "conflict_neutralized"):
        assert torch.isfinite(torch.tensor(report[mode]["nll"]))

    # An untrained random model still has nontrivial gamma/consensus, so the
    # precision-neutralization ablation should generically change the loss.
    assert report["deltas"]["precision_neutralized_delta_nll"] != 0.0
