import torch

from experiments.nano_transformer.preflight import (
    benchmark_belief_ffn_kernel_alone,
    preflight_one_model,
    run_preflight,
)


def test_preflight_one_model_smoke():
    device = torch.device("cpu")
    result = preflight_one_model(
        "cellv0.3",
        vocab_size=64,
        device=device,
        batch_size=2,
        context_length=16,
        warmup=1,
        measured=2,
    )
    assert result["ms_per_step"] > 0
    assert result["tokens_per_sec"] > 0
    assert result["params"] > 0


def test_benchmark_belief_ffn_kernel_alone_smoke():
    device = torch.device("cpu")
    result = benchmark_belief_ffn_kernel_alone(
        d_model=16, d_hidden=8, batch=2, context_length=8, device=device, warmup=1, measured=2
    )
    assert result["belief_ffn_ms_per_fwd_bwd"] > 0
    assert result["swiglu_ffn_ms_per_fwd_bwd"] > 0
    assert result["belief_over_swiglu_ratio"] > 0


def test_run_preflight_covers_all_models():
    device = torch.device("cpu")
    results = run_preflight(
        vocab_size=64, device=device, batch_size=2, context_length=16, warmup=1, measured=2
    )
    assert set(results["per_model"].keys()) == {"swiglu", "cellv0.3", "fixed_confidence"}
    for r in results["per_model"].values():
        assert "slowdown_vs_swiglu" in r
    assert "belief_ffn_kernel_alone" in results
