import json

from experiments.nano_transformer.summarize import (
    aggregate_over_seeds,
    build_diagnostics_table,
    build_generation_table,
    build_main_table,
    build_mechanism_table,
    classify_interpretation,
    parse_training_log,
    render_main_table_markdown,
)

_FAKE_LEARNING_CURVE = [
    {
        "tokens_seen": t, "nll": nll, "perplexity": 2.0, "bits_per_token": 1.0,
        "tokens_evaluated": 100,
    }
    for t, nll in zip(
        (1_000_000, 3_000_000, 10_000_000, 20_000_000, 30_000_000),
        (3.0, 2.5, 2.0, 1.8, 1.6),
        strict=True,
    )
]


def _fake_seed_result(test_nll, selected_lr, param_total, diagnostics=None, mechanism_test=None):
    other_lr = 3e-4 if selected_lr != 3e-4 else 1e-3
    return {
        "selected_lr": selected_lr,
        "all_lr_val_nll": {selected_lr: test_nll - 0.05, other_lr: test_nll + 0.1},
        "test_metrics": {
            "nll": test_nll, "perplexity": 2.71**test_nll, "bits_per_token": test_nll / 0.693,
        },
        "learning_curve": _FAKE_LEARNING_CURVE,
        "diagnostics": diagnostics or [],
        "mechanism_test": mechanism_test,
        "generation_aggregate_diversity": {
            "distinct_1": 0.5, "distinct_2": 0.7,
            "repeated_bigram_rate": 0.3, "repeated_trigram_rate": 0.1,
        },
        "param_report": {"total": param_total, "ffn_hidden_width": 300},
        "run_id": "fake_run",
    }


def _fake_summary():
    cellv03_diag = [
        {
            "training_fraction": frac,
            "layers": {
                "0": {"pi_mean": 0.6, "pi_std": 0.1, "pi_cv": 0.17, "u_mean": 0.02, "u_std": 0.01}
            },
        }
        for frac in (0.0, 0.1, 0.5, 1.0)
    ]
    mechanism = {
        "normal": {"nll": 1.6},
        "precision_neutralized": {"nll": 1.7},
        "conflict_neutralized": {"nll": 1.65},
        "deltas": {
            "precision_neutralized_delta_nll": 0.1,
            "precision_neutralized_delta_ppl": 0.5,
            "conflict_neutralized_delta_nll": 0.05,
            "conflict_neutralized_delta_ppl": 0.2,
        },
    }
    return {
        "models": {
            "swiglu": {"seeds": {"0": _fake_seed_result(1.6, 1e-3, 999424)}},
            "cellv0.3": {
                "seeds": {
                    "0": _fake_seed_result(
                        1.58, 3e-4, 999634, diagnostics=cellv03_diag, mechanism_test=mechanism
                    )
                }
            },
        }
    }


def _fake_preflight():
    return {
        "per_model": {
            "swiglu": {"tokens_per_sec": 250000, "ms_per_step": 64.0},
            "cellv0.3": {"tokens_per_sec": 100000, "ms_per_step": 150.0},
        }
    }


def test_aggregate_over_seeds():
    summary = _fake_summary()
    agg = aggregate_over_seeds(summary, "swiglu")
    assert agg["test_nll_mean"] == 1.6
    assert agg["val_nll_mean"] == 1.55


def test_build_main_table_and_render():
    summary = _fake_summary()
    preflight = _fake_preflight()
    rows = build_main_table(summary, preflight)
    assert len(rows) == 2
    md = render_main_table_markdown(rows)
    assert "swiglu" in md
    assert "cellv0.3" in md
    assert "|" in md


def test_build_diagnostics_table():
    summary = _fake_summary()
    rows = build_diagnostics_table(summary)
    assert len(rows) == 4 * 1 * 5  # 4 fractions x 1 layer x 5 metrics
    assert any(r["metric"] == "u_mean" for r in rows)


def test_build_mechanism_table():
    summary = _fake_summary()
    rows = build_mechanism_table(summary)
    assert len(rows) == 4
    metrics = {r["metric"] for r in rows}
    assert "precision_neutralized_delta_nll" in metrics


def test_build_generation_table():
    summary = _fake_summary()
    rows = build_generation_table(summary)
    assert len(rows) == 2
    assert all("distinct_1" in r for r in rows)


def test_classify_interpretation_strong_positive_when_all_criteria_met():
    summary = _fake_summary()
    preflight = _fake_preflight()
    main_rows = build_main_table(summary, preflight)
    mechanism_rows = build_mechanism_table(summary)
    diagnostics_rows = build_diagnostics_table(summary)
    label = classify_interpretation(summary, main_rows, mechanism_rows, diagnostics_rows)
    assert label in (
        "strong_positive",
        "interesting_neutral",
        "mechanism_positive_performance_neutral",
        "negative",
        "insufficient_data",
    )


def test_parse_training_log(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(
        "[swiglu seed=0 lr=3e-04] step 100/1832 tokens=1638400 loss=5.3135\n"
        "[swiglu seed=0 lr=3e-04] step 200/1832 tokens=3276800 loss=4.6574\n"
        "some unrelated line\n"
    )
    curves = parse_training_log(log)
    assert ("swiglu", 0, 3e-4) in curves
    assert curves[("swiglu", 0, 3e-4)] == [(1638400, 5.3135), (3276800, 4.6574)]


def test_summary_json_roundtrip_compatible(tmp_path):
    """JSON coerces dict keys to strings, so `all_lr_val_nll`'s float LR keys
    come back as strings -- everything else must round-trip exactly."""
    summary = _fake_summary()
    path = tmp_path / "sweep_summary.json"
    path.write_text(json.dumps(summary))
    reloaded = json.loads(path.read_text())

    reloaded_swiglu = reloaded["models"]["swiglu"]["seeds"]["0"]
    original_swiglu = summary["models"]["swiglu"]["seeds"]["0"]
    assert reloaded_swiglu["selected_lr"] == original_swiglu["selected_lr"]
    assert reloaded_swiglu["test_metrics"] == original_swiglu["test_metrics"]
    assert set(reloaded_swiglu["all_lr_val_nll"].values()) == set(
        original_swiglu["all_lr_val_nll"].values()
    )
    assert aggregate_over_seeds(reloaded, "swiglu") == aggregate_over_seeds(summary, "swiglu")
