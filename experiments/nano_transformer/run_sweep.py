"""Orchestrates the full task Sec 10-13 comparison: for every (model kind,
seed) pair, train both LR-grid runs for the full token budget, select the
better one by validation NLL alone, then run its single held-out test
evaluation, the critical mechanism test (Sec 12, cellv0.3 only), and the
generation evaluation (Sec 13) -- once per (model, seed), not per LR run.

Raw per-generation output is saved separately (task Sec 19: "Save raw run
records separately") under `results/raw/generations/`; this module's return
value / `results/processed/sweep_summary.json` holds only the aggregated
numbers `summarize.py` needs to build the final results doc.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.nano_transformer.data import load_token_ids, prepare_dataset
from experiments.nano_transformer.generation import run_generation_eval
from experiments.nano_transformer.interventions import run_critical_mechanism_test
from experiments.nano_transformer.training import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    LR_GRID,
    SEEDS,
    TOKEN_BUDGET,
    RunConfig,
    evaluate_nll,
    select_best_lr,
    train_one_run,
)
from src.training.logging import write_run_record
from src.utilities.device import get_device

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"


def run_full_sweep(
    model_kinds: tuple[str, ...] = ("swiglu", "cellv0.3", "fixed_confidence"),
    seeds: tuple[int, ...] = SEEDS,
    lrs: tuple[float, ...] = LR_GRID,
    token_budget: int = TOKEN_BUDGET,
    batch_size: int = BATCH_SIZE,
    context_length: int = CONTEXT_LENGTH,
    device_override: str | None = None,
    results_dir: Path = RESULTS_DIR,
    log_every_steps: int = 100,
    dataset: Any | None = None,
) -> dict[str, Any]:
    dataset = dataset if dataset is not None else prepare_dataset()
    test_ids = load_token_ids(dataset.test_ids_path)
    device = get_device(device_override)

    summary: dict[str, Any] = {"models": {}}
    for model_kind in model_kinds:
        summary["models"][model_kind] = {"seeds": {}}
        for seed in seeds:
            run_record_dir = results_dir / "raw" / "run_records"
            lr_runs = []
            for lr in lrs:
                config = RunConfig(
                    model_kind=model_kind,
                    seed=seed,
                    learning_rate=lr,
                    token_budget=token_budget,
                    batch_size=batch_size,
                    context_length=context_length,
                    device_override=device_override,
                    checkpoint_dir=results_dir / "raw" / "checkpoints",
                    run_record_dir=run_record_dir,
                )
                result = train_one_run(config, dataset, log_every_steps=log_every_steps)
                lr_runs.append(result)
                val_nll = result["record"].validation_metrics["nll"]
                print(f"[sweep] {model_kind} seed={seed} lr={lr:.0e}: val_nll={val_nll:.4f}")

            best = select_best_lr(lr_runs)
            model = best["model"]

            test_metrics = evaluate_nll(model, test_ids, batch_size, context_length, device)
            best["record"].test_metrics = test_metrics
            write_run_record(best["record"], run_record_dir)  # persist test_metrics too

            mechanism_test = run_critical_mechanism_test(
                model, test_ids, batch_size, context_length, device
            )

            gen_eval = run_generation_eval(model, dataset.tokenizer, device, context_length)
            gen_path = results_dir / "raw" / "generations" / f"{model_kind}_seed{seed}.json"
            gen_path.parent.mkdir(parents=True, exist_ok=True)
            gen_path.write_text(json.dumps(gen_eval.generations, indent=2))

            summary["models"][model_kind]["seeds"][str(seed)] = {
                "selected_lr": best["record"].learning_rate,
                "all_lr_val_nll": {
                    r["record"].learning_rate: r["record"].validation_metrics["nll"]
                    for r in lr_runs
                },
                "test_metrics": test_metrics,
                "learning_curve": best["learning_curve"],
                "diagnostics": best["diagnostics"],
                "mechanism_test": mechanism_test,
                "generation_aggregate_diversity": gen_eval.aggregate_diversity,
                "param_report": best["param_report"],
                "run_id": best["record"].run_id,
                "train_wall_clock_seconds": best["record"].train_wall_clock_seconds,
            }
            print(
                f"[sweep] {model_kind} seed={seed}: selected lr={best['record'].learning_rate:.0e}"
                f" test_nll={test_metrics['nll']:.4f}"
            )

    summary_path = results_dir / "processed" / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    print(f"[sweep] wrote {summary_path}")
    return summary


if __name__ == "__main__":
    run_full_sweep()
