"""Optional scaling experiment (task Sec 17): ONLY after the full 1M sweep
finishes. ModernTransformer vs CellV03Transformer at ~500k and ~2M
parameters, seed=0 only, reusing each model's selected-best LR from the 1M
experiment (`sweep_summary.json`) -- tokenizer, dataset, context length,
training-token budget, and backbone design stay identical; only FFN hidden
width is re-solved via the existing param_solver for each new target.

Does not touch FixedConfidenceCellTransformer (Sec 17 only asks about
ModernTransformer vs CellV03Transformer) and does not re-run the LR grid at
the new scales, per the spec: "Do not tune architectures independently at
each scale."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.nano_transformer.data import load_token_ids, prepare_dataset
from experiments.nano_transformer.training import (
    CONTEXT_LENGTH,
    TOKEN_BUDGET,
    RunConfig,
    evaluate_nll,
    train_one_run,
)
from src.utilities.device import get_device

_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"
SCALING_TARGETS: tuple[int, ...] = (500_000, 2_000_000)
SCALING_MODEL_KINDS: tuple[str, ...] = ("swiglu", "cellv0.3")


def run_scaling_experiment(
    sweep_summary: dict[str, Any],
    targets: tuple[int, ...] = SCALING_TARGETS,
    model_kinds: tuple[str, ...] = SCALING_MODEL_KINDS,
    seed: int = 0,
    token_budget: int = TOKEN_BUDGET,
    context_length: int = CONTEXT_LENGTH,
    device_override: str | None = None,
    results_dir: Path = RESULTS_DIR,
    dataset: Any | None = None,
    log_every_steps: int = 100,
) -> dict[str, Any]:
    dataset = dataset if dataset is not None else prepare_dataset()
    test_ids = load_token_ids(dataset.test_ids_path)
    device = get_device(device_override)

    results: dict[str, Any] = {"targets": {}}
    for target in targets:
        results["targets"][str(target)] = {}
        for model_kind in model_kinds:
            selected_lr = sweep_summary["models"][model_kind]["seeds"][str(seed)]["selected_lr"]
            config = RunConfig(
                model_kind=model_kind,
                seed=seed,
                learning_rate=selected_lr,
                token_budget=token_budget,
                context_length=context_length,
                target_params=target,
                device_override=device_override,
                checkpoint_dir=results_dir / "raw" / "checkpoints" / "scaling",
                run_record_dir=results_dir / "raw" / "run_records" / "scaling",
            )
            result = train_one_run(config, dataset, log_every_steps=log_every_steps)
            test_metrics = evaluate_nll(
                result["model"], test_ids, config.batch_size, config.context_length, device
            )
            results["targets"][str(target)][model_kind] = {
                "reused_lr": selected_lr,
                "param_report": result["param_report"],
                "val_nll": result["record"].validation_metrics["nll"],
                "test_metrics": test_metrics,
                "learning_curve": result["learning_curve"],
            }
            print(
                f"[scaling] target={target} {model_kind}: params="
                f"{result['param_report']['total']} test_nll={test_metrics['nll']:.4f}"
            )

    out_path = results_dir / "processed" / "scaling_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
    print(f"[scaling] wrote {out_path}")
    return results


if __name__ == "__main__":
    from experiments.nano_transformer.summarize import load_sweep_summary

    run_scaling_experiment(load_sweep_summary())
