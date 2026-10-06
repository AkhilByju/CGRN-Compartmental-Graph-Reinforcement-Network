"""Drives summarize.py end-to-end: loads the real sweep + preflight results,
renders the plots, and writes `cellv03_transformer_results.md` (task Sec 19).
Run once the full sweep (`run_sweep.py`) has finished.
"""

from __future__ import annotations

from pathlib import Path

from experiments.nano_transformer.summarize import (
    FIGURES_DIR,
    RESULTS_DIR,
    build_diagnostics_table,
    build_generation_table,
    build_main_table,
    build_mechanism_table,
    classify_interpretation,
    load_preflight,
    load_sweep_summary,
    parse_training_log,
    plot_learning_curves,
    plot_training_loss,
    render_main_table_markdown,
)

_TRAINING_LOG_PATH = RESULTS_DIR / "raw" / "run_sweep_console.log"


def build_all(out_dir: Path = RESULTS_DIR) -> dict:
    summary = load_sweep_summary()
    preflight = load_preflight()

    main_rows = build_main_table(summary, preflight)
    diagnostics_rows = build_diagnostics_table(summary)
    mechanism_rows = build_mechanism_table(summary)
    generation_rows = build_generation_table(summary)
    interpretation = classify_interpretation(summary, main_rows, mechanism_rows, diagnostics_rows)

    learning_curve_path = plot_learning_curves(summary, FIGURES_DIR / "learning_curves.png")

    training_loss_path = None
    if _TRAINING_LOG_PATH.exists():
        curves = parse_training_log(_TRAINING_LOG_PATH)
        if curves:
            training_loss_path = plot_training_loss(
                curves, summary, FIGURES_DIR / "training_loss.png"
            )

    return {
        "main_table_markdown": render_main_table_markdown(main_rows),
        "main_rows": main_rows,
        "diagnostics_rows": diagnostics_rows,
        "mechanism_rows": mechanism_rows,
        "generation_rows": generation_rows,
        "interpretation": interpretation,
        "learning_curve_path": learning_curve_path,
        "training_loss_path": training_loss_path,
    }


if __name__ == "__main__":
    import json

    result = build_all()
    print(result["main_table_markdown"])
    print(f"\ninterpretation bucket: {result['interpretation']}")
    print(json.dumps(result["mechanism_rows"], indent=2))
