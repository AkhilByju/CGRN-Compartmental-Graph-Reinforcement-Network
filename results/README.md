# Results

Generated output only — nothing here is hand-authored, and nothing under
`raw/`, `processed/`, `figures/`, or `tables/` is committed to git except
this README and the `.gitkeep` placeholders (see `.gitignore`). Regenerate
by re-running the relevant script under `scripts/` against a config under
`configs/`.

- `raw/` — per-run outputs: `RunRecord` JSON (`src/training/logging.py`) and
  checkpoints, named by run ID (never by hand).
- `processed/` — aggregated/derived data produced from `raw/` (e.g. by
  `scripts/summarize_results.py`).
- `figures/` — plots generated from `processed/`.
- `tables/` — tables generated from `processed/` (e.g. for paper drafts).
