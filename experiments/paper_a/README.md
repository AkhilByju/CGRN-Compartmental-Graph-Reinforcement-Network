# Paper A — CellV0.1 standalone validation (Phase 1)

**Question this phase answers:** does the frozen CellV0.1 primitive
(`BeliefNetwork` with `scale_stable_precision` aggregation) deserve a paper?
This is a **screening** experiment — implementation and runs only. No CellV0.1
redesign, no new math, no architecture tuning after seeing results
(Paper-A task Sec 12).

CellV0.1 is **frozen**. Nothing in this folder modifies
`src/models/architecture_v0/` — it only composes those modules with
`src/training`, `src/evaluation`, `src/utilities`, and public-dataset loaders.

## Layout

| File | Role |
|---|---|
| `datasets.py` | Frozen 7-dataset list, deterministic seeded splits, train-only preprocessing, 25%/100% subsampling. `prepare_dataset(name, seed, train_fraction)`. |
| `models.py` | The three model families + the fixed-confidence ablation, parameter-matching, CellV0.1 internal diagnostics. |
| `training.py` | One shared protocol: AdamW, lr=1e-2, weight_decay=0, best-val checkpoint restore, early stopping, 15000-step cap. Not tuned per model/dataset. |
| `harness.py` | `run_one(dataset, train_fraction, family, seed)` → trains, evaluates, writes a full-provenance `RunRecord` + validation curve. |
| `run_phase1.py` | The ≤126-run screening grid: 7 datasets × {25%, 100%} × 3 families × 3 seeds. |
| `run_ablation.py` | Sec 11 fixed-confidence ablation: Digits / Diabetes / Fashion-MNIST at 25%, seeds 0–2. |
| `duplication_experiment.py` | Sec 10 deterministic duplication-invariance check (no training). |
| `summarize.py` | Builds report Tables A–G from `results/raw/` → `phase1_results.md`. |

## Benchmark suite (frozen before looking at any CellV0.1 result)

| Dataset | Task | Param budget | Source |
|---|---|---|---|
| Breast Cancer Wisconsin | binary classification | ~10k | `sklearn.datasets.load_breast_cancer` |
| Wine | 3-class classification | ~10k | `sklearn.datasets.load_wine` |
| Digits | 10-class classification | ~25k | `sklearn.datasets.load_digits` |
| Diabetes | regression | ~10k | `sklearn.datasets.load_diabetes` |
| California Housing | regression | ~10k | `sklearn.datasets.fetch_california_housing` (downloaded; skipped **with an explicit notice** if unavailable) |
| MNIST | 10-class classification (flattened) | ~150k | `fetch_openml('mnist_784')`, official 60k/10k split |
| Fashion-MNIST | 10-class classification (flattened) | ~150k | `fetch_openml('Fashion-MNIST')`, official 60k/10k split |

Images are flattened and globally standardized (train-subset mean/std) — **no
CNNs, no feature extractors** in Phase 1; this tests the neuron primitive, not
convolutional architecture.

## Model families

* **A — CellV0.1**: `BeliefNetwork(scale_stable_precision)`, largest hidden-cell
  count within the param budget. Frozen.
* **B — MLP (parameter-matched)**: `Linear → SiLU → Linear`, hidden width chosen
  so the parameter count is within 2% of CellV0.1's actual count.
* **C — MLP (state-count control)**: same 1-hidden-layer SiLU MLP with hidden
  width `3 × CellV0.1_hidden_cells`. **Not** parameter-matched — its parameter
  count (larger on the image datasets, smaller on the low-feature tabular ones,
  because CellV0.1 spends two parameters per connection) is reported, not
  equalized.

## Running

```bash
# full screen (auto device; ~1-2h, MNIST/Fashion-MNIST CellV0.1 runs dominate)
python experiments/paper_a/run_phase1.py

# deterministic, seconds:
python experiments/paper_a/duplication_experiment.py

# fixed-confidence ablation (9 runs):
python experiments/paper_a/run_ablation.py

# build the report tables from whatever is in results/raw/:
python experiments/paper_a/summarize.py
```

Raw `RunRecord`s land in `results/raw/` (gitignored). The committed artifact is
`phase1_results.md` plus the dated entry in `docs/research_log.md`.

## Anti-cherry-picking rules in force (Paper-A task Sec 12)

Once the screen starts: do not alter CellV0.1, do not drop datasets where it
loses, do not change the train fractions, do not add dataset-specific CellV0.1
hyperparameters, do not rerun only bad CellV0.1 seeds, do not tune the
evidence/uncertainty formulas, do not create CellV0.2 or pull in CellV1.x.
Negative and null results stay in the report.
