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

## CellV0.2 follow-up (separate, later, user-commissioned)

`docs/architecture_v0.md` §9 — **CellV0.2**, the Conservative Precision-Gain
Cell — was specified in full by the user on 2026-09-09 as its own
architecture line and run through this **identical frozen protocol** (same
datasets, splits, fractions, seeds, AdamW settings, early stopping). This is
not a modification of the screen above: CellV0.1 and the matched MLP are not
re-run; their recorded `paper_a_phase1` records are reused for comparison.

| File | Role |
|---|---|
| `run_phase1_v02.py` | The 42 CellV0.2 runs (experiment id `paper_a_phase1_cellv02`). |
| `summarize_v02.py` | CellV0.2 tables + paired diffs vs the recorded arms → `phase1_v02_results.md`. |
| `bench_cellv02_layer.py` | Isolated layer forward/backward timing vs CellV0.1 and `Linear+Tanh`. |

CellV0.2 is deliberately **not** in `models.MODEL_FAMILIES`, so
`run_phase1.py` never sweeps it into the original screen.

## CellV0.3 follow-up (separate, later, user-commissioned)

`docs/architecture_v0.md` §10 — **CellV0.3**, the Conflict-Normalized Belief
Cell — was specified in full by the user on 2026-09-09 as its own
architecture line and run through this **identical frozen protocol**. Same
one-signed-matrix parameterization as CellV0.2, but CellV0.2's
population-relative precision gain is removed and each cell folds
`sqrt(precision)` into its own activation. CellV0.1, CellV0.2 and the matched
MLP are not re-run; their recorded records are reused for comparison.

| File | Role |
|---|---|
| `run_phase1_v03.py` | The 42 CellV0.3 runs (experiment id `paper_a_phase1_cellv03`). |
| `summarize_v03.py` | CellV0.3 tables + paired diffs vs the recorded MLP / CellV0.1 / CellV0.2, plus the init-vs-best-checkpoint precision-mechanism report → `phase1_v03_results.md`. |
| `bench_cellv03_layer.py` | Isolated layer forward/backward timing vs CellV0.1, CellV0.2, and `Linear+Tanh`. |

CellV0.3 is deliberately **not** in `models.MODEL_FAMILIES`, so
`run_phase1.py` never sweeps it into the original screen.

## Phase 2 — reliability / corruption benchmark (separate sub-package)

`reliability/` — the **primary go/no-go experiment** for the CellV0.3 paper
direction (user-commissioned 2026-09-09). Frozen CellV0.3 is put in the regime
it was designed for: inputs with heterogeneous, **known** reliability. Four
parameter-matched families (Plain MLP, Confidence-Augmented MLP,
Reliability-Gated MLP, CellV0.3) × two corruption families (missing-feature,
heterogeneous Gaussian) × 4 datasets (MNIST, Fashion-MNIST, Digits, California
Housing) × 3 seeds = 96 runs. For Phase 2 only, CellV0.3's *input belief* is
`(μ = x_corrupted, e = reliability, u = 0)` — input data, not an architecture
change; `src/models/architecture_v0/conflict_normalized.py` is untouched.

See `reliability/README.md` and `reliability/reliability_results.md`. Result
(2026-09-09): **positive but not uniform** — a real robustness advantage over
the reliability-aware baselines on the image tasks, a loss on Digits/missing,
mixed elsewhere. `docs/research_log.md` 2026-09-09.
