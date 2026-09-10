# Paper A — Phase 2: reliability / corruption benchmark for CellV0.3

**The primary go/no-go experiment for the CellV0.3 paper direction.** CellV0.3
is **frozen** (Phase-1 result: `../phase1_v03_results.md`, `docs/research_log.md`
2026-09-09). This phase is implementation + experiment only — no CellV0.3
redesign, no equation changes, no CellV0.4, no confidence-formula tuning, no
auxiliary losses.

## The question

> When input information has heterogeneous and **known** reliability, does
> explicitly propagating reliability/conflict through CellV0.3 provide a useful
> inductive bias beyond conventional neural networks given the same corrupted
> observations and the same reliability information?

A CellV0.3 win over a plain MLP is **not** sufficient — the plain MLP has no
reliability signal. The meaningful comparisons are against the two
reliability-aware baselines (Confidence-Augmented MLP, Reliability-Gated MLP).

## What is and isn't changed

CellV0.3 stays exactly `state = (mu, e, u)` with its conflict-normalized
equations. For Phase 2 **only**, its *input belief* is initialized from the
known observation reliability:

```
mu = x_corrupted     e = reliability c     u = 0
```

This is **input data**, not an architecture modification — `BeliefNetworkV03`
in `src/models/architecture_v0/conflict_normalized.py` is untouched;
`models.ReliabilityCellV03` is a thin wrapper that composes its existing
`layer1 / layer2 / readout` with the reliability-carrying input belief.

CellV0.1, CellV0.2, CellV0.3, and every Phase-1 dataset / split / preprocessing
/ result are reused verbatim, never re-run or modified.

## Frozen benchmark (fixed before looking at any result)

| Dataset | Task | Param budget | Why (already decided) |
|---|---|---|---|
| MNIST | 10-class | 150k | public image problem where V0.3 was competitive |
| Fashion-MNIST | 10-class | 150k | public image problem where V0.3 was competitive |
| Digits | 10-class | 25k | smaller image-like problem, hard for V0.1 |
| California Housing | regression | 10k | reasonably stable public regression |

Diabetes is **excluded** — Phase 1 showed substantial baseline instability there.
Same clean splits / preprocessing / parameter budgets as Phase 1; corruption is
applied **after** preprocessing so a Gaussian `sigma` is in standardized units.

## Two corruption families

| | Training regime (per-example draw) | Reliability `c` | Eval grid |
|---|---|---|---|
| **Missing-feature** | mask each feature w.p. `p ~ U{0, .1, .2, .3}`; missing → `x = 0` | `1.0` observed / `1e-3` missing (frozen) | `p ∈ {0, .1, .3, .5, .7}` |
| **Heterogeneous Gaussian** | per feature `sigma_j ~ U(0, s)`, `x_j += N(0, sigma_j^2)`; `s ~ U{0, .25, .5, .75}` | `c_j = 1 / (1 + sigma_j^2)` (frozen) | `s ∈ {0, .25, .5, .75, 1.0, 1.5}` |

`p = .5/.7` and `s = 1.0/1.5` are reliability-severity **extrapolation** (outside
the training regime). Corruption is a deterministic function of
`(experiment_seed, split, epoch, replica)` — **model identity never enters the
RNG**, so every model in a `(dataset, family, seed)` cell sees byte-identical
`x_corrupted` and `c`. Training corruption varies per epoch; validation and test
are pinned; each test severity uses 3 deterministic replicas (identical across
models), averaged before any across-seed statistic.

## Four model families (all at the CellV0.3 parameter count)

| Family | Underlying net input | Reliability seen | Role |
|---|---|---|---|
| **A — Plain MLP** | `x_corrupted` | none | how hard is the corruption itself |
| **B — Confidence MLP** | `concat(x_corrupted, c)` (dim `2·D`) | exact `c` | can a conventional net *learn* what V0.3 hard-codes? |
| **C — Reliability-Gated MLP** | `c · x_corrupted` (dim `D`) | exploited directly | is hidden-state propagation better than one input-side gate? |
| **D — CellV0.3** | belief `(mu=x_corrupted, e=c, u=0)` | exact `c` | the architecture under test |

A/B/C are the frozen Phase-1 `Linear → SiLU → Linear` MLP; hidden widths are
chosen so every family lands within 2% of CellV0.3's actual parameter count
(B needs a different width — its input is twice as wide; fairness is by
parameters, not width).

## Layout

| File | Role |
|---|---|
| `corruption.py` | Deterministic missing-feature + heterogeneous-Gaussian corruption, reliability maps, `shuffle_confidence`. |
| `models.py` | The four families (uniform `forward(x_corrupted, c)`), parameter matching against the CellV0.3 target, CellV0.3 belief diagnostics. |
| `training.py` | Per-epoch corrupted training; checkpoint selection on the averaged **in-distribution** corrupted-validation primary metric (Sec 9). |
| `evaluate.py` | Per-severity sweep (3 replicas), `corruption_AUC` (trapezoidal), `OOD_drop`, per-layer belief diagnostics, reliability-response. |
| `interventions.py` | Sec 15 confidence-intervention test (true / all-ones / shuffled `c`), evaluation only. |
| `harness.py` | `run_one(dataset, corruption_family, model_family, seed)` → trains, sweeps, writes a full-provenance `RunRecord`. |
| `run_reliability.py` | The 96-run grid: 4 datasets × 2 corruption families × 4 models × 3 seeds. |
| `summarize.py` | Builds `reliability_results.md` (Tables A–G + failures + the predeclared go/no-go read). |

## Running

```bash
# full grid (auto device; ~1h, MNIST/Fashion-MNIST runs dominate)
python experiments/paper_a/reliability/run_reliability.py

# a slice:
python experiments/paper_a/reliability/run_reliability.py \
    --datasets digits --corruptions missing --seeds 0

# rebuild the report from whatever is in results/raw/:
python experiments/paper_a/reliability/summarize.py
```

Raw `RunRecord`s land in `results/raw/` (gitignored). The committed artifact is
`reliability_results.md` plus the dated entry in `docs/research_log.md`.

## Predeclared go/no-go (Sec 19) — report the evidence, do not redesign

* **Strong positive** — CellV0.3 approximately competitive on clean/in-dist
  data *and* a consistent robustness advantage over **both** reliability-aware
  MLP baselines, especially under severe / OOD corruption, across
  datasets/seeds (positive corruption-AUC difference, smaller OOD degradation,
  correct confidence beats shuffled/all-ones, internal `pi` responds to
  severity).
* **Weak / neutral** — CellV0.3 beats the plain MLP but ties the
  confidence-aware / gated MLPs. Reliability information helps; the special
  cell is not clearly necessary.
* **Negative** — the confidence-aware / gated MLPs match or beat CellV0.3
  throughout. In that case: **do not** create CellV0.4 or tune V0.3 — record
  the result.

Do not claim uncertainty calibration: `e`/`u`/`pi` are internal computational
reliability variables, not calibrated predictive probabilities. Do not write
the paper yet. Do not alter V0.3 after the results.
