# Architecture V2 frozen benchmark — Belief Dendritic Network

Separate from Paper A. Evaluates the architecture specified and implemented
in `docs/architecture_v2.md` / `src/models/architecture_v2/belief_dendrite.py`
against five comparison points, on MNIST/Fashion-MNIST under two structured
image corruption families. This is the **frozen** experiment referenced by
`docs/architecture_v2.md` Sec 10 — do not add/remove models, datasets, or
corruption conditions after seeing results (Sec Q).

## What this experiment answers (docs/architecture_v2.md Sec P)

1. **Does the dendritic structure itself help?** `scalar_dendrite` vs `plain_mlp`.
2. **Is input reliability gating sufficient?** `scalar_dendrite_reliability_gated` vs `belief_dendrite`.
3. **Does hierarchical belief propagation beat dense belief propagation?** `belief_dendrite` vs `cellv0.3`.
4. **Is any gain just because the conventional model lacks reliability?** `belief_dendrite` vs `confidence_mlp`.
5. **Does the mechanism actually localize corruption?** correlation between a layer-1
   branch's fraction of receptive field inside the corrupted patch and that
   branch's own precision (expected: negative — more overlap, lower `pi_branch`).

## Datasets

MNIST, Fashion-MNIST — the frozen Paper-A loader
(`experiments/paper_a/datasets.py::prepare_dataset`, `train_fraction=1.0`),
reused unmodified: official 60k/10k train/test split, a seeded 10k
validation carve-out from the 60k, global mean/std standardization fit on
the training subset only.

## Corruption families (`corruption.py`, spec Sec L/M)

Both are a single contiguous square patch per image, not independent
per-pixel corruption — the point is to test whether damage *localized to
one region* selectively lowers the dendrites whose receptive field
overlaps it.

* **`missing_patch`** — patch zeroed, `reliability=1e-3` inside /
  `1` outside. Train side `∈ {0,4,7,10}`; eval side
  `∈ {0,4,7,10,14,18}`.
* **`noisy_patch`** — fixed side `10`, additive `N(0, sigma^2)` (one `sigma`
  per example) inside, `reliability = 1/(1+sigma^2)` inside / `1` outside.
  Train `sigma ∈ {0,0.5,1.0}`; eval `sigma ∈ {0,0.5,1.0,1.5,2.0}`.

Corruption realizations are a pure function of `(experiment_seed, split,
epoch, replica)` — model identity never enters the RNG stream, so every
model in a `(dataset, corruption_family, seed)` cell sees byte-identical
corrupted inputs. 3 deterministic test replicas per severity are averaged
before any other statistic.

## Models (`models.py`, spec Sec N)

| family | sees `c` how | ~params |
|---|---|---|
| `plain_mlp` | not at all | ~150k |
| `confidence_mlp` | `concat(x, c)` | ~150k |
| `cellv0.3` | belief `(mu=x, e=c, u=0)`, dense fan-in (`BeliefNetworkV03`) | ~150k |
| `scalar_dendrite` | not at all, sparse dendritic topology | ~150k |
| `scalar_dendrite_reliability_gated` | `c * x` once, at the input, same topology | ~150k |
| `belief_dendrite` | belief `(mu=x, e=c, u=0)`, propagated through the topology | ~150k |

Every family independently targets ~150k trainable parameters (not matched
to any one family's exact count). The three dendritic families share one
`DendriticConnectivity` pair per `seed` (`B=4` branches/soma, first layer
`7×7` local patches, second layer `K=min(32,H)=32` sparse random, uniform
hidden width solved once from the 150k budget via
`param_count.solve_hidden_width_for_budget`) — so families 4-6 are compared
over identical sparse topology, isolating belief propagation from structure.

## Training protocol (`training.py`)

The frozen Paper-A protocol, reused unmodified — "one common predeclared LR
protocol... not tuned per architecture" (spec Sec K): AdamW, `lr=1e-2`,
`weight_decay=0`, batch 128, up to 15000 steps, corrupted-validation
early stopping (`val_every=200`, patience 8 checks ⇒ ~1600-step patience
window at this val-set size), best-checkpoint restore. Training data is
corrupted fresh each epoch (deterministic per `(seed, epoch)`); checkpoint
selection uses accuracy averaged over the in-distribution severities
(`{0,4,7,10}` / `{0,0.5,1.0}`), never the severe/OOD points.

## Metrics (`evaluate.py`, spec Sec O)

Accuracy, macro-F1, `corruption_AUC` (trapezoidal area under the
severity/accuracy curve), `OOD_drop` (accuracy at max training severity
minus accuracy at the most severe evaluated point). For `cellv0.3` and
`belief_dendrite`: per-layer belief-state diagnostics (chunked, so a full
10k-row test split stays memory-flat); `belief_dendrite` additionally logs
branch/soma precision distributions, the branch-routing effective count
(`1/sum_b r_b^2`), somatic conflict (`u_soma`), and the corruption-overlap
↔ branch-precision correlation. Diagnostics are observational only — never
optimized against.

## Running it

```bash
# smoke test (tiny step cap, no records written)
python -c "from experiments.belief_dendrite.harness import run_one; \
  print(run_one('mnist', 'missing_patch', 'plain_mlp', 0, max_steps=50, write_record=False))"

# the full frozen grid: 2 datasets x 2 corruption families x 6 families x 3 seeds = 72 runs
python -m experiments.belief_dendrite.run_belief_dendrite

# after the grid finishes
python -m experiments.belief_dendrite.summarize
```

Results land in `results/raw/*.json` (one `RunRecord` + validation-curve
history per cell) and `results/processed/belief_dendrite_report.md`
(the combined tables + Sec P answers).

## Interpretation rules (spec Sec Q — do not redesign after the first frozen result)

**Strong signal**: `belief_dendrite` is competitive clean, its advantage
over `scalar_dendrite_reliability_gated` and `cellv0.3` grows with
corruption severity, branch precision correlates with branch corruption
overlap, and the soma visibly downweights corrupted branches.
**Interesting but incomplete**: dendritic models improve generally but
`belief_dendrite ≈` the scalar controls — structure works, hierarchical
belief propagation hasn't earned itself. **Negative**: a scalar control or
`confidence_mlp` matches/beats `belief_dendrite`, branch precision fails to
localize corruption, or training is dramatically slower/unstable. Whatever
the outcome: record it in `docs/architecture_v2.md` and
`docs/research_log.md`; do not spin up a V2.1 automatically off this
evidence (same discipline Paper A closed Phase 3 with).
