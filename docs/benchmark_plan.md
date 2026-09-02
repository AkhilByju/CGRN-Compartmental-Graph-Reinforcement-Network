# Benchmark Plan

## Three evaluation tracks

### Track A — Classical Machine Learning

Tests whether the architecture behaves like a general ML architecture on
conventional supervised problems, not just on tasks designed to flatter it.

- **Regression tasks:** synthetic linear regression, nonlinear synthetic
  regression, interaction-heavy regression, real tabular regression (e.g.
  housing-price datasets).
- **Baselines:** Linear Regression, Ridge/Lasso where appropriate, Random
  Forest, Gradient Boosting / XGBoost, conventional MLP, proposed
  architecture.
- **Metrics:** MAE, RMSE, R², sample efficiency, training time, inference
  time, parameter count.
- Goal is *not* to beat boosted trees — it's to characterize where the
  architecture works. Corresponds to Experiment 007.

### Track B — Classification

- **Tasks:** binary classification, multiclass classification, XOR, nonlinear
  decision boundaries, interaction-heavy synthetic data, real tabular
  datasets.
- **Baselines:** Logistic Regression, Random Forest, Gradient Boosting /
  XGBoost, MLP, proposed architecture.
- **Metrics:** accuracy, F1, AUROC where appropriate, calibration, sample
  efficiency, convergence speed, runtime, parameter count. Corresponds to
  Experiment 007.

### Track C — Reasoning, World Modeling, and Language

- **Tasks:** sequence transformations, arithmetic, sorting, relational
  reasoning, graph reasoning, compositional generalization, counterfactuals,
  causal reasoning, synthetic worlds, natural language, TinyStories,
  BabyLM-type evaluation.
- **Baselines:** MLP where relevant, GRU/LSTM, Transformer, eventually
  Mamba/SSM, eventually other recurrent/cellular alternatives. Corresponds to
  Experiments 008–011.

## Benchmark ladder (simplest to hardest)

Run in order; do not skip a level to chase an interesting result at a higher
level before the lower ones are understood (see `experiment_protocol.md`).

- **Level 0 — Sanity checks.** "Is the implementation fundamentally capable
  of learning?" Linear functions, nonlinear functions, XOR, simple binary
  classification, toy regression. These are debugging/unit tests, not paper
  results.
- **Level 1 — Classical ML.** "Does the model work as a general supervised
  learner?" Tabular regression/classification, real and synthetic controlled
  datasets. (= Track A/B.)
- **Level 2 — Representation / pattern learning.** "Can the architecture
  discover useful representations?" Denoising, sequence mapping, feature
  composition, pattern completion, structured nonlinear functions.
- **Level 3 — Algorithmic reasoning.** "Can recurrent computation implement
  reusable algorithms?" Addition, sorting, simple algorithms, graph
  traversal, shortest-path, relational logic, object tracking.
- **Level 4 — Out-of-distribution generalization.** "Did the model learn a
  procedure, or memorize the training distribution?" Train on short
  sequences / small graphs / short reasoning chains; evaluate on longer
  sequences / larger graphs / longer chains (e.g. train on 5-node graphs,
  test on 10/20/40 nodes; train on 2-step relational inference, test on
  5/10 steps).
- **Level 5 — World modeling.** "Does the model construct useful latent
  representations of underlying situations?" Paraphrase invariance,
  counterfactuals, interventions, causal relationships, hidden-state
  inference, composition, consistency. See §"Synthetic world dataset" below.
- **Level 6 — Language.** Train on TinyStories, then BabyLM-style restricted
  language data. Evaluate language loss / perplexity where appropriate,
  grammatical competence, semantic tasks, reasoning, generation quality,
  representations, sample efficiency.

## Synthetic regression and classification progressions

Not arbitrary toy datasets — designed to reveal *where* the architecture
becomes useful.

**Regression:** `y = 3x + 2` (simple linear) → `y = sin(x)` (basic
nonlinear) → `y = x1*x2 + sin(x3) + x4^2` (feature interactions) →
hierarchical interactions among variable groups → relational regression
(output depends on relationships, not individual features).

**Classification:** linearly separable → XOR → nonlinear boundaries →
hierarchical features → relational classification → graph-derived
classification.

This progression is what lets us pinpoint when richer computational cells
start providing an advantage (ties to Q2 in `hypotheses.md`).

## Synthetic world dataset (for Level 5)

Example underlying world (a small relation graph):

```text
Bob --owns--> Red Key
Bob --located_at--> Kitchen
Red Key --opens--> Laboratory
```

Generate multiple textual descriptions of the *same* world, e.g.:

- A: "Bob is holding the red key. The red key opens the laboratory. Bob is
  in the kitchen."
- B: "The laboratory is accessible using the red key currently possessed by
  Bob, who is standing in the kitchen."

Same world, different wording — the architecture should ideally develop
compatible internal representations for A and B.

### World-representation tests

- **Paraphrase invariance** — different descriptions of the same world
  produce similar internal representations.
- **Intervention sensitivity** — changing a fact (e.g. "Bob gives the key to
  Alice") should appropriately change the internal state.
- **Counterfactual reasoning** — e.g. "If Bob had kept the key, who could
  open the laboratory?"
- **Relational composition** — train relationships separately, test
  combinations never seen during training.
- **Causal reasoning** — e.g. "A pushes B. B hits C. C moves." → what caused
  C to move?
- **Hidden-state inference** — partial observations, infer an unobserved
  state.
- **Consistency** — equivalent questions in multiple linguistic forms should
  yield consistent responses.

## Sample efficiency

Because biological intelligence appears highly sample-efficient, vary
training-data availability and measure performance curves: 100, 500, 1,000,
5,000, 10,000, 50,000 examples. Key question:

> Does the architecture learn useful structure from fewer examples?

An architecture that provides most of its advantage in low-data regimes,
even if performance converges at high data volume, would be a notable
result (ties to H5).

## Parameter efficiency

Compare architectures at approximately matched parameter counts, across
scales practical for the primary hardware (Apple M4-class): ~100K, ~300K,
~1M, ~3M, ~10M parameters. Key questions: does the proposed architecture
outperform conventional networks when very small? Does the advantage grow,
shrink, or disappear with scale? Are richer cells particularly useful below
some parameter budget?

## Compute efficiency — report multiple fairness regimes

Parameter count alone is insufficient — a recurrent model can reuse 1M
parameters 100 times and perform far more computation than a 1M-parameter
feed-forward model. Report:

- **Parameter-matched** — same approximate learned parameter count. Answers:
  which architecture uses parameters more effectively?
- **Compute-matched** — same approximate training/inference FLOPs. Answers:
  does the architecture provide better computational efficiency?
- **Data-matched** — same number of training examples/tokens. Answers: which
  architecture learns more from the same data?
- **Wall-clock** — actual training time, inference time, memory usage.
  Irregular graph architectures may be theoretically efficient but hardware
  inefficient; document that tradeoff rather than hiding it.

## Repeated reasoning compute

A particularly important evaluation: performance vs. number of refinement
iterations, `T = 1, 2, 4, 8, 16, 32`. Plot performance vs. iteration count
*and* performance vs. actual compute. This distinguishes "the architecture
benefits from additional thinking" from "the architecture just wastes more
computation." (Experiment 005; see also `architecture_v0.md` §2.)

## Monte Carlo / multiple hypotheses (later stage — Experiment 012+)

Legitimate motivation, not added merely because it's mathematically
interesting: a deterministic latent system (`Z0 -> Z1 -> Z2 -> ...`) commits
to one internal trajectory, but many reasoning tasks involve uncertainty. The
*first* stochastic experiment should be multiple latent particles, not full
MCMC: initialize `Z0^1, ..., Z0^K` (possibly with small stochastic
differences), let each trajectory refine independently/semi-independently,
then score/aggregate. Sweep `K = 1, 2, 4, 8`. Questions: do multiple
trajectories improve ambiguous reasoning, robustness, or discover alternative
interpretations — and is the improvement worth the extra inference compute?
**If no, do not pursue MCMC/energy-based inference** (Experiment 013 is
gated on a positive result here). See `architecture_v0.md` §8 for why
full MCMC is deferred.

## Language modeling stays a core goal

The architecture should not become *only* a graph-reasoning model, a
continual-learning system, or a synthetic-world solver — a general language
model architecture is a long-term goal. Ordinary causal language modeling
must remain in the language experiments (not replaced immediately by
alternative objectives) so that a clean comparison is possible: same
objective + same data + similar parameters + different architecture. Later
experiments may compare next-token prediction, masked reconstruction,
denoising, latent-state prediction, bidirectional reconstruction, and
predictive world-state learning against each other — but not as a
replacement for the baseline comparison.

- **Early language test:** TinyStories or another highly constrained small
  corpus — verifies real language acquisition, tests generation, compares
  small models.
- **Serious small-scale evaluation:** BabyLM-style restricted-data
  evaluation — measures language acquisition under limited data, compares
  against parameter-matched Transformer baselines, evaluates grammar,
  semantics, conceptual knowledge, and generalization.
