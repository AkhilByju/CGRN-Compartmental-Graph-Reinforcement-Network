# Hypotheses

All claims this project makes should be falsifiable. Avoid unfalsifiable
philosophical framing (e.g. "Transformers are not intelligent," "our model
actually understands").

## Primary hypotheses (H1–H6)

Each should be tested independently; a negative result on one does not
invalidate the others.

- **H1 — Computational Unit Hypothesis.** Structured compartmental units
  provide greater representational efficiency than conventional point-like
  units under certain classes of problems. Tested by: Experiments 002–004.
- **H2 — Recurrent Computation Hypothesis.** Reusing parameters over multiple
  reasoning iterations provides stronger systematic generalization than
  increasing feed-forward depth, at matched parameter budgets. Tested by:
  Experiments 005, 009.
- **H3 — Circuit Hypothesis.** Clustered local computation with selective
  global communication provides useful inductive biases for relational
  problems. Tested by: Experiment 006.
- **H4 — World-State Hypothesis.** Iteratively refined latent states
  represent underlying relational worlds more consistently than models
  optimized only for surface prediction. Tested by: Experiment 010.
- **H5 — Sample-Efficiency Hypothesis.** The proposed architecture learns
  certain structured problems from fewer examples. Tested by: the
  sample-efficiency protocol in `benchmark_plan.md`, applied across
  Experiments 004, 007–010.
- **H6 — Language Hypothesis.** The architecture can support meaningful
  language modeling despite not being organized as a Transformer stack.
  Tested by: Experiment 011.

## CellV0 research questions (Q1–Q6)

These are the *only* questions the first experiments need to answer — resist
scope creep beyond them until they're settled (see `architecture_v0.md` §7–8
and `experiment_protocol.md` Experiments 002–004).

- **Q1.** Can CellV0 approximate ordinary functions as reliably as
  conventional networks?
- **Q2.** Does CellV0 improve learning of interaction-heavy functions?
- **Q3.** Does CellV0 improve parameter efficiency?
- **Q4.** Does CellV0 improve sample efficiency?
- **Q5.** Does CellV0 remain trainable as compartment count increases?
- **Q6.** Does any advantage come simply from additional computation (i.e.
  disappears once compute-matched, not just parameter-matched)?

## What counts as each kind of result

See `research_thesis.md` §2 for the philosophy behind this section.

### Positive result
The architecture demonstrates better parameter efficiency, reasoning,
generalization, or learning behavior than matched baselines.

### Mixed result (still valuable)
E.g. weak on conventional prediction tasks but strong on iterative relational
reasoning. A specific, explicable pattern of wins/losses across tracks is
itself the finding — see the illustrative pattern below.

### Negative result (still valuable, if explained)
Poor performance, with a rigorous investigation of *why*: oversmoothing
during recurrent graph computation, unstable latent dynamics, optimization
difficulty, excessive computation requirements, poor information
propagation, representation collapse, inability to scale computation depth,
or failure of richer cells to provide useful inductive bias. When results are
bad, explicitly check for and report on:

- Do all cell states converge toward the same representation (oversmoothing)?
- Do gradients vanish/explode over refinement iterations?
- Does compartmentalization reduce information sharing?
- Does sparse graph connectivity create information bottlenecks?
- Does repeated parameter reuse limit memorization capacity?
- Does additional inference depth become unstable rather than helpful?
- Is the architecture massively less hardware-efficient (wall-clock), even if
  theoretically FLOP-efficient?
- Does graph locality prevent global structure formation?

A rigorous characterization of a recurring failure mode is itself a
publishable contribution.

## Illustrative pattern of a compelling (hypothetical, non-binding) result

```text
Classical tabular regression:  XGBoost > MLP > ArchitectureV0
Simple classification:         MLP ~ ArchitectureV0
Language modeling:              Transformer ~ ArchitectureV0
Relational reasoning:           ArchitectureV0 > Transformer
OOD reasoning:                  ArchitectureV0 >> Transformer
Low-data learning:              ArchitectureV0 > Transformer
Inference scaling:               ArchitectureV0 improves with more refinement steps
```

This pattern would point to a specific architectural advantage — iterative
relational computation — which is a useful scientific conclusion even though
the architecture loses on two of the tracks shown.

## Operationalizing "understanding" / world representation

Do not claim a model "understands." Instead, use a measurable definition:

> A system demonstrates stronger world representation if it maintains similar
> internal states when surface descriptions change but underlying meaning
> remains constant, while appropriately changing its internal state when the
> underlying world itself changes.

See `benchmark_plan.md` §"World-representation tests" for the concrete tests
(paraphrase invariance, intervention sensitivity, counterfactual reasoning,
relational composition, causal reasoning, hidden-state inference,
consistency) that operationalize this definition.
