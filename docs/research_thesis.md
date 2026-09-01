# Research Thesis

Status: living document. Last major revision: repository initialization.

## 1. Objective

Investigate a fundamentally different neural-network architecture inspired by
computational principles observed in biological neural systems, motivated by
the hypothesis that current artificial architectures — especially
Transformers — may over-rely on:

- extremely large parameter counts,
- simple artificial computational units,
- fixed computational layers,
- dense global communication,
- static parameters during inference,
- token-oriented representations,
- and next-token prediction as the primary formulation of intelligence.

**This is not a claim that Transformers are ineffective.** Transformers are
extraordinarily capable. The research question is:

> Can we construct a substantially different neural architecture that
> achieves stronger learning, reasoning, representation, or parameter
> efficiency at small scale by using richer computational units, recurrent
> graph-based interaction, persistent internal state, and iterative latent
> refinement?

This is exploratory foundational ML research. The target is **not** to
outperform frontier models. The target is to determine whether a different
computational architecture exhibits useful properties under controlled,
parameter-matched and compute-matched experiments.

## 2. Central research philosophy

The architecture should not be judged only by language perplexity, benchmark
accuracy, or whether it beats a Transformer on one task. Instead, the
research should characterize:

> **What kinds of computation does this architecture perform well?**
> **What kinds of computation does it perform poorly?**

All three outcomes below are valid research results:

- **Positive** — better parameter efficiency, reasoning, generalization, or
  learning behavior.
- **Mixed** — e.g. weak on conventional prediction, strong on iterative
  relational reasoning. Scientifically valuable on its own.
- **Negative** — poor performance, but a careful investigation reveals why
  (oversmoothing, unstable latent dynamics, optimization difficulty, excessive
  compute requirements, poor information propagation, representation
  collapse, inability to scale computation depth, failure of richer cells to
  provide useful inductive bias). A rigorously analyzed failure is still
  useful research. See `hypotheses.md` §"What would constitute a useful
  negative result."

**Guiding rule:** Do not try to prove the architecture is superior. Determine
what properties it actually possesses.

## 3. The biological-neuron argument — and the version we do NOT make

A frontier model with a trillion parameters does **not** have a trillion
artificial neurons — most parameters are connection weights. So this project
does **not** use the argument "humans have ~86–100B neurons while frontier
models have trillions of neurons." That comparison is incorrect and should
never appear in this project's writing.

The scientifically useful observation is different:

> Biological neurons are dramatically more computationally complex than the
> simple units generally used inside artificial neural networks.

A conventional artificial unit is approximately: input → weighted combination
→ nonlinear activation. A biological neuron can include thousands of synaptic
inputs, complex dendritic trees, nonlinear computation within dendritic
branches, recurrent interactions, temporal dynamics, local state, multiple
plasticity mechanisms, varying connection strengths, structural organization,
and multiple timescales of adaptation.

**This project does not attempt to simulate biological neurons.** It asks:

> Are there computational principles from biological neural systems that
> artificial neural architectures have oversimplified?

## 4. Working name

**CGRN — Compartmental Graph Refinement Network.** Temporary working name
only, not a claimed final/publication name. `ArchitectureV0` may be a safer
internal/code name until novelty and mathematical details are finalized. The
name currently reflects three principles — see `architecture_v0.md`:

- **Compartmental** — each basic cell contains multiple internal
  computational components rather than behaving like a simple point neuron.
- **Graph** — cells interact through a graph/circuit structure rather than
  purely sequential feed-forward layers.
- **Refinement** — the system repeatedly updates an internal latent state
  rather than performing a single fixed-depth forward pass.

## 5. Core architectural thesis

> Complex cognition may be more parameter-efficient when computation is
> performed by stateful, compartmentalized units organized into sparse
> recurrent circuits that iteratively construct and refine latent
> representations, rather than exclusively by homogeneous feed-forward layers
> operating directly over token sequences.

Four major ideas the architecture should eventually investigate:

1. richer computational cells,
2. structured graph/circuit communication,
3. repeated recurrent computation using shared parameters,
4. internal latent/world-state refinement.

Adaptive learning, plasticity, Monte Carlo reasoning, and other mechanisms
are potential *later* extensions and must not all be introduced at once (see
next section).

## 6. Isolation of variables — what must NOT change initially

This is the project's most important experimental discipline.

`ArchitectureV0` should continue to use standard modern training machinery:
backpropagation, autodiff, AdamW (or another conventional optimizer),
conventional losses, ordinary floating-point computation.

**Do NOT initially introduce:** STDP, biologically realistic spikes, dopamine
simulation, local-only learning, permanent structural plasticity, changing
learned weights during inference, replay, metaplasticity, Monte Carlo chains,
energy-based training, dynamic topology, novel optimizers, novel training
objectives, and novel architectures — simultaneously.

If everything changes at once and the model performs poorly, the experiment
becomes scientifically uninterpretable. The first experiments must answer:

> Does the proposed architecture itself provide useful computational behavior
> when trained using conventional optimization?

Only after that is established should the learning process itself be
modified. See `experiment_protocol.md` for the ordered experiment sequence
that enforces this.

## 7. A paper does not require the whole architecture

The project may naturally generate multiple papers rather than one grand
unified result. Hypothetical (non-binding) examples:

- *Compartmental Neural Units for Parameter-Efficient Learning*
- *Clustered Recurrent Neural Circuits for Iterative Reasoning*
- *Latent World-State Refinement for Compositional Generalization*
- *Stochastic Hypothesis Search in Recurrent Neural Systems*
- *Language Modeling With Compartmental Graph Networks*

A strong result on one component is publishable without completing the
entire long-term vision. Don't gate progress on doing everything at once.

## 8. Long-term vision (context, not a build spec)

```text
                    INPUT / OBSERVATIONS
                             |
                             v
                         ENCODER
                             |
                             v
               initialize distributed state
                             |
                             v
       +=====================================+
       |                                     |
       |     PERSISTENT COMPUTATIONAL        |
       |             SUBSTRATE               |
       |                                     |
       |   +------------+   +------------+   |
       |   |  Cluster A |<->|  Cluster B |   |
       |   | rich cells |   | rich cells |   |
       |   +-----+------+   +-----+------+   |
       |         |                |          |
       |         v                v          |
       |   +------------+   +------------+   |
       |   |  Cluster C |<->|  Cluster D |   |
       |   +------------+   +------------+   |
       |                                     |
       +==================+==================+
                          |
                     refinement (repeat)
                          |
                  stable / useful state
                          |
                          v
                       DECODER
                +----------+----------+
                |          |          |
           regression  language    actions
          classification reasoning
```

The system is intended to investigate intelligence as **iterative
transformation of a persistent internal representation by a structured
population of computationally rich units**, rather than exclusively as **a
fixed stack of layers repeatedly predicting the next token**.

Note: `next-token prediction is not removed as a baseline objective` — see
`benchmark_plan.md` §"Language" for why it stays in the comparison.
