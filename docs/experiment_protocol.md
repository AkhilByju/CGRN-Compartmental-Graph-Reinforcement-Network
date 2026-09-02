# Experiment Protocol

## Ordered experiment sequence

Follow this order. Each experiment changes the minimum necessary relative to
the previous one (isolation of variables — see `research_thesis.md` §6).
Directories already exist under `experiments/`; each has its own README with
status and links back to the hypotheses it tests.

| # | Name | Purpose |
|---|---|---|
| 001 | Baseline Validation | Implement tiny MLP, tiny recurrent model, tiny Transformer, and a common training/evaluation framework. Establish trusted baselines. |
| 002 | CellV0 | Introduce the compartmental cell alone. No graph, no stochasticity, no dynamic connections, no novel learning rule. Determine whether the richer cell deserves to exist (Q1–Q4). |
| 003 | Cell Ablations | Vary number of compartments, integration mechanism, hidden-state dimensions, compartment sharing, gating. Determine which aspects actually matter (Q5–Q6). |
| 004 | CellV0 Scaling | Model-size and dataset-size scaling of the same frozen CellV0 (`precision` aggregation) vs. a parameter-matched MLP: does the belief mechanism's relative advantage grow, shrink, or stay parallel with scale (Q3–Q4), and does Experiment 003D's evidence/uncertainty intervention effect change with model size? No architecture, depth, optimizer, or training-procedure changes. |
| 005 | Recurrent Reuse | Apply the same computational system repeatedly (shared parameters across iterations). Determine whether parameter reuse + extra inference compute improves learning/reasoning (H2). |
| 006 | Cluster Structure | Organize cells into local circuits with a **fixed** topology. Determine whether local specialization + sparse global communication adds useful inductive bias (H3). |
| 007 | Classical Regression and Classification | Evaluate broadly against Track A/B baselines (linear/logistic regression, random forest, boosted trees, MLP). Characterize the architecture as a general ML model. |
| 008 | Algorithmic / Relational Reasoning | Arithmetic, sequence transformations, relational inference, graphs, sorting, composition. Determine whether iterative computation helps on structured tasks. |
| 009 | OOD Generalization | Increase problem complexity beyond training (Level 4 of the benchmark ladder). Determine whether reusable computation generalizes systematically (H2). |
| 010 | Latent World Modeling | Introduce procedural relational environments (Level 5). Test whether the architecture develops stable world representations (H4). |
| 011 | Language | TinyStories, then BabyLM-style restricted language data (Level 6). Determine whether the architecture can operate as a genuine language model (H6). |
| 012 | Multiple Latent Trajectories | Introduce stochastic particles (multi-hypothesis latent state). Test whether maintaining multiple internal hypotheses improves reasoning. |
| 013 | MCMC / Energy-Based Inference | Only if Experiment 012 provides strong justification. |

Do not start an experiment before the ones above it in the sequence have
produced results, unless the user explicitly directs otherwise.

## What every run must record

Runs are identified by a config content-hash, never a hand-edited filename
(no `final_model_v2_REAL_fixed_new.pt`). See `src/utilities/config.py`
(hashing) and `src/training/logging.py` (`RunRecord`). Every run logs:

- architecture (name/version)
- full configuration (the YAML, verbatim or by reference)
- parameter count
- dataset identity (name, version/generation seed)
- random seed
- optimizer and its hyperparameters (learning rate, weight decay, etc.)
- training steps / examples / tokens seen
- refinement iterations used (where applicable)
- approximate compute (FLOPs estimate)
- wall-clock time (train and, separately, inference)
- validation metrics
- test metrics
- git commit hash
- checkpoint location

## Config conventions

- One YAML config per run, under `configs/<family>/`, where `<family>`
  matches the experiment's domain (`baselines/`, `cells/`, `regression/`,
  `classification/`, `reasoning/`, `world_modeling/`, `language/`).
- A config's hash (see `src/utilities/config.py::config_hash`) plus a
  timestamp forms the run ID; checkpoints and logs are named after the run
  ID, never manually.
- When comparing architectures within an experiment, produce both a
  parameter-matched config and a compute-matched config where feasible (see
  `benchmark_plan.md` §"Compute efficiency").

## Model scales (practical for Apple M4-class hardware)

- **Debug model:** 100K–300K parameters — rapid iteration, debugging, sanity
  tests, architecture exploration.
- **Small research model:** ~1M parameters — meaningful comparisons,
  reasoning, regression/classification, ablations.
- **Main small-scale model:** ~3M–10M parameters — stronger benchmarks,
  TinyStories, restricted language experiments, paper-level evaluations.

The architecture should be designed so that small-model results are
scientifically meaningful; the project does not depend on demonstrating
7B+-parameter scaling.

## Framework and hardware

- **Python + PyTorch.** Fast research iteration, autodiff, strong MPS support
  on Apple Silicon, easy CPU/GPU portability, straightforward implementation
  of unusual computational graphs. Do not begin in C++ — that solves a later
  problem (optimizing a validated architecture), not the current one (does
  the architecture work at all). If profiling later reveals a specific
  bottleneck, optimized kernels (C++/CUDA/Metal/Triton) can be added after
  the architecture demonstrates value.
- **Primary development hardware:** Apple M4-class Mac. Early experiments
  should deliberately stay within the model-scale budget above.
