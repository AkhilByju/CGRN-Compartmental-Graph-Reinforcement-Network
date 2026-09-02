# CLAUDE.md — Agent Guide for CGRN

This file is the entry point for any coding agent (or human) working in this
repository. Read it fully before making changes. It is intentionally more
directive than a typical README because the single biggest risk in this
project is an agent inventing architecture details prematurely.

## 1. What this project is

An exploratory ML-architecture research project testing whether a
compartmental-cell, graph-structured, iteratively-refined neural architecture
(working name **CGRN** / **ArchitectureV0**) has useful properties relative
to conventional feed-forward/Transformer baselines, under parameter-matched
and compute-matched conditions. Full motivation: `docs/research_thesis.md`.

This is **not** a "beat the Transformer" project. It is a controlled,
falsifiable-hypothesis-driven investigation. A well-characterized negative
result is a valid and valuable outcome. See `docs/hypotheses.md`.

## 2. Current phase: DESIGN, NOT IMPLEMENTATION

**As of the current state of this repo, `CellV0` — the architecture's basic
computational unit — has not been mathematically specified.** Until
`docs/architecture_v0.md` contains a finalized spec (state, compartment
computation, integration rule, update rule, parameter count), the following
is true:

### An agent working in this repo MAY:
- Edit/extend documentation in `docs/`.
- Add or refine experiment folders under `experiments/`.
- Build out **generic, architecture-agnostic** infrastructure: config
  loading/validation, run-ID hashing, seeding, device selection, checkpoint
  naming, run-metadata logging, generic metric functions (MAE/RMSE/R²,
  accuracy/F1/AUROC), generic training-loop plumbing that takes an arbitrary
  `nn.Module`.
- Add or refine **placeholder/stub** files for baselines (`src/models/baselines/`)
  and for the novel architecture (`src/models/architecture_v0/`) — signatures,
  docstrings, `NotImplementedError`, TODOs referencing the relevant doc section.
- Write tests for the infrastructure above.
- Track literature in `docs/related_work.md`.
- Append dated entries to `docs/research_log.md`.

### An agent working in this repo MUST NOT (until told the spec is finalized):
- Invent CellV0's math (compartment functions, integration operator, update
  rule) — this is the user's research decision, not the agent's.
- Implement compartment logic, graph/cluster message-passing, dynamic-graph
  routing, fast contextual association (`A_ij(t)`), Monte Carlo / multi-particle
  inference, energy-based/MCMC inference, or any plasticity/adaptive-learning
  mechanism (STDP, dopamine simulation, local-only learning, replay,
  metaplasticity, structural plasticity, weight changes during inference).
- Introduce a novel optimizer or novel training objective. Use AdamW (or
  another conventional optimizer) and standard losses — see
  `docs/research_thesis.md` §"Isolation of variables".
- Fill in `src/models/architecture_v0/*.py` with real logic beyond what the
  user has explicitly specified in `docs/architecture_v0.md`. As of this
  writing: CellV0's *state* is specified (`cell.py`'s `BeliefCell`) and its
  *aggregation operator* has three implemented, unchosen candidates
  (`integration.py`'s `BeliefLayer`) pending Experiment 002/003. Do not add
  a fourth aggregation method or pick a "winner" without that experiment.
  Everything downstream (`cluster.py`, `graph.py`, `dynamics.py`,
  `encoder.py`, `decoder.py`, `model.py`) must stay `NotImplementedError`
  stubs until the user finalizes them — check `docs/architecture_v0.md`'s
  per-section status before assuming something is decided.
- Fully implement the baseline models in `src/models/baselines/` beyond
  placeholders — that's Experiment 001, which starts only once the user says
  to begin it.
- Introduce dynamic/learned topology before a fixed topology has been tested
  (§ "Fixed graph before dynamic graph" in `docs/architecture_v0.md`).

If asked to "just implement the architecture," push back and point to this
section — the correct next research step is **Architecture Specification
V0.1** (a math document), not code. See `docs/architecture_v0.md`'s open
questions.

## 3. Map of the docs

| Doc | Contents |
|---|---|
| `docs/research_thesis.md` | Objective, philosophy, why this isn't "Transformers are bad," the biological-neuron clarification, what must NOT change initially (isolation of variables) |
| `docs/architecture_v0.md` | The architecture design space (compartmental cells, layers→persistent substrate, iterative refinement, graph/cluster organization, fixed-before-dynamic graph, fast associations, latent world-state) — **currently unspecified**, framed as open design questions |
| `docs/hypotheses.md` | H1–H6 falsifiable hypotheses, CellV0 research questions Q1–Q6, what counts as a positive/mixed/negative result |
| `docs/benchmark_plan.md` | Three evaluation tracks (Classical ML / Classification / Reasoning-World-Language), the 7-level benchmark ladder, sample- and parameter- and compute-efficiency protocols |
| `docs/experiment_protocol.md` | The 001–013 experiment sequence, and the exact metadata every run must record |
| `docs/related_work.md` | Literature tracker (fill in as papers are read) |
| `docs/research_log.md` | Dated research diary — append, don't rewrite history |

## 4. Reproducibility rules (non-negotiable)

1. **No hand-named checkpoints.** Never `final_model_v2_REAL_fixed.pt`. Every
   run is identified by a config content-hash + timestamp, produced by
   `src/utilities/config.py`. See `src/training/checkpointing.py`.
2. **Every run logs full metadata**: architecture name, full config, parameter
   count, dataset identity, seed, optimizer + LR, steps/tokens seen,
   refinement-iteration count (if applicable), approximate FLOPs, wall-clock
   time, val/test metrics, git commit hash, checkpoint path. Use
   `src/training/logging.py`'s `RunRecord`.
3. **Configs are YAML, not hardcoded constants**, and live under `configs/<family>/`.
4. **Isolate variables.** When adding an experiment, change the minimum
   number of things relative to the previous experiment in the sequence
   (`docs/experiment_protocol.md`). Don't bundle a new optimizer with a new
   architecture change in the same experiment.
5. **Parameter counts and compute must be comparable across architectures**
   being compared in a given experiment — matched-parameter and
   matched-compute variants are both expected (`docs/benchmark_plan.md`).

## 5. Commit discipline

Commit in small, logically-scoped stages (docs vs. infra vs. experiment
scaffolding vs. tests) rather than one large commit. Write commit messages
that describe the *why* (which research step this serves), not just the
diff.

## 6. Environment

- Python ≥3.10, PyTorch ≥2.2 (MPS backend for local Apple Silicon dev, CUDA/CPU
  as fallback — see `src/utilities/device.py`).
- Install: `pip install -e ".[dev]"`. Run tests: `pytest`.
- Classical-ML baselines (Track A, Experiment 007) need the optional
  `classical-ml` extra (`xgboost`) — not installed by default.

## 7. When in doubt

If a task requires deciding architecture math, choosing among the mutually
exclusive design options listed in `docs/architecture_v0.md`, or skipping
ahead in the experiment sequence in `docs/experiment_protocol.md` — stop and
ask the user rather than guessing. Everything else (docs, infra, stubs,
tests, experiment tracking) can proceed without asking.
