# CGRN — Compartmental Graph Refinement Network (working name)

Exploratory ML-architecture research investigating whether a neural architecture
built from **richer computational cells**, **sparse recurrent graph
communication**, **shared-parameter iterative refinement**, and a **persistent
latent world-state** exhibits useful properties relative to conventional
feed-forward / Transformer baselines — under parameter-matched and
compute-matched conditions.

`CGRN` and `ArchitectureV0` are placeholder names. Nothing about the name,
the math, or the final architecture is settled yet — see
[`docs/architecture_v0.md`](docs/architecture_v0.md).

## Project status

**Design / specification phase.** No model architecture has been implemented
yet, and none should be until `CellV0` is mathematically specified (see
[`docs/architecture_v0.md`](docs/architecture_v0.md) and
[`CLAUDE.md`](CLAUDE.md)). What exists today is the reproducibility
scaffolding: directory structure, run/config conventions, generic training
and evaluation infrastructure, and documentation of the research plan.

## Start here

| If you want to... | Read... |
|---|---|
| Understand the research motivation and philosophy | [`docs/research_thesis.md`](docs/research_thesis.md) |
| See the (currently unspecified) architecture design space and open questions | [`docs/architecture_v0.md`](docs/architecture_v0.md) |
| See the falsifiable hypotheses this project tests | [`docs/hypotheses.md`](docs/hypotheses.md) |
| See how the architecture will be evaluated (tracks, benchmark ladder, efficiency axes) | [`docs/benchmark_plan.md`](docs/benchmark_plan.md) |
| See the experiment sequence and reproducibility requirements for a run | [`docs/experiment_protocol.md`](docs/experiment_protocol.md) |
| Track papers / prior work relevant to each idea | [`docs/related_work.md`](docs/related_work.md) |
| See the running research diary | [`docs/research_log.md`](docs/research_log.md) |
| Work on this repo as a coding agent | [`CLAUDE.md`](CLAUDE.md) |

## Repository layout

```text
docs/          research framing, architecture spec, hypotheses, protocol, log
configs/       YAML experiment configs, grouped by experiment family
src/
  models/
    baselines/       conventional MLP / RNN / Transformer baselines (stubs)
    architecture_v0/ CGRN itself — NOT YET IMPLEMENTED, math pending
  data/          dataset generators, grouped by track/level
  training/      generic trainer, optimizer builder, checkpointing, run logging
  evaluation/    metrics, grouped by track (regression/classification/reasoning/...)
  utilities/     seeding, device selection, config loading + hashing
experiments/   one directory per numbered experiment (001-012), each with a README
results/       raw/processed run outputs, figures, tables (gitignored, .gitkeep only)
scripts/       CLI entry points: train.py, evaluate.py, benchmark.py, summarize_results.py
notebooks/     exploratory only — never a source of reproducible results
tests/         tests for the infrastructure in src/
```

## Setup

Primary development hardware is Apple Silicon (M4-class); PyTorch's MPS
backend is used for local acceleration, with CPU/CUDA as fallbacks.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Reproducibility conventions

- Every run is defined by a YAML config under `configs/` and identified by a
  content hash of that config (see `src/utilities/config.py`) — never by a
  hand-edited filename like `final_model_v2_REAL_fixed.pt`.
- Every run records: architecture, full config, parameter count, dataset,
  seed, optimizer, learning rate, steps/tokens seen, refinement iterations
  (where applicable), approximate compute, wall-clock time, validation/test
  metrics, git commit hash, and checkpoint location. See
  [`docs/experiment_protocol.md`](docs/experiment_protocol.md).

## License

MIT — see [`LICENSE`](LICENSE).
