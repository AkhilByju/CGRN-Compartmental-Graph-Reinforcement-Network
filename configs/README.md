# Configs

One YAML file per run, grouped by family into the subdirectories here
(`baselines/`, `cells/`, `regression/`, `classification/`, `reasoning/`,
`world_modeling/`, `language/`). See `docs/experiment_protocol.md` "Config
conventions" for the full rule set. Summary:

- A config is loaded into `src.utilities.config.ExperimentConfig`. Fields
  not in that dataclass are preserved under `extra` rather than rejected, so
  architecture-specific hyperparameters (e.g. compartment count once CellV0
  is specified) don't require touching the loader.
- A run's identity is `src.utilities.config.make_run_id(config)` — a hash of
  the config's contents plus a timestamp. Never name a checkpoint or log
  file by hand.
- When comparing architectures within an experiment, prefer providing both a
  parameter-matched config and a compute-matched config where feasible
  (`docs/benchmark_plan.md` "Compute efficiency").

## Minimal example

```yaml
experiment_id: "001_baseline_validation"
architecture: "mlp_baseline"
dataset: "synthetic_regression_linear"
seed: 0
optimizer: "adamw"
learning_rate: 0.001
batch_size: 32
max_steps: 2000
# refinement_iterations: null   # only set for iterative-refinement architectures
# extra:
#   hidden_dim: 64
#   num_layers: 2
```

No configs exist yet in the subdirectories below — they populate as each
experiment in `docs/experiment_protocol.md` begins.
