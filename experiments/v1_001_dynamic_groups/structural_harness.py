"""CellV1.5's first experiment: does *slow structural plasticity* earn its
keep, on top of the same persistent substrate held frozen?
(`docs/architecture_v1.md` §16.)

Three arms, and only three -- the user's exact specification:

| Arm | What it is |
|---|---|
| `cellv0.1` | `BeliefNetwork("scale_stable_precision")`, parameter-matched to CellV1.5 |
| `cellv1.5_frozen` | `StructuralBeliefGraph`, bootstrap topology (§16.8) fixed all run |
| `cellv1.5_plastic` | The identical model, with §16.9's prune/grow schedule executing |

No `mlp`, no `field_*`, no `association_*`, no `cellv1_local`/`cellv1_full`
-- §16.1 supersedes those for this variant, and they stay in the repo as
frozen historical baselines (their numbers are already on record in
`docs/research_log.md`).

**The controlled variable is exactly one bit.** `cellv1.5_frozen` and
`cellv1.5_plastic` are built from the same RNG state, so their initial
parameters are bit-identical; they are bootstrapped with a dedicated,
identically-seeded generator, so their starting topology is bit-identical;
they draw the same minibatches in the same order; and both call
`update_edge_utility()` every step (it is pure bookkeeping -- it touches
no parameter and no gradient, so leaving it on in the frozen arm keeps
even the edge-metadata trajectory comparable). The *only* difference is
whether `maybe_run_structural_plasticity` is allowed to fire. Plasticity's
own LSH candidate retrieval draws from a separate generator so it cannot
desynchronize the global RNG stream the two arms share -- meaning the two
arms are provably identical up to the first plasticity event
(`tests/test_structural_harness.py::test_frozen_and_plastic_are_identical_before_first_event`).

**Nothing in `src/models/architecture_v1/structural*.py` is modified by
this module** -- CellV1.5 is frozen for this evaluation. Everything here
is measurement and training-loop wiring, sitting outside the model.

## Implementation choices this harness has to make (flagged, not silent)

- **`num_steps` (T).** §16 never pins `T`. Default `2` here, matching the
  experimental setting of CellV1.4 (`association_local_global_t2`), the
  immediately-preceding variant evaluated on this exact task under this
  exact protocol -- so CellV1.5's numbers sit next to a comparable
  refinement budget rather than a differently-chosen one. Exposed as
  `--num-steps`; identical across both CellV1.5 arms by construction.
- **Encoder.** §16.10 proposes reusing `PopulationEncoder` unwrapped. This
  task is object-structured, and every prior CellV1 arm evaluated on it
  uses `ObjectSeededEncoder` (`harness.py`'s `OBJECT_SEEDED_TASKS`) -- a
  dense global-broadcast adapter would let every cell see the whole input
  immediately, partially bypassing the organization being measured. Using
  `ObjectSeededEncoder` here keeps CellV1.5 comparable to the arms it is
  meant to be read against; the substitution is exactly the one
  `harness.py` already applies to every other CellV1 variant on this task.
- **`total_steps` for §16.9's end-of-training freeze window.** The freeze
  is defined as "the last `freeze_fraction` of total training," but a
  convergence protocol does not know its own total in advance. `max_steps`
  (the training *budget*) is passed as `total_steps`, so the freeze window
  is the last 20% of the budget -- entered only by runs that actually get
  that far. Each run records `plasticity_events` and
  `entered_freeze_window` so this is visible rather than assumed.
- **Best-validation checkpointing through a resizable `EdgeRegistry`.**
  §16.2/`structural.py` flag that `load_state_dict` only round-trips when
  the edge count matches at load time -- and under plasticity it does not
  (the in-degree-1 safety fallback can add edges beyond the pruned count).
  `_restore_model_state` below re-points the three registry buffers to the
  checkpoint's shapes *before* `load_state_dict`, which is a harness-level
  workaround, not a change to CellV1.5.
- **"Functional edge-use" needs a definition; `_functional_edge_use`
  documents the one used** (an edge is functionally in use for an input
  when it supplies at least `tau / in_degree(target)` of its target's
  incoming precision -- i.e. at least `tau` times what a uniform
  contribution would be). Reported at three `tau` values plus a
  threshold-free effective-in-degree, so no conclusion rests on the cutoff.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from harness import (  # noqa: E402
    ASSOCIATION_DIM,
    CELLV1_HIDDEN_DIM,
    OBJECT_SEEDED_TASKS,
    TASKS,
    _build_splits,
    _compute_metrics,
    _is_regression,
    _match_belief_hidden_cells,
)
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.data.synthetic.dynamic_groups import N_OBJECTS as DEFAULT_N_OBJECTS  # noqa: E402
from src.data.synthetic.utils import standardize  # noqa: E402
from src.evaluation.classification import accuracy as accuracy_metric  # noqa: E402
from src.evaluation.efficiency import count_parameters  # noqa: E402
from src.evaluation.regression import r_squared  # noqa: E402
from src.models.architecture_v0.belief_network import BeliefNetwork  # noqa: E402
from src.models.architecture_v1.object_encoder import ObjectSeededEncoder  # noqa: E402
from src.models.architecture_v1.structural_model import StructuralBeliefGraph  # noqa: E402
from src.models.architecture_v1.structural_plasticity import (  # noqa: E402
    StructuralPlasticityConfig,
)
from src.training.logging import RunRecord, get_git_commit, write_run_record  # noqa: E402
from src.utilities.config import ExperimentConfig, make_run_id  # noqa: E402
from src.utilities.device import get_device  # noqa: E402
from src.utilities.seeding import set_seed  # noqa: E402

STRUCTURAL_ARCHITECTURES: tuple[str, ...] = ("cellv0.1", "cellv1.5_frozen", "cellv1.5_plastic")
CELLV1_5_ARCHITECTURES: tuple[str, ...] = ("cellv1.5_frozen", "cellv1.5_plastic")

# §16.10's proposed default (d_s), and CellV1.4's assoc_dim -- CellV1.5
# reuses §15's `AssociationFunction` unmodified for `a_ij(t)` (§16.4), so
# `assoc_dim` keeps that variant's value rather than a new one.
D_S = 16
ASSOC_DIM = 32
# T -- see this module's docstring; not pinned by §16.
NUM_STEPS = 2

# `tau` values for `_functional_edge_use`'s "is this edge in use right
# now" cutoff, as multiples of a uniform contribution.
#
# `tau = 1.0` is the headline and the least arbitrary of the four: "this
# edge carries at least as much of its target's incoming precision as it
# would under an even split," i.e. no free parameter beyond the target's
# own degree. The others bracket it -- a smoke run showed `tau <= 0.5`
# saturating (essentially every edge counts as in use for essentially
# every input, so the statistic can't distinguish anything), which is
# exactly why the cutoff is reported as a profile rather than a number.
EDGE_USE_TAUS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
HEADLINE_TAU = 1.0

# The complexity conditions this experiment runs, matching
# `run_complexity_scaling_association.py`'s definitions exactly so the
# numbers are comparable to the CellV1.4 sweep already on record.
COMPLEXITY_LEVELS: dict[str, dict[str, int]] = {
    "hard": {"n_objects": 96, "k_min": 8, "k_max": 14},
    "very_hard": {"n_objects": 192, "k_min": 12, "k_max": 20},
}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def _make_structural_model(
    task: str,
    in_features: int,
    out_features: int,
    seed: int,
    n_cells: int,
    n_objects: int,
    association_dim: int,
    assoc_dim: int,
    d_s: int,
    hidden_dim: int,
    num_steps: int,
    plasticity_config: StructuralPlasticityConfig,
) -> StructuralBeliefGraph:
    """One `StructuralBeliefGraph`, built from a freshly-reset RNG state.

    Re-seeding immediately before construction (rather than once for the
    whole `_build_structural_comparison_models` call) is what makes the
    frozen and plastic arms bit-identical at initialization -- otherwise
    the second model built would draw from an RNG stream the first had
    already advanced, and "identical except whether rewiring executes"
    would be false before training even started."""
    set_seed(seed)
    encoder = (
        ObjectSeededEncoder(n_objects=n_objects, n_cells=n_cells, association_dim=association_dim)
        if task in OBJECT_SEEDED_TASKS
        else None
    )
    return StructuralBeliefGraph(
        in_features=in_features,
        out_features=out_features,
        n_cells=n_cells,
        association_dim=association_dim,
        d_s=d_s,
        assoc_dim=assoc_dim,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        encoder=encoder,
        plasticity_config=plasticity_config,
    )


def _build_structural_comparison_models(
    task: str,
    in_features: int,
    out_features: int,
    seed: int,
    n_cells: int,
    n_objects: int = DEFAULT_N_OBJECTS,
    association_dim: int = ASSOCIATION_DIM,
    assoc_dim: int = ASSOC_DIM,
    d_s: int = D_S,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    num_steps: int = NUM_STEPS,
    plasticity_config: StructuralPlasticityConfig | None = None,
) -> tuple[dict[str, torch.nn.Module], dict[str, int | float | str]]:
    """`cellv0.1` (parameter-matched) + the two CellV1.5 arms.

    CellV1.5 is the size-defining target, the same convention
    `_build_global_comparison_models`/`_build_association_comparison_models`
    already use. Both CellV1.5 arms have identical parameter counts by
    construction -- topology lives in buffers, never in parameters
    (§16.2/§16.3) -- so "which arm is bigger" is not a confound here, and
    the reported `param_match_error_fraction` makes `cellv0.1`'s residual
    mismatch (the `BeliefNetwork` size grid is coarse at these
    `in_features`) explicit rather than implied to be exact."""
    config = plasticity_config or StructuralPlasticityConfig()
    common = dict(
        task=task,
        in_features=in_features,
        out_features=out_features,
        seed=seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        assoc_dim=assoc_dim,
        d_s=d_s,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        plasticity_config=config,
    )
    frozen = _make_structural_model(**common)
    plastic = _make_structural_model(**common)
    target_params = count_parameters(frozen)

    set_seed(seed)
    hidden_cells = _match_belief_hidden_cells(target_params, in_features, out_features)
    cellv0_1 = BeliefNetwork(
        in_features, hidden_cells, out_features, aggregation="scale_stable_precision"
    )

    models = {"cellv0.1": cellv0_1, "cellv1.5_frozen": frozen, "cellv1.5_plastic": plastic}
    matched_params = count_parameters(cellv0_1)
    sizing: dict[str, int | float | str] = {
        "cellv1.5_frozen__params": target_params,
        "cellv1.5_plastic__params": count_parameters(plastic),
        "cellv0.1__params": matched_params,
        "cellv0.1__hidden_cells": hidden_cells,
        "param_match_error_fraction": abs(matched_params - target_params) / target_params,
        "d_s": d_s,
        "assoc_dim": assoc_dim,
        "num_steps": num_steps,
    }
    return models, sizing


# ---------------------------------------------------------------------------
# Checkpointing across a runtime-resizable EdgeRegistry
# ---------------------------------------------------------------------------


_REGISTRY_BUFFERS = ("core.edges.edge_index", "core.edges.utility", "core.edges.age")


def _clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _restore_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    """`load_state_dict` is strict about buffer shapes, and CellV1.5's
    `EdgeRegistry` buffers genuinely change shape during training
    (`structural.py`'s own docstring flags that its `state_dict()` "only
    round-trips correctly when the edge count at load time matches the
    saved one"). Re-point the three registry buffers to the checkpoint's
    shapes first -- assigning a tensor to a name already registered via
    `register_buffer` updates `_buffers[name]` in place, exactly as
    `EdgeRegistry.add_edges`/`remove_edges` already do -- then load
    normally. Harness-level workaround; CellV1.5 itself is untouched."""
    registry = getattr(getattr(model, "core", None), "edges", None)
    if registry is not None and all(k in state for k in _REGISTRY_BUFFERS):
        registry.edge_index = state["core.edges.edge_index"].clone()
        registry.utility = state["core.edges.utility"].clone()
        registry.age = state["core.edges.age"].clone()
    model.load_state_dict(state)


# ---------------------------------------------------------------------------
# Structural-substrate measurement (§16's own claims, made checkable)
# ---------------------------------------------------------------------------


def _edge_set(edge_index: torch.Tensor) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in edge_index.tolist()}


def _degree_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean/spread/tails of a degree array, plus a Gini coefficient --
    §16.8 explicitly declines to constrain per-cell degree ("a cell may
    end up with 2 edges or 30"), so how unequal the final distribution
    actually is, is a result, not a diagnostic."""
    sorted_v = np.sort(values.astype(np.float64))
    n = sorted_v.size
    total = sorted_v.sum()
    weights = 2 * np.arange(1, n + 1) - n - 1
    gini = 0.0 if total <= 0 else float(weights.dot(sorted_v) / (n * total))
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_min": int(values.min()),
        f"{prefix}_max": int(values.max()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p99": float(np.quantile(values, 0.99)),
        f"{prefix}_zero_fraction": float((values == 0).mean()),
        f"{prefix}_gini": gini,
    }


def _capture_step_states(
    model: StructuralBeliefGraph, x: torch.Tensor
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Runs `model` on `x` in eval mode and returns, for each of the `T`
    refinement steps, the `(mu, evidence, uncertainty, z, w)` that step
    *actually saw*.

    Via a forward-pre-hook on the single shared `StructuralRefinementStep`
    (which is called `T` times per forward pass) rather than by
    reimplementing the loop -- reimplementing it would risk measuring a
    slightly different model than the one that produced the R^2, which is
    the failure mode this whole harness exists to avoid."""
    captured: list[tuple[torch.Tensor, ...]] = []

    def hook(_module, args):
        cells, _edge_index, w = args[0], args[1], args[2]
        captured.append((cells.mu, cells.evidence, cells.uncertainty, cells.z, w))

    handle = model.core.step.register_forward_pre_hook(hook)
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            model(x)
        model.train(was_training)
    finally:
        handle.remove()
    return captured  # type: ignore[return-value]


def _functional_edge_use(
    model: StructuralBeliefGraph,
    x_eval: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    taus: tuple[float, ...] = EDGE_USE_TAUS,
    max_jaccard_pairs: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """§16.0's actual claim -- "different inputs activate different
    subsets of that learned substrate" -- turned into numbers.

    **The definition.** For receiving cell `j`, §16.5 fuses its incoming
    structural edges weighted by precision `p_ij = a_ij(t) * e_i /
    (u_i^2 + eps)`. The share of `j`'s incoming precision that edge
    `(i, j)` supplies, `share_ij = p_ij / sum_{i'} p_i'j`, is therefore
    the model's *own* statement of how much that edge counts for this
    input -- no new mechanism is introduced to measure it. An edge counts
    as **functionally in use** for an input when
    `share_ij >= tau / in_degree(j)`, i.e. at least `tau` times a uniform
    contribution, and as in use *for that input* if it clears the bar at
    any of the `T` refinement steps. `tau` is reported at three values
    because the cutoff is a measurement choice, not something §16 defines.

    Also reported, threshold-free: `effective_in_degree`, the perplexity
    `exp(H(share_.j))` of each target's precision-share distribution,
    averaged over targets/steps/inputs. If it sits near the mean
    in-degree the substrate is being used near-uniformly (every edge
    always on); well below it means the fusion is genuinely selecting a
    subset.

    Returns per-edge use rates aggregated into scalars; the caller saves
    the raw per-edge arrays alongside."""
    eps = model.core.step.eps
    edge_index = model.core.edges.edge_index
    n_edges = int(edge_index.shape[0])
    out: dict[str, float] = {}
    if n_edges == 0:
        return {f"edge_use_rate_mean__tau{t}": float("nan") for t in taus}

    source = edge_index[:, 0].to(device)
    target = edge_index[:, 1].to(device)
    n_cells = model.core.edges.n_cells
    in_degree = model.core.edges.in_degree().to(device).clamp(min=1)
    in_degree_at_edge = in_degree[target].to(torch.float32)  # (E,)

    # Per-tau: (E,) count of inputs in which the edge was in use, and a
    # per-input active mask kept for the Jaccard/overlap statistics.
    use_counts = {t: torch.zeros(n_edges, dtype=torch.float64) for t in taus}
    masks_for_jaccard: dict[float, list[torch.Tensor]] = {t: [] for t in taus}
    active_per_input = {t: [] for t in taus}
    eff_in_degree_sum, eff_in_degree_count = 0.0, 0
    n_seen = 0

    for start in range(0, x_eval.shape[0], batch_size):
        xb = x_eval[start : start + batch_size].to(device)
        states = _capture_step_states(model, xb)
        batch = xb.shape[0]
        n_seen += batch
        # (steps, batch, E) of per-edge precision shares.
        step_shares = []
        for mu, evidence, uncertainty, z, _w in states:
            phi = model.core.step.assoc_fn(mu, evidence, uncertainty, z)
            a = (phi[:, source] * phi[:, target]).sum(dim=-1)  # (batch, E) -- a_ij(t), §16.4
            # §16.5's p_ij = a_ij(t) * e_i / (u_i^2 + eps).
            precision = a * evidence[:, source] / (uncertainty[:, source] ** 2 + eps)
            denom = torch.zeros(batch, n_cells, device=device, dtype=precision.dtype)
            denom.index_add_(1, target, precision)
            share = precision / (denom[:, target] + eps)
            step_shares.append(share)
            # Threshold-free: perplexity of each target's share distribution.
            ent = torch.zeros(batch, n_cells, device=device, dtype=share.dtype)
            ent.index_add_(1, target, -(share * torch.log(share.clamp_min(1e-12))))
            eff = torch.exp(ent)  # (batch, n_cells)
            has_edges = model.core.edges.in_degree().to(device) > 0
            eff_in_degree_sum += float(eff[:, has_edges].sum().item())
            eff_in_degree_count += int(batch * int(has_edges.sum().item()))

        shares = torch.stack(step_shares, dim=0)  # (steps, batch, E)
        for tau in taus:
            bar = tau / in_degree_at_edge  # (E,)
            active = (shares >= bar).any(dim=0)  # (batch, E) -- in use at any refinement step
            use_counts[tau] += active.sum(dim=0).double().cpu()
            active_per_input[tau].append(active.float().mean(dim=1).cpu())
            masks_for_jaccard[tau].append(active.cpu())

    generator = torch.Generator().manual_seed(seed)
    for tau in taus:
        rate = (use_counts[tau] / max(n_seen, 1)).numpy()  # (E,) fraction of inputs using this edge
        masks = torch.cat(masks_for_jaccard[tau], dim=0)  # (N, E) bool
        per_input_active = torch.cat(active_per_input[tau]).numpy()
        n_inputs = masks.shape[0]
        if n_inputs >= 2:
            idx_a = torch.randint(0, n_inputs, (max_jaccard_pairs,), generator=generator)
            idx_b = torch.randint(0, n_inputs, (max_jaccard_pairs,), generator=generator)
            keep = idx_a != idx_b
            ma, mb = masks[idx_a[keep]], masks[idx_b[keep]]
            inter = (ma & mb).sum(dim=1).double()
            union = (ma | mb).sum(dim=1).double().clamp(min=1)
            jaccard = float((inter / union).mean().item())
        else:
            jaccard = float("nan")
        suffix = f"__tau{tau}"
        out.update(
            {
                f"edge_use_rate_mean{suffix}": float(rate.mean()),
                f"edge_use_rate_std{suffix}": float(rate.std()),
                # The three buckets that answer "is the substrate used
                # input-dependently, or is it just always-on / dead weight":
                f"edges_never_used_fraction{suffix}": float((rate == 0.0).mean()),
                f"edges_always_used_fraction{suffix}": float((rate == 1.0).mean()),
                f"edges_input_dependent_fraction{suffix}": float(
                    ((rate > 0.0) & (rate < 1.0)).mean()
                ),
                f"mean_active_edge_fraction_per_input{suffix}": float(per_input_active.mean()),
                f"mean_pairwise_jaccard{suffix}": jaccard,
            }
        )
        out[f"_edge_use_rate_array{suffix}"] = rate  # type: ignore[assignment]

    out["mean_effective_in_degree"] = (
        eff_in_degree_sum / eff_in_degree_count if eff_in_degree_count else float("nan")
    )
    out["n_eval_inputs_for_edge_use"] = float(n_seen)
    return out


def _structural_metrics(
    model: StructuralBeliefGraph,
    bootstrap_edge_index: torch.Tensor,
    x_eval: torch.Tensor,
    device: torch.device,
    seed: int,
    batch_size: int = 64,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Everything the user asked to be recorded about the substrate,
    measured on the *restored best-validation checkpoint* -- the same
    weights and the same topology that produced the reported test R^2,
    not wherever training happened to stop.

    Returns `(scalars, arrays)`; `arrays` is saved to `results/raw` so the
    degree distribution and per-edge use rates can be re-analyzed without
    re-running anything."""
    final_index = model.core.edges.edge_index.detach().cpu()
    boot_index = bootstrap_edge_index.detach().cpu()
    final_set, boot_set = _edge_set(final_index), _edge_set(boot_index)
    n_final, n_boot = len(final_set), len(boot_set)

    in_degree = model.core.edges.in_degree().cpu().numpy()
    out_degree = np.zeros(model.core.edges.n_cells, dtype=np.int64)
    if final_index.shape[0] > 0:
        np.add.at(out_degree, final_index[:, 0].numpy(), 1)

    scalars: dict[str, float] = {
        "n_edges_bootstrap": float(n_boot),
        "n_edges_final": float(n_final),
        # The headline: how much of the substrate is no longer what
        # bootstrap handed it (§16.8 calls the bootstrap topology
        # "disposable" -- this is whether it actually got disposed of).
        "structural_edges_changed_fraction": (
            len(final_set - boot_set) / n_final if n_final else float("nan")
        ),
        "bootstrap_edges_retained_fraction": (
            len(final_set & boot_set) / n_boot if n_boot else float("nan")
        ),
        "n_edges_added_vs_bootstrap": float(len(final_set - boot_set)),
        "n_edges_removed_vs_bootstrap": float(len(boot_set - final_set)),
        "min_in_degree": float(in_degree.min()),  # §16.8's one safety constraint, checked
    }
    scalars.update(_degree_summary(in_degree, "in_degree"))
    scalars.update(_degree_summary(out_degree, "out_degree"))

    use = _functional_edge_use(model, x_eval, device, batch_size=batch_size, seed=seed)
    arrays: dict[str, np.ndarray] = {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "final_edge_index": final_index.numpy(),
        "bootstrap_edge_index": boot_index.numpy(),
        "final_edge_utility": model.core.edges.utility.detach().cpu().numpy(),
        "final_edge_age": model.core.edges.age.detach().cpu().numpy(),
    }
    for key, value in list(use.items()):
        if key.startswith("_edge_use_rate_array"):
            arrays["edge_use_rate" + key[len("_edge_use_rate_array") :]] = value
        else:
            scalars[key] = value
    return scalars, arrays


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_until_convergence_structural(
    model: torch.nn.Module,
    loss_fn,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    seed: int,
    regression: bool,
    batch_size: int,
    lr: float,
    val_every: int,
    patience_steps: int,
    max_steps: int,
    plasticity_enabled: bool,
    is_structural: bool,
) -> tuple[dict[str, float | int], torch.Tensor | None]:
    """`harness.py::_train_until_convergence`'s protocol -- validate every
    `val_every` steps, stop after `patience_steps` without improvement,
    restore the best-validation state before returning -- extended with
    `StructuralRefinementCore`'s documented integration contract
    (forward -> backward -> `update_edge_utility` -> `optimizer.step` ->
    `maybe_run_structural_plasticity`) and with checkpointing that
    survives a resized `EdgeRegistry`.

    Not a modification of the shared loop: kept separate so the CellV1.4/
    ORFF-field results already on record stay reproducible from the
    function that produced them, byte for byte.

    `plasticity_enabled=False` is the frozen arm -- bootstrap still runs
    (there has to *be* a substrate), utility is still tracked (pure
    bookkeeping; it changes no parameter and no gradient), only
    `maybe_run_structural_plasticity` is withheld.

    Returns `(convergence_stats, bootstrap_edge_index)`."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    x_val_dev, y_val_dev = x_val.to(device), y_val.to(device)

    bootstrap_edge_index: torch.Tensor | None = None
    plasticity_generator: torch.Generator | None = None
    if is_structural:
        # A dedicated generator for both bootstrap and plasticity: the
        # frozen arm never draws from it, so if plasticity shared the
        # global stream the two arms' minibatch order would diverge and
        # "identical except rewiring" would be false.
        bootstrap_generator = torch.Generator().manual_seed(seed)
        plasticity_generator = torch.Generator().manual_seed(seed + 10_000)
        model.bootstrap_structural_graph(
            x_train[:batch_size].to(device), generator=bootstrap_generator
        )
        bootstrap_edge_index = model.core.edges.edge_index.detach().cpu().clone()

    model.train()
    best_val = -float("inf")
    best_step, best_wall_clock = 0, 0.0
    best_state: dict[str, torch.Tensor] | None = None
    best_plasticity_events = 0
    steps_since_improvement = 0
    plasticity_events = 0

    start = time.perf_counter()
    step = 0
    data_iter = iter(loader)
    while step < max_steps:
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)

        loss = loss_fn(model(xb), yb)
        loss.backward()
        if is_structural:
            model.update_edge_utility()  # §16.6 -- after backward, before zero_grad
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if is_structural and plasticity_enabled:
            # `total_steps=max_steps`: §16.9's freeze window is the last
            # 20% of the training *budget* -- see this module's docstring.
            ran = model.maybe_run_structural_plasticity(
                xb, step, max_steps, generator=plasticity_generator
            )
            plasticity_events += int(ran)

        if step % val_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_dev)
                metric_fn = r_squared if regression else accuracy_metric
                val_metric = metric_fn(val_pred, y_val_dev)
            model.train()

            if val_metric > best_val:
                best_val = val_metric
                best_step = step
                best_wall_clock = time.perf_counter() - start
                best_state = _clone_model_state(model)
                best_plasticity_events = plasticity_events
                steps_since_improvement = 0
            else:
                steps_since_improvement += val_every

            if steps_since_improvement >= patience_steps:
                break

    total_wall_clock = time.perf_counter() - start
    if best_state is not None:
        _restore_model_state(model, best_state)

    core = getattr(model, "core", None)
    plasticity = getattr(core, "plasticity_config", None) or StructuralPlasticityConfig()
    freeze_start = int((1.0 - plasticity.freeze_fraction) * max_steps)
    return (
        {
            "steps_to_convergence": best_step,
            "wall_clock_to_convergence_seconds": best_wall_clock,
            "total_steps_run": step,
            "total_wall_clock_seconds": total_wall_clock,
            "best_val_metric": best_val,
            "plasticity_events_total": plasticity_events,
            "plasticity_events_at_best_checkpoint": best_plasticity_events,
            "entered_freeze_window": bool(
                is_structural and plasticity_enabled and step > freeze_start
            ),
        },
        bootstrap_edge_index,
    )


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_structural_comparison(
    task: str,
    seed: int,
    batch_size: int = 64,
    lr: float = 1e-2,
    n_train: int = 3_000,
    n_val: int = 500,
    n_test: int = 500,
    val_every: int = 100,
    patience_steps: int = 2_000,
    max_steps: int = 15_000,
    results_dir: str | Path = "results/raw",
    graph_dir: str | Path = "results/raw/structural_substrate",
    n_cells: int | None = None,
    n_objects: int = DEFAULT_N_OBJECTS,
    k_min: int = 2,
    k_max: int = 5,
    association_dim: int = ASSOCIATION_DIM,
    assoc_dim: int = ASSOC_DIM,
    d_s: int = D_S,
    hidden_dim: int = CELLV1_HIDDEN_DIM,
    num_steps: int = NUM_STEPS,
    device: torch.device | None = None,
    plasticity_config: StructuralPlasticityConfig | None = None,
    edge_use_eval_n: int = 500,
) -> dict:
    """One `(task, seed, complexity)` cell of the CellV1.5 evaluation:
    `cellv0.1` / `cellv1.5_frozen` / `cellv1.5_plastic`, convergence-
    trained, best-validation checkpointed, then measured.

    `patience_steps=2000` / `max_steps=15000` are deliberately far more
    generous than the `500`/`5000` used for the CellV1.3/1.4 sweeps. That
    setting produced a documented artifact (`docs/research_log.md`,
    2026-09-03 CellV1.3.1 entry, finding 3): `field_t2` at
    `very_hard`/seed 0 early-stopped at R^2=0.0667 while the same
    architecture reached 0.79/0.82 on the other two seeds -- a model
    killed on a plateau before it broke out, reported as an architectural
    result. A patience of 2000 steps is four times the longest
    plateau-to-improvement gap anywhere in that sweep's recorded
    steps-to-convergence (which topped out around 2400 steps *total*), and
    the 15000-step cap is 3x the old one, so a run that stops has visibly
    stopped improving rather than run out of budget. Each run records
    `total_steps_run` against `max_steps` so a cap-limited run is
    identifiable rather than silently misread as converged."""
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Expected one of {TASKS}.")

    set_seed(seed)
    device = device if device is not None else get_device()
    regression = _is_regression(task)
    is_object_seeded = task in OBJECT_SEEDED_TASKS
    n_cells = n_cells if n_cells is not None else n_objects

    train_split, val_split, test_split = _build_splits(
        task, seed, n_train, n_val, n_test, n_objects=n_objects, k_min=k_min, k_max=k_max
    )
    if is_object_seeded:
        x_train, y_train, _ = train_split
        x_val, y_val, _ = val_split
        x_test, y_test, _ = test_split
    else:
        x_train, y_train = train_split
        x_val, y_val = val_split
        x_test, y_test = test_split
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    in_features = x_train.shape[1]
    out_features = 1 if regression else 2
    loss_fn = torch.nn.MSELoss() if regression else torch.nn.CrossEntropyLoss()
    headline = "r2" if regression else "accuracy"

    config_plasticity = plasticity_config or StructuralPlasticityConfig()
    models, sizing = _build_structural_comparison_models(
        task,
        in_features,
        out_features,
        seed,
        n_cells=n_cells,
        n_objects=n_objects,
        association_dim=association_dim,
        assoc_dim=assoc_dim,
        d_s=d_s,
        hidden_dim=hidden_dim,
        num_steps=num_steps,
        plasticity_config=config_plasticity,
    )

    results: dict[str, float | int | str | bool] = {
        "task": task,
        "seed": seed,
        "n_cells": n_cells,
        "n_objects": n_objects,
        "k_min": k_min,
        "k_max": k_max,
        "headline": headline,
        "max_steps": max_steps,
        "patience_steps": patience_steps,
        "k_bar": config_plasticity.k_bar,
        "warmup_steps": config_plasticity.warmup_steps,
        "update_interval": config_plasticity.update_interval,
        "prune_fraction": config_plasticity.prune_fraction,
        "freeze_fraction": config_plasticity.freeze_fraction,
        "device": str(device),
        **sizing,
    }

    x_test_dev, y_test_dev = x_test.to(device), y_test.to(device)
    graph_path = Path(graph_dir)

    for arch_name, model in models.items():
        is_structural = arch_name in CELLV1_5_ARCHITECTURES
        convergence, bootstrap_edge_index = _train_until_convergence_structural(
            model,
            loss_fn,
            x_train,
            y_train,
            x_val,
            y_val,
            device,
            seed,
            regression,
            batch_size,
            lr,
            val_every=val_every,
            patience_steps=patience_steps,
            max_steps=max_steps,
            plasticity_enabled=(arch_name == "cellv1.5_plastic"),
            is_structural=is_structural,
        )

        model.eval()
        with torch.no_grad():
            infer_start = time.perf_counter()
            test_pred = model(x_test_dev)
            inference_wall_clock = time.perf_counter() - infer_start
            test_metrics = _compute_metrics(regression, test_pred, y_test_dev)

        for name, value in test_metrics.items():
            results[f"{arch_name}__{name}"] = value
        results[f"{arch_name}__inference_wall_clock_seconds"] = inference_wall_clock
        results[f"{arch_name}__hit_step_cap"] = convergence["total_steps_run"] >= max_steps
        for name, value in convergence.items():
            results[f"{arch_name}__{name}"] = value

        if is_structural and bootstrap_edge_index is not None:
            scalars, arrays = _structural_metrics(
                model,
                bootstrap_edge_index,
                x_test[:edge_use_eval_n],
                device,
                seed=seed,
                batch_size=batch_size,
            )
            for name, value in scalars.items():
                results[f"{arch_name}__{name}"] = value
            graph_path.mkdir(parents=True, exist_ok=True)
            npz_name = f"v1_002_{arch_name}_{task}_n{n_objects}_seed{seed}.npz"
            np.savez_compressed(graph_path / npz_name, **arrays)

        config = ExperimentConfig(
            experiment_id="v1_002_structural_substrate",
            architecture=arch_name,
            dataset=task,
            seed=seed,
            optimizer="adamw",
            learning_rate=lr,
            batch_size=batch_size,
            max_steps=max_steps,
            extra={
                **sizing,
                "n_cells": n_cells,
                "n_objects": n_objects,
                "k_min": k_min,
                "k_max": k_max,
                "association_dim": association_dim,
                "val_every": val_every,
                "patience_steps": patience_steps,
                "structural_plasticity_enabled": arch_name == "cellv1.5_plastic",
                **{k: v for k, v in convergence.items()},
            },
        )
        record = RunRecord(
            run_id=make_run_id(config),
            experiment_id=config.experiment_id,
            architecture=arch_name,
            config=config.to_dict(),
            parameter_count=count_parameters(model),
            dataset=task,
            seed=seed,
            optimizer=config.optimizer,
            learning_rate=lr,
            steps_completed=int(convergence["total_steps_run"]),
            examples_or_tokens_seen=int(convergence["total_steps_run"]) * batch_size,
            refinement_iterations=num_steps if is_structural else None,
            approximate_flops=None,
            train_wall_clock_seconds=float(convergence["total_wall_clock_seconds"]),
            inference_wall_clock_seconds=inference_wall_clock,
            validation_metrics={"best": float(convergence["best_val_metric"])},
            test_metrics=test_metrics,
            git_commit=get_git_commit(),
            checkpoint_path=None,
        )
        write_run_record(record, Path(results_dir))

    return results
