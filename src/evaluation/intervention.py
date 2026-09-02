"""Inference-time interventions on an already-computed `BeliefCell`'s
evidence/uncertainty channels (docs/research_log.md "Experiment 002 initial
results" follow-up 2; Experiment 003D, docs/experiment_protocol.md).

Generic diagnostic tooling, not new architecture logic: every function here
takes a `BeliefCell` produced by an already-trained `BeliefLayer` (any
aggregation method) and returns a new one with `evidence`/`uncertainty`
replaced by a fixed, non-learned rule -- `mu` is always left untouched, and
none of `integration.py`'s aggregation formulas are touched. This is a
controlled-corruption/permutation-style test, the same idea as permutation
feature importance, applied to a cell's internal reliability state instead
of a model's raw inputs.

Used to answer: does a downstream `BeliefLayer`'s aggregation actually use
`evidence`/`uncertainty` in a way that matters for the final prediction, or
has the network learned to route around them, making `(evidence,
uncertainty)` decorative? If a perturbation here barely changes prediction
quality, that channel isn't pulling weight for that cell.
"""

from __future__ import annotations

import torch

from src.models.architecture_v0.cell import BeliefCell

PERTURBATION_MODES: tuple[str, ...] = (
    "baseline",
    "evidence_ones",
    "uncertainty_ones",
    "shuffle_evidence_uncertainty",
    "uncertainty_random",
)


def perturb_belief(belief: BeliefCell, mode: str, seed: int) -> BeliefCell:
    """Returns a new `BeliefCell` with `evidence`/`uncertainty` perturbed
    per `mode`; `mu` is always unchanged so any prediction-quality change is
    attributable to the evidence/uncertainty channels alone. Randomized
    modes draw from their own CPU generator seeded by `seed`, independent of
    any training/data RNG state, so a perturbed evaluation is reproducible
    on its own.

    - "baseline": identity -- the unperturbed reference every other mode is
      compared against.
    - "evidence_ones": every cell's `evidence` set to 1 (matching the fixed
      value `BeliefCell.from_observed_features` gives raw inputs);
      `uncertainty` is kept as computed.
    - "uncertainty_ones": every cell's `uncertainty` set to 1; `evidence` is
      kept as computed.
    - "shuffle_evidence_uncertainty": for each example independently, the
      `(evidence, uncertainty)` pairs are randomly permuted across cells --
      every cell keeps its own `mu` but receives another cell's evidence/
      uncertainty, breaking any learned correspondence between a cell's
      content and its own reliability state while leaving the marginal
      distribution of evidence/uncertainty values unchanged.
    - "uncertainty_random": `uncertainty` replaced with values drawn
      uniformly from the batch's own observed `[min, max]` range (so the
      substitute sits on a realistic scale rather than being adversarially
      out-of-distribution) but otherwise independent of the input and the
      cell.
    """
    if mode not in PERTURBATION_MODES:
        raise ValueError(
            f"Unknown perturbation mode '{mode}'. Expected one of {PERTURBATION_MODES}."
        )

    if mode == "baseline":
        return belief

    if mode == "evidence_ones":
        return BeliefCell(
            mu=belief.mu,
            evidence=torch.ones_like(belief.evidence),
            uncertainty=belief.uncertainty,
        )

    if mode == "uncertainty_ones":
        return BeliefCell(
            mu=belief.mu,
            evidence=belief.evidence,
            uncertainty=torch.ones_like(belief.uncertainty),
        )

    # Randomized modes: generate on CPU (a `torch.Generator` used with
    # `.uniform_`/`torch.randperm` must live on the same device as the
    # generated tensor, and CPU generators are the common denominator across
    # this project's CPU/MPS/CUDA targets), then move the result to
    # `belief`'s device.
    generator = torch.Generator().manual_seed(seed)
    batch, cells = belief.shape

    if mode == "shuffle_evidence_uncertainty":
        perm = torch.stack(
            [torch.randperm(cells, generator=generator) for _ in range(batch)]
        ).to(belief.device)
        evidence = torch.gather(belief.evidence, -1, perm)
        uncertainty = torch.gather(belief.uncertainty, -1, perm)
        return BeliefCell(mu=belief.mu, evidence=evidence, uncertainty=uncertainty)

    # mode == "uncertainty_random"
    lo = belief.uncertainty.min().item()
    hi = belief.uncertainty.max().item()
    random_uncertainty = (
        torch.empty(batch, cells, dtype=belief.dtype).uniform_(lo, hi, generator=generator)
    ).to(belief.device)
    return BeliefCell(mu=belief.mu, evidence=belief.evidence, uncertainty=random_uncertainty)
