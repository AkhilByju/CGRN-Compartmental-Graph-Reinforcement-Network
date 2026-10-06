"""Critical mechanism test (task Sec 12): at evaluation only, no retraining,
compare a trained CellV0.3 model's normal inference against two ablations
applied purely inside the `BeliefFFN` activation:

- precision-neutralized: `sqrt(pi_hidden) -> 1`
- conflict-neutralized: force `u_hidden = 0` when computing hidden precision

Both are `BeliefFFN.forward` keyword flags (`belief_ffn.py`); this module
just wires `training.evaluate_nll` up to them and reports the deltas that
tell us whether the learned belief state is causally load-bearing.
"""

from __future__ import annotations

from typing import Any

import torch

from experiments.nano_transformer.belief_ffn import BeliefFFN
from experiments.nano_transformer.training import evaluate_nll


def model_supports_belief_interventions(model: torch.nn.Module) -> bool:
    return isinstance(model.blocks[0].ffn, BeliefFFN)


def run_critical_mechanism_test(
    model: torch.nn.Module,
    ids,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Returns `None` for models without a belief mechanism (Model A / C);
    otherwise `{"normal": {...}, "precision_neutralized": {...},
    "conflict_neutralized": {...}, "deltas": {...}}`, each inner dict the
    `evaluate_nll` metrics for that inference mode, on the full `ids` split
    (intended for the held-out test set, Sec 12/10)."""
    if not model_supports_belief_interventions(model):
        return None

    normal = evaluate_nll(model, ids, batch_size, context_length, device)
    precision_neutralized = evaluate_nll(
        model, ids, batch_size, context_length, device, neutralize_precision=True
    )
    conflict_neutralized = evaluate_nll(
        model, ids, batch_size, context_length, device, neutralize_conflict=True
    )

    deltas = {
        "precision_neutralized_delta_nll": precision_neutralized["nll"] - normal["nll"],
        "precision_neutralized_delta_ppl": (
            precision_neutralized["perplexity"] - normal["perplexity"]
        ),
        "conflict_neutralized_delta_nll": conflict_neutralized["nll"] - normal["nll"],
        "conflict_neutralized_delta_ppl": conflict_neutralized["perplexity"] - normal["perplexity"],
    }

    return {
        "normal": normal,
        "precision_neutralized": precision_neutralized,
        "conflict_neutralized": conflict_neutralized,
        "deltas": deltas,
    }
