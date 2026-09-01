"""Conventional optimizer construction.

Per docs/research_thesis.md's isolation-of-variables rule, ArchitectureV0
experiments use standard optimizers (AdamW by default). Do not add a novel
optimizer here without a deliberate, logged research decision
(docs/research_log.md) -- see CLAUDE.md Sec 2.
"""

from __future__ import annotations

import torch

from src.utilities.config import ExperimentConfig

_SUPPORTED_OPTIMIZERS = {"adamw", "adam", "sgd"}


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    name = config.optimizer.lower()
    if name not in _SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"Unsupported optimizer '{config.optimizer}'. Supported: "
            f"{sorted(_SUPPORTED_OPTIMIZERS)}. Introducing a novel optimizer is a "
            "deliberate research decision -- see docs/research_thesis.md "
            "'Isolation of variables' before adding one."
        )
    params = model.parameters()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.SGD(params, lr=config.learning_rate, weight_decay=config.weight_decay)
