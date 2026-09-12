"""CellV0.3 belief-state diagnostics (task Sec 11): per-layer hidden `pi`
mean/std/coefficient-of-variation and `u` mean/std, snapshotted on a fixed
sample so successive snapshots (init / 10% / 50% / 100% of training) are
directly comparable. No-ops for models whose FFN isn't `BeliefFFN` (Model A
and Model C have no belief state).
"""

from __future__ import annotations

import torch

from experiments.nano_transformer.backbone import TransformerLM

_DIAGNOSTIC_SAMPLE_BATCH = 8


@torch.no_grad()
def collect_belief_diagnostics(
    model: TransformerLM,
    ids,
    device: torch.device,
    context_length: int,
    batch_size: int = _DIAGNOSTIC_SAMPLE_BATCH,
) -> dict[int, dict[str, float]] | None:
    """Runs a fixed, deterministic sample (the first `batch_size` blocks of
    `ids`, independent of the current training step) through the model,
    replicating `TransformerBlock.forward`'s attention step exactly but
    calling `ffn.forward_with_diagnostics` in place of `ffn.forward` so each
    layer's belief-state summary can be captured. Returns `None` if this
    model's FFN doesn't expose belief diagnostics (Model A / Model C)."""
    first_ffn = model.blocks[0].ffn
    if not hasattr(first_ffn, "forward_with_diagnostics"):
        return None

    was_training = model.training
    model.eval()

    stride = context_length + 1
    n_blocks_available = (len(ids) - stride) // context_length + 1
    batch_size = min(batch_size, max(1, n_blocks_available))

    batch_x = torch.empty((batch_size, context_length), dtype=torch.long)
    for row in range(batch_size):
        start = row * context_length
        window = ids[start : start + stride]
        batch_x[row] = torch.from_numpy(window[:-1].astype("int64"))
    batch_x = batch_x.to(device)

    x = model.tok_emb(batch_x)
    rope_freqs = model.rope_freqs[:context_length].to(x.device)

    layer_diagnostics: dict[int, dict[str, float]] = {}
    for i, block in enumerate(model.blocks):
        x = x + block.attn(block.attn_norm(x), rope_freqs)
        normed = block.ffn_norm(x)
        ffn_out, diag = block.ffn.forward_with_diagnostics(normed)
        x = x + ffn_out
        layer_diagnostics[i] = diag

    if was_training:
        model.train()

    return layer_diagnostics
