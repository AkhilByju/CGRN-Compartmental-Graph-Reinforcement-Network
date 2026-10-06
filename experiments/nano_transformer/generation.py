"""Generation evaluation (task Sec 13): a fixed prompt set, identical
sampling procedure for every model, saved raw generations, and the
diversity metrics computed on the generated continuation only (not the
prompt). Distinct-n follows the standard word-level definition (Li et al.
2016): unique n-grams / total n-grams over whitespace-split words of the
decoded text -- BPE-token-level distinct-n would conflate tokenizer
granularity with generation diversity, especially at this project's tiny
512-token vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from experiments.nano_transformer.tokenizer import TrainedTokenizer

PROMPTS: tuple[str, ...] = (
    "Once upon a time, there was a little girl named Lily.",
    "Tom and his dog went for a walk in the park.",
    "One sunny day, a rabbit found a big carrot.",
    "The little boy was very happy because",
    "Once upon a time, there was a cat who loved to",
    "Sara and her mom went to the store to buy",
    "In a small house, there lived a friendly bear who",
    "The sun was shining, and all the animals were",
    "One day, a bird flew over the forest and saw",
    "Once there was a kind old man who lived in a",
)
GENERATION_SEEDS: tuple[int, ...] = (0, 1, 2)
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_P = 0.95


@torch.no_grad()
def _sample_next_token(logits: torch.Tensor, top_p: float, generator: torch.Generator) -> int:
    """Nucleus (top-p) sampling from `logits` (already temperature-scaled),
    shape `[vocab_size]`."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # Keep the smallest prefix whose cumulative probability >= top_p; a
    # token is dropped only once the *previous* tokens already reached top_p.
    drop_mask = (cumulative - sorted_probs) > top_p
    sorted_probs = sorted_probs.masked_fill(drop_mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum()
    choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
    return int(sorted_idx[choice].item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer: TrainedTokenizer,
    prompt: str,
    device: torch.device,
    context_length: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    seed: int = 0,
) -> dict[str, Any]:
    """Autoregressive generation with a per-call `torch.Generator` (so
    sampling is reproducible without disturbing global RNG state, and is
    identical across models given the same seed/prompt -- Sec 13: "identical
    for every model"). Returns the prompt, the generated continuation only,
    and the full text."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)

    prompt_ids = tokenizer.encode(prompt)
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        idx = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
        logits = model(idx)[0, -1, :] / temperature
        next_id = _sample_next_token(logits.to("cpu"), top_p, generator)
        ids.append(next_id)

    if was_training:
        model.train()

    continuation_ids = ids[len(prompt_ids) :]
    return {
        "prompt": prompt,
        "seed": seed,
        "prompt_ids": prompt_ids,
        "continuation_ids": continuation_ids,
        "continuation_text": tokenizer.decode(continuation_ids),
        "full_text": tokenizer.decode(ids),
    }


def _ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def distinct_n(words: list[str], n: int) -> float:
    grams = _ngrams(words, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repeated_ngram_rate(words: list[str], n: int) -> float:
    grams = _ngrams(words, n)
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def diversity_metrics(text: str) -> dict[str, float]:
    words = text.split()
    return {
        "distinct_1": distinct_n(words, 1),
        "distinct_2": distinct_n(words, 2),
        "repeated_bigram_rate": repeated_ngram_rate(words, 2),
        "repeated_trigram_rate": repeated_ngram_rate(words, 3),
    }


@dataclass(frozen=True)
class GenerationEvalResult:
    generations: list[dict[str, Any]]
    aggregate_diversity: dict[str, float]


def run_generation_eval(
    model: torch.nn.Module,
    tokenizer: TrainedTokenizer,
    device: torch.device,
    context_length: int,
    prompts: tuple[str, ...] = PROMPTS,
    seeds: tuple[int, ...] = GENERATION_SEEDS,
) -> GenerationEvalResult:
    """Sec 13: every (prompt, seed) pair, identical sampling config for
    every model. Every raw generation is kept (`generations`); diversity
    metrics are also averaged across all of them (`aggregate_diversity`)."""
    generations: list[dict[str, Any]] = []
    per_gen_metrics: list[dict[str, float]] = []

    for prompt in prompts:
        for seed in seeds:
            result = generate(model, tokenizer, prompt, device, context_length, seed=seed)
            metrics = diversity_metrics(result["continuation_text"])
            generations.append({**result, **metrics})
            per_gen_metrics.append(metrics)

    aggregate = {
        key: sum(m[key] for m in per_gen_metrics) / len(per_gen_metrics)
        for key in per_gen_metrics[0]
    }
    return GenerationEvalResult(generations=generations, aggregate_diversity=aggregate)
