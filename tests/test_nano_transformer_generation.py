import torch

from experiments.nano_transformer.generation import (
    distinct_n,
    diversity_metrics,
    generate,
    repeated_ngram_rate,
    run_generation_eval,
)
from experiments.nano_transformer.tokenizer import train_byte_level_bpe_tokenizer
from experiments.nano_transformer.training import build_model_and_spec

_SAMPLE_TEXT = (
    "Once upon a time there was a little dog named Spot. "
    "Spot liked to play in the park with his friend Kitty. "
    "Kitty and Spot found a shiny red ball. They played all day and became "
    "best friends forever.\n"
) * 30


def _make_tokenizer(tmp_path):
    text_path = tmp_path / "train.txt"
    text_path.write_text(_SAMPLE_TEXT)
    return train_byte_level_bpe_tokenizer(text_path, vocab_size=300)


def test_distinct_n_and_repeated_ngram_rate_basic_cases():
    words = ["a", "b", "a", "b", "a", "b"]
    assert distinct_n(words, 1) == 2 / 6
    assert distinct_n(words, 2) == 2 / 5  # bigrams: ab,ba,ab,ba,ab -> {ab,ba}
    assert repeated_ngram_rate(words, 2) == 1 - 2 / 5


def test_distinct_n_empty_input_is_zero():
    assert distinct_n([], 1) == 0.0
    assert repeated_ngram_rate(["only_one"], 2) == 0.0


def test_diversity_metrics_keys():
    metrics = diversity_metrics("the cat sat on the mat the cat sat")
    for key in ("distinct_1", "distinct_2", "repeated_bigram_rate", "repeated_trigram_rate"):
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0


def test_generate_is_reproducible_given_same_seed(tmp_path):
    torch.manual_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    model, _ = build_model_and_spec("swiglu", vocab_size=tokenizer.vocab_size, context_length=64)
    device = torch.device("cpu")

    out_a = generate(
        model, tokenizer, "Spot the dog", device, context_length=64, max_new_tokens=20, seed=42
    )
    out_b = generate(
        model, tokenizer, "Spot the dog", device, context_length=64, max_new_tokens=20, seed=42
    )
    assert out_a["continuation_ids"] == out_b["continuation_ids"]


def test_generate_differs_across_seeds_generically(tmp_path):
    torch.manual_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    model, _ = build_model_and_spec("swiglu", vocab_size=tokenizer.vocab_size, context_length=64)
    device = torch.device("cpu")

    out_a = generate(
        model, tokenizer, "Spot the dog", device, context_length=64, max_new_tokens=20, seed=0
    )
    out_b = generate(
        model, tokenizer, "Spot the dog", device, context_length=64, max_new_tokens=20, seed=1
    )
    assert out_a["continuation_ids"] != out_b["continuation_ids"]


def test_run_generation_eval_covers_all_prompts_and_seeds(tmp_path):
    torch.manual_seed(0)
    tokenizer = _make_tokenizer(tmp_path)
    model, _ = build_model_and_spec("cellv0.3", vocab_size=tokenizer.vocab_size, context_length=64)
    device = torch.device("cpu")

    prompts = ("Spot the dog", "Kitty played")
    seeds = (0, 1)
    result = run_generation_eval(
        model, tokenizer, device, context_length=64, prompts=prompts, seeds=seeds
    )
    assert len(result.generations) == len(prompts) * len(seeds)
    for key in ("distinct_1", "distinct_2", "repeated_bigram_rate", "repeated_trigram_rate"):
        assert key in result.aggregate_diversity
