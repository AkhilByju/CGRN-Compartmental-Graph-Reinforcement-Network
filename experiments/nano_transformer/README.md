# CellV0.3 Nano Transformer — exploratory, separate from Paper A

**Branch:** `cellv03_nano_transformer`. Not pushed.

Tests whether CellV0.3 (`src/models/architecture_v0/conflict_normalized.py`'s
`ConflictNormalizedLayer`, frozen, unmodified) functions as a useful
hidden-neuron primitive inside an otherwise conventional decoder-only
Transformer, at ~1,000,000 parameters, on TinyStories. Does not modify
CellV0.3's equations and does not redesign it based on these results. Does
not touch the existing BeliefDendrite (`experiments/belief_dendrite/`) or
Paper A (`experiments/paper_a/`) experiments.

## The three models (Sec 5-7)

Identical backbone (`backbone.py`): pre-norm blocks, RMSNorm, RoPE, causal
self-attention via `F.scaled_dot_product_attention(is_causal=True)`, 4
heads, `d_model=128`, `n_layers=5`, `context_length=256`, tied
embedding/LM-head, no dropout, no attention bias. The **only** thing that
differs between models is the FFN hidden-neuron primitive:

| model | FFN | file |
|---|---|---|
| `swiglu` (Model A) | `SiLU(W_gate x) * (W_value x) -> W_out` | `models.py::SwiGLUFFN` |
| `cellv0.3` (Model B) | frozen CellV0.3, raw-moment kernel, `mu_in=x, e_in=1, u_in=0` every call | `belief_ffn.py::BeliefFFN` |
| `fixed_confidence` (Model C) | same `V`/`gain`/`bias`/`out_proj` shapes as B, no belief mechanism | `belief_ffn.py::FixedConfidenceFFN` |

Each model's FFN hidden width is independently solved
(`param_solver.py::solve_hidden_width_for_target`) to land within +/-1% of
1,000,000 total parameters -- SwiGLU and the two CellV0.3-shaped FFNs have
different per-hidden-unit parameter costs, so their widths differ by design.

## Data (Sec 2)

Official TinyStories (`roneneldan/TinyStories`, the original release, not
the GPT4-augmented V2) `train`/`valid` text files. TinyStories ships no
official test split, so `data.py::split_validation_into_val_and_test`
derives both a validation set (used only for LR selection during training)
and a held-out test set (used once, at the end) from the official
validation file via a fixed, deterministic 50/50 story-index-parity split --
both stay entirely held out from training; only the official train split is
ever tokenized into the training stream. This is a disclosed design choice,
not an assumption.

One byte-level BPE tokenizer, vocab size 512 (one special token,
`<|endoftext|>`, matching TinyStories' own document separator), trained on
the training split only and frozen for every model (`tokenizer.py`). Token
streams are cached once (`cache/tokens/*.npy`) and walked deterministically,
non-overlapping, non-shuffled (`data.py::iterate_batches`) -- the same
example order regardless of model, seed, or LR.

## Training (Sec 10)

AdamW (`betas=(0.9, 0.95)`, `weight_decay=0.1`), grad-clip 1.0, cosine decay
with 5% warmup, batch 64 x context 256 (16,384 tokens/step), exactly
30,000,000 processed training tokens (never epoch count). LR grid `{3e-4,
1e-3}` x seeds `{0, 1, 2}` x 3 model kinds = 18 runs; the better LR per
`(model, seed)` is selected by validation NLL alone, then evaluated once on
the held-out test split (`training.py`, `run_sweep.py`).

## Real preflight results (this machine, Apple M4 Max, MPS, batch=64, context=256)

| model | ms/step | tokens/sec | vs swiglu |
|---|---|---|---|
| swiglu | 63.54 | 257,841 | 1.00x |
| cellv0.3 | 156.27 | 104,847 | 2.46x |
| fixed_confidence | 102.06 | 160,528 | 1.61x |

Under Sec 15's 3x threshold -- no implementation-only optimization pass was
required. Full JSON: `results/processed/mps_speed_preflight.json`.

## Running it

```bash
# one-time: download + tokenize + cache (idempotent, safe to re-run)
python -c "from experiments.nano_transformer.data import prepare_dataset; prepare_dataset()"

# MPS speed preflight (Sec 15)
python -m experiments.nano_transformer.preflight

# the full 18-run sweep (Sec 10-13)
python -m experiments.nano_transformer.run_sweep

# optional, only after the sweep above finishes (Sec 17)
python -m experiments.nano_transformer.run_scaling

# build the results doc from the sweep's output
python -m experiments.nano_transformer.build_report
```

Aggregated results: `results/processed/sweep_summary.json` (+
`scaling_summary.json` if Sec 17 was run). Every raw generation:
`results/raw/generations/<model>_seed<seed>.json`. Per-run provenance
(config, parameter count, git commit, checkpoint path, metrics):
`results/raw/run_records/*.json`. Narrative results: `cellv03_transformer_results.md`.

## Design choices made without further sign-off (generic infra, not architecture)

- Tokenizer library (`tokenizers`, byte-level BPE) and dataset download
  (direct HTTPS from the official HF dataset repo).
- Val/test derivation from TinyStories' single official validation file
  (above).
- Learning-efficiency curve (Sec 14) evaluates a fixed 500,000-token
  validation subset at each token checkpoint (not the full ~4.1M-token
  validation split, which is instead used in full for the Sec 10 LR-
  selection NLL and the single final test evaluation) -- a standard
  compute/rigor tradeoff for a 5-point curve, disclosed rather than silent.
- Distinct-n / repeated-n-gram-rate (Sec 13) use the standard word-level
  definition (Li et al. 2016: unique n-grams / total n-grams over
  whitespace-split decoded text), not BPE-token-level n-grams, since the
  latter would conflate tokenizer granularity with generation diversity at
  this project's 512-token vocabulary.
- No KV cache in generation (`generation.py::generate` recomputes the full
  forward pass each step) -- simplicity over speed at 200 generated tokens.
- Sec 18's interpretation buckets are computed mechanically from that
  section's own stated criteria (`summarize.py::classify_interpretation`),
  not asserted by hand.
