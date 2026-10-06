# CellV0.3 Nano Transformer — Results

Exploratory work, separate from Paper A. Branch `cellv03_nano_transformer`,
not merged/pushed. See `README.md` for the full protocol and disclosed
design choices; `docs/architecture_v0.md` Sec 10 for CellV0.3's frozen
equations. Run 2026-09-12 on an Apple M4 Max (MPS).

## Scientific question and one-line answer

**Does CellV0.3 function as a useful hidden-neuron primitive inside an
otherwise conventional decoder-only Transformer at ~1M parameters, on
TinyStories?** No. It underperforms both a parameter-matched SwiGLU
Transformer and its own mechanism-stripped control by a wide, consistent
margin, at 2.46x the per-step compute — and the disadvantage *widens*
monotonically as capacity scales from 500k to 2M parameters (Sec 8), while
SwiGLU improves steadily. It also shows genuine, nontrivial internal
belief-state evolution that measurably (if not sufficiently) matters when
ablated. See Sec "Interpretation" below.

## 1. Main comparison table (Sec 19)

Mean over 3 seeds; each seed's number is from the LR (of `{3e-4, 1e-3}`)
selected by validation NLL alone, evaluated once on the held-out test split.
`1e-3` won for every model/seed (6/6).

| model | params | hidden width | val NLL | test NLL | PPL | bits/token | tokens/sec | ms/step |
|---|---|---|---|---|---|---|---|---|
| swiglu | 999,424 | 315 | 1.8410 | 1.8418 | 6.31 | 2.657 | 257,841 | 63.54 |
| fixed_confidence | 999,634 | 469 | 2.1992 | 2.1992 | 9.02 | 3.173 | 160,528 | 102.06 |
| cellv0.3 | 999,634 | 469 | 2.3266 | 2.3259 | 10.24 | 3.356 | 104,847 | 156.27 |

Per-seed test NLL (all selected `lr=1e-3`):

| model | seed 0 | seed 1 | seed 2 | mean | std |
|---|---|---|---|---|---|
| swiglu | 1.8422 | 1.8362 | 1.8471 | 1.8418 | 0.0045 |
| fixed_confidence | 2.1825 | 2.2088 | 2.2064 | 2.1992 | 0.0119 |
| cellv0.3 | 2.2808 | 2.3435 | 2.3535 | 2.3259 | 0.0322 |

CellV0.3's seed-to-seed spread (std 0.032) is ~3x SwiGLU's (0.0045) and ~3x
`fixed_confidence`'s (0.0119) — the belief mechanism doesn't just perform
worse on average, it's also the least stable of the three across seeds.

## 2. Parameter fairness (Sec 9)

| model | total | embedding | attention | FFN | norms | FFN hidden width |
|---|---|---|---|---|---|---|
| swiglu | 999,424 | 65,536 | 327,680 | 604,800 | 1,408 | 315 |
| cellv0.3 | 999,634 | 65,536 | 327,680 | 605,010 | 1,408 | 469 |
| fixed_confidence | 999,634 | 65,536 | 327,680 | 605,010 | 1,408 | 469 |

All three within the required 990k-1.010M band (actual: 999.4k-999.6k,
well inside +/-1%). Embedding, attention, and norm parameter counts are
*identical* across all three (same backbone, same tied embedding) — the
entire, tiny 210-parameter difference between SwiGLU and the two CellV0.3-
shaped models is a rounding artifact of the two FFN kinds' different
per-hidden-unit cost hitting slightly different closest integer widths.
`cellv0.3` and `fixed_confidence` are *exactly* parameter- and width-matched
(469 hidden units each, identical `V`/`gain_raw`/`bias`/`out_proj` shapes),
which is what makes their comparison in Sec "Interpretation" below
meaningful.

## 3. Learning efficiency (Sec 14)

![validation NLL vs processed tokens](results/figures/learning_curves.png)

Validation NLL (mean over 3 seeds) at each token checkpoint, on the fixed
500k-token validation subset (see `README.md`'s disclosed choice):

| tokens | swiglu | fixed_confidence | cellv0.3 |
|---|---|---|---|
| 1,000,000 | 5.28 | 5.28 | 5.28 |
| 3,000,000 | 3.90 | 4.06 | 4.19 |
| 10,000,000 | 2.25 | 2.70 | 2.83 |
| 20,000,000 | 1.92 | 2.31 | 2.44 |
| 30,000,000 | 1.83 | 2.20 | 2.33 |

The ranking `swiglu < fixed_confidence < cellv0.3` is set by 3M tokens and
never changes — CellV0.3 is not merely behind at convergence, it is behind
at *every* measured point on the curve, i.e. worse sample efficiency, not
just a worse asymptote (Sec 18's "equal or better sample efficiency"
criterion for a strong-positive result clearly fails).

![training loss vs processed tokens](results/figures/training_loss.png)

Training loss (reconstructed from the sweep's own per-step console log,
selected-LR runs only) shows the same ordering and tight within-family seed
clustering, confirming the validation-curve ranking isn't a validation-set
artifact.

## 4. CellV0.3 belief-state diagnostics (Sec 11)

Mean over 3 seeds, one fixed validation sample, per Transformer layer
(0-indexed, 5 layers total):

| training fraction | layer | pi mean | pi std | pi CV | u mean | u std |
|---|---|---|---|---|---|---|
| 0.0 (init) | 0 | 0.503 | 0.018 | 0.036 | 0.989 | 0.071 |
| 0.0 (init) | 4 | 0.503 | 0.018 | 0.036 | 0.989 | 0.071 |
| 0.1 | 0 | 0.508 | 0.019 | 0.037 | 0.972 | 0.072 |
| 0.1 | 4 | 0.513 | 0.016 | 0.031 | 0.950 | 0.060 |
| 0.5 | 0 | 0.524 | 0.022 | 0.042 | 0.910 | 0.080 |
| 0.5 | 4 | 0.546 | 0.019 | 0.034 | 0.834 | 0.063 |
| 1.0 (final) | 0 | 0.530 | 0.023 | 0.042 | 0.889 | 0.080 |
| 1.0 (final) | 4 | 0.551 | 0.020 | 0.036 | 0.817 | 0.065 |

(Full 5-layer table in `results/processed/sweep_summary.json`'s
`diagnostics` field per seed.)

**Key structural finding**: because every `BeliefFFN` call resets the input
belief to neutral (`mu_in = x, e_in = 1, u_in = 0`, task Sec 6), the raw-
moment identity `e_hidden = E/L` reduces to *exactly* 1 for every token,
every layer, every step — verified both analytically
(`belief_ffn.py::_raw_moment_finish`, `pi_j=1` for all input `j` makes
`E = sum_j |V_ij| = L`) and empirically here: `pi_mean` tracks
`1/(1 + u_mean)` to 3 decimal places at every row above (e.g. init:
`1/(1+0.989) = 0.503`, exact match). So in this architecture,
CellV0.3's confidence/precision signal is carried *entirely* by conflict
(`u`), never by evidence (`e`) — the "no externally supplied confidence
signal" framing (Sec 1) forces this, and it is a real, disclosed consequence
worth stating plainly rather than burying in a table.

The belief state is **not inert**: `u_mean` drops substantially during
training (layer 4: `0.989 -> 0.817`, i.e. internal disagreement among a
hidden unit's input dimensions shrinks as `V` is learned) and `pi_mean`
rises correspondingly (`0.503 -> 0.551`). Later layers (layer 4) develop
more pronounced belief structure than earlier layers (layer 0) throughout
training. This is genuine, non-trivial belief-state development, not a
frozen artifact of initialization.

## 5. Critical mechanism test: neutralization (Sec 12)

Mean over 3 seeds, evaluated once on the held-out test split, no retraining:

| intervention | delta test NLL | delta PPL |
|---|---|---|
| precision-neutralized (`sqrt(pi_hidden) -> 1`) | +0.0631 (std 0.0050) | +0.668 |
| conflict-neutralized (`u_hidden -> 0`) | +0.0631 (std 0.0050) | +0.668 |

Both ablations **raise** test NLL — the learned belief state is causally
load-bearing, not decorative. But note the two rows are numerically
identical to 4 decimal places: this is the direct consequence of Sec 4's
finding above. Since `e_hidden` is always exactly 1, conflict-neutralization
(`u -> 0` implies `pi = e/(1+e*0) = e = 1`) and precision-neutralization
(`sqrt(pi) -> 1` directly) force the *exact same* activation-scale value.
**Sec 12's two interventions are not independent tests in this specific
application of CellV0.3** — they collapse to one question ("does
`sqrt(pi_hidden)` matter at all?"), not two ("does evidence matter?" /
"does conflict matter?", since evidence never varies here). The honest
answer to that one question is yes, modestly: removing it costs ~0.063
nats/token (~3.4% relative), consistent with, but far smaller than, the
~0.48-nat gap to SwiGLU.

## 6. Generation evaluation (Sec 13)

10 fixed prompts x 3 seeds x 200 tokens, temperature 0.8, top-p 0.95,
identical sampling code for every model. Mean over all 30 generations per
model (word-level distinct-n, `generation.py`):

| model | distinct-1 | distinct-2 | repeated-bigram rate | repeated-trigram rate |
|---|---|---|---|---|
| swiglu | 0.7026 | 0.9613 | 0.0387 | 0.0063 |
| fixed_confidence | 0.7100 | 0.9627 | 0.0373 | 0.0060 |
| cellv0.3 | 0.7163 | 0.9627 | 0.0373 | 0.0079 |

Diversity metrics are essentially indistinguishable across models (all
within ~1.5 points of each other) — at this perplexity level (6-10) none of
the three produces qualitatively different *kinds* of text, just text of
different quality. One representative sample per model (same prompt, seed
0; full set in `results/raw/generations/`):

> **Prompt**: "Once upon a time, there was a little girl named Lily."
>
> **swiglu**: "She loved to play outside in the high in the garden. One
> day, she saw a big hot wide and then said, 'I will clean up to your
> cave.' But one of the grass. She felt sad, and so she put the direction
> in her hands. [...]"
>
> **fixed_confidence**: "She loved to play outside together. One day, Lily
> and her mommy felt better. She was excited to make anymore. Suddenly, her
> dad catched. She would pick up the sounds, and round it. [...]"
>
> **cellv0.3**: "She loved to play together. She had a big boy street. She
> was also clearing and hopless. Tom and Sam was waiting, but they could
> stay. They had a table and the sun. [...]"

All three are recognizably TinyStories-shaped but grammatically broken at
this scale/budget — not a basis for a qualitative ranking, hence the
diversity metrics (not cherry-picked prose) are the reported result.

## 7. MPS speed preflight (Sec 15)

50 warmup + 200 measured steps, batch 64, context 256, this machine:

| model | ms/step | tokens/sec | vs swiglu |
|---|---|---|---|
| swiglu | 63.54 | 257,841 | 1.00x |
| fixed_confidence | 102.06 | 160,528 | 1.61x |
| cellv0.3 | 156.27 | 104,847 | 2.46x |

Under Sec 15's 3x flag threshold, so no mandatory implementation-only
optimization pass was triggered (and none was applied). `BeliefFFN`-alone
kernel benchmark (matched hidden width 469, batch 64, context 256):
21.29 ms/fwd+bwd vs SwiGLU-at-the-same-width's 3.65 ms/fwd+bwd (5.84x) —
the isolated FFN-kernel ratio is larger than the whole-model ratio because
attention and the rest of the block cost is shared and identical, diluting
the FFN-only gap once folded into a full model step. Full JSON:
`results/processed/mps_speed_preflight.json`.

## 8. Optional scaling experiment (Sec 17)

ModernTransformer vs CellV03Transformer at ~500k and ~2M parameters, seed 0
only, reusing each model's selected `lr=1e-3` from the 1M sweep (no
independent re-tuning). Test NLL:

| params | swiglu | cellv0.3 | gap (cellv0.3 - swiglu) |
|---|---|---|---|
| 500,224 / 500,404 | 2.1087 | 2.3236 | +0.2149 |
| 999,424 / 999,634 | 1.8418 | 2.3259 | +0.4841 |
| 1,999,744 / 1,999,384 | 1.7318 | 2.3440 | +0.6122 |

**The gap widens monotonically and substantially with scale — this is the
clearest single result in this experiment.** SwiGLU improves steadily and
predictably with capacity, exactly as expected (2.11 -> 1.84 -> 1.73 nats
across a 4x parameter range). CellV0.3 does not: its test NLL is
essentially flat, even slightly *worse* at 2M than at 500k (2.3236 -> 2.3259
-> 2.3440) despite its FFN hidden width growing from 82 to 469 to 1,244
units over the same range. Adding capacity to the CellV0.3 hidden layer does
not translate into better language modeling here, unlike an ordinary FFN.
This directly answers Sec 17's question ("does any advantage/disadvantage
widen or shrink with capacity") in the least ambiguous way it could have
come out: the disadvantage widens, monotonically, at both ends of the range
tested.

## 9. Interpretation (Sec 18)

Sec 18's four buckets, checked against this run's actual numbers (not
asserted): **matches/beats test NLL** — no (cellv0.3 is 26% higher
perplexity than swiglu, not "matching" by any reasonable margin).
**Equal/better sample efficiency** — no (behind at every learning-curve
checkpoint, Sec 3 above). **Nontrivial internal pi/u variation** — yes,
clearly (Sec 4: `u_mean` moves from 0.99 to ~0.82-0.89 over training,
consistently across layers and seeds). **Neutralization measurably hurts**
— yes, modestly (+0.063 nats/token, though the two prescribed ablations
turn out to be the same test here, Sec 5). **Reasonable compute overhead**
— borderline: 2.46x is under Sec 15's 3x flag, but is a real, substantial
cost for a model that performs worse, not better.

`summarize.py::classify_interpretation`'s mechanical rule (which treats
"under 3x" as sufficient for "reasonable compute") lands this in
**mechanism-positive-but-performance-neutral**. Read against Sec 18's own
prose definitions rather than that one threshold, the more honest label is
**Negative, with a disclosed mechanism-positive footnote**: "ModernTransformer
clearly wins in perplexity" is unambiguously true here (not a close call,
not "slightly loses"), and CellV0.3 "adds substantial compute" is also true
in absolute terms (more than double the per-step cost). The Sec 17 scaling
result (Sec 8 above) removes any remaining ambiguity: the disadvantage
*widens* with capacity rather than shrinking, which is the opposite of what
"performance-neutral, might close with more capacity" would predict. The
footnote worth recording rather than discarding: the belief mechanism is not
inert dead weight — it develops real internal structure during training and
that structure is causally used at inference — it simply isn't enough to
close, or even meaningfully narrow, the gap to a conventional FFN at any
scale tested. `fixed_confidence` (same parameter count, same hidden width,
same `V`/`gain`/`bias`/`out_proj`, *no* belief computation) beats full
`cellv0.3` by 0.127 nats/token at 1M, directly answering Sec 7's question:
at least part of CellV0.3's shortfall here is attributable to the belief
computation itself, not merely the shared normalized-weight/tanh
parameterization (which *also* underperforms SwiGLU on its own, by 0.358
nats/token).

**Per Sec 18's stopping rule for a negative result: stop.** No CellV0.4, no
retuning CellV0.3 toward language modeling based on this run. This result
is scoped to one architecture line (dense CellV0.3 as a per-token FFN
primitive with a neutral, per-call-reset input belief), one dataset
(TinyStories), and one scale (~1M parameters, 30M training tokens) — it
says nothing about `BeliefDendriteNetwork` (Architecture V2, a materially
different, already separately and positively evaluated mechanism, see
`docs/architecture_v2.md`) or about CellV0.3 in non-language-modeling
settings (Paper A).

## 10. Addendum: LLM-judge qualitative scoring (post-hoc, not in the original Sec 1-19 protocol)

Distinct-n (Sec 13) measures lexical variety, not quality — it can't tell
you whether a continuation is grammatical or makes narrative sense. To get
an actual quality signal beyond loss, the same 30-generation-per-model set
used for Sec 13 was graded on the TinyStories paper's own axes (grammar,
creativity, consistency, plot; 1-10 each) by an LLM judge, blinded to model
identity. **Judge was Claude, not GPT-4** — a deliberate, disclosed
deviation from the original paper's methodology, run interactively in one
session rather than as a rerunnable API script (see
`results/raw/llm_judge/rubric.md` for the exact prompt and blinding
procedure). Mean over 30 generations, std in parens:

| model | grammar | creativity | consistency | plot | overall |
|---|---|---|---|---|---|
| swiglu | 3.87 (1.43) | 3.40 (0.88) | 1.80 (0.70) | 2.17 (0.90) | **2.81** (0.80) |
| fixed_confidence | 1.97 (0.87) | 2.13 (0.50) | 1.23 (0.56) | 1.13 (0.34) | 1.62 (0.46) |
| cellv0.3 | 1.87 (0.62) | 1.97 (0.48) | 1.23 (0.42) | 1.20 (0.40) | 1.57 (0.39) |

Welch's t on the overall (4-axis mean) score, n=30 each: swiglu vs
fixed_confidence t=6.95, swiglu vs cellv0.3 t=7.54 (both decisive) —
**fixed_confidence vs cellv0.3 t=0.45** (indistinguishable from noise).

Two things this adds beyond Sec 13 and the main loss table:

1. **All three models are qualitatively poor in absolute terms** — even
   swiglu's 2.81/10 overall reflects real, frequent breakdowns (abrupt
   unrelated-story restarts mid-passage, character/name swaps, invented
   non-words). At 1M params / 30M tokens, none of these three produce
   reliably grammatical, consistent short stories — a useful absolute
   anchor that a relative "cellv0.3 loses" framing alone doesn't convey.
2. **The judge confirms the loss ranking's top split but not its bottom
   split.** swiglu is judged decisively better than both alternatives on
   every axis, consistent with Sec 1's test-NLL gap. But fixed_confidence's
   0.127-nat/token loss advantage over cellv0.3 (Sec 9) does **not** show up
   as a perceptible quality difference here (t=0.45) — to a blinded reader,
   the two are equally broken. The belief mechanism's cost is real in loss
   terms but not in a way a human/LLM reader would notice at this
   checkpoint; consistent with Sec 5's finding that the mechanism's causal
   effect (+0.063 nats from neutralization) is small relative to the
   ~0.48-nat gap to swiglu.

Full per-item scores and one-line justifications:
`results/raw/llm_judge/graded_items.json`. Rubric and blinding procedure:
`results/raw/llm_judge/rubric.md`. Aggregate stats:
`results/processed/llm_judge_summary.json`.

## Reproducing this

See `README.md`. Raw run records: `results/raw/run_records/*.json`. Raw
generations: `results/raw/generations/*.json`. Checkpoints:
`results/raw/checkpoints/*.pt`. Full console log:
`results/raw/run_sweep_console.log`. Aggregated JSON:
`results/processed/sweep_summary.json`,
`results/processed/mps_speed_preflight.json`,
`results/processed/scaling_summary.json`,
`results/processed/llm_judge_summary.json`. LLM-judge raw data:
`results/raw/llm_judge/`.
