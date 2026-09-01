# Related Work Tracker

Living literature tracker. Add a row when a paper is read (even partially);
don't wait for a "complete" summary. Link papers to the hypothesis or
architecture-design section they bear on so the connection isn't lost later.

## How to add an entry

```markdown
### <Author(s), Year> — <Short Title>

- **Link:** <url or arXiv id>
- **Relevant to:** <e.g. H1, H3, architecture_v0.md §3 (graph/cluster organization)>
- **Summary:** 1-3 sentences.
- **Relation to CGRN:** what it confirms, contradicts, or suggests we should
  try/avoid.
```

## Candidate areas to track (fill in as read — none of these are read yet)

- **Transformers / attention** — the baseline this project positions itself
  against (not "beat," but "characterize differences from").
- **State-space models / Mamba** — alternative to dense global attention;
  relevant to `architecture_v0.md` §3 (selective vs. dense communication) and
  Track C baselines.
- **Graph neural networks / message passing** — directly relevant to the
  graph/cluster organization design space and to known GNN failure modes
  (oversmoothing, over-squashing) that `hypotheses.md` explicitly asks this
  project to check for.
- **Recurrent/iterative reasoning models** (e.g. universal transformers,
  recurrent depth, "thinking" at inference time) — directly relevant to H2
  and Experiment 004/008.
- **Dendritic / compartmental computation in ANNs** — directly relevant to
  `architecture_v0.md` §1 (CellV0) and H1.
- **World models** — relevant to H4 and Experiment 009.
- **Diffusion models / iterative denoising** — relevant to
  `architecture_v0.md` §6.
- **Energy-based models / MCMC inference** — relevant to Experiment 012 (only
  if gated open by Experiment 011).
- **BabyLM / TinyStories and other restricted-data language evaluation** —
  relevant to Experiment 010 and H6.
- **Compositional / systematic generalization benchmarks** — relevant to
  Experiment 008 (Level 4 of the benchmark ladder).

## Entries

_(none yet — this section fills in as literature is read)_
