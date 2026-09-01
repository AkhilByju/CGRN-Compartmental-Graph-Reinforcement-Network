# Experiment 012 — MCMC / Energy-Based Inference

**Status:** Blocked — only proceed if Experiment 011 provides strong
justification (per `docs/architecture_v0.md` Sec 8 and
`docs/benchmark_plan.md` "Monte Carlo / multiple hypotheses"). Do not begin
this experiment as a default next step.

**Purpose (conditional):** Define an energy function `E_theta(Z, X)` where
low-energy states correspond to internal representations more compatible
with observed evidence; explore latent states stochastically rather than
deterministically. Known risks: difficult optimization, poor mixing,
rapidly growing compute, training instability, and much harder analysis —
all reasons this sits last in the sequence.

**Hypotheses tested:** exploratory — not one of H1–H6.
