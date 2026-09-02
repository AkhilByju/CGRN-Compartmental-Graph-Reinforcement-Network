# Experiment 012 — Multiple Latent Trajectories

**Status:** Blocked — depends on Experiment 010 (a single deterministic
latent trajectory must be well understood first).

**Purpose:** Introduce stochastic particles: initialize `Z_0^1, ...,
Z_0^K` (possibly with small stochastic differences), let each trajectory
refine independently/semi-independently, then score/aggregate. Sweep `K =
1, 2, 4, 8`. Test whether maintaining multiple internal hypotheses improves
ambiguous reasoning or robustness, and whether the improvement is worth the
extra inference compute. See `docs/benchmark_plan.md` "Monte Carlo /
multiple hypotheses."

**Gate for Experiment 013:** if this experiment shows no benefit, do not
proceed to Experiment 013 (per `docs/architecture_v0.md` Sec 8).

**Hypotheses tested:** exploratory — not one of H1–H6; motivates whether
energy-based/MCMC inference is worth pursuing at all.
