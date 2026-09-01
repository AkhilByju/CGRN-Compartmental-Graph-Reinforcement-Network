# Experiment 010 — Language

**Status:** Blocked — depends on Experiment 005 at minimum, and ideally
009's world-representation results to interpret language-model behavior.

**Purpose:** Start with TinyStories or an equivalent highly constrained
small corpus (verify real language acquisition, test generation, compare
small models), then BabyLM-style restricted-data evaluation (measure
language acquisition under limited data vs. parameter-matched Transformer
baselines; evaluate grammar, semantics, conceptual knowledge,
generalization). Ordinary causal language modeling must remain in the
comparison — see `docs/benchmark_plan.md` "Language modeling stays a core
goal."

**Prerequisite:** `src/data/language/`, `src/evaluation/language.py`,
and `src/models/architecture_v0/{encoder,decoder}.py` (all not yet
implemented).

**Hypotheses tested:** H6 (Language Hypothesis).
