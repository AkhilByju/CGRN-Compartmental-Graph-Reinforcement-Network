"""World-representation tests (Level 5, docs/benchmark_plan.md
"World-representation tests"): paraphrase invariance, intervention
sensitivity, counterfactual reasoning, relational composition, causal
reasoning, hidden-state inference, and consistency.

Not yet implemented: these all require a defined internal-state
representation to compare (e.g. similarity between `Z_T` for two
paraphrases), which depends on ArchitectureV0's latent state
(docs/architecture_v0.md Sec 5) and the synthetic-world generator
(Experiment 009). Implement alongside that experiment. See
docs/hypotheses.md "Operationalizing understanding" for the definition
these tests must satisfy -- do not implement a proxy that doesn't match it.
"""

from __future__ import annotations
