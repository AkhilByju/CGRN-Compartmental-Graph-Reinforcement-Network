"""Language-modeling evaluation (Level 6, docs/benchmark_plan.md "Language
modeling stays a core goal"): loss/perplexity, and eventually grammatical
and semantic competence tasks for TinyStories / BabyLM-style evaluation.

Not yet implemented: begins with Experiment 010
(docs/experiment_protocol.md), once a tokenizer/data pipeline and a decoder
(docs/architecture_v0.md Sec 5) exist for both the baselines and
ArchitectureV0.
"""

from __future__ import annotations
