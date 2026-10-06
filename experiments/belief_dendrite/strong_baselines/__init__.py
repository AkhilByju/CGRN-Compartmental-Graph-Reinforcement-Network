"""Architecture V2 -- Strong CNN + Transformer Baseline Benchmark (CIFAR-10).

A new, separate benchmark from the frozen MNIST/Fashion-MNIST comparison in
``experiments/belief_dendrite/`` (that one is recorded and closed -- see its
own README). This package answers a different, harder question: does
``BeliefDendriteNetwork`` (frozen, reused unmodified from
``src/models/architecture_v2/belief_dendrite.py``) provide robustness to
localized missing information beyond what conventional spatial architectures
(a small CNN, a small Vision Transformer) already provide, including when
those conventional architectures are given the exact same reliability
signal.

Nothing in ``src/models/architecture_v2/`` is modified by this package. The
three dendritic model families here (``ScalarDendriteNetwork``,
``BeliefDendriteNetwork``) are the frozen classes, only re-sized and re-wired
for RGB 32x32 input.
"""
