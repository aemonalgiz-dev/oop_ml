"""Transformers that reshape the inputs before a model sees them.

A transformer learns from the inputs alone and never sees a target, which is why
everything here derives from ``Transformer`` rather than from ``Estimator``.
"""
