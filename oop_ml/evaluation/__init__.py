"""Predictions paired with the truth, and the metrics that follow from them.

Three classes, siblings rather than a hierarchy, because they share the idea of
aligning a prediction with what happened and nothing else. R^2 says nothing
about a label; a confusion matrix says nothing about a quantity; and the binary
matrix hands back one precision where the multi-class one has to hand back a
vector. No caller wanting any of the three would accept another.

Each aligns its pair once at construction, counts once, and exposes every
metric as a property read off that. A metric whose denominator is empty raises
rather than returning nan, because "this model never fired" and "this model
scored zero" are different claims.
"""
