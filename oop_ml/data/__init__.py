"""The vocabulary every package in the library speaks.

A :class:`~oop_ml.data.column.Column` is a validated float vector and the one
place raw input is coerced. A :class:`~oop_ml.data.feature.Feature` is a column
that knows its own name, which is the whole premise of the library's API. A
:class:`~oop_ml.data.feature_set.FeatureSet` is a validated group of them, and
:class:`~oop_ml.data.coefficients.Coefficients` is the same idea pointed the
other way: names bound to what a fit learned rather than to what went in.

These live together because they are the types that cross every boundary.
Value objects used by exactly one package -- the scalings a Standardizer
learns, the terms a PolynomialFeatures builds, the Dataset the splitters carve
up -- stay beside the thing that uses them.
"""
