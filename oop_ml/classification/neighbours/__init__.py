"""Predicting a class from whoever is nearby.

No boundary is fitted. The decision surface is whatever falls out of where the
training rows happen to sit, which is why it can be any shape at all and why it
follows noise as happily as signal when k is small.

The same model serves two classes or twenty, because a vote does not care how
many candidates are on the ballot -- which is the one place this family is
simpler than logistic regression rather than stranger.
"""
