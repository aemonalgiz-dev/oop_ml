"""Powers and products of the features, as ordinary columns.

The insight this package exists to make concrete is that fitting a curve needs
no new model. "Linear" describes the coefficients, not the shape of the
prediction, so once x**2 is a column of the design matrix the estimator has
nothing new to learn.

features holds the transformer; terms holds a single term -- which knows the
name of the column it produces and how to compute it -- and the ordered group
of them.
"""
