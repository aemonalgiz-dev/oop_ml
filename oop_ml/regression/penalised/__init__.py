"""Squared error plus a price on the size of the coefficients.

The penalty is the entire difference from :mod:`~oop_ml.regression.least_squares`,
and its *shape* is the entire difference between these two. Ridge squares the
coefficients, which shrinks them smoothly and never quite to zero. Lasso takes
their absolute value, whose corner at zero is what lets a coefficient land
exactly there and select the feature out of the model altogether.

That corner is also why one of these has a closed form and the other does not.
"""
