"""Squared error plus a price on the size of the coefficients.

The penalty is the entire difference from :mod:`~oop_ml.numpy.regression.least_squares`,
and its *shape* is the entire difference between these. Ridge squares the
coefficients, which shrinks them smoothly and never quite to zero. Lasso takes
their absolute value, whose corner at zero is what lets a coefficient land
exactly there and select the feature out of the model altogether. The elastic
net charges both at once, in a proportion the caller names, because each of the
first two has a failure the other does not.

That corner is also why one of these has a closed form and the others do not.
The two that do not share their search, in
:mod:`~oop_ml.numpy.regression.penalised.coordinate_descent`, and differ only
in what one coefficient's own optimum is.
"""
