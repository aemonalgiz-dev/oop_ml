"""Squared error and nothing else added to it.

Three models minimising the same objective by three different routes: the
closed form for one predictor, the closed form for several, and the walk
downhill that arrives at the same place without solving anything. What they
share is the objective; what differs is only how the coefficients come out of
it, which is the distinction the whole package is built around.
"""
