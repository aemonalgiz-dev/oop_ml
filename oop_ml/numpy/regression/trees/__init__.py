"""Predicting a quantity by which box a row falls in.

The second regressor here that assumes no shape, and the opposite of the first
in what it costs. A neighbour model defers everything to prediction time and
carries the training set forever; a tree pays an exhaustive search once and then
answers from a dozen comparisons, holding nothing but the questions.

The surface it fits is piecewise constant either way, so both flatten out beyond
the edge of the data instead of extending a trend.
"""
