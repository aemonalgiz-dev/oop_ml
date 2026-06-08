"""Fits that do not let one bad row decide the answer.

Least squares squares the residuals, so a row twice as far off pulls four
times as hard and a row ten times as far off pulls a hundred times as hard.
That is the right weighting when the errors really are Gaussian and a
catastrophe when a handful of them are not, since a single mistyped
measurement can move the whole line. Everything here changes the shape of the
penalty on a residual so that being far off stops making a row more
influential past some point.
"""
