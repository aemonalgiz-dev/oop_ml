"""One boundary, two classes, and two ways of finding it.

Both models here maximise the same concave log-likelihood and land on the same
coefficients. What differs is only what each knows on the way: gradient ascent
knows which way is uphill and has to be told how far to step, Newton reads the
curvature as well and works the distance out instead. That is hundreds of
epochs against single-digit iterations for an identical answer.
"""
