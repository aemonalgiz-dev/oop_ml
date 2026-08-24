"""The logistic function, written once because it is easy to write badly.

Every model that maps a log-odds onto a probability wants this, and the naive
spelling of it is a trap. ``1 / (1 + exp(-z))`` overflows once ``z`` drops below
about -709: the value that comes back is still correct, since ``1 / inf`` is 0,
so what it costs is not a wrong answer but an accumulating stream of
``RuntimeWarning`` from exactly the rows the model is most certain about.

Keeping it here rather than on a model is a judgement about what it is. The
*link* belongs to the model, since a probit would reach for the normal CDF and
mean something different by it. This particular function does not: it is a
numerical primitive, and two logistic models spelling it separately is two
chances to spell it wrong.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.types import FloatArray


def sigmoid(linear_predictor: FloatArray) -> FloatArray:
    """Map log-odds onto probabilities in ``[0, 1]``, without overflowing.

    Branch on the sign so the exponential is only ever handed a non-positive
    argument. ``exp(-|z|)`` is the only exponential either half needs: the
    denominator is ``1 + exp(-|z|)`` outright, and the numerator, ``exp(min(z,
    0))``, is 1 where ``z`` is non-negative and ``exp(z)`` otherwise -- which
    for negative ``z`` is ``exp(-|z|)`` again.

    Parameters
    ----------
    linear_predictor:
        ``z = X b``, one entry per observation.

    Returns
    -------
    FloatArray
        Probabilities in ``[0, 1]``, the same shape as the input.
    """
    decay = np.exp(-np.abs(linear_predictor))
    return np.where(linear_predictor >= 0.0, 1.0, decay) / (1.0 + decay)
