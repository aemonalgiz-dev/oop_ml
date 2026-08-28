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

from oop_ml.core.data.probabilities import Probabilities, ProbabilityMatrix
from oop_ml.core.types import FloatArray


def sigmoid(linear_predictor: FloatArray) -> Probabilities:
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
    Probabilities
        One chance per row, each in ``[0, 1]``. The type carries the bound the
        formula guarantees, so nothing downstream has to re-establish it.
    """
    decay = np.exp(-np.abs(linear_predictor))

    return Probabilities(np.where(linear_predictor >= 0.0, 1.0, decay) / (1.0 + decay))


def softmax(scores: FloatArray) -> ProbabilityMatrix:
    """Normalise each row of ``scores`` into a distribution over the columns.

    The multi-class generalisation of :func:`sigmoid`, and the same trap wearing
    different clothes. Written literally, ``exp(z) / sum(exp(z))`` overflows to
    ``inf / inf`` and hands back ``nan`` once any score passes about 709: at
    ``z = 800`` the naive form gives ``[nan, 0, 0]`` for what is plainly
    ``[1, 0, 0]``.

    Subtracting the row maximum first fixes it exactly rather than
    approximately. The constant cancels between numerator and denominator, for
    the same reason that adding a constant to every class's weights leaves the
    probabilities untouched, and it guarantees the largest exponent is
    ``exp(0)``.

    Parameters
    ----------
    scores:
        ``(n_samples, n_classes)``, one score per class per row.

    Returns
    -------
    ProbabilityMatrix
        The same shape, every row non-negative and summing to 1. The row sum
        is the whole point of the shift above, so the type is where it should
        be recorded.
    """
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)

    return ProbabilityMatrix(
        exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
    )
