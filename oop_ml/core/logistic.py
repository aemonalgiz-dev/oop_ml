"""The logistic and softmax functions on arrays, written once.

Both are numerical primitives, and the naive spelling of each is a trap.
``1 / (1 + exp(-z))`` overflows once ``z`` drops below about -709, and
``exp(z) / sum(exp(z))`` overflows to ``inf / inf`` and answers ``nan`` once any
score passes about 709. Neither failure is a wrong formula; both are the right
formula written in the order that loses.

These live in ``core`` because two unrelated parts of the library now want
them. :mod:`oop_ml.classification.logistic` wraps them into ``Probabilities``
and ``ProbabilityMatrix``, the bounded types a classifier's output deserves,
and :mod:`oop_ml.core.network.activation` wants the same arithmetic with no
such claim attached, since a hidden neuron's sigmoid output is a coordinate in
a learned space and not a chance of anything.

That distinction is the whole reason the split exists. The stable arithmetic is
shared; what the numbers *mean* afterwards is not, and a function returning
``Probabilities`` cannot serve a caller for whom the values are not
probabilities. Core holds the arithmetic, and each caller wraps it in the type
that tells the truth about its own numbers.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.types import FloatArray


def stable_logistic(values: FloatArray) -> FloatArray:
    """Map log-odds onto ``[0, 1]`` without overflowing.

    Branch on the sign so the exponential is only ever handed a non-positive
    argument. ``exp(-|z|)`` is the only exponential either half needs: the
    denominator is ``1 + exp(-|z|)`` outright, and the numerator, ``exp(min(z,
    0))``, is 1 where ``z`` is non-negative and ``exp(z)`` otherwise -- which
    for negative ``z`` is ``exp(-|z|)`` again.

    Parameters
    ----------
    values:
        Any shape. Scores, log-odds, or pre-activation sums.

    Returns
    -------
    FloatArray
        The same shape, every entry in ``[0, 1]``.
    """
    decay = np.exp(-np.abs(values))

    return np.where(values >= 0.0, 1.0, decay) / (1.0 + decay)


def stable_softmax(scores: FloatArray) -> FloatArray:
    """Normalise each row of ``scores`` into a distribution over the columns.

    The multi-class generalisation of :func:`stable_logistic`, and the same trap
    wearing different clothes. Written literally, ``exp(z) / sum(exp(z))``
    overflows to ``inf / inf`` and hands back ``nan`` once any score passes
    about 709: at ``z = 800`` the naive form gives ``[nan, 0, 0]`` for what is
    plainly ``[1, 0, 0]``.

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
    FloatArray
        The same shape, every row non-negative and summing to 1.
    """
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)

    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)
