"""The logistic function as a classifier's link, carrying the bound it earns.

The arithmetic moved to :mod:`oop_ml.core.logistic` once a second caller
appeared for it, and what stays here is the part that is genuinely about
classification: the promise that these particular numbers are chances.

Keeping the wrapper rather than pointing every model at the core function is a
judgement about what the return type is worth. ``Probabilities`` and
``ProbabilityMatrix`` carry the bound and the row sum that the formulas
guarantee, so nothing downstream re-establishes either, and a hidden layer's
sigmoid output -- which is a coordinate and not a chance -- cannot be passed
where a classifier's output is expected. The *link* still belongs to the model,
since a probit would reach for the normal CDF and mean something different by
it; only the numerics are shared.
"""

from __future__ import annotations

from oop_ml.core.data.probabilities import Probabilities, ProbabilityMatrix
from oop_ml.core.logistic import stable_logistic, stable_softmax
from oop_ml.core.types import FloatArray


def sigmoid(linear_predictor: FloatArray) -> Probabilities:
    """Map log-odds onto probabilities in ``[0, 1]``, without overflowing.

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
    return Probabilities(stable_logistic(linear_predictor))


def softmax(scores: FloatArray) -> ProbabilityMatrix:
    """Normalise each row of ``scores`` into a distribution over the columns.

    Parameters
    ----------
    scores:
        ``(n_samples, n_classes)``, one score per class per row.

    Returns
    -------
    ProbabilityMatrix
        The same shape, every row non-negative and summing to 1. The row sum
        is the whole point of the shift underneath, so the type is where it
        should be recorded.
    """
    return ProbabilityMatrix(stable_softmax(scores))
