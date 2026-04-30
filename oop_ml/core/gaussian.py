"""The log density of a Gaussian, at the array level.

Here for the reason :mod:`oop_ml.core.logistic` is here: two unrelated callers
want this arithmetic and neither may import the other. Both backends' naive
Bayes reads the same three summaries, a prior, a mean and a variance per class
per column, and turns them into the same score, so writing that sum twice would
be two chances to write it differently.

Why logs rather than densities
-------------------------------
A row's likelihood under a class is a product of one density per column, and a
density is often well below one, so twenty columns multiply to something near
the smallest number a float can hold and fifty underflow to exactly zero.
Adding logarithms is the same comparison with none of the underflow, which is
why nothing here multiplies densities.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.types import FloatArray

LOG_TWO_PI = float(np.log(2.0 * np.pi))
"""The constant term in a Gaussian's log density, computed once."""


def gaussian_log_scores(
    rows: FloatArray,
    means: FloatArray,
    variances: FloatArray,
    class_priors: FloatArray,
) -> FloatArray:
    """The log of each class's prior times its density, one row per query.

    The quantity a naive Bayes decision is actually made on. Unnormalised, so
    it is negative and its rows do not sum to anything in particular; turning
    it into a probability adds nothing to the ranking.

    Parameters
    ----------
    rows:
        ``(n_rows, n_features)``, the queries.
    means:
        ``(n_classes, n_features)``, each class's average per column.
    variances:
        ``(n_classes, n_features)``, each class's spread per column, already
        floored so that none of them is zero.
    class_priors:
        ``(n_classes,)``, how common each class was, summing to one.

    Returns
    -------
    FloatArray
        ``(n_rows, n_classes)``.
    """
    gaps = rows[:, None, :] - means[None, :, :]
    exponent = (gaps * gaps) / variances[None, :, :]
    normaliser = np.log(variances[None, :, :]) + LOG_TWO_PI

    densities = -0.5 * (exponent + normaliser).sum(axis=2)
    return densities + np.log(class_priors)[None, :]


def normalised_from_log_scores(scores: FloatArray) -> FloatArray:
    """Log scores turned into rows that sum to one.

    Each row's largest score is subtracted before exponentiating, which is the
    same answer and cannot overflow, the same guard
    :func:`~oop_ml.core.logistic.stable_softmax` applies for the same reason.
    """
    shifted = np.exp(scores - scores.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)
