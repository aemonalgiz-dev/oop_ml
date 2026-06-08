"""The log density of a Gaussian, at the array level.

Here for the reason :mod:`oop_ml.core.logistic` is here: two unrelated callers
want this arithmetic and neither may import the other. Both backends' naive
Bayes reads the same three summaries, a prior, a mean and a variance per class
per column, and turns them into the same score, so writing that sum twice would
be two chances to write it differently. The same holds for the discriminant
below, whose summaries are a prior and a mean per class over one shared
covariance.

Two models, one family
-----------------------
Naive Bayes gives every class its own spread and assumes the columns do not
vary together; the discriminant lets them vary together and makes every class
share one matrix saying how. The first buys a score that is quadratic in the
row, the second a score that is linear in it, and the arithmetic here is where
that difference actually appears.

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
from numpy.linalg import LinAlgError

from oop_ml.core.exceptions import CollinearFeaturesError
from oop_ml.core.solving.positive_definite import solve_positive_definite
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


def discriminant_weights(
    means: FloatArray, pooled_covariance: FloatArray
) -> FloatArray:
    """``means @ inverse(pooled_covariance)``, one weight vector per class.

    The whole of what makes linear discriminant analysis linear. Both classes'
    scores carry the same ``-0.5 * row' S^-1 row``, since they share ``S``, so
    that term is identical in every class's score and cancels out of every
    comparison. What is left depends on the row only through ``row' S^-1 mu_k``,
    which is a plane.

    Solved rather than inverted, and by Cholesky rather than by a general
    solve, because a pooled within-class covariance is symmetric positive
    *semi*-definite by construction and the factorisation fails exactly when
    the "semi" is load-bearing. That failure is the guard: a zero direction
    means some column is a combination of the others within every class, so
    infinitely many discriminants separate the classes equally well and none
    of them is the answer.

    Parameters
    ----------
    means:
        ``(n_classes, n_features)``, each class's average per column.
    pooled_covariance:
        ``(n_features, n_features)``, the within-class covariance every class
        shares.

    Returns
    -------
    FloatArray
        ``(n_classes, n_features)``.

    Raises
    ------
    CollinearFeaturesError
        If the pooled covariance is singular, which is to say if some column
        carries no information the others do not already carry.
    """
    try:
        solved = solve_positive_definite(pooled_covariance, means.T)
    except LinAlgError:
        raise CollinearFeaturesError(
            "the pooled within-class covariance is singular, so no unique "
            "discriminant exists. Some column is a linear combination of the "
            "others within the classes, or there are fewer rows than columns; "
            "drop the redundant column, or reduce the columns first with "
            "PrincipalComponentAnalysis"
        ) from None

    return np.asarray(solved).T


def linear_discriminant_scores(
    rows: FloatArray,
    means: FloatArray,
    pooled_covariance: FloatArray,
    class_priors: FloatArray,
) -> FloatArray:
    """Each class's discriminant, one row per query.

    ``row . (S^-1 mu_k) - 0.5 mu_k' S^-1 mu_k + log(prior_k)``, which is the
    log of the class's prior times its density with every term that does not
    depend on ``k`` dropped. Those terms are the same for every class here
    because the covariance is shared, so dropping them shifts a row's whole
    set of scores by one constant and changes neither the ranking nor the
    probabilities that come out of :func:`normalised_from_log_scores`.

    Returns
    -------
    FloatArray
        ``(n_rows, n_classes)``, unnormalised and not a log density.

    Raises
    ------
    CollinearFeaturesError
        If the pooled covariance is singular.
    """
    weights = discriminant_weights(means, pooled_covariance)
    offsets = -0.5 * np.einsum("kj,kj->k", weights, means) + np.log(class_priors)

    return rows @ weights.T + offsets[None, :]
