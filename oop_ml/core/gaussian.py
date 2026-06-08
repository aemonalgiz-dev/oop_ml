"""The log density of a Gaussian, at the array level.

Here for the reason :mod:`oop_ml.core.logistic` is here: two unrelated callers
want this arithmetic and neither may import the other. Both backends' naive
Bayes reads the same three summaries, a prior, a mean and a variance per class
per column, and turns them into the same score, so writing that sum twice would
be two chances to write it differently. The same holds for the discriminant
below, whose summaries are a prior and a mean per class over one shared
covariance.

Three models, one family, and one dial
----------------------------------------
All three ask which class most plausibly produced a row, and they differ only
in how much they will spend describing a class's shape:

- naive Bayes gives every class its own spread and assumes the columns do not
  vary together, which is one number per column per class;
- linear discriminant analysis lets them vary together but makes every class
  share one matrix saying how, which is one matrix in total;
- quadratic discriminant analysis gives every class a matrix of its own.

Only the middle one produces a score that is linear in the row, and only
because the shared matrix makes the quadratic term identical in every class's
score. The other two keep it, and the arithmetic below is where that
difference actually appears.

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

from oop_ml.core.exceptions import CollinearFeaturesError
from oop_ml.core.solving.positive_definite import solve_positive_definite
from oop_ml.core.types import FloatArray

LOG_TWO_PI = float(np.log(2.0 * np.pi))
"""The constant term in a Gaussian's log density, computed once."""

MINIMUM_CLASS_ROWS = 2
"""How many rows a class needs before it has a covariance at all.

One row deviates from its own mean by exactly zero, in every direction, so
a class of one describes no shape whatever the columns are. Two is the
floor, not the requirement: a class also needs more rows than columns
before its covariance can be inverted, and that refusal is the singular
one raised by :func:`quadratic_discriminant_scores`.
"""


COVARIANCE_RANK_THRESHOLD = float(np.finfo(np.float64).eps)
"""How small a covariance's least direction may be before it counts as none.

Relative to its largest, so the test says nothing about the units the columns
were measured in. Machine epsilon is the tightest threshold float64 admits.
"""


def is_invertible(covariance: FloatArray) -> bool:
    """Whether this covariance describes a spread in every direction.

    A Cholesky factorisation is *not* a sufficient test, which is the reason
    this exists. LAPACK refuses a matrix with a negative pivot, and an exactly
    singular one has a zero pivot rather than a negative one, so it can pass:
    measured, ``[[8, 4], [4, 2]]`` has determinant exactly 0 and eigenvalues
    ``[0, 10]``, and ``cho_factor`` accepts it. What follows is not a failure
    but an answer: numbers around 1e15 that do not solve the system, and inside
    a quadratic discriminant fit those came back as ``inf`` scores from a model
    reporting itself fitted. So the rank is tested here, rather than inferred
    from a factorisation that never promised to test it.

    Symmetric by construction at every call site, so the eigenvalues are real
    and ``eigvalsh`` returns them in ascending order.
    """
    eigenvalues = np.linalg.eigvalsh(covariance)

    return bool(eigenvalues[0] > eigenvalues[-1] * COVARIANCE_RANK_THRESHOLD)


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
    if not is_invertible(pooled_covariance):
        raise CollinearFeaturesError(
            "the pooled within-class covariance is singular, so no unique "
            "discriminant exists. Some column is a linear combination of the "
            "others within the classes, or there are fewer rows than columns; "
            "drop the redundant column, or reduce the columns first with "
            "PrincipalComponentAnalysis"
        )

    return np.asarray(solve_positive_definite(pooled_covariance, means.T)).T


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


def quadratic_discriminant_scores(
    rows: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    class_priors: FloatArray,
) -> FloatArray:
    """Each class's discriminant when every class has a covariance of its own.

    ``-0.5 (x - mu_k)' S_k^-1 (x - mu_k) - 0.5 log det S_k + log prior_k``, one
    value per row per class. The only term dropped is ``-p/2 log(2 pi)``, which
    is the same for every class whatever the covariances are.

    Nothing else drops out here, and that is the whole difference from
    :func:`linear_discriminant_scores`. With one shared matrix the term in
    ``x' S^-1 x`` is identical across classes and cancels; with a matrix per
    class it does not, so the score keeps a quadratic term and the boundary
    between two classes is a conic rather than a plane.

    Parameters
    ----------
    rows:
        ``(n_rows, n_features)``, the queries.
    means:
        ``(n_classes, n_features)``, each class's average per column.
    covariances:
        ``(n_classes, n_features, n_features)``, one symmetric positive
        definite matrix per class, already regularised if it is going to be.
    class_priors:
        ``(n_classes,)``, how common each class was.

    Returns
    -------
    FloatArray
        ``(n_rows, n_classes)``, unnormalised.

    Raises
    ------
    CollinearFeaturesError
        If any class's covariance is singular, naming the class. A class needs
        more rows than columns before it can describe a shape at all.
    """
    n_classes = means.shape[0]
    scores = np.empty((rows.shape[0], n_classes), dtype=np.float64)

    for label in range(n_classes):
        covariance = covariances[label]
        gaps = rows - means[label]

        if not is_invertible(covariance):
            raise CollinearFeaturesError(
                f"the covariance of class {label} is singular, so that class "
                "describes no shape in some direction. A class needs more rows "
                "than there are columns, and no column may be a combination of "
                "the others within it; raise shrinkage above zero to pull every "
                "class's covariance toward the identity instead"
            )

        solved = solve_positive_definite(covariance, gaps.T)
        quadratic = np.einsum("ij,ji->i", gaps, np.asarray(solved))
        _, log_determinant = np.linalg.slogdet(covariance)
        scores[:, label] = -0.5 * (quadratic + log_determinant)

    return scores + np.log(class_priors)[None, :]
