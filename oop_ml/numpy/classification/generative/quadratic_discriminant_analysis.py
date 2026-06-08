"""Classify generatively when the classes are not the same shape as each other.

The third corner of the square this package draws, and the one that spends the
most on describing a class. All three models here ask which class most
plausibly produced a row; what separates them is how much freedom a class has
to say what it looks like::

    naive Bayes  one number per column per class   diagonal, per class
    linear       one matrix for every class        full, shared
    quadratic    one matrix per class              full, per class

Naive Bayes gives every class its own spread but refuses to let the columns
vary together. Linear discriminant analysis lets them vary together and then
insists that every class varies the same way. This one gives up neither, and
pays for it in parameters and in rows.

Why the boundary curves
------------------------
The score of a row under class ``k``, dropping only the constant that is the
same for every class::

    delta_k(x) = -0.5 (x - mu_k)' S_k^-1 (x - mu_k) - 0.5 log det S_k
                 + log prior_k

Expand the first term and the piece in ``x' S_k^-1 x`` survives, because
``S_k`` depends on ``k``. In linear discriminant analysis that piece is
identical in every class's score and cancels out of every comparison, which is
what leaves a plane. Here it does not cancel, so the boundary between two
classes is whatever conic the difference of two quadratic forms describes: an
ellipse when one class is tighter than the other, a hyperbola when they are
tighter along different directions, and a plane in the one case where the two
covariances happen to be equal.

That is not a free improvement. A boundary that can curve can also curve around
noise, and the parameter count says why: with ``p`` columns and ``K`` classes,
the shared matrix costs ``p (p + 1) / 2`` numbers once and the per-class
matrices cost that ``K`` times over. At twenty columns and three classes that
is 210 against 630.

Why a class needs more rows than columns
------------------------------------------
Each class's covariance is estimated from that class's rows alone::

    S_k = sum_{i in k} (x_i - mu_k)(x_i - mu_k)'  /  (n_k - 1)

which is the ordinary sample covariance, with ``n_k - 1`` for the one mean it
already spent. A sum of ``n_k`` outer products of vectors that are constrained
to sum to zero has rank at most ``n_k - 1``, so a class with no more rows than
columns describes no shape in at least one direction and its covariance cannot
be inverted. Linear discriminant analysis pools every class's deviations into
one matrix and so only needs ``n`` rows in total; this one needs them in each
class separately, and that is usually the reason to prefer the other.

What shrinkage does about it
-----------------------------
``shrinkage`` pulls every class's covariance toward the identity::

    S_k(s) = (1 - s) S_k  +  s I

At ``s = 0`` nothing changes and a class short of rows is refused. Above zero
the matrix is positive definite whatever the data did, so the fit goes through,
and the model is trading its freedom to describe a shape for the ability to
describe one at all. At ``s = 1`` every class is a unit sphere, ``log det`` is
zero for all of them, and what is left is a nearest-mean classifier tilted by
the priors, which is the far end of that trade and is worth knowing as the
place the dial actually goes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores, ProbabilityMatrix
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import TooFewValuesError
from oop_ml.core.gaussian import (
    MINIMUM_CLASS_ROWS,
    normalised_from_log_scores,
    quadratic_discriminant_scores,
)
from oop_ml.core.types import FloatArray


class QuadraticDiscriminantAnalysis(MultiClassClassifier[Sequence[Feature], Feature]):
    """One mean and one covariance per class, and a class prior.

    Parameters
    ----------
    shrinkage:
        How far each class's covariance is pulled toward the identity, from 0
        for the sample covariance itself to 1 for a unit sphere per class. The
        default leaves the estimate alone; anything above zero makes every
        covariance invertible, which is what lets a class short of rows be
        fitted at all.

    Raises
    ------
    NotFittedError
        From any learned property, or from ``predict``, before ``fit``.
    """

    shrinkage: float = Field(default=0.0, ge=0.0, le=1.0)

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_class_priors",
        "_means",
        "_covariances",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _class_priors: FloatArray | None = PrivateAttr(default=None)
    _means: FloatArray | None = PrivateAttr(default=None)
    _covariances: FloatArray | None = PrivateAttr(default=None)

    @property
    def n_classes(self) -> int:
        """How many classes the fit saw."""
        self._check_fitted()
        assert self._class_priors is not None
        return int(self._class_priors.shape[0])

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The columns this model was fitted on, in order."""
        self._check_fitted()
        assert self._feature_names is not None
        return self._feature_names

    @property
    def class_priors(self) -> FloatArray:
        """How common each class was, one per class, summing to one."""
        self._check_fitted()
        assert self._class_priors is not None
        return self._class_priors

    @property
    def means(self) -> FloatArray:
        """Each class's average for each column, ``(n_classes, n_features)``."""
        self._check_fitted()
        assert self._means is not None
        return self._means

    @property
    def covariances(self) -> FloatArray:
        """One covariance per class, ``(n_classes, n_features, n_features)``.

        Already shrunk, so this is the matrix the discriminant actually uses
        rather than the raw sample estimate. At ``shrinkage = 0`` the two are
        the same thing.
        """
        self._check_fitted()
        assert self._covariances is not None
        return self._covariances

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Learn a prior, a mean and a covariance per class.

        No iteration and nothing to converge, as in the other two generative
        models. What is different is that every summary is estimated from one
        class's rows alone, so a class short of rows is a fit that cannot be
        made rather than one that comes out imprecise.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
        target_values:
            The classes, as whole positions running ``0 .. K - 1``.

        Returns
        -------
        Self
            This model, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        NonBinaryLabelsError
            If the target holds a negative or fractional value.
        SingleClassError
            If the target holds fewer than two classes, or leaves a gap in the
            run from zero.
        TooFewValuesError
            If any class holds fewer than two rows, so it has no deviation from
            its own mean to measure at all.
        CollinearFeaturesError
            If any class's covariance comes out singular at this shrinkage.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        rows = feature_set.feature_matrix
        labels = np.asarray(target_column.values, dtype=np.int64)
        n_classes = target_column.n_classes
        n_features = feature_set.n_features

        priors = np.empty(n_classes, dtype=np.float64)
        means = np.empty((n_classes, n_features), dtype=np.float64)
        covariances = np.empty((n_classes, n_features, n_features), dtype=np.float64)
        identity = np.eye(n_features, dtype=np.float64)

        for label in range(n_classes):
            group = rows[labels == label]
            n_belonging = group.shape[0]

            if n_belonging < MINIMUM_CLASS_ROWS:
                raise TooFewValuesError(
                    f"class {label} holds {n_belonging} row(s), and a covariance "
                    "needs at least two: with one row there is no deviation from "
                    "the class mean to measure"
                )

            priors[label] = n_belonging / rows.shape[0]
            means[label] = group.mean(axis=0)

            deviations = group - means[label]
            sample = deviations.T @ deviations / (n_belonging - 1)
            covariances[label] = (
                1.0 - self.shrinkage
            ) * sample + self.shrinkage * identity

        # Not the scores this discards but the refusal they can raise. A class
        # too small to describe a shape is found here rather than on the first
        # prediction, so a refused fit leaves the model unfitted.
        quadratic_discriminant_scores(means, means, covariances, priors)

        self._feature_names = tuple(one.name for one in feature_set)
        self._class_priors = priors
        self._means = means
        self._covariances = covariances

        self._mark_fitted()
        return self

    def discriminant_scores(self, input_values: Sequence[Feature]) -> FloatArray:
        """Each class's discriminant, one row per query.

        Unnormalised, and quadratic in the row rather than linear, which is the
        whole of the difference from
        :class:`~oop_ml.numpy.classification.generative.linear_discriminant_analysis.LinearDiscriminantAnalysis`.
        Only the term that is constant across classes whatever the covariances
        are has been dropped.

        Returns
        -------
        FloatArray
            ``(n_rows, n_classes)``, of either sign.
        """
        block = self._matched_rows(input_values)
        assert self._means is not None
        assert self._covariances is not None
        assert self._class_priors is not None

        return quadratic_discriminant_scores(
            block.values, self._means, self._covariances, self._class_priors
        )

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """Each class's share of the total plausibility, one row per query.

        Returns
        -------
        ProbabilityMatrix
            Rows summing to one.
        """
        return ProbabilityMatrix(
            normalised_from_log_scores(self.discriminant_scores(input_values))
        )

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The most plausible class for each row."""
        return Predictions.already_checked(
            np.argmax(self.discriminant_scores(input_values), axis=1).astype(np.float64)
        )

    def _matched_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        """The query columns, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a fitted feature is missing, since every column enters every
            class's quadratic form.
        """
        self._check_fitted()
        assert self._feature_names is not None

        matched = FeatureSet.matching(self._feature_names, input_values)
        return rows_of(matched.feature_matrix, self._feature_names)
