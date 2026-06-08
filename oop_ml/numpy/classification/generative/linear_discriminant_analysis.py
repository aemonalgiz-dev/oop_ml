"""Classify by asking which class produced the row, when every class has one shape.

The second generative classifier here and the sibling of
:mod:`~oop_ml.numpy.classification.generative.gaussian_naive_bayes`. Both build
a small model of what each class looks like and then ask which of those models
most plausibly produced a new row. What they disagree about is the shape.

Naive Bayes gives every class its own spread and assumes that inside a class
the columns do not vary together, which is a diagonal covariance per class.
This one lets the columns vary together, which is a full covariance, and pays
for that by making every class share one. Neither assumption is true; they are
different economies with the same budget.

Why sharing the covariance makes the boundary a plane
------------------------------------------------------
Score a row under class ``k`` by the log of its prior times its density::

    log prior_k  -  0.5 (x - mu_k)' S_k^-1 (x - mu_k)  -  0.5 log det S_k  + c

Expanding the middle term gives ``x' S_k^-1 x``, ``-2 x' S_k^-1 mu_k`` and
``mu_k' S_k^-1 mu_k``. With a covariance *per class* the first of those depends
on ``k``, so the score is quadratic in ``x`` and the boundary between two
classes is a conic. Share one ``S`` and that term becomes identical in every
class's score, as does ``log det S``, so both cancel out of every comparison
and what is left is::

    delta_k(x) = x' S^-1 mu_k  -  0.5 mu_k' S^-1 mu_k  +  log prior_k

which is a plane in ``x``. That cancellation is the whole of the word "linear"
in the name, and it is worth seeing that the model is not linear by
construction; it is generative, and the linearity falls out.

The same argument says what is given up. Quadratic discriminant analysis is
this model with the sharing dropped, and it can curve a boundary around a class
that is tighter or differently tilted than its neighbours. It costs a full
covariance per class, which is where the parameter count goes: five columns
need fifteen numbers per class rather than fifteen once.

The pooled covariance
---------------------
Every class contributes its own deviations from its own mean, and the sum is
divided once::

    S = sum_k sum_{i in k} (x_i - mu_k)(x_i - mu_k)'  /  (n - K)

The denominator is ``n - K`` and not ``n``, because ``K`` means have already
been estimated from these rows and each one costs a degree of freedom. This is
the same correction that divides a sample variance by ``n - 1``, applied once
per class. It matters more than it looks: at the sizes a worked example uses
the two differ by a fifth, and the difference lands directly in the
discriminant.

Why a singular pooled covariance is refused
---------------------------------------------
The discriminant needs ``S^-1``. If some column is a linear combination of the
others within the classes, or if there are fewer rows than columns, ``S`` has a
direction of no spread and infinitely many discriminants separate the classes
equally well. Answering would mean choosing one arbitrarily, so this raises
``CollinearFeaturesError`` instead, the same position
:class:`~oop_ml.numpy.regression.least_squares.multiple_feature_regression.MultipleLinearRegression`
takes on the same question. Reduce the columns first if that is what is wanted.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import PrivateAttr

from oop_ml.core.base.estimator import MultiClassClassifier
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.probabilities import ClassScores, ProbabilityMatrix
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import TooFewValuesError
from oop_ml.core.gaussian import (
    discriminant_weights,
    linear_discriminant_scores,
    normalised_from_log_scores,
)
from oop_ml.core.types import FloatArray


class LinearDiscriminantAnalysis(MultiClassClassifier[Sequence[Feature], Feature]):
    """One mean per class, one covariance for all of them, and a class prior.

    Takes no hyperparameters, which is a fact about the model rather than an
    omission and makes it the only classifier here with none. The means, the
    pooled covariance and the priors are all counts and sums over the training
    rows, so the fit has no starting point, no step size and no stopping rule,
    and there is nothing to choose.

    Raises
    ------
    NotFittedError
        From any learned property, or from ``predict``, before ``fit``.
    """

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_class_priors",
        "_means",
        "_pooled_covariance",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _class_priors: FloatArray | None = PrivateAttr(default=None)
    _means: FloatArray | None = PrivateAttr(default=None)
    _pooled_covariance: FloatArray | None = PrivateAttr(default=None)

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
    def pooled_covariance(self) -> FloatArray:
        """The within-class covariance every class shares.

        ``(n_features, n_features)``, symmetric, divided by ``n - K``. This is
        the one summary naive Bayes does not have, and holding it is what lets
        the columns vary together.
        """
        self._check_fitted()
        assert self._pooled_covariance is not None
        return self._pooled_covariance

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Learn a prior and a mean per class, and one covariance for all.

        There is no iteration and nothing to converge, as in naive Bayes. The
        difference is the second pass: each class's rows are centred on their
        own mean, and the deviations of every class go into one matrix.

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
            If there are no more rows than classes, so the pooled covariance
            has no degrees of freedom left to be estimated from.
        CollinearFeaturesError
            If the pooled covariance comes out singular.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        rows = feature_set.feature_matrix
        labels = np.asarray(target_column.values, dtype=np.int64)
        n_classes = target_column.n_classes
        n_rows = rows.shape[0]

        if n_rows <= n_classes:
            raise TooFewValuesError(
                f"a pooled covariance over {n_classes} classes needs more than "
                f"{n_classes} rows, and {n_rows} were supplied. Every class "
                "spends one degree of freedom on its own mean"
            )

        priors = np.empty(n_classes, dtype=np.float64)
        means = np.empty((n_classes, feature_set.n_features), dtype=np.float64)
        deviations = np.empty_like(rows)

        for label in range(n_classes):
            belonging = labels == label
            group = rows[belonging]
            priors[label] = group.shape[0] / n_rows
            means[label] = group.mean(axis=0)
            # Each class is centred on its own mean before pooling, which is
            # what makes this a *within*-class covariance. Centring everything
            # on one overall mean would fold the separation between the classes
            # into the matrix meant to describe the spread inside them.
            deviations[belonging] = group - means[label]

        pooled = deviations.T @ deviations / (n_rows - n_classes)

        # Not the weights this discards but the refusal it can raise. A fit
        # that succeeded and left a model raising on its first prediction is
        # worse than one that refused, so the singular case is found here.
        discriminant_weights(means, pooled)

        self._feature_names = tuple(one.name for one in feature_set)
        self._class_priors = priors
        self._means = means
        self._pooled_covariance = pooled

        self._mark_fitted()
        return self

    def discriminant_scores(self, input_values: Sequence[Feature]) -> FloatArray:
        """Each class's discriminant, one row per query.

        Named for what it is rather than borrowing naive Bayes' ``log_scores``,
        because these are not the logs of anything. Every term that the shared
        covariance makes the same across classes has been dropped, which shifts
        a row's whole set of scores by one constant and leaves both the ranking
        and the probabilities exactly where they were.

        Returns
        -------
        FloatArray
            ``(n_rows, n_classes)``, unnormalised, and of either sign.
        """
        block = self._matched_rows(input_values)
        assert self._means is not None
        assert self._pooled_covariance is not None
        assert self._class_priors is not None

        return linear_discriminant_scores(
            block.values, self._means, self._pooled_covariance, self._class_priors
        )

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """Each class's share of the total plausibility, one row per query.

        These are better calibrated than naive Bayes', for the reason that
        model's docstring gives: treating correlated columns as independent
        counts the same evidence twice, and this one does not.

        Returns
        -------
        ProbabilityMatrix
            Rows summing to one.
        """
        return ProbabilityMatrix(
            normalised_from_log_scores(self.discriminant_scores(input_values))
        )

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The most plausible class for each row.

        Read off the discriminants rather than the probabilities, since
        normalising cannot change which is largest.
        """
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
            If a fitted feature is missing, since the discriminant is a plane
            over all of them and cannot be evaluated with one absent.
        """
        self._check_fitted()
        assert self._feature_names is not None

        matched = FeatureSet.matching(self._feature_names, input_values)
        return rows_of(matched.feature_matrix, self._feature_names)
