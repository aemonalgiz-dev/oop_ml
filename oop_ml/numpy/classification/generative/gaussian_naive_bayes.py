"""Classify by asking which class most plausibly produced the row.

Every classifier before this one draws a boundary. This one does something
different in kind: it builds a small model of what each class looks like, and
then asks, of a new row, which of those models would most likely have produced
it. That is what makes it generative rather than discriminative, and it is the
first classifier here that could in principle be run backwards to invent a row.

Why naive, and why it works anyway
-----------------------------------
The honest version would model how the features vary *together* within a
class, which needs a full covariance matrix per class, so a class with five
columns needs fifteen numbers to describe its shape and a class with fifty
needs over a thousand. The naive assumption is that inside a class the columns
are independent, which reduces that to a mean and a variance per column.

The assumption is almost always false. Height and weight are plainly related
within any group of people. What rescues the method is that the decision only
needs the *ranking* of the class scores to be right, and treating correlated
evidence as independent mostly inflates every class's score together, so the
ranking survives what the probabilities do not. That is the honest summary: a
good classifier and a bad estimator of probability, and this class says so
rather than quietly presenting the second as though it were the first.

Why the arithmetic is done in logs
------------------------------------
A row's likelihood under a class is a product of one density per column, and a
density is often well below one, so twenty columns multiply to something near
the smallest number a float can hold and fifty underflow to exactly zero. Adding
logarithms instead is the same comparison with none of the underflow, which is
why nothing here multiplies densities and why the scores it reports are log
scores until the very last step.

Why a variance floor
---------------------
A class whose column never varies has a variance of zero, and the density is
then a division by zero. That is not a broken fit, it is a fact about the
training rows, and the usual repair is to add a small share of the largest
variance in the data to every variance. The share is a field rather than a
constant because the right size depends on the data, and the default follows
the engine this library's other backend wraps so the two agree.
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
from oop_ml.core.gaussian import gaussian_log_scores, normalised_from_log_scores
from oop_ml.core.types import FloatArray


class GaussianNaiveBayes(MultiClassClassifier[Sequence[Feature], Feature]):
    """One Gaussian per class per column, and a class prior.

    Parameters
    ----------
    variance_smoothing:
        The share of the largest column variance added to every variance, so a
        class whose column never varies still has a density. The default is
        the one scikit-learn uses, so both backends answer alike.

    Raises
    ------
    NotFittedError
        From any learned property, or from ``predict``, before ``fit``.
    """

    variance_smoothing: float = Field(default=1e-9, ge=0.0)

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_class_priors",
        "_means",
        "_variances",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _class_priors: FloatArray | None = PrivateAttr(default=None)
    _means: FloatArray | None = PrivateAttr(default=None)
    _variances: FloatArray | None = PrivateAttr(default=None)

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
    def variances(self) -> FloatArray:
        """Each class's spread for each column, floored, same shape as the means."""
        self._check_fitted()
        assert self._variances is not None
        return self._variances

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Learn a prior, a mean and a variance per class.

        There is no iteration and nothing to converge. Each class's rows are
        summarised once, which is why this is the quickest fit in the library.

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
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        target_column = target_values.column
        target_column.check_is_label_encoded()

        rows = feature_set.feature_matrix
        labels = np.asarray(target_column.values, dtype=np.int64)
        n_classes = target_column.n_classes

        # A constant column is fine here, unlike in a linear fit: it simply
        # contributes the same density to every class and cancels out of the
        # comparison. Only a zero variance is a problem, and the floor is what
        # answers that.
        floor = self.variance_smoothing * float(np.var(rows, axis=0).max(initial=0.0))

        priors = np.empty(n_classes, dtype=np.float64)
        means = np.empty((n_classes, feature_set.n_features), dtype=np.float64)
        variances = np.empty_like(means)

        for label in range(n_classes):
            belonging = rows[labels == label]
            priors[label] = belonging.shape[0] / rows.shape[0]
            means[label] = belonging.mean(axis=0)
            variances[label] = belonging.var(axis=0) + floor

        self._feature_names = tuple(one.name for one in feature_set)
        self._class_priors = priors
        self._means = means
        self._variances = variances

        self._mark_fitted()
        return self

    def log_scores(self, input_values: Sequence[Feature]) -> FloatArray:
        """The log of each class's prior times its density, one row per query.

        The quantity the decision is actually made on, exposed because it is
        the honest one. Turning it into a probability adds nothing to the
        ranking and invites a reader to trust a number the naive assumption
        does not support.

        Returns
        -------
        FloatArray
            ``(n_rows, n_classes)``, unnormalised and negative.
        """
        block = self._matched_rows(input_values)
        assert self._means is not None
        assert self._variances is not None
        assert self._class_priors is not None

        return gaussian_log_scores(
            block.values, self._means, self._variances, self._class_priors
        )

    def predict_probabilities(self, input_values: Sequence[Feature]) -> ClassScores:
        """Each class's share of the total plausibility, one row per query.

        Normalised by subtracting each row's largest log score before
        exponentiating, which is the same answer and cannot overflow.

        Returns
        -------
        ProbabilityMatrix
            Rows summing to one. The stronger type is honest about the
            arithmetic; the module docstring is honest about the assumption
            behind it.
        """
        return ProbabilityMatrix(
            normalised_from_log_scores(self.log_scores(input_values))
        )

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The most plausible class for each row.

        Read off the log scores rather than the probabilities, since
        normalising cannot change which is largest and this way nothing is
        exponentiated to answer a question about order.
        """
        return Predictions.already_checked(
            np.argmax(self.log_scores(input_values), axis=1).astype(np.float64)
        )

    def _matched_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        """The query columns, in the order the fit saw them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If a fitted feature is missing, since a density needs every column
            the fit described.
        """
        self._check_fitted()
        assert self._feature_names is not None

        matched = FeatureSet.matching(self._feature_names, input_values)
        return rows_of(matched.feature_matrix, self._feature_names)
