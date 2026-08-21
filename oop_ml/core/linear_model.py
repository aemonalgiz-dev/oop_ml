"""The machinery every model that is linear in its coefficients needs.

Ordinary least squares, ridge, lasso and logistic regression have almost nothing
in common at the point where they actually compute something, and almost
everything in common everywhere else. All four build a design matrix, optionally
prepend a ones column so the bias is just another coefficient, split that bias
back off the head of the solution, pair the remaining weights with the names of
the features they came from, and later evaluate ``b0 + sum(b_j x_j)`` by looking
those weights up by name.

That shared part lives here, and it deliberately says nothing about what the
target is. Least squares fits against a number, logistic regression fits against
a label, so if this class knew about targets at all it could only serve one of
them. It knows about coefficients and about the design matrix, which is the
part they genuinely share.

What sits on top of it is the task. ``LinearFeatureRegressor`` mixes this with
``Regressor``, and the classification frame mixes the very same thing with
``Classifier``. The word "linear" means linear in the coefficients,
which is why it can span two tasks that are not otherwise related.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import PrivateAttr

from oop_ml.core.base import Fittable
from oop_ml.core.coefficients import Coefficient, Coefficients
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.feature import Feature
from oop_ml.core.feature_set import FeatureSet
from oop_ml.core.types import FloatArray


class LinearModel(Fittable):
    """Coefficients keyed by feature name, and the design matrix behind them.

    Parameters
    ----------
    fit_intercept:
        Learn a bias term, which is the default. Setting it prepends a ones
        column to ``X`` so the bias is solved for like any other coefficient.
        With it off, the column is omitted, the surface is forced through the
        origin, and ``intercept_`` reports ``0.0``.
    """

    fit_intercept: bool = True

    _intercept: float | None = PrivateAttr(default=None)
    _coefficients: Coefficients | None = PrivateAttr(default=None)

    @property
    def coefficients_(self) -> Coefficients:
        """The learned weights, available once ``fit`` has run.

        You can read one by name, as in ``model.coefficients_["age"]``, or
        iterate the :class:`~oop_ml.core.coefficients.Coefficient` objects
        themselves. The collection is immutable, so handing it out to a caller
        cannot corrupt the fitted state.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._coefficients is not None
        return self._coefficients

    @property
    def intercept_(self) -> float:
        """Learned bias term, or ``0.0`` when ``fit_intercept`` is ``False``.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._intercept is not None
        return self._intercept

    def _parameter_count(self, feature_set: FeatureSet) -> int:
        """How many unknowns the fit has to solve for.

        One weight per feature, plus the intercept when it is being learned,
        which is exactly the width of the design matrix. A fit needs at least
        this many observations or the system is underdetermined.
        """
        return feature_set.n_features + (1 if self.fit_intercept else 0)

    def _design_matrix(self, feature_set: FeatureSet) -> FloatArray:
        """``X``: the feature columns, with a leading ones column if wanted."""
        design_matrix = feature_set.feature_matrix

        if not self.fit_intercept:
            return design_matrix

        return np.column_stack([np.ones(feature_set.n_samples), design_matrix])

    def _store_solution(self, feature_set: FeatureSet, solution: FloatArray) -> None:
        """Split the intercept off the solution and pair the rest with names.

        The ones column sits first in the design matrix, so its weight is the
        intercept and everything after it lines up with the feature columns, in
        the order the feature set fixed at construction. Every solver in the
        library hands its answer back through here, which is what keeps that
        ordering assumption in one place rather than in each of them.
        """
        if self.fit_intercept:
            intercept = float(solution[0])
            weights = solution[1:]
        else:
            intercept = 0.0
            weights = solution

        self._intercept = intercept
        self._coefficients = Coefficients(
            [
                Coefficient(feature.name, weight)
                # strict: a length mismatch here would mean the design matrix
                # and the feature set had drifted out of step.
                for feature, weight in zip(feature_set, weights, strict=True)
            ]
        )

    def _check_names_match_the_fit(self, input_values: Sequence[Feature]) -> None:
        """Raise unless exactly the fitted feature names were supplied."""
        supplied_names = {feature.name for feature in input_values}
        fitted_names = {coefficient.name for coefficient in self.coefficients_}

        if supplied_names != fitted_names:
            raise InvalidValuesError(
                f"expected features {', '.join(sorted(fitted_names))}; "
                f"got {', '.join(sorted(supplied_names)) or 'none'}"
            )

    def _linear_predictor(self, input_values: Sequence[Feature]) -> FloatArray:
        """``intercept_ + sum(coefficients_[name] * values)`` over the features.

        For a regressor this is already the prediction. For a classifier it is
        the log-odds, which still has to go through a sigmoid before it means
        anything, and that difference is the only thing the two tasks disagree
        about at this point.

        Features are matched to coefficients by name rather than by position, so
        a caller may pass them in whatever order is convenient although they do
        have to supply exactly the names seen during ``fit``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        self._check_names_match_the_fit(input_values)

        # Deliberately not a FeatureSet. Its zero-variance guard protects the
        # fit, since a constant column is collinear with the intercept, whereas
        # a constant column is perfectly legal to predict on.
        reference_column = input_values[0].column
        for feature in input_values[1:]:
            reference_column.check_equal_length(feature.column)

        # Matching by name rather than by position is what lets the caller pass
        # the features in any order, and no design matrix is rebuilt here.
        predictions = np.full(reference_column.n_samples, self.intercept_)
        for feature in input_values:
            predictions = (
                predictions + self.coefficients_[feature.name] * feature.values
            )

        return predictions
