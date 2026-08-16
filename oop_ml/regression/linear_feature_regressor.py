"""Shared frame for every feature-first linear model.

Ordinary least squares, ridge, and gradient descent differ in exactly one place:
how the coefficient vector is obtained from ``X`` and ``y``. Everything around
that is identical, which is to say validating the features, building the design
matrix, splitting the intercept off the head of the solution, pairing the
remaining weights with their feature names, and later evaluating the hyperplane
by name.

:class:`LinearFeatureRegressor` owns all of that and leaves a single abstract
method, :meth:`LinearFeatureRegressor._solve`. A new linear model is then only
its own hyperparameters plus its own answer to "given ``X`` and ``y``, what is
``beta``?", and that second part is the one carrying the actual idea.

This is the template-method pattern rather than a solver-strategy object: the
hyperparameters that vary (``penalty``, ``learning_rate``) belong to the model
the user constructs, so ``RidgeRegression(penalty=1.0)`` reads better than
``LinearRegression(solver=RidgeSolver(penalty=1.0))``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import PrivateAttr

from oop_ml.core.base import Regressor
from oop_ml.core.coefficients import Coefficient, Coefficients
from oop_ml.core.column import Column
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.feature import Feature
from oop_ml.core.feature_set import FeatureSet
from oop_ml.core.types import FloatArray


class LinearFeatureRegressor(Regressor[Sequence[Feature], Feature]):
    """A hyperplane fit over named features; weights are read back by name.

    The feature-first API is this library's OOP alternative to an anonymous
    design matrix: predictors arrive as :class:`~oop_ml.core.feature.Feature` objects
    that keep their identity through fitting, so a coefficient is retrieved as
    ``model.coefficients_["age"]`` rather than ``model.coef_[2]``.

    Subclasses implement :meth:`_solve` and nothing else.

    Parameters
    ----------
    fit_intercept:
        Learn a bias term (default), i.e. prepend a ones column to ``X``. When
        ``False`` the column is omitted and the hyperplane is forced through the
        origin, with ``intercept_`` reported as ``0.0``.
    """

    fit_intercept: bool = True

    _intercept: float | None = PrivateAttr(default=None)
    _coefficients: Coefficients | None = PrivateAttr(default=None)

    @abstractmethod
    def _solve(self, design_matrix: FloatArray, target_column: Column) -> FloatArray:
        """Return the coefficient vector for ``design_matrix`` against the target.

        The one thing a linear model has to answer for itself.

        Parameters
        ----------
        design_matrix:
            ``X``, shape ``(n_samples, parameter_count)``. Already carries the
            leading ones column when ``fit_intercept`` is set, so implementations
            never handle the intercept as a special case.
        target_column:
            ``y``, already validated and aligned with the rows of ``X``.

        Returns
        -------
        FloatArray
            ``beta``, with one entry per column of ``design_matrix`` and in the
            same order, the intercept coming first whenever there is one.
        """

    @property
    def coefficients_(self) -> Coefficients:
        """The learned weights (available after ``fit``).

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

    def fit(self, input_values: Sequence[Feature], target_values: Feature) -> Self:
        """Fit the hyperplane, delegating the solve itself to the subclass.

        Parameters
        ----------
        input_values:
            One or more predictor columns, all the same length as the target.
            Their names key the learned coefficients and must be unique.
        target_values:
            The response column being regressed on.

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
        TooFewValuesError
            If there are fewer observations than parameters to estimate.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()
        feature_set.check_aligned_with(target_values)
        feature_set.check_supports_parameter_count(self._parameter_count(feature_set))

        solution = self._solve(self._design_matrix(feature_set), target_values.column)

        # The ones column sits first in the design matrix, so its weight is the
        # intercept and everything after it lines up with the feature columns --
        # in the order the feature set fixed at construction.
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

        self._mark_fitted()
        return self

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """Evaluate the fitted hyperplane on the given features.

        Computes ``intercept_ + sum(coefficients_[name] * values)`` over the
        features, matched to the fitted coefficients *by name* rather than by
        position, so a caller may pass them in whatever order is convenient
        although they do have to supply exactly the names seen during ``fit``.

        Parameters
        ----------
        input_values:
            Predictor columns to score, all of equal length.

        Returns
        -------
        FloatArray
            One prediction per observation.

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

        # Deliberately not a FeatureSet: its zero-variance guard protects the
        # *fit* (a constant column is collinear with the intercept), but a
        # constant column is perfectly legal to predict on.
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

    def _check_names_match_the_fit(self, input_values: Sequence[Feature]) -> None:
        """Raise unless exactly the fitted feature names were supplied."""
        supplied_names = {feature.name for feature in input_values}
        fitted_names = {coefficient.name for coefficient in self.coefficients_}

        if supplied_names != fitted_names:
            raise InvalidValuesError(
                f"expected features {', '.join(sorted(fitted_names))}; "
                f"got {', '.join(sorted(supplied_names)) or 'none'}"
            )

    def _parameter_count(self, feature_set: FeatureSet) -> int:
        """How many unknowns the fit has to solve for.

        One weight per feature, plus the intercept when it is being learned --
        which is exactly the width of the design matrix. Least squares needs at
        least this many observations, or the system is underdetermined.
        """
        return feature_set.n_features + (1 if self.fit_intercept else 0)

    @staticmethod
    def _normal_equations_matrix(design_matrix: FloatArray) -> FloatArray:
        """``X.T X``, the (parameter_count, parameter_count) system matrix."""
        return design_matrix.T @ design_matrix

    @staticmethod
    def _normal_equations_vector(
        design_matrix: FloatArray, target_column: Column
    ) -> FloatArray:
        """``X.T y``, the right-hand side, carrying one entry per parameter."""
        return design_matrix.T @ target_column.values

    def _design_matrix(self, feature_set: FeatureSet) -> FloatArray:
        """``X``: the feature columns, with a leading ones column if wanted."""
        design_matrix = feature_set.feature_matrix

        if not self.fit_intercept:
            return design_matrix

        return np.column_stack([np.ones(feature_set.n_samples), design_matrix])
