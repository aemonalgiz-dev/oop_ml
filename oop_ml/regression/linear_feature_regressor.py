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

from collections.abc import Sequence
from typing import Self

from oop_ml.core.base import Regressor
from oop_ml.core.column import Column
from oop_ml.core.feature import Feature
from oop_ml.core.linear_model import LinearModel
from oop_ml.core.types import FloatArray


class LinearFeatureRegressor(LinearModel, Regressor[Sequence[Feature], Feature]):
    """A hyperplane fit over named features; weights are read back by name.

    The feature-first API is this library's OOP alternative to an anonymous
    design matrix: predictors arrive as :class:`~oop_ml.core.feature.Feature` objects
    that keep their identity through fitting, so a coefficient is retrieved as
    ``model.coefficients["age"]`` rather than ``model.coef_[2]``.

    Subclasses implement :meth:`_solve` and nothing else.

    Parameters
    ----------
    fit_intercept:
        Learn a bias term (default), i.e. prepend a ones column to ``X``. When
        ``False`` the column is omitted and the hyperplane is forced through the
        origin, with ``intercept`` reported as ``0.0``.
    """

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
        AllSameValuesError
            If any predictor is constant. This was raised before it was
            documented; the guard has always been here.
        TooFewValuesError
            If there are fewer observations than parameters to estimate.
        """
        return self._fit_linear_model(input_values, target_values)

    def predict(self, input_values: Sequence[Feature]) -> FloatArray:
        """Evaluate the fitted hyperplane on the given features.

        Computes ``intercept + sum(coefficients[name] * values)`` over the
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
        return self._linear_predictor(input_values)

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
