"""The machinery every model that is linear in its coefficients needs.

Ordinary least squares, ridge, lasso and logistic regression have almost nothing
in common at the point where they actually compute something, and almost
everything in common everywhere else. All four build a design matrix, optionally
prepend a ones column so the bias is just another coefficient, split that bias
back off the head of the solution, pair the remaining weights with the names of
the features they came from, and later evaluate ``b0 + sum(b_j x_j)`` by looking
those weights up by name.

That shared part lives here, including the fitting skeleton itself. An earlier version
of this module argued that it should say nothing about the target at all, on the grounds
that least squares fits against a number while logistic regression fits against a label.
That reasoning confused two different things. Both of them fit against a
:class:`~oop_ml.core.data.column.Column`; what differs is only which
*constraints* that column has to satisfy, and a single abstract hook covers that.
Holding the line cost three byte-identical copies of ``fit``, one per concrete frame,
which is a worse trade than the one it was avoiding.

So the skeleton is here and :meth:`LinearModel._validated_target_column` is the
one seam in it. A regressor takes the column as it comes; a classifier insists
it is binary and carries both classes. Neither has to restate the six lines
around that decision.

What sits on top of it is the task. ``LinearFeatureRegressor`` mixes this with
``Regressor``, and the classification frame mixes the very same thing with
``Classifier``. The word "linear" means linear in the coefficients,
which is why it can span two tasks that are not otherwise related.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Self

import numpy as np
from pydantic import PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.data.coefficients import Coefficient, Coefficients
from oop_ml.core.data.column import Column
from oop_ml.core.data.design_matrix import DesignMatrix
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.exceptions import InvalidValuesError, NonUniqueFeaturesError
from oop_ml.core.types import FloatArray


class LinearModel(Fittable):
    """Coefficients keyed by feature name, and the design matrix behind them.

    Parameters
    ----------
    fit_intercept:
        Learn a bias term, which is the default. Setting it prepends a ones
        column to ``X`` so the bias is solved for like any other coefficient.
        With it off, the column is omitted, the surface is forced through the
        origin, and ``intercept`` reports ``0.0``.
    """

    fit_intercept: bool = True

    _intercept: float | None = PrivateAttr(default=None)
    _coefficients: Coefficients | None = PrivateAttr(default=None)

    @property
    def coefficients(self) -> Coefficients:
        """The learned weights, available once ``fit`` has run.

        You can read one by name, as in ``model.coefficients["age"]``, or iterate the
        :class:`~oop_ml.core.data.coefficients.Coefficient` objects
        themselves. The collection is immutable, so handing it out to a caller cannot
        corrupt the fitted state.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._coefficients is not None
        return self._coefficients

    @property
    def intercept(self) -> float:
        """Learned bias term, or ``0.0`` when ``fit_intercept`` is ``False``.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._intercept is not None
        return self._intercept

    @abstractmethod
    def _solve(self, design_matrix: DesignMatrix, target_column: Column) -> FloatArray:
        """Return the coefficient vector for ``design_matrix`` against the target.

        The one thing a linear model has to answer for itself, and the only
        abstract method that carries an actual idea rather than plumbing.

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

    def _validated_target_column(self, target_values: Feature) -> Column:
        """The target as a column, checked against whatever the task requires.

        The seam between the two frames built on this class. The default is the
        regression answer -- a :class:`~oop_ml.core.data.column.Column` is
        already numeric, finite and non-empty, and least squares asks nothing further of
        it. Classification overrides this to insist on 0/1 labels with both classes
        present.

        Parameters
        ----------
        target_values:
            The target column as supplied to ``fit``.

        Returns
        -------
        Column
            The validated column, ready to hand to :meth:`_solve`.
        """
        return target_values.column

    def _fit_linear_model(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Self:
        """Validate, solve, pair the weights with their names, mark fitted.

        The whole of fitting a linear model, minus the two decisions a subclass
        owns: what counts as a valid target
        (:meth:`_validated_target_column`) and how ``beta`` is obtained
        (:meth:`_solve`). Public ``fit`` methods delegate here rather than
        restating it, so the order of the guards is fixed in one place.

        Not itself named ``fit``, because the public method is where each task
        documents the exceptions it raises, and those lists genuinely differ.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_columns_vary()
        feature_set.check_aligned_with(target_values)
        feature_set.check_supports_parameter_count(self._parameter_count(feature_set))

        solution = self._solve(
            self._design_matrix(feature_set),
            self._validated_target_column(target_values),
        )
        self._store_solution(feature_set, solution)

        self._mark_fitted()
        return self

    def _parameter_count(self, feature_set: FeatureSet) -> int:
        """How many unknowns the fit has to solve for.

        One weight per feature, plus the intercept when it is being learned,
        which is exactly the width of the design matrix. A fit needs at least
        this many observations or the system is underdetermined.
        """
        return feature_set.n_features + (1 if self.fit_intercept else 0)

    def _design_matrix(self, feature_set: FeatureSet) -> DesignMatrix:
        """``X``: the feature columns, with a leading ones column if wanted.

        Column-major, for the reason ``FeatureSet`` gives: the solvers here are
        dominated by ``X.T @ v``, and that product wants contiguous columns.
        Allocating in that order costs nothing over ``column_stack``, whereas
        converting afterwards would cost more than the layout saves on a model
        that only reads the matrix once.

        Returns a :class:`~oop_ml.core.data.design_matrix.DesignMatrix` rather
        than the array, so that whether column zero is the ones column travels
        with the numbers instead of being re-derived from ``self.fit_intercept``
        at every site that cares.
        """
        names = [feature.name for feature in feature_set]

        if not self.fit_intercept:
            return DesignMatrix(feature_set.feature_matrix, names, False)

        values = np.empty(
            (feature_set.n_samples, feature_set.n_features + 1), order="F"
        )
        values[:, 0] = 1.0
        values[:, 1:] = feature_set.feature_matrix

        return DesignMatrix(values, names, True)

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
        """Raise unless exactly the fitted feature names were supplied, once each.

        The duplicate check comes first and is not cosmetic: the set
        comparison alone would pass a list holding one feature twice, and the
        linear predictor then adds that coefficient's contribution once per
        copy -- a silently doubled effect, which is a plausible number rather
        than an error.
        """
        supplied_names = {feature.name for feature in input_values}

        if len(supplied_names) != len(input_values):
            counted: dict[str, int] = {}
            for feature in input_values:
                counted[feature.name] = counted.get(feature.name, 0) + 1
            repeated = sorted(name for name, count in counted.items() if count > 1)
            raise NonUniqueFeaturesError(
                f"each feature may be supplied once; got {', '.join(repeated)} "
                f"more than once"
            )

        fitted_names = {coefficient.name for coefficient in self.coefficients}

        if supplied_names != fitted_names:
            raise InvalidValuesError(
                f"expected features {', '.join(sorted(fitted_names))}; "
                f"got {', '.join(sorted(supplied_names)) or 'none'}"
            )

    def _linear_predictor(self, input_values: Sequence[Feature]) -> FloatArray:
        """``intercept + sum(coefficients[name] * values)`` over the features.

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
        predictions = np.full(reference_column.n_samples, self.intercept)
        for feature in input_values:
            predictions = predictions + self.coefficients[feature.name] * feature.values

        return predictions
