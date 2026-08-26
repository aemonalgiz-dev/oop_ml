"""Simple linear regression: one predictor, fit by ordinary least squares.

Aligned onto the shared ``core`` frame:

* inherits :class:`~oop_ml.core.base.estimator.Regressor`, so fitted-state tracking, the
  ``fit``/``predict`` contract, ``evaluate``, and ``score`` come from one place;
* data arrives in ``fit(input_values, target_values)``, not the constructor --
  construction only configures, fitting learns;
* learned parameters live on private attributes and are read back through
  ``slope`` / ``intercept``, which raise ``NotFittedError`` before ``fit``;
* inputs become :class:`~oop_ml.core.data.column.Column` objects at the
boundary, so this
  module never coerces an array itself, and the mean/deviation arithmetic the
  slope needs is asked of the column rather than recomputed here;
* metrics are not re-exposed here. ``evaluate(input_values, actual_values)``
  returns a :class:`~oop_ml.core.evaluation.regression.RegressionEvaluation`
  carrying residuals, RSS, MSE, TSS and R^2 off a single ``predict``, so that
  there is one way to ask the question and every other regressor answers it
  identically.
"""

from typing import Self

from pydantic import PrivateAttr

from oop_ml.core.base.estimator import Regressor
from oop_ml.core.data.column import Column
from oop_ml.core.solving.normal_equations import LeastSquaresLine
from oop_ml.core.types import FloatArray, NumericInput
from oop_ml.core.validation import ValueRole

MINIMUM_SAMPLES = 2
"""Two points determine a line, and anything fewer cannot pin down a slope."""


class SimpleLinearRegression(Regressor[NumericInput, NumericInput]):
    """Least-squares line over a single predictor.

    Predicts ``intercept + slope * input_value`` for each observation.
    """

    _slope: float | None = PrivateAttr(default=None)
    _intercept: float | None = PrivateAttr(default=None)

    @property
    def slope(self) -> float:
        """Learned slope (available after ``fit``)."""
        self._check_fitted()
        assert self._slope is not None
        return self._slope

    @property
    def intercept(self) -> float:
        """Learned intercept (available after ``fit``)."""
        self._check_fitted()
        assert self._intercept is not None
        return self._intercept

    def fit(self, input_values: NumericInput, target_values: NumericInput) -> Self:
        """Learn ``slope`` and ``intercept`` from the training pairs."""
        input_column = Column.of(input_values, ValueRole.INPUT_VALUES)
        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)

        input_column.check_equal_length(target_column)
        input_column.check_min_length(MINIMUM_SAMPLES)
        input_column.check_has_variance()

        # The slope is the two columns' co-variation over the predictor's own
        # variation; the intercept then anchors the line at the pair of means.
        input_target_covariance = float(
            (input_column.deviations * target_column.deviations).sum()
        )

        self._slope = input_target_covariance / input_column.sum_of_squared_deviations
        self._intercept = target_column.mean - self._slope * input_column.mean

        self._mark_fitted()

        return self

    def least_squares_line(
        self, input_values: NumericInput, target_values: NumericInput
    ) -> LeastSquaresLine:
        """The two sums behind the line, rather than only the line.

        The observed route beside :meth:`fit`. Same arithmetic; it keeps the
        means, the covariation and the predictor's own variation instead of
        collapsing them into two numbers.

        Worth naming because every larger least-squares fit is this
        generalised, and because the denominator is where the one structural
        failure lives: a constant predictor has zero variation, and no amount
        of target makes the slope defined.

        Records rather than mutates: this does not fit the model.

        Returns
        -------
        LeastSquaresLine
            ``line.result`` is the ``(slope, intercept)`` pair :meth:`fit`
            stores.

        Raises
        ------
        NonEqualArrayLengthError
            If the two columns differ in length.
        TooFewValuesError
            If there are fewer than two rows.
        AllSameValuesError
            If the predictor does not vary.
        """
        input_column = Column.of(input_values, ValueRole.INPUT_VALUES)
        target_column = Column.of(target_values, ValueRole.TARGET_VALUES)

        input_column.check_equal_length(target_column)
        input_column.check_min_length(MINIMUM_SAMPLES)
        input_column.check_has_variance()

        covariation = float((input_column.deviations * target_column.deviations).sum())
        variation = float(input_column.sum_of_squared_deviations)
        slope = covariation / variation

        return LeastSquaresLine(
            float(input_column.mean),
            float(target_column.mean),
            covariation,
            variation,
            slope,
            float(target_column.mean - slope * input_column.mean),
        )

    def predict(self, input_values: NumericInput) -> FloatArray:
        """Evaluate the fitted line at each input value."""
        self._check_fitted()

        input_column = Column.of(input_values, ValueRole.INPUT_VALUES)

        return input_column.values * self.slope + self.intercept
