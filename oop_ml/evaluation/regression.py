"""Predicted values paired with the truth, and every metric that follows from it.

Every regression metric is a question about the same two aligned vectors, namely
what the model predicted and what actually happened. When I had these written as
free functions, each one was re-establishing that pairing on every single call,
coercing both inputs, proving they lined up, handing back a tuple, and then the
next metric turned around and did the whole thing again.

:class:`RegressionEvaluation` is that pairing modelled as a value object.
Aligning the two columns happens once in the constructor, and the residuals are
computed there as well, so every metric becomes a property reading state the
object is already holding. Asking it for four numbers costs you one alignment
and one subtraction rather than four of each.

The metric formulas live here. They remain model-free, in the sense that an
evaluation is built from two vectors and knows nothing whatsoever about
estimators, although you now reach them as behaviour on the data rather than as
functions you pass the data into.
"""

from __future__ import annotations

import numpy as np

from oop_ml.data.column import Column, ColumnSource
from oop_ml.exceptions import UndefinedMetricError
from oop_ml.types import FloatArray
from oop_ml.validation import ValueRole


class RegressionEvaluation:
    """What a model predicted, against what actually happened.

    Parameters
    ----------
    actual_values:
        The observed target values.
    predicted_values:
        What the model predicted for the same observations.

    Raises
    ------
    NonEqualArrayLengthError
        If the two inputs are not the same length.
    EmptyValuesError
        If either input has no observations.
    """

    __slots__ = ("_actual_column", "_predicted_column", "_residuals")

    def __init__(
        self,
        actual_values: ColumnSource,
        predicted_values: ColumnSource,
    ) -> None:
        self._actual_column = Column.of(actual_values, ValueRole.ACTUAL_VALUES)
        self._predicted_column = Column.of(predicted_values, ValueRole.PREDICTED_VALUES)
        self._actual_column.check_equal_length(self._predicted_column)

        self._residuals = self._actual_column.values - self._predicted_column.values

    @property
    def actual_values(self) -> FloatArray:
        """The observed targets."""
        return self._actual_column.values

    @property
    def predicted_values(self) -> FloatArray:
        """What the model predicted."""
        return self._predicted_column.values

    @property
    def n_samples(self) -> int:
        """Number of observations scored."""
        return self._actual_column.n_samples

    @property
    def residuals(self) -> FloatArray:
        """Signed error per observation: ``actual - predicted``."""
        return self._residuals

    @property
    def squared_errors(self) -> FloatArray:
        """Squared error per observation."""
        return np.square(self._residuals)

    @property
    def residual_sum_of_squares(self) -> float:
        """Total squared error left unexplained by the predictions.

        Always >= 0, and zero exactly when every prediction is perfect.
        """
        return float(np.sum(self.squared_errors))

    @property
    def mean_squared_error(self) -> float:
        """Average squared error per observation: ``RSS / n``."""
        return self.residual_sum_of_squares / self.n_samples

    @property
    def total_sum_of_squares(self) -> float:
        """Squared error of the dumbest useful baseline: always guessing the mean.

        The yardstick R^2 measures against, and exactly the observed column's
        total variation.
        """
        return self._actual_column.sum_of_squared_deviations

    @property
    def r2_score(self) -> float:
        """Coefficient of determination, ``1 - RSS / TSS``.

        A value of 1.0 is a perfect fit, 0.0 means the model is doing no better
        than predicting the mean, and anything negative means it is doing worse
        than that.

        Raises
        ------
        UndefinedMetricError
            If the observed targets have zero variance, which leaves nothing to
            explain and a denominator of zero.
        """
        if self.total_sum_of_squares == 0.0:
            raise UndefinedMetricError(
                "R^2 is undefined when the true values have zero variance"
            )

        return 1.0 - self.residual_sum_of_squares / self.total_sum_of_squares

    def __repr__(self) -> str:
        return f"RegressionEvaluation(n_samples={self.n_samples})"
