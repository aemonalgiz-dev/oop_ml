"""Spec for RegressionEvaluation -- one aligned pairing, every metric off it.

Also the home of the metric specs: RSS, MSE and R^2 are properties of this
object rather than free functions, so this is where their behaviour is pinned.
"""

import numpy as np
import pytest

from oop_ml.core.evaluation import RegressionEvaluation
from oop_ml.data.column import Column
from oop_ml.exceptions import (
    EmptyValuesError,
    NonEqualArrayLengthError,
    UndefinedMetricError,
)
from oop_ml.validation import ValueRole


class TestConstruction:
    def test_accepts_raw_sequences(self):
        assert RegressionEvaluation([1, 2, 3], [1, 2, 3]).n_samples == 3

    def test_accepts_already_validated_columns(self):
        evaluation = RegressionEvaluation(
            Column([1, 2, 3], ValueRole.ACTUAL_VALUES),
            Column([1, 2, 3], ValueRole.PREDICTED_VALUES),
        )

        assert evaluation.n_samples == 3

    def test_mismatched_lengths_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            RegressionEvaluation([1, 2, 3], [1, 2])

    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            RegressionEvaluation([], [])


class TestPerObservation:
    def test_residuals(self):
        np.testing.assert_allclose(
            RegressionEvaluation([6, 6, 10], [5, 7, 9]).residuals,
            [1.0, -1.0, 1.0],
        )

    def test_squared_errors(self):
        np.testing.assert_allclose(
            RegressionEvaluation([6, 6, 10], [5, 7, 9]).squared_errors,
            [1.0, 1.0, 1.0],
        )

    def test_exposes_both_vectors(self):
        evaluation = RegressionEvaluation([1, 2], [3, 4])

        np.testing.assert_allclose(evaluation.actual_values, [1.0, 2.0])
        np.testing.assert_allclose(evaluation.predicted_values, [3.0, 4.0])


class TestResidualSumOfSquares:
    def test_zero_when_perfect(self):
        evaluation = RegressionEvaluation([1, 2, 3], [1, 2, 3])

        assert evaluation.residual_sum_of_squares == pytest.approx(0.0)

    def test_sum_of_squared_residuals(self):
        # residuals = (-1, 0, 1) -> 1 + 0 + 1
        evaluation = RegressionEvaluation([1, 2, 3], [2, 2, 2])

        assert evaluation.residual_sum_of_squares == pytest.approx(2.0)


class TestMeanSquaredError:
    def test_zero_when_perfect(self):
        assert RegressionEvaluation(
            [1, 2, 3], [1, 2, 3]
        ).mean_squared_error == pytest.approx(0.0)

    def test_mean_of_squared_residuals(self):
        # RSS = 2 over n = 3
        assert RegressionEvaluation(
            [1, 2, 3], [2, 2, 2]
        ).mean_squared_error == pytest.approx(2.0 / 3.0)

    def test_equals_rss_over_n(self):
        evaluation = RegressionEvaluation([3, -0.5, 2, 7], [2.5, 0.0, 2, 8])

        assert evaluation.mean_squared_error == pytest.approx(
            evaluation.residual_sum_of_squares / evaluation.n_samples
        )


class TestTotalSumOfSquares:
    def test_is_the_variation_in_the_truth(self):
        # mean of [1,2,3] is 2 -> deviations (-1, 0, 1) -> 2.0
        evaluation = RegressionEvaluation([1, 2, 3], [9, 9, 9])

        assert evaluation.total_sum_of_squares == pytest.approx(2.0)

    def test_is_zero_when_the_truth_is_constant(self):
        evaluation = RegressionEvaluation([5, 5, 5], [1, 2, 3])

        assert evaluation.total_sum_of_squares == pytest.approx(0.0)


class TestR2Score:
    def test_perfect_is_one(self):
        assert RegressionEvaluation([1, 2, 3], [1, 2, 3]).r2_score == pytest.approx(1.0)

    def test_predicting_the_mean_is_zero(self):
        # mean of [1,2,3] is 2; predicting 2 everywhere -> R^2 == 0
        assert RegressionEvaluation([1, 2, 3], [2, 2, 2]).r2_score == pytest.approx(0.0)

    def test_worse_than_mean_is_negative(self):
        # RSS = 8, TSS = 2 -> 1 - 4 = -3
        assert RegressionEvaluation([1, 2, 3], [3, 2, 1]).r2_score == pytest.approx(
            -3.0
        )

    def test_undefined_when_no_variance(self):
        with pytest.raises(UndefinedMetricError):
            _ = RegressionEvaluation([5, 5, 5], [5, 5, 6]).r2_score
