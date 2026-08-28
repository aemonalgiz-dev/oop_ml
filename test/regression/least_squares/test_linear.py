"""Spec for SimpleLinearRegression.

The line fitted to ``TRAINING_INPUTS``/``TRAINING_TARGETS`` is ``y = 3 + 2x``
exactly, so predictions and metrics are checkable by hand. ``KNOWN_LINES``
collects several datasets whose least-squares answer is known, including one
imperfect fit.
"""

import numpy as np
import pytest

from oop_ml.core.data.predictions import Predictions
from oop_ml.core.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NotFittedError,
    TooFewValuesError,
    UndefinedMetricError,
)
from oop_ml.core.types import NumericValues
from oop_ml.regression.least_squares.simple_linear_regression import (
    SimpleLinearRegression,
)

TRAINING_INPUTS: NumericValues = [1, 2, 3, 4]
TRAINING_TARGETS: NumericValues = [5, 7, 9, 11]

# inputs, targets, expected slope, expected intercept
KNOWN_LINES = [
    ([1, 2, 3, 4], [5, 7, 9, 11], 2.0, 3.0),
    ([0, 1, 2, 3], [4, 3, 2, 1], -1.0, 4.0),
    ([-2, -1, 0, 1, 2], [-6, -3, 0, 3, 6], 3.0, 0.0),
    ([1, 3, 7, 10], [3, 7, 15, 21], 2.0, 1.0),
    ([1, 2, 3, 4, 5], [2, 4, 5, 4, 5], 0.6, 2.2),
    ([1, 2], [10, 10], 0.0, 10.0),
]
KNOWN_LINE_IDS = [
    "positive slope",
    "negative slope",
    "zero intercept",
    "uneven spacing",
    "imperfect fit",
    "flat target",
]

# Predicting [1, 2, 3] gives [5, 7, 9]; against [6, 6, 10] the residuals are
# (1, -1, 1), so RSS = 3, MSE = 1, TSS = 32/3 and R^2 = 0.71875.
SCORED_INPUTS: NumericValues = [1, 2, 3]
SCORED_TARGETS: NumericValues = [6, 6, 10]


def fitted_model(
    inputs: NumericValues = TRAINING_INPUTS,
    targets: NumericValues = TRAINING_TARGETS,
) -> SimpleLinearRegression:
    return SimpleLinearRegression().fit(inputs, targets)


def scored_evaluation():
    return fitted_model().evaluate(SCORED_INPUTS, SCORED_TARGETS)


class TestFit:
    @pytest.mark.parametrize(
        ("inputs", "targets", "expected_slope", "expected_intercept"),
        KNOWN_LINES,
        ids=KNOWN_LINE_IDS,
    )
    def test_recovers_the_known_line(
        self, inputs, targets, expected_slope, expected_intercept
    ):
        model = fitted_model(inputs, targets)

        assert model.slope == pytest.approx(expected_slope)
        assert model.intercept == pytest.approx(expected_intercept)

    def test_fit_returns_self(self):
        model = SimpleLinearRegression()

        assert model.fit(TRAINING_INPUTS, TRAINING_TARGETS) is model

    def test_is_fitted_after_fit(self):
        assert fitted_model().is_fitted is True


class TestBeforeFit:
    @pytest.mark.parametrize("attribute", ["slope", "intercept"])
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(SimpleLinearRegression(), attribute)

    def test_is_not_fitted_before_fit(self):
        assert SimpleLinearRegression().is_fitted is False

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            SimpleLinearRegression().predict([4])


class TestFitValidation:
    @pytest.mark.parametrize(
        ("inputs", "targets", "expected_error"),
        [
            ([2, 2, 2], [1, 2, 3], AllSameValuesError),
            ([1, 2, 3], [1, 2], NonEqualArrayLengthError),
            ([], [], EmptyValuesError),
            ([1], [3], TooFewValuesError),
            ([1, 2, np.nan], [3, 5, 7], InvalidValuesError),
            ([1, 2, np.inf], [3, 5, 7], InvalidValuesError),
            ([1, 2, 3], [3, 5, np.nan], InvalidValuesError),
        ],
        ids=[
            "constant input",
            "length mismatch",
            "empty",
            "one sample",
            "not-a-number input",
            "infinite input",
            "not-a-number target",
        ],
    )
    def test_rejects_unusable_training_data(self, inputs, targets, expected_error):
        with pytest.raises(expected_error):
            fitted_model(inputs, targets)


class TestPredict:
    @pytest.mark.parametrize(
        ("inputs", "expected"),
        [
            ([0, 5, 10], [3.0, 13.0, 23.0]),
            ([1, 2, 3], [5.0, 7.0, 9.0]),
            ([-1], [1.0]),
            ([2.5], [8.0]),
        ],
        ids=["beyond the data", "on the data", "negative", "between points"],
    )
    def test_evaluates_the_line(self, inputs, expected):
        np.testing.assert_allclose(fitted_model().predict(inputs), expected)

    def test_returns_a_predictions_object(self):
        assert isinstance(fitted_model().predict([1, 2, 3]), Predictions)

    def test_predict_empty_values_raises(self):
        with pytest.raises(EmptyValuesError):
            fitted_model().predict([])

    @pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
    def test_predict_invalid_values_raises(self, invalid_value):
        with pytest.raises(InvalidValuesError):
            fitted_model().predict([1, 2, invalid_value])


class TestEvaluate:
    @pytest.mark.parametrize(
        ("attribute", "expected"),
        [
            ("residuals", [1.0, -1.0, 1.0]),
            ("squared_errors", [1.0, 1.0, 1.0]),
        ],
    )
    def test_per_observation_vectors(self, attribute, expected):
        np.testing.assert_allclose(getattr(scored_evaluation(), attribute), expected)

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("residual_sum_of_squares", 3.0),
            ("mean_squared_error", 1.0),
            ("total_sum_of_squares", 32 / 3),
            ("r2_score", 0.71875),
            ("n_samples", 3),
        ],
    )
    def test_metrics(self, metric, expected):
        assert getattr(scored_evaluation(), metric) == pytest.approx(expected)

    def test_mean_squared_error_is_rss_over_observation_count(self):
        evaluation = scored_evaluation()

        assert evaluation.mean_squared_error == pytest.approx(
            evaluation.residual_sum_of_squares / evaluation.n_samples
        )

    def test_one_evaluation_answers_every_metric(self):
        # The point of evaluate(): predictions are computed once, not once per
        # metric as the old per-metric methods did.
        evaluation = scored_evaluation()

        assert evaluation.residual_sum_of_squares == pytest.approx(3.0)
        assert evaluation.mean_squared_error == pytest.approx(1.0)
        assert evaluation.r2_score == pytest.approx(0.71875)

    def test_evaluate_with_non_equal_lengths_raises(self):
        with pytest.raises(NonEqualArrayLengthError):
            fitted_model().evaluate([1, 2], [5])


class TestScore:
    @pytest.mark.parametrize(
        ("training_inputs", "training_targets", "inputs", "targets", "expected"),
        [
            (TRAINING_INPUTS, TRAINING_TARGETS, [1, 2, 3, 4], [5, 7, 9, 11], 1.0),
            (TRAINING_INPUTS, TRAINING_TARGETS, [1, 2, 3], [6, 6, 10], 0.71875),
            ([1, 2, 3], [4, 4, 4], [10, 20, 30], [2, 4, 6], 0.0),
            ([1, 2, 3], [4, 4, 4], [10, 20, 30], [1, 2, 3], -6.0),
        ],
        ids=["perfect", "imperfect", "no better than the mean", "worse than the mean"],
    )
    def test_r2_over_the_whole_range(
        self, training_inputs, training_targets, inputs, targets, expected
    ):
        model = fitted_model(training_inputs, training_targets)

        assert model.score(inputs, targets) == pytest.approx(expected)

    def test_r2_is_undefined_when_targets_are_constant(self):
        with pytest.raises(UndefinedMetricError):
            fitted_model([1, 2, 3], [5, 7, 9]).score([1, 2, 3], [5, 5, 5])
