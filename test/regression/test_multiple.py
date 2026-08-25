"""Spec for MultipleLinearRegression.

Datasets come from :mod:`test.fixtures` -- the exact plane, the same plane with
an orthogonal displacement (imperfect fit, unchanged solution), and the
through-the-origin plane for the no-intercept path.
"""

import numpy as np
import pytest

from oop_ml.core.feature import Feature
from oop_ml.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
)
from oop_ml.regression.multiple_feature_regression import MultipleLinearRegression
from test.fixtures import (
    DISPLACED_PLANE,
    DISPLACED_PLANE_RESIDUAL_SUM_OF_SQUARES,
    DISPLACED_PLANE_RESIDUALS,
    DISPLACED_PLANE_TOTAL_SUM_OF_SQUARES,
    EXACT_PLANE,
    ORIGIN_PLANE,
)


def fitted_model(fixture=EXACT_PLANE, fit_intercept=True) -> MultipleLinearRegression:
    return MultipleLinearRegression(fit_intercept=fit_intercept).fit(
        fixture.input_features, fixture.target_feature
    )


class TestConstruction:
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_fit_intercept_is_configurable(self, fit_intercept):
        model = MultipleLinearRegression(fit_intercept=fit_intercept)

        assert model.fit_intercept is fit_intercept

    def test_fit_intercept_defaults_true(self):
        assert MultipleLinearRegression().fit_intercept is True


class TestBeforeFit:
    @pytest.mark.parametrize("attribute", ["coefficients", "intercept"])
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(MultipleLinearRegression(), attribute)

    def test_is_not_fitted_before_fit(self):
        assert MultipleLinearRegression().is_fitted is False


class TestFit:
    @pytest.mark.parametrize(
        ("fixture", "fit_intercept"),
        [
            (EXACT_PLANE, True),
            # No plane passes through the displaced points, so the solver has to
            # minimise rather than interpolate -- an interpolating bug fails here.
            (DISPLACED_PLANE, True),
            (ORIGIN_PLANE, False),
        ],
        ids=["exact plane", "imperfect fit", "through the origin"],
    )
    def test_recovers_the_known_solution(self, fixture, fit_intercept):
        model = fitted_model(fixture, fit_intercept)

        assert model.intercept == pytest.approx(fixture.expected_intercept)
        assert model.coefficients["x1"] == pytest.approx(fixture.expected_first_weight)
        assert model.coefficients["x2"] == pytest.approx(fixture.expected_second_weight)

    def test_coefficients_keyed_by_feature_name(self):
        learned = fitted_model().coefficients

        assert {coefficient.name for coefficient in learned} == {"x1", "x2"}

    def test_is_fitted_after_fit(self):
        assert fitted_model().is_fitted is True

    def test_fit_returns_self(self):
        model = MultipleLinearRegression()

        assert (
            model.fit(EXACT_PLANE.input_features, EXACT_PLANE.target_feature) is model
        )

    def test_target_length_mismatch_raises(self):
        with pytest.raises(NonEqualArrayLengthError):
            MultipleLinearRegression().fit(
                [Feature("x1", [1, 2, 3])], Feature("y", [1, 2])
            )

    def test_no_features_raises(self):
        with pytest.raises(EmptyValuesError):
            MultipleLinearRegression().fit([], EXACT_PLANE.target_feature)

    def test_duplicate_feature_names_raise(self):
        first, second = EXACT_PLANE.input_features

        with pytest.raises(NonUniqueFeaturesError):
            MultipleLinearRegression().fit(
                [first, Feature("x1", second.values)], EXACT_PLANE.target_feature
            )

    def test_too_few_samples_raises(self):
        # Two rows cannot determine two coefficients plus an intercept.
        with pytest.raises(TooFewValuesError):
            MultipleLinearRegression().fit(
                [Feature("x1", [1, 2]), Feature("x2", [3, 5])], Feature("y", [1, 2])
            )


class TestPredict:
    @pytest.mark.parametrize(
        ("first_values", "second_values", "expected"),
        [
            ([10, 0], [0, 10], [21.0, 31.0]),
            ([0, 0], [0, 0], [1.0, 1.0]),
            ([1, 2, 3], [1, 1, 1], [6.0, 8.0, 10.0]),
            ([-1], [-1], [-4.0]),
        ],
        ids=["new samples", "all zeros", "three rows", "negative values"],
    )
    def test_evaluates_the_plane(self, first_values, second_values, expected):
        predictions = fitted_model().predict(
            [Feature("x1", first_values), Feature("x2", second_values)]
        )

        np.testing.assert_allclose(predictions, expected)

    def test_returns_numpy_array(self):
        predictions = fitted_model().predict([Feature("x1", [1]), Feature("x2", [1])])

        assert isinstance(predictions, np.ndarray)

    def test_predict_before_fit_raises(self):
        with pytest.raises(NotFittedError):
            MultipleLinearRegression().predict(EXACT_PLANE.input_features)

    def test_matches_features_by_name_not_position(self):
        # Positional matching would give [31, 21] instead.
        predictions = fitted_model().predict(
            [Feature("x2", [0, 10]), Feature("x1", [10, 0])]
        )

        np.testing.assert_allclose(predictions, [21.0, 31.0])

    @pytest.mark.parametrize(
        "supplied",
        [
            [Feature("nope", [1]), Feature("x2", [1])],
            [Feature("x1", [1])],
            [Feature("x1", [1]), Feature("x2", [1]), Feature("extra", [1])],
        ],
        ids=["unknown name", "missing feature", "extra feature"],
    )
    def test_feature_names_must_match_the_fit(self, supplied):
        with pytest.raises(InvalidValuesError):
            fitted_model().predict(supplied)

    def test_misaligned_features_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            fitted_model().predict([Feature("x1", [1, 2]), Feature("x2", [1])])


class TestScoring:
    def test_scores_one_on_an_exact_fit(self):
        model = fitted_model()

        assert model.score(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        ) == pytest.approx(1.0)

    def test_residuals_of_an_imperfect_fit(self):
        model = fitted_model(DISPLACED_PLANE)

        evaluation = model.evaluate(
            DISPLACED_PLANE.input_features, DISPLACED_PLANE.target_feature
        )

        # atol: the last residual is exactly zero in theory, 9e-16 in floating
        # point, which a purely relative tolerance can never accept.
        np.testing.assert_allclose(
            evaluation.residuals, DISPLACED_PLANE_RESIDUALS, atol=1e-12
        )

    @pytest.mark.parametrize(
        ("metric", "expected"),
        [
            ("residual_sum_of_squares", DISPLACED_PLANE_RESIDUAL_SUM_OF_SQUARES),
            ("total_sum_of_squares", DISPLACED_PLANE_TOTAL_SUM_OF_SQUARES),
            ("mean_squared_error", DISPLACED_PLANE_RESIDUAL_SUM_OF_SQUARES / 5),
            ("r2_score", 73 / 83),
        ],
    )
    def test_metrics_of_an_imperfect_fit(self, metric, expected):
        model = fitted_model(DISPLACED_PLANE)

        evaluation = model.evaluate(
            DISPLACED_PLANE.input_features, DISPLACED_PLANE.target_feature
        )

        assert getattr(evaluation, metric) == pytest.approx(expected)
