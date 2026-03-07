"""Spec for GradientDescentRegression.

The closed-form answer for every fixture is known, so the walk can be checked
against the jump -- but only ever approximately, which is the defining
difference between the two.
"""

import numpy as np
import pytest

from oop_ml.core.feature import Feature
from oop_ml.exceptions import NotFittedError
from oop_ml.regression.gradient_descent_regression import GradientDescentRegression
from test.fixtures import EXACT_PLANE, ORIGIN_PLANE

CONVERGED_SETTINGS = {"max_epochs": 50_000, "tolerance": 1e-12}


def converged_model(
    learning_rate: float = 0.05, fixture=EXACT_PLANE, fit_intercept=True
):
    return GradientDescentRegression(
        learning_rate=learning_rate, fit_intercept=fit_intercept, **CONVERGED_SETTINGS
    ).fit(fixture.input_features, fixture.target_feature)


class TestConstruction:
    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [("learning_rate", 0.01), ("max_epochs", 10_000), ("tolerance", 1e-8)],
    )
    def test_defaults(self, field_name, expected):
        assert getattr(GradientDescentRegression(), field_name) == pytest.approx(
            expected
        )

    @pytest.mark.parametrize(
        ("field_name", "invalid"),
        [
            ("learning_rate", 0.0),
            ("learning_rate", -0.1),
            ("max_epochs", 0),
            ("max_epochs", -5),
            ("tolerance", 0.0),
        ],
    )
    def test_non_positive_settings_are_rejected(self, field_name, invalid):
        with pytest.raises(ValueError):
            GradientDescentRegression(**{field_name: invalid})


class TestBeforeFit:
    @pytest.mark.parametrize("attribute", ["epochs_run", "converged", "coefficients"])
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(GradientDescentRegression(), attribute)


class TestConvergence:
    @pytest.mark.parametrize("learning_rate", [0.01, 0.02, 0.05, 0.1])
    def test_every_workable_step_size_reaches_the_same_answer(self, learning_rate):
        # The step size changes how long the walk takes, never where it ends.
        model = converged_model(learning_rate)

        assert model.intercept == pytest.approx(
            EXACT_PLANE.expected_intercept, abs=1e-6
        )
        assert model.coefficients["x1"] == pytest.approx(
            EXACT_PLANE.expected_first_weight, abs=1e-6
        )
        assert model.coefficients["x2"] == pytest.approx(
            EXACT_PLANE.expected_second_weight, abs=1e-6
        )

    def test_reports_convergence(self):
        model = converged_model()

        assert model.converged is True
        assert model.epochs_run < model.max_epochs

    def test_a_larger_step_needs_fewer_epochs(self):
        assert converged_model(0.1).epochs_run < converged_model(0.01).epochs_run

    @pytest.mark.parametrize(
        ("first_values", "second_values", "expected"),
        [([10, 0], [0, 10], [21.0, 31.0]), ([1, 2], [1, 1], [6.0, 8.0])],
    )
    def test_predicts_like_the_closed_form(self, first_values, second_values, expected):
        predictions = converged_model().predict(
            [Feature("x1", first_values), Feature("x2", second_values)]
        )

        np.testing.assert_allclose(predictions, expected, atol=1e-5)


class TestEpochCap:
    @pytest.mark.parametrize("max_epochs", [1, 5, 20])
    def test_stopping_early_is_reported_not_hidden(self, max_epochs):
        # Unfinished coefficients must not be presented as an answer.
        model = GradientDescentRegression(
            learning_rate=0.05, max_epochs=max_epochs
        ).fit(EXACT_PLANE.input_features, EXACT_PLANE.target_feature)

        assert model.converged is False
        assert model.epochs_run == max_epochs

    def test_more_epochs_get_closer(self):
        def error_after(max_epochs: int) -> float:
            model = GradientDescentRegression(
                learning_rate=0.05, max_epochs=max_epochs, tolerance=1e-15
            ).fit(EXACT_PLANE.input_features, EXACT_PLANE.target_feature)
            return abs(model.coefficients["x1"] - EXACT_PLANE.expected_first_weight)

        assert error_after(500) < error_after(50)

    def test_a_divergent_step_terminates_instead_of_hanging(self):
        # Far above the 2/L threshold, so the walk blows up -- the cap is what
        # makes that a reported non-convergence rather than an infinite loop.
        model = GradientDescentRegression(learning_rate=5.0, max_epochs=50).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert model.converged is False


class TestWithoutIntercept:
    def test_is_forced_through_the_origin(self):
        model = converged_model(fixture=ORIGIN_PLANE, fit_intercept=False)

        assert model.intercept == pytest.approx(ORIGIN_PLANE.expected_intercept)
        assert model.coefficients["x1"] == pytest.approx(
            ORIGIN_PLANE.expected_first_weight, abs=1e-6
        )
        assert model.coefficients["x2"] == pytest.approx(
            ORIGIN_PLANE.expected_second_weight, abs=1e-6
        )
