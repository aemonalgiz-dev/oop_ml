"""Spec for RidgeRegression.

Every expected figure below was computed from ``(X.T X + penalty * I) b = X.T y``
with the intercept slot of ``I`` zeroed, and checked against a reference solve.
"""

import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import NotFittedError
from oop_ml.regression.multiple_feature_regression import MultipleLinearRegression
from oop_ml.regression.ridge_regression import RidgeRegression
from test.fixtures import EXACT_PLANE, ORIGIN_PLANE

# penalty -> (intercept, x1 weight, x2 weight) on the exact plane.
PENALTY_SWEEP = [
    (0.0, 1.000000, 2.000000, 3.000000),
    (0.5, 2.157270, 1.679525, 2.409496),
    (1.0, 2.953704, 1.453704, 2.009259),
    (5.0, 5.368557, 0.724227, 0.847938),
    (10.0, 6.181724, 0.452752, 0.487020),
]

# penalty -> (x1 weight, x2 weight) with no intercept, where every column is penalised.
ORIGIN_PENALTY_SWEEP = [
    (0.0, 2.000000, 3.000000),
    (1.0, 1.939394, 2.606061),
    (5.0, 1.523810, 1.809524),
]


def fitted_model(penalty: float = 1.0) -> RidgeRegression:
    return RidgeRegression(penalty=penalty).fit(
        EXACT_PLANE.input_features, EXACT_PLANE.target_feature
    )


class TestConstruction:
    def test_penalty_defaults_to_one(self):
        assert RidgeRegression().penalty == pytest.approx(1.0)

    @pytest.mark.parametrize("penalty", [0.0, 0.5, 1.0, 100.0])
    def test_penalty_is_configurable(self, penalty):
        assert RidgeRegression(penalty=penalty).penalty == pytest.approx(penalty)

    @pytest.mark.parametrize("penalty", [-1.0, -0.0001])
    def test_negative_penalty_is_rejected(self, penalty):
        with pytest.raises(ValueError):
            RidgeRegression(penalty=penalty)

    def test_coefficients_raise_before_fit(self):
        with pytest.raises(NotFittedError):
            _ = RidgeRegression().coefficients


class TestPenaltySweep:
    @pytest.mark.parametrize(
        ("penalty", "intercept", "first_weight", "second_weight"), PENALTY_SWEEP
    )
    def test_matches_the_penalised_normal_equations(
        self, penalty, intercept, first_weight, second_weight
    ):
        model = fitted_model(penalty)

        assert model.intercept == pytest.approx(intercept, abs=1e-6)
        assert model.coefficients["x1"] == pytest.approx(first_weight, abs=1e-6)
        assert model.coefficients["x2"] == pytest.approx(second_weight, abs=1e-6)

    def test_zero_penalty_reproduces_ordinary_least_squares(self):
        ridge = fitted_model(penalty=0.0)
        ordinary = MultipleLinearRegression().fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert ridge.intercept == pytest.approx(ordinary.intercept)
        assert ridge.coefficients["x1"] == pytest.approx(ordinary.coefficients["x1"])
        assert ridge.coefficients["x2"] == pytest.approx(ordinary.coefficients["x2"])

    @pytest.mark.parametrize("feature_name", ["x1", "x2"])
    def test_shrinkage_is_monotone_in_the_penalty(self, feature_name):
        weights = [
            abs(fitted_model(penalty).coefficients[feature_name])
            for penalty, *_ in PENALTY_SWEEP
        ]

        assert weights == sorted(weights, reverse=True)

    def test_shrinks_toward_but_never_to_zero(self):
        assert fitted_model(penalty=1000.0).coefficients["x1"] > 0.0


class TestInterceptIsNotPenalised:
    @pytest.mark.parametrize("offset", [100.0, -50.0, 0.5])
    def test_shifting_the_target_shifts_only_the_intercept(self, offset):
        # Adding a constant to every y moves the plane bodily. A penalised
        # intercept could not follow, and the slopes would distort instead.
        base = fitted_model(penalty=1.0)
        shifted = RidgeRegression(penalty=1.0).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.shifted_target(offset)
        )

        assert shifted.intercept == pytest.approx(base.intercept + offset)
        assert shifted.coefficients["x1"] == pytest.approx(base.coefficients["x1"])
        assert shifted.coefficients["x2"] == pytest.approx(base.coefficients["x2"])


class TestWithoutIntercept:
    @pytest.mark.parametrize(
        ("penalty", "first_weight", "second_weight"), ORIGIN_PENALTY_SWEEP
    )
    def test_every_feature_is_penalised(self, penalty, first_weight, second_weight):
        # With no ones column, column 0 is an ordinary predictor. Exempting it
        # would leave x1 unshrunk and load the whole penalty onto x2.
        model = RidgeRegression(penalty=penalty, fit_intercept=False).fit(
            ORIGIN_PLANE.input_features, ORIGIN_PLANE.target_feature
        )

        assert model.intercept == pytest.approx(0.0)
        assert model.coefficients["x1"] == pytest.approx(first_weight, abs=1e-6)
        assert model.coefficients["x2"] == pytest.approx(second_weight, abs=1e-6)


class TestCollinearFeatures:
    @pytest.mark.parametrize("multiplier", [2.0, 5.0])
    def test_solves_where_ordinary_least_squares_is_singular(self, multiplier):
        # x2 is an exact multiple of x1, so X.T X is singular and OLS has no
        # unique answer. The penalty lifts the diagonal and makes it invertible,
        # splitting the weight in proportion to the columns.
        first, _ = EXACT_PLANE.input_features
        collinear = Feature("x2", [value * multiplier for value in first.values])

        model = RidgeRegression(penalty=1.0).fit(
            [first, collinear], EXACT_PLANE.target_feature
        )

        assert model.coefficients["x1"] == pytest.approx(
            model.coefficients["x2"] / multiplier, abs=1e-6
        )
