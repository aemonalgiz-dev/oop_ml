"""Spec for LassoRegression.

Every expected coefficient below was produced by a reference coordinate-descent
solve and then confirmed to be the objective's minimum by a brute-force search
around it, so a passing test means the numbers are right rather than merely
self-consistent.
"""

import pytest

from oop_ml.core.exceptions import NotFittedError
from oop_ml.regression.lasso_regression import LassoRegression
from oop_ml.regression.multiple_feature_regression import MultipleLinearRegression
from oop_ml.regression.ridge_regression import RidgeRegression
from test.fixtures import EXACT_PLANE, ORIGIN_PLANE

# penalty -> (intercept, x1 weight, x2 weight) on the exact plane.
PENALTY_SWEEP = [
    (0.0, 1.000000, 2.000000, 3.000000),
    (1.0, 1.547619, 1.833333, 2.738095),
    (2.0, 2.095238, 1.666667, 2.476190),
    (4.0, 3.190476, 1.333333, 1.952381),
    (8.0, 5.380952, 0.666667, 0.904762),
]

# penalty -> (x1 weight, x2 weight) with no intercept, every column penalised.
ORIGIN_PENALTY_SWEEP = [
    (0.0, 2.0, 3.0),
    (2.0, 1.9, 2.9),
    (10.0, 1.5, 2.5),
]

# Beyond this the second predictor is selected out entirely.
PENALTY_THAT_ZEROES_ONE = 12.0
PENALTY_THAT_ZEROES_BOTH = 16.0


def fitted_model(penalty: float = 1.0) -> LassoRegression:
    return LassoRegression(penalty=penalty).fit(
        EXACT_PLANE.input_features, EXACT_PLANE.target_feature
    )


class TestConstruction:
    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [("penalty", 1.0), ("max_iterations", 1_000), ("tolerance", 1e-10)],
    )
    def test_defaults(self, field_name, expected):
        assert getattr(LassoRegression(), field_name) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("field_name", "invalid"),
        [
            ("penalty", -1.0),
            ("max_iterations", 0),
            ("max_iterations", -3),
            ("tolerance", 0.0),
        ],
    )
    def test_invalid_settings_are_rejected(self, field_name, invalid):
        with pytest.raises(ValueError):
            LassoRegression(**{field_name: invalid})


class TestBeforeFit:
    @pytest.mark.parametrize(
        "attribute", ["coefficients", "intercept", "iterations_run", "converged"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(LassoRegression(), attribute)


class TestSoftThreshold:
    @pytest.mark.parametrize(
        ("value", "threshold", "expected"),
        [
            (5.0, 2.0, 3.0),
            (-5.0, 2.0, -3.0),
            (1.5, 2.0, 0.0),
            (-1.5, 2.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
            (5.0, 0.0, 5.0),
        ],
        ids=[
            "positive shrinks",
            "negative shrinks",
            "below threshold clamps",
            "negative below threshold clamps",
            "exactly at threshold clamps",
            "zero stays zero",
            "no threshold is identity",
        ],
    )
    def test_moves_toward_zero_and_stops_there(self, value, threshold, expected):
        assert LassoRegression._soft_threshold(value, threshold) == pytest.approx(
            expected
        )

    def test_clamped_result_is_exactly_zero(self):
        # Not merely small: the whole point is landing on zero.
        assert LassoRegression._soft_threshold(1.5, 2.0) == 0.0


class TestPenaltySweep:
    @pytest.mark.parametrize(
        ("penalty", "intercept", "first_weight", "second_weight"), PENALTY_SWEEP
    )
    def test_matches_the_reference_solve(
        self, penalty, intercept, first_weight, second_weight
    ):
        model = fitted_model(penalty)

        assert model.intercept == pytest.approx(intercept, abs=1e-6)
        assert model.coefficients["x1"] == pytest.approx(first_weight, abs=1e-6)
        assert model.coefficients["x2"] == pytest.approx(second_weight, abs=1e-6)

    def test_zero_penalty_reproduces_ordinary_least_squares(self):
        lasso = fitted_model(penalty=0.0)
        ordinary = MultipleLinearRegression().fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert lasso.intercept == pytest.approx(ordinary.intercept)
        assert lasso.coefficients["x1"] == pytest.approx(ordinary.coefficients["x1"])
        assert lasso.coefficients["x2"] == pytest.approx(ordinary.coefficients["x2"])

    @pytest.mark.parametrize("feature_name", ["x1", "x2"])
    def test_shrinkage_is_monotone_in_the_penalty(self, feature_name):
        weights = [
            abs(fitted_model(penalty).coefficients[feature_name])
            for penalty, *_ in PENALTY_SWEEP
        ]

        assert weights == sorted(weights, reverse=True)


class TestFeatureSelection:
    def test_a_large_enough_penalty_zeroes_one_feature_exactly(self):
        model = fitted_model(PENALTY_THAT_ZEROES_ONE)

        assert model.coefficients["x2"] == 0.0
        assert model.coefficients["x1"] != 0.0

    def test_a_larger_penalty_zeroes_every_feature(self):
        model = fitted_model(PENALTY_THAT_ZEROES_BOTH)

        assert model.coefficients["x1"] == 0.0
        assert model.coefficients["x2"] == 0.0

    def test_with_no_slopes_left_the_intercept_is_the_target_mean(self):
        # Predicting a constant, the best constant is the mean -- and the
        # unpenalised intercept is free to go there.
        model = fitted_model(PENALTY_THAT_ZEROES_BOTH)

        assert model.intercept == pytest.approx(EXACT_PLANE.target_feature.column.mean)

    @pytest.mark.parametrize("penalty", [PENALTY_THAT_ZEROES_ONE, 40.0, 100.0])
    def test_ridge_never_reaches_zero_where_lasso_does(self, penalty):
        # The defining difference: ridge's penalty goes limp near zero, so its
        # coefficients shrink forever without arriving.
        ridge = RidgeRegression(penalty=penalty).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert ridge.coefficients["x1"] != 0.0
        assert ridge.coefficients["x2"] != 0.0


class TestConvergence:
    def test_reports_convergence(self):
        model = fitted_model()

        assert model.converged is True
        assert model.iterations_run <= model.max_iterations

    def test_stopping_early_is_reported_not_hidden(self):
        model = LassoRegression(penalty=1.0, max_iterations=1, tolerance=1e-15).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert model.converged is False
        assert model.iterations_run == 1

    def test_converges_well_inside_the_iteration_cap(self):
        model = fitted_model()

        assert model.iterations_run < model.max_iterations

    @pytest.mark.parametrize(
        ("lighter_penalty", "heavier_penalty"), [(1.0, 12.0), (12.0, 16.0)]
    )
    def test_a_heavier_penalty_converges_sooner(self, lighter_penalty, heavier_penalty):
        # Coefficients clamped to zero stop moving, so the sweeps settle sooner:
        # 137 sweeps at penalty 1, 47 at penalty 12, 2 once everything is zeroed.
        assert (
            fitted_model(heavier_penalty).iterations_run
            < fitted_model(lighter_penalty).iterations_run
        )


class TestWithoutIntercept:
    @pytest.mark.parametrize(
        ("penalty", "first_weight", "second_weight"), ORIGIN_PENALTY_SWEEP
    )
    def test_every_feature_is_penalised(self, penalty, first_weight, second_weight):
        model = LassoRegression(penalty=penalty, fit_intercept=False).fit(
            ORIGIN_PLANE.input_features, ORIGIN_PLANE.target_feature
        )

        assert model.intercept == pytest.approx(0.0)
        assert model.coefficients["x1"] == pytest.approx(first_weight, abs=1e-6)
        assert model.coefficients["x2"] == pytest.approx(second_weight, abs=1e-6)

    def test_a_huge_penalty_zeroes_everything(self):
        model = LassoRegression(penalty=60.0, fit_intercept=False).fit(
            ORIGIN_PLANE.input_features, ORIGIN_PLANE.target_feature
        )

        assert model.coefficients["x1"] == 0.0
        assert model.coefficients["x2"] == 0.0
