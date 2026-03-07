"""Spec for Standardizer.

Fixture: the usual ``x1``, whose mean is 1.4 and whose population standard
deviation is sqrt(5.2 / 5) = 1.019804, so every standardized value is checkable
by hand.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NotFittedError,
)
from oop_ml.preprocessing.standardizer import Standardizer
from test.fixtures import (
    EXACT_PLANE,
    FIRST_PREDICTOR,
    FIRST_PREDICTOR_MEAN,
    FIRST_PREDICTOR_STANDARD_DEVIATION,
    SECOND_PREDICTOR,
    STANDARDIZED_FIRST_PREDICTOR,
)


def make_input_features() -> list[Feature]:
    return EXACT_PLANE.input_features


def fitted_standardizer() -> Standardizer:
    return Standardizer().fit(make_input_features())


class TestBeforeFit:
    def test_scalings_raise_before_fit(self):
        with pytest.raises(NotFittedError):
            _ = Standardizer().scalings

    def test_transform_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            Standardizer().transform(make_input_features())

    def test_is_not_fitted_before_fit(self):
        assert Standardizer().is_fitted is False


class TestFit:
    def test_learns_a_scaling_per_feature(self):
        scalings = fitted_standardizer().scalings

        assert {scaling.name for scaling in scalings} == {"x1", "x2"}

    @pytest.mark.parametrize(
        ("feature_name", "expected_mean", "expected_spread"),
        [("x1", 1.4, 1.019804), ("x2", 1.2, 0.748331)],
    )
    def test_learns_each_column_mean_and_population_spread(
        self, feature_name, expected_mean, expected_spread
    ):
        # Population spread: sqrt(sum_of_squared_deviations / n), not / (n - 1).
        scaling = fitted_standardizer().scalings[feature_name]

        assert scaling.mean == pytest.approx(expected_mean)
        assert scaling.standard_deviation == pytest.approx(expected_spread, abs=1e-6)

    def test_returns_self(self):
        standardizer = Standardizer()

        assert standardizer.fit(make_input_features()) is standardizer

    def test_is_fitted_after_fit(self):
        assert fitted_standardizer().is_fitted is True

    def test_constant_feature_raises(self):
        with pytest.raises(AllSameValuesError):
            Standardizer().fit([Feature("flat", [7, 7, 7])])

    def test_misaligned_features_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            Standardizer().fit([Feature("x1", [1, 2, 3]), Feature("x2", [1, 2])])


class TestTransform:
    def test_centres_and_rescales_each_column(self):
        standardized = fitted_standardizer().transform(make_input_features())

        np.testing.assert_allclose(
            standardized[0].values, STANDARDIZED_FIRST_PREDICTOR, atol=1e-6
        )

    def test_result_has_zero_mean_and_unit_spread(self):
        for feature in fitted_standardizer().transform(make_input_features()):
            assert feature.column.mean == pytest.approx(0.0, abs=1e-12)
            assert feature.column.standard_deviation == pytest.approx(1.0)

    def test_names_are_preserved(self):
        standardized = fitted_standardizer().transform(make_input_features())

        assert [feature.name for feature in standardized] == ["x1", "x2"]

    def test_matches_features_by_name_not_position(self):
        standardizer = fitted_standardizer()

        reversed_order = standardizer.transform(
            [Feature("x2", SECOND_PREDICTOR), Feature("x1", FIRST_PREDICTOR)]
        )

        assert reversed_order[1].name == "x1"
        np.testing.assert_allclose(
            reversed_order[1].values, STANDARDIZED_FIRST_PREDICTOR, atol=1e-6
        )

    def test_unknown_feature_raises(self):
        with pytest.raises(InvalidValuesError):
            fitted_standardizer().transform([Feature("nope", FIRST_PREDICTOR)])

    def test_a_subset_of_the_fitted_features_is_allowed(self):
        # Unlike predict, transforming one column of a held-out set is legitimate.
        standardized = fitted_standardizer().transform([Feature("x1", FIRST_PREDICTOR)])

        assert [feature.name for feature in standardized] == ["x1"]

    @pytest.mark.parametrize(
        "held_out_values",
        [[11, 11, 12, 10, 13], [0, 0, 0], [-5, 100], [1.4]],
        ids=["shifted by ten", "all zeros", "far outside", "exactly the mean"],
    )
    def test_uses_training_statistics_not_the_new_columns(self, held_out_values):
        # The heart of fit/transform: held-out data is centred with the *training*
        # mean. Standardizing against its own mean would give zero mean; against
        # x1's training mean of 1.4 it must not.
        held_out = fitted_standardizer().transform([Feature("x1", held_out_values)])

        np.testing.assert_allclose(
            held_out[0].values,
            [
                (value - FIRST_PREDICTOR_MEAN) / FIRST_PREDICTOR_STANDARD_DEVIATION
                for value in held_out_values
            ],
            atol=1e-6,
        )


class TestFitTransform:
    def test_matches_fit_then_transform(self):
        combined = Standardizer().fit_transform(make_input_features())
        separate = fitted_standardizer().transform(make_input_features())

        for from_combined, from_separate in zip(combined, separate, strict=True):
            np.testing.assert_allclose(from_combined.values, from_separate.values)
