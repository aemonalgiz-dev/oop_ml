"""Spec for PolynomialFeatures -- red until ``_build_terms`` lands.

Column counts follow ``C(p + d, d) - 1``: the number of terms of total degree at
most ``d`` over ``p`` features, less the constant term, which the model's
intercept already supplies.
"""

import numpy as np
import pytest

from oop_ml.core.feature import Feature
from oop_ml.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
    NotFittedError,
)
from oop_ml.preprocessing.polynomial_features import PolynomialFeatures
from oop_ml.regression.multiple_feature_regression import MultipleLinearRegression

SINGLE_PREDICTOR = [Feature("x1", [1, 2, 3, 4])]
TWO_PREDICTORS = [Feature("x1", [1, 2, 3]), Feature("x2", [4, 5, 6])]


def fitted_expansion(
    degree: int = 2, include_interactions: bool = True, features=None
) -> PolynomialFeatures:
    return PolynomialFeatures(
        degree=degree, include_interactions=include_interactions
    ).fit(TWO_PREDICTORS if features is None else features)


class TestConstruction:
    def test_degree_defaults_to_two(self):
        assert PolynomialFeatures().degree == 2

    def test_interactions_are_included_by_default(self):
        assert PolynomialFeatures().include_interactions is True

    @pytest.mark.parametrize("degree", [0, -1])
    def test_degree_below_one_is_rejected(self, degree):
        with pytest.raises(ValueError):
            PolynomialFeatures(degree=degree)


class TestBeforeFit:
    def test_terms_raise_before_fit(self):
        with pytest.raises(NotFittedError):
            _ = PolynomialFeatures().terms

    def test_transform_raises_before_fit(self):
        with pytest.raises(NotFittedError):
            PolynomialFeatures().transform(TWO_PREDICTORS)


class TestTermGeneration:
    @pytest.mark.parametrize(
        ("degree", "expected_names"),
        [
            (1, ["x1", "x2"]),
            (2, ["x1", "x2", "x1^2", "x1*x2", "x2^2"]),
            (
                3,
                [
                    "x1",
                    "x2",
                    "x1^2",
                    "x1*x2",
                    "x2^2",
                    "x1^3",
                    "x1^2*x2",
                    "x1*x2^2",
                    "x2^3",
                ],
            ),
        ],
    )
    def test_orders_by_degree_then_by_feature(self, degree, expected_names):
        assert list(fitted_expansion(degree).terms.names) == expected_names

    @pytest.mark.parametrize(
        ("degree", "expected_names"),
        [
            (2, ["x1", "x2", "x1^2", "x2^2"]),
            (3, ["x1", "x2", "x1^2", "x2^2", "x1^3", "x2^3"]),
        ],
    )
    def test_without_interactions_only_pure_powers_survive(
        self, degree, expected_names
    ):
        assert (
            list(fitted_expansion(degree, include_interactions=False).terms.names)
            == expected_names
        )

    @pytest.mark.parametrize(
        ("n_features", "degree", "expected_count"),
        [(1, 1, 1), (1, 9, 9), (2, 2, 5), (2, 3, 9), (3, 2, 9), (3, 3, 19)],
    )
    def test_column_count_matches_the_combinatorial_formula(
        self, n_features, degree, expected_count
    ):
        features = [
            Feature(f"x{index}", [index + 1, index + 3, index + 6, index + 10])
            for index in range(n_features)
        ]

        expansion = PolynomialFeatures(degree=degree).fit(features)

        assert expansion.terms.n_terms == expected_count

    def test_degree_one_is_the_original_features(self):
        assert list(fitted_expansion(1).terms.names) == ["x1", "x2"]


class TestFitValidation:
    def test_returns_self(self):
        expansion = PolynomialFeatures()

        assert expansion.fit(TWO_PREDICTORS) is expansion

    def test_no_features_raises(self):
        with pytest.raises(EmptyValuesError):
            PolynomialFeatures().fit([])

    def test_duplicate_names_raise(self):
        with pytest.raises(NonUniqueFeaturesError):
            PolynomialFeatures().fit(
                [Feature("x1", [1, 2, 3]), Feature("x1", [4, 5, 6])]
            )

    def test_misaligned_features_raise(self):
        with pytest.raises(NonEqualArrayLengthError):
            PolynomialFeatures().fit([Feature("x1", [1, 2, 3]), Feature("x2", [4, 5])])

    def test_constant_feature_raises(self):
        with pytest.raises(AllSameValuesError):
            PolynomialFeatures().fit([Feature("flat", [7, 7, 7])])


class TestTransform:
    @pytest.mark.parametrize(
        ("term_name", "expected_values"),
        [
            ("x1", [1.0, 2.0, 3.0]),
            ("x2", [4.0, 5.0, 6.0]),
            ("x1^2", [1.0, 4.0, 9.0]),
            ("x1*x2", [4.0, 10.0, 18.0]),
            ("x2^2", [16.0, 25.0, 36.0]),
        ],
    )
    def test_computes_each_expanded_column(self, term_name, expected_values):
        expanded = fitted_expansion().transform(TWO_PREDICTORS)
        by_name = {feature.name: feature for feature in expanded}

        np.testing.assert_allclose(by_name[term_name].values, expected_values)

    def test_column_order_matches_the_fitted_terms(self):
        expansion = fitted_expansion()

        expanded = expansion.transform(TWO_PREDICTORS)

        assert [feature.name for feature in expanded] == list(expansion.terms.names)

    def test_matches_features_by_name_not_position(self):
        expansion = fitted_expansion()

        reversed_order = expansion.transform(list(reversed(TWO_PREDICTORS)))

        assert [feature.name for feature in reversed_order] == list(
            expansion.terms.names
        )

    def test_applies_to_held_out_rows_of_a_different_length(self):
        expansion = fitted_expansion()

        expanded = expansion.transform([Feature("x1", [10, 20]), Feature("x2", [1, 2])])
        by_name = {feature.name: feature for feature in expanded}

        np.testing.assert_allclose(by_name["x1^2"].values, [100.0, 400.0])
        np.testing.assert_allclose(by_name["x1*x2"].values, [10.0, 40.0])

    def test_a_missing_feature_raises(self):
        # Unlike Standardizer, a subset is not enough: x1*x2 needs both columns.
        with pytest.raises(InvalidValuesError):
            fitted_expansion().transform([Feature("x1", [1, 2, 3])])


class TestFitTransform:
    def test_matches_fit_then_transform(self):
        combined = PolynomialFeatures().fit_transform(TWO_PREDICTORS)
        separate = fitted_expansion().transform(TWO_PREDICTORS)

        for from_combined, from_separate in zip(combined, separate, strict=True):
            assert from_combined.name == from_separate.name
            np.testing.assert_allclose(from_combined.values, from_separate.values)


class TestFittingACurve:
    def test_a_quadratic_is_recovered_exactly(self):
        # y = 2 + 3*x - x^2. A straight line cannot fit this; the expansion
        # turns it into ordinary multiple regression, which recovers it exactly.
        inputs = Feature("x1", [-2, -1, 0, 1, 2, 3])
        targets = Feature(
            "y", [2 + 3 * value - value**2 for value in [-2, -1, 0, 1, 2, 3]]
        )

        expansion = PolynomialFeatures(degree=2).fit([inputs])
        model = MultipleLinearRegression().fit(expansion.transform([inputs]), targets)

        assert model.intercept == pytest.approx(2.0)
        assert model.coefficients["x1"] == pytest.approx(3.0)
        assert model.coefficients["x1^2"] == pytest.approx(-1.0)

    def test_coefficients_are_readable_by_term_name(self):
        expansion = PolynomialFeatures(degree=2).fit(SINGLE_PREDICTOR)
        targets = Feature("y", [3, 8, 15, 24])

        model = MultipleLinearRegression().fit(
            expansion.transform(SINGLE_PREDICTOR), targets
        )

        assert "x1^2" in model.coefficients
