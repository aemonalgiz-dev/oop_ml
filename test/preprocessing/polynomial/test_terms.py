"""Spec for PolynomialTerm / PolynomialTerms -- expansion terms as objects."""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.preprocessing.polynomial.terms import PolynomialTerm, PolynomialTerms

INPUT_FEATURES = [Feature("x1", [1, 2, 3]), Feature("x2", [4, 5, 6])]


class TestPolynomialTermName:
    @pytest.mark.parametrize(
        ("powers", "expected_name"),
        [
            ({"x1": 1}, "x1"),
            ({"x1": 2}, "x1^2"),
            ({"x1": 9}, "x1^9"),
            ({"x1": 1, "x2": 1}, "x1*x2"),
            ({"x1": 2, "x2": 1}, "x1^2*x2"),
            ({"x1": 2, "x2": 3}, "x1^2*x2^3"),
        ],
    )
    def test_reads_as_the_algebra_does(self, powers, expected_name):
        assert PolynomialTerm(powers).name == expected_name


class TestPolynomialTermValidation:
    def test_no_powers_raises(self):
        with pytest.raises(EmptyValuesError):
            PolynomialTerm({})

    @pytest.mark.parametrize("exponent", [0, -1, 1.5])
    def test_non_positive_integer_exponent_raises(self, exponent):
        with pytest.raises(InvalidValuesError):
            PolynomialTerm({"x1": exponent})


class TestPolynomialTermDegree:
    @pytest.mark.parametrize(
        ("powers", "expected_degree"),
        [
            ({"x1": 1}, 1),
            ({"x1": 3}, 3),
            ({"x1": 1, "x2": 1}, 2),
            ({"x1": 2, "x2": 3}, 5),
        ],
    )
    def test_total_degree_sums_the_exponents(self, powers, expected_degree):
        assert PolynomialTerm(powers).total_degree == expected_degree


class TestPolynomialTermEvaluate:
    @pytest.mark.parametrize(
        ("powers", "expected_values"),
        [
            ({"x1": 1}, [1.0, 2.0, 3.0]),
            ({"x1": 2}, [1.0, 4.0, 9.0]),
            ({"x2": 2}, [16.0, 25.0, 36.0]),
            ({"x1": 1, "x2": 1}, [4.0, 10.0, 18.0]),
            ({"x1": 2, "x2": 1}, [4.0, 20.0, 54.0]),
        ],
    )
    def test_multiplies_the_raised_columns(self, powers, expected_values):
        result = PolynomialTerm(powers).evaluate(INPUT_FEATURES)

        np.testing.assert_allclose(result.values, expected_values)

    def test_result_is_named_for_the_term(self):
        result = PolynomialTerm({"x1": 2, "x2": 1}).evaluate(INPUT_FEATURES)

        assert result.name == "x1^2*x2"

    def test_matches_features_by_name_not_position(self):
        reversed_order = list(reversed(INPUT_FEATURES))

        result = PolynomialTerm({"x1": 2}).evaluate(reversed_order)

        np.testing.assert_allclose(result.values, [1.0, 4.0, 9.0])

    def test_missing_feature_raises(self):
        with pytest.raises(InvalidValuesError):
            PolynomialTerm({"nope": 1}).evaluate(INPUT_FEATURES)


class TestPolynomialTermEquality:
    def test_equal_when_powers_match(self):
        assert PolynomialTerm({"x1": 2}) == PolynomialTerm({"x1": 2})

    def test_unequal_when_exponents_differ(self):
        assert PolynomialTerm({"x1": 2}) != PolynomialTerm({"x1": 3})

    def test_order_of_construction_does_not_matter_for_equality(self):
        assert PolynomialTerm({"x1": 1, "x2": 2}) == PolynomialTerm({"x2": 2, "x1": 1})


class TestPolynomialTerms:
    def make_terms(self) -> PolynomialTerms:
        return PolynomialTerms(
            [
                PolynomialTerm({"x1": 1}),
                PolynomialTerm({"x2": 1}),
                PolynomialTerm({"x1": 2}),
                PolynomialTerm({"x1": 1, "x2": 1}),
            ]
        )

    def test_reports_its_column_names_in_order(self):
        assert self.make_terms().names == ("x1", "x2", "x1^2", "x1*x2")

    def test_counts_its_terms(self):
        terms = self.make_terms()

        assert terms.n_terms == 4
        assert len(terms) == 4

    def test_iterates_term_objects(self):
        assert all(isinstance(term, PolynomialTerm) for term in self.make_terms())

    def test_reports_the_source_features_without_repeats(self):
        assert self.make_terms().source_feature_names == ("x1", "x2")

    def test_expand_produces_one_column_per_term_in_order(self):
        expanded = self.make_terms().expand(INPUT_FEATURES)

        assert [feature.name for feature in expanded] == ["x1", "x2", "x1^2", "x1*x2"]
        np.testing.assert_allclose(expanded[2].values, [1.0, 4.0, 9.0])
        np.testing.assert_allclose(expanded[3].values, [4.0, 10.0, 18.0])

    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            PolynomialTerms([])

    def test_duplicate_term_names_raise(self):
        with pytest.raises(NonUniqueFeaturesError):
            PolynomialTerms([PolynomialTerm({"x1": 2}), PolynomialTerm({"x1": 2})])
