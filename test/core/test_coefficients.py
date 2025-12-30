"""Spec for Coefficient / Coefficients -- learned weights as objects."""

import pytest

from oop_ml.core.coefficients import Coefficient, Coefficients
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)


def make_coefficients() -> Coefficients:
    return Coefficients([Coefficient("x1", 2.0), Coefficient("x2", 3.0)])


class TestCoefficient:
    def test_carries_its_feature_name(self):
        assert Coefficient("age", 1.5).name == "age"

    def test_carries_its_value(self):
        assert Coefficient("age", 1.5).value == pytest.approx(1.5)

    def test_name_is_stripped(self):
        assert Coefficient("  age  ", 1.5).name == "age"

    def test_value_is_a_python_float(self):
        assert isinstance(Coefficient("age", 2).value, float)

    def test_empty_name_raises(self):
        with pytest.raises(InvalidValuesError):
            Coefficient("", 1.0)

    def test_equal_when_name_and_value_match(self):
        assert Coefficient("age", 1.0) == Coefficient("age", 1.0)

    def test_unequal_when_values_differ(self):
        assert Coefficient("age", 1.0) != Coefficient("age", 2.0)


class TestCoefficients:
    def test_reads_a_weight_by_feature_name(self):
        assert make_coefficients()["x1"] == pytest.approx(2.0)

    def test_value_for_matches_subscript(self):
        learned = make_coefficients()

        assert learned.value_for("x2") == learned["x2"]

    def test_unknown_feature_raises(self):
        with pytest.raises(InvalidValuesError):
            make_coefficients()["nope"]

    def test_membership_is_by_name(self):
        assert "x1" in make_coefficients()
        assert "nope" not in make_coefficients()

    def test_iterates_coefficient_objects(self):
        assert all(
            isinstance(coefficient, Coefficient) for coefficient in make_coefficients()
        )

    def test_counts_its_weights(self):
        learned = make_coefficients()

        assert learned.n_coefficients == 2
        assert len(learned) == 2

    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            Coefficients([])

    def test_duplicate_names_raise(self):
        with pytest.raises(NonUniqueFeaturesError):
            Coefficients([Coefficient("x1", 1.0), Coefficient("x1", 2.0)])
