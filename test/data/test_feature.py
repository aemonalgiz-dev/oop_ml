import numpy as np
import pytest

from oop_ml.data.feature import Feature
from oop_ml.exceptions import EmptyValuesError, InvalidValuesError


class TestConstruction:
    def test_coerces_sequence_to_float_array(self):
        feature = Feature("age", [25, 30, 35])

        assert isinstance(feature.values, np.ndarray)
        assert feature.values.dtype == np.float64
        np.testing.assert_allclose(feature.values, [25.0, 30.0, 35.0])

    def test_accepts_numpy_array(self):
        feature = Feature("age", np.array([1.0, 2.0, 3.0]))

        np.testing.assert_allclose(feature.values, [1.0, 2.0, 3.0])

    def test_name_is_stored(self):
        assert Feature("price", [1, 2]).name == "price"

    def test_name_is_stripped(self):
        assert Feature("  price  ", [1, 2]).name == "price"

    def test_n_samples_and_len_agree(self):
        feature = Feature("x", [1, 2, 3, 4])

        assert feature.n_samples == 4
        assert len(feature) == 4


class TestValidation:
    def test_empty_name_raises(self):
        with pytest.raises(InvalidValuesError):
            Feature("", [1, 2, 3])

    def test_whitespace_name_raises(self):
        with pytest.raises(InvalidValuesError):
            Feature("   ", [1, 2, 3])

    def test_non_string_name_raises(self):
        with pytest.raises(InvalidValuesError):
            Feature(42, [1, 2, 3])  # type: ignore[arg-type]

    def test_empty_values_raises(self):
        with pytest.raises(EmptyValuesError):
            Feature("x", [])

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_raise(self, bad):
        with pytest.raises(InvalidValuesError):
            Feature("x", [1.0, 2.0, bad])

    def test_two_dimensional_values_raise(self):
        with pytest.raises(InvalidValuesError):
            Feature("x", [[1, 2], [3, 4]])  # type: ignore[arg-type]


class TestImmutability:
    def test_values_array_is_read_only(self):
        feature = Feature("x", [1, 2, 3])

        with pytest.raises(ValueError):
            feature.values[0] = 99.0

    def test_freezing_does_not_touch_the_callers_array(self):
        original = np.array([1.0, 2.0, 3.0])
        Feature("x", original)

        # The caller's array must remain writable after construction.
        original[0] = 99.0
        assert original[0] == 99.0

    def test_name_cannot_be_reassigned(self):
        feature = Feature("x", [1, 2, 3])

        with pytest.raises(AttributeError):
            feature.name = "y"  # type: ignore[misc]


class TestEquality:
    def test_equal_when_name_and_values_match(self):
        assert Feature("x", [1, 2, 3]) == Feature("x", [1, 2, 3])

    def test_unequal_when_names_differ(self):
        assert Feature("x", [1, 2, 3]) != Feature("y", [1, 2, 3])

    def test_unequal_when_values_differ(self):
        assert Feature("x", [1, 2, 3]) != Feature("x", [1, 2, 4])

    def test_not_equal_to_non_feature(self):
        assert Feature("x", [1, 2, 3]) != [1, 2, 3]

    def test_equal_features_share_a_hash(self):
        a = Feature("x", [1, 2, 3])
        b = Feature("x", [1, 2, 3])

        assert hash(a) == hash(b)
        assert len({a, b}) == 1
