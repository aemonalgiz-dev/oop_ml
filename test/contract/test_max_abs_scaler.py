"""The contract every backend's MaxAbsScaler keeps.

The fixture carries a negative value on purpose. ``[-4, 2, 1]`` has a largest
magnitude of 4, belonging to the negative entry, so it lands on
``[-1, 0.5, 0.25]`` with the sign kept, and a scaler dividing by the largest
*value* would divide by 2 and be caught. The second column ``[0, 0, 8]`` is
the structural-zero case, where the zeros must come back as exactly zero
because nothing was subtracted from them.

A third fixture carries the same three balances measured in units 1e16 times
smaller. Its largest magnitude is a real 4e-16 and it must still land on
``[-1, 0.5, 0.25]``, which is the assertion that separates a scaler from an
engine substituting a magnitude of one whenever the true one falls under ten
machine epsilons.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    NotFittedError,
)

from .harness import provided

_BALANCES = [-4.0, 2.0, 1.0]
_COUNTS = [0.0, 0.0, 8.0]
FEATURES = [Feature("balance", _BALANCES), Feature("count", _COUNTS)]

BALANCE_SPREAD = 4.0
COUNT_SPREAD = 8.0
SCALED_BALANCES = [-1.0, 0.5, 0.25]
SCALED_COUNTS = [0.0, 0.0, 1.0]

_IN_SMALL_UNITS = [-4e-16, 2e-16, 1e-16]
SMALL_UNITS_SPREAD = 4e-16


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_without_arguments(backend: ModuleType) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")

    assert MaxAbsScaler().is_fitted is False


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    model = MaxAbsScaler()

    assert model.fit(FEATURES) is model


def test_its_scalings_are_addressable_by_feature_name(backend: ModuleType) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    scalings = MaxAbsScaler().fit(FEATURES).scalings

    assert scalings.n_features == 2
    assert scalings["balance"].centre == pytest.approx(0.0)
    assert scalings["balance"].spread == pytest.approx(BALANCE_SPREAD)
    assert scalings["count"].centre == pytest.approx(0.0)
    assert scalings["count"].spread == pytest.approx(COUNT_SPREAD)


def test_it_divides_by_the_largest_magnitude_and_keeps_the_sign(
    backend: ModuleType,
) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    model = MaxAbsScaler().fit(FEATURES)

    transformed = model.transform(FEATURES)

    assert [feature.name for feature in transformed] == ["balance", "count"]
    assert np.allclose(column_of(transformed, "balance"), SCALED_BALANCES)


def test_a_structural_zero_stays_exactly_zero(backend: ModuleType) -> None:
    """Nothing is subtracted, so a zero is divided and stays a zero to the
    last bit rather than to a tolerance."""
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    transformed = MaxAbsScaler().fit(FEATURES).transform(FEATURES)

    assert list(column_of(transformed, "count")) == SCALED_COUNTS


def test_inverse_transform_undoes_transform(backend: ModuleType) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    model = MaxAbsScaler().fit(FEATURES)

    restored = model.inverse_transform(model.transform(FEATURES))

    assert [feature.name for feature in restored] == ["balance", "count"]
    assert np.allclose(column_of(restored, "balance"), _BALANCES)
    assert np.allclose(column_of(restored, "count"), _COUNTS)


def test_it_accepts_a_subset_of_the_fitted_features_in_any_order(
    backend: ModuleType,
) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    model = MaxAbsScaler().fit(FEATURES)

    only_count = model.transform([Feature("count", _COUNTS)])
    reversed_order = model.transform([FEATURES[1], FEATURES[0]])

    assert [feature.name for feature in only_count] == ["count"]
    assert np.allclose(column_of(only_count, "count"), SCALED_COUNTS)
    assert [feature.name for feature in reversed_order] == ["count", "balance"]
    assert np.allclose(column_of(reversed_order, "balance"), SCALED_BALANCES)


def test_it_refuses_a_feature_it_never_learned(backend: ModuleType) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    model = MaxAbsScaler().fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([Feature("height", _BALANCES)])


def test_it_refuses_an_all_zero_column_at_fit(backend: ModuleType) -> None:
    """A constant column of sevens is accepted, since seven is its magnitude.
    The column with no magnitude at all is the all-zero one, and the engine
    underneath would divide it by one; the contract is the refusal."""
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    zeros = [Feature("balance", _BALANCES), Feature("empty", [0.0, 0.0, 0.0])]

    with pytest.raises(AllSameValuesError):
        MaxAbsScaler().fit(zeros)


def test_a_column_in_small_units_is_divided_by_its_own_tiny_magnitude(
    backend: ModuleType,
) -> None:
    """Changing the unit a column is measured in cannot change where its
    extreme value lands. An engine underneath substitutes a magnitude of one
    for anything below ten machine epsilons, which would hand the column back
    almost unmoved and call its magnitude 1.0."""
    MaxAbsScaler = provided(backend, "MaxAbsScaler")
    model = MaxAbsScaler().fit([Feature("in_small_units", _IN_SMALL_UNITS)])

    scaling = model.scalings["in_small_units"]
    transformed = model.transform([Feature("in_small_units", _IN_SMALL_UNITS)])

    assert scaling.centre == pytest.approx(0.0)
    assert scaling.spread == pytest.approx(SMALL_UNITS_SPREAD)
    assert np.allclose(column_of(transformed, "in_small_units"), SCALED_BALANCES)


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    MaxAbsScaler = provided(backend, "MaxAbsScaler")

    with pytest.raises(NotFittedError):
        MaxAbsScaler().transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = MaxAbsScaler().scalings
