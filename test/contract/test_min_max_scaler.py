"""The contract every backend's MinMaxScaler keeps.

The fixture was chosen so the answer is exact. ``[1, 3, 5]`` has a smallest
value of 1 and a range of 4, so it lands on ``[0, 0.5, 1]``; ``[10, 20, 40]``
has a smallest value of 10 and a range of 30, so it lands on ``[0, 1/3, 1]``.
The second column is there because a scaler that read the wrong column's
statistics would still put the first one on ``[0, 1]``.

A third fixture carries the same three lengths measured in units 1e16 times
smaller. Its range is a real 4e-16 and it must still land on ``[0, 0.5, 1]``,
which is the assertion that separates a scaler from an engine substituting a
range of one whenever the true one falls under ten machine epsilons.
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

_LENGTHS = [1.0, 3.0, 5.0]
_WEIGHTS = [10.0, 20.0, 40.0]
FEATURES = [Feature("length", _LENGTHS), Feature("weight", _WEIGHTS)]

LENGTH_CENTRE, LENGTH_SPREAD = 1.0, 4.0
WEIGHT_CENTRE, WEIGHT_SPREAD = 10.0, 30.0
SCALED_LENGTHS = [0.0, 0.5, 1.0]
SCALED_WEIGHTS = [0.0, 1.0 / 3.0, 1.0]

_IN_SMALL_UNITS = [1e-16, 3e-16, 5e-16]
SMALL_UNITS_CENTRE, SMALL_UNITS_SPREAD = 1e-16, 4e-16


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_without_arguments(backend: ModuleType) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")

    assert MinMaxScaler().is_fitted is False


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")
    model = MinMaxScaler()

    assert model.fit(FEATURES) is model


def test_its_scalings_are_addressable_by_feature_name(backend: ModuleType) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")
    scalings = MinMaxScaler().fit(FEATURES).scalings

    assert scalings.n_features == 2
    assert scalings["length"].centre == pytest.approx(LENGTH_CENTRE)
    assert scalings["length"].spread == pytest.approx(LENGTH_SPREAD)
    assert scalings["weight"].centre == pytest.approx(WEIGHT_CENTRE)
    assert scalings["weight"].spread == pytest.approx(WEIGHT_SPREAD)


def test_it_transforms_every_feature_onto_the_unit_interval(
    backend: ModuleType,
) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")
    model = MinMaxScaler().fit(FEATURES)

    transformed = model.transform(FEATURES)

    assert [feature.name for feature in transformed] == ["length", "weight"]
    assert np.allclose(column_of(transformed, "length"), SCALED_LENGTHS)
    assert np.allclose(column_of(transformed, "weight"), SCALED_WEIGHTS)


def test_inverse_transform_undoes_transform(backend: ModuleType) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")
    model = MinMaxScaler().fit(FEATURES)

    restored = model.inverse_transform(model.transform(FEATURES))

    assert [feature.name for feature in restored] == ["length", "weight"]
    assert np.allclose(column_of(restored, "length"), _LENGTHS)
    assert np.allclose(column_of(restored, "weight"), _WEIGHTS)


def test_it_accepts_a_subset_of_the_fitted_features_in_any_order(
    backend: ModuleType,
) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")
    model = MinMaxScaler().fit(FEATURES)

    only_weight = model.transform([Feature("weight", _WEIGHTS)])
    reversed_order = model.transform([FEATURES[1], FEATURES[0]])

    assert [feature.name for feature in only_weight] == ["weight"]
    assert np.allclose(column_of(only_weight, "weight"), SCALED_WEIGHTS)
    assert [feature.name for feature in reversed_order] == ["weight", "length"]
    assert np.allclose(column_of(reversed_order, "length"), SCALED_LENGTHS)


def test_it_refuses_a_feature_it_never_learned(backend: ModuleType) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")
    model = MinMaxScaler().fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([Feature("height", _LENGTHS)])


def test_it_refuses_a_constant_column_at_fit(backend: ModuleType) -> None:
    """A range of zero has nothing to divide by. The engine underneath would
    substitute one and answer all zeros; the contract is the refusal."""
    MinMaxScaler = provided(backend, "MinMaxScaler")
    constant = [Feature("length", _LENGTHS), Feature("flat", [7.0, 7.0, 7.0])]

    with pytest.raises(AllSameValuesError):
        MinMaxScaler().fit(constant)


def test_a_column_in_small_units_is_divided_by_its_own_tiny_range(
    backend: ModuleType,
) -> None:
    """Changing the unit a column is measured in cannot change where it lands
    on the unit interval. An engine underneath substitutes a range of one for
    anything below ten machine epsilons, which would hand the column back
    almost unmoved and call its range 1.0."""
    MinMaxScaler = provided(backend, "MinMaxScaler")
    model = MinMaxScaler().fit([Feature("in_small_units", _IN_SMALL_UNITS)])

    scaling = model.scalings["in_small_units"]
    transformed = model.transform([Feature("in_small_units", _IN_SMALL_UNITS)])

    assert scaling.centre == pytest.approx(SMALL_UNITS_CENTRE)
    assert scaling.spread == pytest.approx(SMALL_UNITS_SPREAD)
    assert np.allclose(column_of(transformed, "in_small_units"), SCALED_LENGTHS)


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    MinMaxScaler = provided(backend, "MinMaxScaler")

    with pytest.raises(NotFittedError):
        MinMaxScaler().transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = MinMaxScaler().scalings
