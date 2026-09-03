"""The contract every backend's RobustScaler keeps.

The fixture's quartiles were placed by hand. ``[1, 2, ..., 9]`` has median 5
and, placing a quantile by linear interpolation between sorted neighbours,
quartiles at positions ``0.25 * 8 = 2`` and ``0.75 * 8 = 6``, which are the
values 3 and 7. So the interquartile range is 4 and the column lands on
``(x - 5) / 4``.

The second column is the same nine values with the last turned to a thousand.
Neither the median nor either quartile is in the tail, so its centre and
spread are the same 5 and 4, and that is the property that makes the scaler
worth having rather than a detail of the fixture.

The third fixture is the same ramp scaled down to 1e-16 a step. Its
interquartile range is a real 4e-16 and both backends must divide by it, which
is the assertion that separates them from an engine substituting a spread of
one whenever the range falls under ten machine epsilons.
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

_READINGS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
_WITH_OUTLIER = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1000.0]
FEATURES = [Feature("reading", _READINGS), Feature("spiked", _WITH_OUTLIER)]

MEDIAN, INTERQUARTILE_RANGE = 5.0, 4.0
SCALED_READINGS = (np.array(_READINGS) - MEDIAN) / INTERQUARTILE_RANGE
SCALED_WITH_OUTLIER = (np.array(_WITH_OUTLIER) - MEDIAN) / INTERQUARTILE_RANGE

_IN_SMALL_UNITS = [position * 1e-16 for position in range(9)]
SMALL_UNITS_CENTRE, SMALL_UNITS_SPREAD = 4e-16, 4e-16
SCALED_IN_SMALL_UNITS = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_without_arguments(backend: ModuleType) -> None:
    RobustScaler = provided(backend, "RobustScaler")

    assert RobustScaler().is_fitted is False


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    model = RobustScaler()

    assert model.fit(FEATURES) is model


def test_its_scalings_are_addressable_by_feature_name(backend: ModuleType) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    scalings = RobustScaler().fit(FEATURES).scalings

    assert scalings.n_features == 2
    assert scalings["reading"].centre == pytest.approx(MEDIAN)
    assert scalings["reading"].spread == pytest.approx(INTERQUARTILE_RANGE)


def test_an_outlier_moves_neither_the_centre_nor_the_spread(
    backend: ModuleType,
) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    scalings = RobustScaler().fit(FEATURES).scalings

    assert scalings["spiked"].centre == pytest.approx(MEDIAN)
    assert scalings["spiked"].spread == pytest.approx(INTERQUARTILE_RANGE)


def test_it_transforms_every_feature_to_the_known_answer(backend: ModuleType) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    model = RobustScaler().fit(FEATURES)

    transformed = model.transform(FEATURES)

    assert [feature.name for feature in transformed] == ["reading", "spiked"]
    assert np.allclose(column_of(transformed, "reading"), SCALED_READINGS)
    assert np.allclose(column_of(transformed, "spiked"), SCALED_WITH_OUTLIER)


def test_inverse_transform_undoes_transform(backend: ModuleType) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    model = RobustScaler().fit(FEATURES)

    restored = model.inverse_transform(model.transform(FEATURES))

    assert [feature.name for feature in restored] == ["reading", "spiked"]
    assert np.allclose(column_of(restored, "reading"), _READINGS)
    assert np.allclose(column_of(restored, "spiked"), _WITH_OUTLIER)


def test_it_accepts_a_subset_of_the_fitted_features_in_any_order(
    backend: ModuleType,
) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    model = RobustScaler().fit(FEATURES)

    only_spiked = model.transform([Feature("spiked", _WITH_OUTLIER)])
    reversed_order = model.transform([FEATURES[1], FEATURES[0]])

    assert [feature.name for feature in only_spiked] == ["spiked"]
    assert np.allclose(column_of(only_spiked, "spiked"), SCALED_WITH_OUTLIER)
    assert [feature.name for feature in reversed_order] == ["spiked", "reading"]
    assert np.allclose(column_of(reversed_order, "reading"), SCALED_READINGS)


def test_it_refuses_a_feature_it_never_learned(backend: ModuleType) -> None:
    RobustScaler = provided(backend, "RobustScaler")
    model = RobustScaler().fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([Feature("height", _READINGS)])


def test_it_refuses_a_column_whose_quartiles_coincide(backend: ModuleType) -> None:
    """Seven identical values out of eight put both quartiles inside the
    repeated run, so the column varies and still has no robust spread. The
    engine underneath would divide by one and say nothing, and a wrapper
    reading its patched spread would report a range of exactly one; the
    contract is the refusal."""
    RobustScaler = provided(backend, "RobustScaler")
    lopsided = [Feature("mostly_five", [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 9.0])]

    with pytest.raises(AllSameValuesError):
        RobustScaler().fit(lopsided)


def test_a_column_in_small_units_is_divided_by_its_own_tiny_spread(
    backend: ModuleType,
) -> None:
    """The same nine readings measured in units 1e16 times smaller are the
    same nine readings, and the scaler must answer the same ramp. An engine
    underneath substitutes a spread of one for any range below ten machine
    epsilons, which returns the column almost exactly as it arrived and calls
    its range 1.0, so a wrapper reading that patched number reports a spread
    2.5e15 times too large and nothing raises."""
    RobustScaler = provided(backend, "RobustScaler")
    model = RobustScaler().fit([Feature("in_small_units", _IN_SMALL_UNITS)])

    scaling = model.scalings["in_small_units"]
    transformed = model.transform([Feature("in_small_units", _IN_SMALL_UNITS)])

    assert scaling.centre == pytest.approx(SMALL_UNITS_CENTRE)
    assert scaling.spread == pytest.approx(SMALL_UNITS_SPREAD)
    assert np.allclose(column_of(transformed, "in_small_units"), SCALED_IN_SMALL_UNITS)


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    RobustScaler = provided(backend, "RobustScaler")

    with pytest.raises(NotFittedError):
        RobustScaler().transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = RobustScaler().scalings
