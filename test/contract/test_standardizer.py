"""The contract every backend's Standardizer keeps.

The fixture's statistics were worked by hand. ``[2, 4, 4, 4, 5, 5, 7, 9]`` has
mean 5 and, dividing by ``n`` rather than ``n - 1``, a standard deviation of
exactly 2, so the standardized column is ``(x - 5) / 2``. The second column is
the first shifted and stretched, ``10 * x + 1``, which keeps every number in
the answer exact.

A third fixture is nine values a few last bits apart above 1.0, whose spread
is a real 1.3009e-15. An engine underneath judges such a column constant and
substitutes a spread of one, so the assertion is that the tiny spread is the
one reported and the one divided by. Its tolerance is looser than the rest of
this module's, at a thousandth rather than a millionth, and deliberately so.
Subtracting a mean of 1.0 from values that differ from it in the last bits is
catastrophic cancellation, so the two backends' variance routines land
1.8e-4 apart on that column, which is precisely the case the contract says is
checked against a known answer rather than against the other backend.

What the contract asserts is the encapsulation. No constructor arguments.
Fit takes a sequence of Features. Transform hands back Features keeping their
names, a subset of the fitted ones is accepted in any order, an unknown one
is refused, a constant column is refused at fit, and an unfitted model refuses
with the library's own exception rather than the engine's.
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

_AGES = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
_SCORES = 10.0 * _AGES + 1.0
FEATURES = [Feature("age", _AGES), Feature("score", _SCORES)]

AGE_MEAN, AGE_DEVIATION = 5.0, 2.0
SCORE_MEAN, SCORE_DEVIATION = 51.0, 20.0
STANDARDIZED = (_AGES - AGE_MEAN) / AGE_DEVIATION

_ALMOST_FLAT = [1.0 + position * 5e-16 for position in range(9)]
ALMOST_FLAT_DEVIATION = 1.3009e-15
CANCELLATION_TOLERANCE = 1e-3


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_without_arguments(backend: ModuleType) -> None:
    Standardizer = provided(backend, "Standardizer")

    assert Standardizer().is_fitted is False


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    Standardizer = provided(backend, "Standardizer")
    model = Standardizer()

    assert model.fit(FEATURES) is model


def test_its_scalings_are_addressable_by_feature_name(backend: ModuleType) -> None:
    Standardizer = provided(backend, "Standardizer")
    scalings = Standardizer().fit(FEATURES).scalings

    assert scalings.n_features == 2
    assert scalings["age"].mean == pytest.approx(AGE_MEAN)
    assert scalings["age"].standard_deviation == pytest.approx(AGE_DEVIATION)
    assert scalings["score"].mean == pytest.approx(SCORE_MEAN)
    assert scalings["score"].standard_deviation == pytest.approx(SCORE_DEVIATION)


def test_it_transforms_every_feature_to_the_known_answer(backend: ModuleType) -> None:
    Standardizer = provided(backend, "Standardizer")
    model = Standardizer().fit(FEATURES)

    transformed = model.transform(FEATURES)

    assert [feature.name for feature in transformed] == ["age", "score"]
    assert np.allclose(column_of(transformed, "age"), STANDARDIZED)
    assert np.allclose(column_of(transformed, "score"), STANDARDIZED)


def test_fit_transform_agrees_with_fit_then_transform(backend: ModuleType) -> None:
    Standardizer = provided(backend, "Standardizer")

    transformed = Standardizer().fit_transform(FEATURES)

    assert np.allclose(column_of(transformed, "age"), STANDARDIZED)


def test_it_accepts_a_subset_of_the_fitted_features_in_any_order(
    backend: ModuleType,
) -> None:
    """The statistics belong to the training rows, and a held-out column is
    rescaled by them whichever column it is and whatever order it arrives in."""
    Standardizer = provided(backend, "Standardizer")
    model = Standardizer().fit(FEATURES)

    only_score = model.transform([Feature("score", _SCORES)])
    reversed_order = model.transform([FEATURES[1], FEATURES[0]])

    assert [feature.name for feature in only_score] == ["score"]
    assert np.allclose(column_of(only_score, "score"), STANDARDIZED)
    assert [feature.name for feature in reversed_order] == ["score", "age"]
    assert np.allclose(column_of(reversed_order, "age"), STANDARDIZED)


def test_it_refuses_a_feature_it_never_learned(backend: ModuleType) -> None:
    Standardizer = provided(backend, "Standardizer")
    model = Standardizer().fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([Feature("height", _AGES)])


def test_it_refuses_a_constant_column_at_fit(backend: ModuleType) -> None:
    """A zero spread has nothing to divide by. The engine underneath would
    substitute a spread of one and carry on; the contract is the refusal."""
    Standardizer = provided(backend, "Standardizer")
    constant = [Feature("age", _AGES), Feature("flat", [3.0] * len(_AGES))]

    with pytest.raises(AllSameValuesError):
        Standardizer().fit(constant)


def test_an_almost_flat_column_is_divided_by_its_own_tiny_spread(
    backend: ModuleType,
) -> None:
    """A column that varies only in the last bits of 1.0 still varies, and
    standardizing it is still a division by 1.3009e-15. The engine underneath
    judges it constant on a relative bound and substitutes a spread of one,
    which would report a spread 7.7e14 times too large and hand the column
    back essentially unchanged, so the standardized column would keep a
    spread of 1.3e-15 where it should carry a spread of one."""
    Standardizer = provided(backend, "Standardizer")
    model = Standardizer().fit([Feature("almost_flat", _ALMOST_FLAT)])

    scaling = model.scalings["almost_flat"]
    standardized = column_of(
        model.transform([Feature("almost_flat", _ALMOST_FLAT)]), "almost_flat"
    )

    assert scaling.standard_deviation == pytest.approx(
        ALMOST_FLAT_DEVIATION, rel=CANCELLATION_TOLERANCE
    )
    assert float(np.std(standardized)) == pytest.approx(1.0, rel=CANCELLATION_TOLERANCE)


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    Standardizer = provided(backend, "Standardizer")

    with pytest.raises(NotFittedError):
        Standardizer().transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = Standardizer().scalings
