"""The contract every backend's PolynomialFeatures keeps.

The fixture is two columns of three values, ``x1 = [1, 2, 3]`` and
``x2 = [4, 5, 6]``, so every expanded column can be written by hand. At
degree 2 the five columns are ``x1, x2, x1^2, x1*x2, x2^2``, in that order,
which is ascending degree and then feature order, and the names are the
numpy backend's rule, ``^`` for a power and ``*`` between factors. Degree 3
adds ``x1^3, x1^2*x2, x1*x2^2, x2^3`` behind them, and it is there because
ordering is the claim that cannot be checked where there is only one mixed
term to place.

What the contract asserts beyond the arithmetic is the encapsulation. Same
constructor keywords. The terms are fixed at fit and addressable in order.
Transform matches by name, so the same columns in the other order give the
same answer, and a missing column is refused rather than guessed at, since
``x1*x2`` cannot be computed without both.
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

_FIRST = [1.0, 2.0, 3.0]
_SECOND = [4.0, 5.0, 6.0]
FEATURES = [Feature("x1", _FIRST), Feature("x2", _SECOND)]

DEGREE_TWO_NAMES = ["x1", "x2", "x1^2", "x1*x2", "x2^2"]
POWERS_ONLY_NAMES = ["x1", "x2", "x1^2", "x2^2"]
DEGREE_THREE_NAMES = [
    "x1",
    "x2",
    "x1^2",
    "x1*x2",
    "x2^2",
    "x1^3",
    "x1^2*x2",
    "x1*x2^2",
    "x2^3",
]
EXPECTED = {
    "x1": [1.0, 2.0, 3.0],
    "x2": [4.0, 5.0, 6.0],
    "x1^2": [1.0, 4.0, 9.0],
    "x1*x2": [4.0, 10.0, 18.0],
    "x2^2": [16.0, 25.0, 36.0],
    "x1^3": [1.0, 8.0, 27.0],
    "x1^2*x2": [4.0, 20.0, 54.0],
    "x1*x2^2": [16.0, 50.0, 108.0],
    "x2^3": [64.0, 125.0, 216.0],
}


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=3, include_interactions=False)

    assert model.degree == 3
    assert model.include_interactions is False


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=2)

    assert model.fit(FEATURES) is model


def test_its_terms_are_fixed_at_fit_in_column_order(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    terms = PolynomialFeatures(degree=2).fit(FEATURES).terms

    assert terms.n_terms == 5
    assert list(terms.names) == DEGREE_TWO_NAMES
    assert terms.source_feature_names == ("x1", "x2")


def test_it_expands_to_every_power_and_product(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=2).fit(FEATURES)

    expanded = model.transform(FEATURES)

    assert [feature.name for feature in expanded] == DEGREE_TWO_NAMES
    for name in DEGREE_TWO_NAMES:
        assert np.allclose(column_of(expanded, name), EXPECTED[name])


def test_it_matches_by_name_and_not_by_position(backend: ModuleType) -> None:
    """The same two columns in the other order must expand to the same
    columns in the fitted order, because ``x1*x2`` is one product whichever
    column arrives first and ``x1^2`` must not become the square of ``x2``."""
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=2).fit(FEATURES)

    expanded = model.transform([FEATURES[1], FEATURES[0]])

    assert [feature.name for feature in expanded] == DEGREE_TWO_NAMES
    assert np.allclose(column_of(expanded, "x1^2"), EXPECTED["x1^2"])
    assert np.allclose(column_of(expanded, "x2^2"), EXPECTED["x2^2"])


def test_without_interactions_only_the_powers_survive(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=2, include_interactions=False).fit(FEATURES)

    expanded = model.transform(FEATURES)

    assert list(model.terms.names) == POWERS_ONLY_NAMES
    assert [feature.name for feature in expanded] == POWERS_ONLY_NAMES
    for name in POWERS_ONLY_NAMES:
        assert np.allclose(column_of(expanded, name), EXPECTED[name])


def test_degree_three_keeps_the_columns_in_the_declared_order(
    backend: ModuleType,
) -> None:
    """Ordering is the assertion, and it needs a degree where the terms can
    be permuted. Degree 2 over two features has one mixed term, so a wrong
    ordering has almost nowhere to show; degree 3 has three of them, and every
    column's values are checked against its own name so that a table read in
    one order and named in another is caught rather than averaged away."""
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=3).fit(FEATURES)

    expanded = model.transform(FEATURES)

    assert list(model.terms.names) == DEGREE_THREE_NAMES
    assert [feature.name for feature in expanded] == DEGREE_THREE_NAMES
    for name in DEGREE_THREE_NAMES:
        assert np.allclose(column_of(expanded, name), EXPECTED[name])


def test_degree_one_returns_the_features_unchanged(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    expanded = PolynomialFeatures(degree=1).fit_transform(FEATURES)

    assert [feature.name for feature in expanded] == ["x1", "x2"]
    assert np.allclose(column_of(expanded, "x1"), _FIRST)
    assert np.allclose(column_of(expanded, "x2"), _SECOND)


def test_it_refuses_to_expand_with_a_column_missing(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    model = PolynomialFeatures(degree=2).fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([FEATURES[0]])


def test_it_refuses_a_constant_column_at_fit(backend: ModuleType) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")
    constant = [Feature("x1", _FIRST), Feature("flat", [2.0, 2.0, 2.0])]

    with pytest.raises(AllSameValuesError):
        PolynomialFeatures(degree=2).fit(constant)


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    PolynomialFeatures = provided(backend, "PolynomialFeatures")

    with pytest.raises(NotFittedError):
        PolynomialFeatures(degree=2).transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = PolynomialFeatures(degree=2).terms
