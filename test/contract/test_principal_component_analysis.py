"""The contract every backend's PrincipalComponentAnalysis keeps.

The fixture's decomposition was done by hand. Centred, its covariance is
``[[2.5, 1.5], [1.5, 2.5]]``, whose eigenvalues are ``2.5 +/- 1.5``, so the
variances are exactly 4.0 and 1.0 along ``(1, 1)/sqrt(2)`` and
``(1, -1)/sqrt(2)``, and the first component's coordinate for the row
centred at ``(2, 2)`` is ``4 / sqrt(2)``.

No test asserts a sign. An eigenvector and its negation are the same
direction, the two engines are free to return either, and everything below
compares absolute values or a quantity that is sign-free.
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
    TooFewValuesError,
)

from .harness import provided

_FIRST = [12.0, 8.0, 11.0, 9.0, 10.0]
_SECOND = [102.0, 98.0, 99.0, 101.0, 100.0]
FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]

VARIANCES = (4.0, 1.0)
TOTAL_VARIANCE = 5.0
SHARES = (0.8, 0.2)
LOADING = 1.0 / np.sqrt(2.0)

#: Centred rows are (2, 2), (-2, -2), (1, -1), (-1, 1), (0, 0); projected on
#: (1, 1)/sqrt(2) the first coordinate is (a + b)/sqrt(2).
FIRST_COORDINATES = np.array([4.0, -4.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)


def column_of(features: list[Feature], name: str) -> np.ndarray:
    """One named column out of a transform's output."""
    return next(feature.values for feature in features if feature.name == name)


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis(n_components=1, standardize=True)

    assert model.n_components == 1
    assert model.standardize is True


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis()

    assert model.fit(FEATURES) is model


def test_its_components_carry_the_known_variances_in_order(
    backend: ModuleType,
) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    components = PrincipalComponentAnalysis().fit(FEATURES).components

    assert components.n_components == 2
    assert [one.variance for one in components] == pytest.approx(list(VARIANCES))
    assert components.total_variance == pytest.approx(TOTAL_VARIANCE)
    assert components.variance_shares == pytest.approx(SHARES)


def test_its_loadings_are_addressable_by_feature_name(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    first = PrincipalComponentAnalysis().fit(FEATURES).components["component_1"]

    assert abs(first.loading_for("first")) == pytest.approx(LOADING)
    assert abs(first.loading_for("second")) == pytest.approx(LOADING)


def test_it_transforms_onto_named_components(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis().fit(FEATURES)

    transformed = model.transform(FEATURES)

    assert [feature.name for feature in transformed] == ["component_1", "component_2"]
    assert np.allclose(
        np.abs(column_of(transformed, "component_1")), np.abs(FIRST_COORDINATES)
    )


def test_truncation_keeps_the_leading_component(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis(n_components=1).fit(FEATURES)

    assert model.components.n_components == 1
    assert model.components.cumulative_shares == pytest.approx((0.8,))
    assert [feature.name for feature in model.transform(FEATURES)] == ["component_1"]


def test_inverse_transform_undoes_a_full_transform(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis().fit(FEATURES)

    restored = model.inverse_transform(model.transform(FEATURES))

    assert [feature.name for feature in restored] == ["first", "second"]
    assert np.allclose(column_of(restored, "first"), _FIRST)
    assert np.allclose(column_of(restored, "second"), _SECOND)


#: Standardised by the population deviation, each column has sample variance
#: ``n / (n - 1) = 5 / 4``, so the covariance is ``5/4 * [[1, .6], [.6, 1]]``
#: and its eigenvalues are ``5/4 * (1 +/- 0.6)``.
STANDARDIZED_VARIANCES = (2.0, 0.5)


def test_standardizing_decomposes_the_correlation_instead(backend: ModuleType) -> None:
    """Standardised, the covariance becomes the correlation ``[[1, .6], [.6, 1]]``
    up to one scale, so the shares are 0.8 and 0.2 by a different route and
    the round trip still lands on the original units.

    The shares alone cannot tell this fit from an unstandardised one, since
    both columns of the fixture have the same variance and scaling them
    together changes nothing but the scale. The variances can: 4.0 and 1.0
    unscaled, 2.0 and 0.5 scaled. A wrapper that ignored ``standardize``
    passed the first version of this test.
    """
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis(standardize=True).fit(FEATURES)

    assert model.components.variance_shares == pytest.approx(SHARES)
    assert [one.variance for one in model.components] == pytest.approx(
        list(STANDARDIZED_VARIANCES)
    )
    assert model.components.total_variance == pytest.approx(sum(STANDARDIZED_VARIANCES))

    restored = model.inverse_transform(model.transform(FEATURES))

    assert np.allclose(column_of(restored, "first"), _FIRST)
    assert np.allclose(column_of(restored, "second"), _SECOND)


def test_writing_into_the_learned_directions_does_not_move_them(
    backend: ModuleType,
) -> None:
    """A fitted model's answers do not change because a caller wrote into an
    array it handed out. ``directions`` and ``direction`` are both writeable
    float arrays and both are written into here.

    Both are rebuilt from the loadings on every read today, so the write lands
    nowhere. The claim is worth stating anyway, because caching that matrix is
    the obvious optimisation and it turns the read into a handout. Measured
    against a version that builds ``directions`` once and returns it again,
    the second read comes back full of 999.0 and this test goes red on both
    backends.
    """
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis().fit(FEATURES)
    before = column_of(model.transform(FEATURES), "component_1")

    model.components.directions[:] = 999.0
    model.components["component_1"].direction[:] = 999.0

    assert np.allclose(column_of(model.transform(FEATURES), "component_1"), before)
    assert abs(model.components["component_1"].loading_for("first")) == pytest.approx(
        LOADING
    )
    assert np.allclose(np.abs(model.components.directions), LOADING)


def test_a_refused_refit_leaves_the_earlier_fit_intact(backend: ModuleType) -> None:
    """Compute into locals, assign at the end, checked rather than intended.

    The constant column is refused while the fit is learning how to prepare
    its rows, which is the step that both writes the means the transform will
    subtract and, on one backend, fits a standardizer. A refit that raises
    there must leave the model as the last successful fit left it.

    One deeper refusal is deliberately not the one used here. Refitting on
    data with no spread at all is refused by both backends and leaves only the
    scikit backend intact, because the numpy backend records the flat data's
    means before the components refuse, so its transform of the original rows
    comes back at 77.781746 where the earlier fit answered 2.828427. That
    divergence is pinned on the backend that keeps it, in
    ``test/scikit/test_unsupervised_translation.py``, rather than asserted
    here where it would be red.
    """
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis(standardize=True).fit(FEATURES)
    before = column_of(model.transform(FEATURES), "component_1")
    constant = [Feature("first", [1.0, 2.0, 3.0]), Feature("second", [4.0, 4.0, 4.0])]

    with pytest.raises(AllSameValuesError):
        model.fit(constant)

    assert model.is_fitted
    assert model.components.n_components == 2
    assert [one.variance for one in model.components] == pytest.approx(
        list(STANDARDIZED_VARIANCES)
    )
    assert np.allclose(column_of(model.transform(FEATURES), "component_1"), before)


def test_it_refuses_a_constant_column_when_standardizing(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    constant = [Feature("first", _FIRST), Feature("flat", [3.0] * 5)]

    with pytest.raises(AllSameValuesError):
        PrincipalComponentAnalysis(standardize=True).fit(constant)


def test_it_refuses_more_components_than_features(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")

    with pytest.raises(InvalidValuesError):
        PrincipalComponentAnalysis(n_components=3).fit(FEATURES)


def test_it_refuses_a_single_row(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")

    with pytest.raises(TooFewValuesError):
        PrincipalComponentAnalysis().fit(
            [Feature("first", [1.0]), Feature("second", [2.0])]
        )


def test_it_refuses_data_with_no_spread_at_all(backend: ModuleType) -> None:
    """Every column constant leaves no variance to share out, and the refusal
    is this library's rather than a division-by-zero warning from an engine."""
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    flat = [Feature("first", [3.0, 3.0, 3.0]), Feature("second", [1.0, 1.0, 1.0])]

    with pytest.raises(InvalidValuesError):
        PrincipalComponentAnalysis().fit(flat)


def test_it_refuses_a_query_over_the_wrong_features(backend: ModuleType) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")
    model = PrincipalComponentAnalysis().fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.transform([FEATURES[0]])


def test_it_refuses_to_transform_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    PrincipalComponentAnalysis = provided(backend, "PrincipalComponentAnalysis")

    with pytest.raises(NotFittedError):
        PrincipalComponentAnalysis().transform(FEATURES)

    with pytest.raises(NotFittedError):
        _ = PrincipalComponentAnalysis().components
