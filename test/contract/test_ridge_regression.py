"""The contract every backend's RidgeRegression keeps, and the template for the rest.

What a contract test asserts, and what it deliberately does not
------------------------------------------------------------------
Two backends solving ridge regression will not agree to the last bit, and the
contract must not ask them to. scikit-learn's Ridge reaches the same normal
equations by a different solver, so coefficients agree to floating point and
not to identity, and a contract written as ``==`` would fail on correct code.

What must be identical is the *encapsulation*. Same constructor keyword. Fit
takes a sequence of Features and a Feature. Predict hands back Predictions,
one per row. Coefficients are addressable by feature name. An unfitted model
refuses to predict with the library's own exception, not the engine's. Those
are the promises a caller relies on, and they are what a swap at the call site
depends on. The numbers are checked against the fixture's known answer, which
both backends must recover, rather than against each other.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: y = 3 * area + 2 * baths + 1, with a little noise, so the answer is known.
_AREAS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_BATHS = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0])
_PRICES = (
    3.0 * _AREAS
    + 2.0 * _BATHS
    + 1.0
    + np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01])
)
FEATURES = [Feature("area", _AREAS), Feature("baths", _BATHS)]
TARGET = Feature("price", _PRICES)

#: A penalty large enough on eight rows to pull both coefficients well away
#: from the plane the fixture was built from.
HEAVY_PENALTY = 50.0

#: What ``(Xc' Xc + penalty I) b = Xc' yc`` gives on the centred columns,
#: which is the definition of the fit both backends make with an intercept
#: left out of the norm. Solved once and written down, since a test that
#: rebuilds the implementation's own arithmetic asserts nothing.
COEFFICIENTS_AT_A_LIGHT_PENALTY = {"area": 2.968257952, "baths": 2.059924172}
COEFFICIENTS_AT_A_HEAVY_PENALTY = {"area": 1.631640625, "baths": 0.789203125}

#: The same system on the raw columns, which is what the fit without an
#: intercept solves, at the light penalty.
THROUGH_THE_ORIGIN = {"area": 2.691945673, "baths": 2.897783302}


def test_it_is_constructed_by_the_same_keyword(backend: ModuleType) -> None:
    RidgeRegression = provided(backend, "RidgeRegression")

    assert RidgeRegression(penalty=0.5).penalty == pytest.approx(0.5)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    RidgeRegression = provided(backend, "RidgeRegression")
    model = RidgeRegression(penalty=0.01)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    RidgeRegression = provided(backend, "RidgeRegression")
    model = RidgeRegression(penalty=0.01).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_PRICES)
    assert np.allclose(np.asarray(predictions), _PRICES, atol=0.2)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    RidgeRegression = provided(backend, "RidgeRegression")
    model = RidgeRegression(penalty=0.01).fit(FEATURES, TARGET)

    assert model.coefficients["area"] == pytest.approx(3.0, abs=0.1)
    assert model.coefficients["baths"] == pytest.approx(2.0, abs=0.1)


def test_a_large_penalty_shrinks_every_coefficient(backend: ModuleType) -> None:
    """The one field this model has, made to change an answer.

    Everything above fits at 0.01 and checks the coefficients to ``abs=0.1``,
    which a wrapper handing its engine a penalty of zero also passes, so
    ``penalty`` was a constructor round-trip as far as this spec went. The
    two answers here are far enough apart that no single number satisfies
    both, and each is the closed form rather than the other backend's answer.
    """
    RidgeRegression = provided(backend, "RidgeRegression")
    light = RidgeRegression(penalty=0.01).fit(FEATURES, TARGET)
    heavy = RidgeRegression(penalty=HEAVY_PENALTY).fit(FEATURES, TARGET)

    for name, expected in COEFFICIENTS_AT_A_LIGHT_PENALTY.items():
        assert light.coefficients[name] == pytest.approx(expected, abs=1e-6)
    for name, expected in COEFFICIENTS_AT_A_HEAVY_PENALTY.items():
        assert heavy.coefficients[name] == pytest.approx(expected, abs=1e-6)
        assert abs(heavy.coefficients[name]) < abs(light.coefficients[name])


def test_without_an_intercept_the_plane_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    """``fit_intercept`` is the other field, and it is fitted nowhere else here.

    A wrapper that dropped the flag reports ``intercept`` as 0.0 anyway,
    because the frame writes that when no intercept was asked for, so only
    the coefficients can tell the two apart.
    """
    RidgeRegression = provided(backend, "RidgeRegression")
    model = RidgeRegression(penalty=0.01, fit_intercept=False).fit(FEATURES, TARGET)

    assert model.intercept == 0.0
    for name, expected in THROUGH_THE_ORIGIN.items():
        assert model.coefficients[name] == pytest.approx(expected, abs=1e-6)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    RidgeRegression = provided(backend, "RidgeRegression")
    model = RidgeRegression(penalty=0.01).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) > 0.99


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    """The engine underneath has its own not-fitted error. A caller who swapped
    backends must not have to catch a different exception."""
    RidgeRegression = provided(backend, "RidgeRegression")

    with pytest.raises(NotFittedError):
        RidgeRegression(penalty=0.01).predict(FEATURES)
