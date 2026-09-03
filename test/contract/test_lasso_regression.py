"""The contract every backend's LassoRegression keeps.

Two things separate a lasso from a ridge, and both are asserted. At a small
penalty it recovers the plane the fixture was built from. At a penalty large
enough, every coefficient is exactly zero and the intercept is the target's
mean, which follows from the definition rather than from either backend:
once nothing else is fitted, the only unpenalised parameter has to carry the
whole level.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: y = 1 + 2 * first + 3 * second, exactly.
_FIRST = np.array([1.0, 1.0, 2.0, 0.0, 3.0])
_SECOND = np.array([1.0, 2.0, 2.0, 1.0, 0.0])
_TARGETS = 1.0 + 2.0 * _FIRST + 3.0 * _SECOND
FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
TARGET = Feature("target", _TARGETS)

#: Above this, on this fixture, both coefficients are driven to zero. The
#: intercept is then the target mean, 37 / 5.
SELECTING_EVERYTHING_OUT = 16.0
TARGET_MEAN = 7.4

#: Forced through the origin at a penalty small enough to shrink almost
#: nothing, the answer is the least-squares plane through the origin. X'X is
#: [[15, 7], [7, 10]] and X'y is [58, 50], whose determinant is 101, so the
#: coefficients are 230 / 101 and 344 / 101 less what the penalty takes.
BARELY_PENALISED = 1e-4
THROUGH_THE_ORIGIN_FIRST = 230.0 / 101.0
THROUGH_THE_ORIGIN_SECOND = 344.0 / 101.0


def test_it_is_constructed_by_the_same_keyword(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")

    assert LassoRegression(penalty=0.5).penalty == pytest.approx(0.5)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=0.01)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=0.01).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=0.05)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=0.01).fit(FEATURES, TARGET)

    assert model.coefficients["first"] == pytest.approx(2.0, abs=0.05)
    assert model.coefficients["second"] == pytest.approx(3.0, abs=0.05)


def test_a_large_enough_penalty_selects_every_feature_out(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=SELECTING_EVERYTHING_OUT).fit(FEATURES, TARGET)

    assert model.coefficients["first"] == pytest.approx(0.0, abs=1e-12)
    assert model.coefficients["second"] == pytest.approx(0.0, abs=1e-12)
    assert model.intercept == pytest.approx(TARGET_MEAN)


def test_without_an_intercept_the_plane_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    """``fit_intercept`` is a keyword this spec otherwise never states.

    The penalty tests above all run at the default, and the intercept they
    assert, the target mean, is what an intercept-fitting lasso gives, so a
    wrapper that hardcoded the flag satisfies every one of them. Through the
    origin the coefficients move to a different pair of numbers, which is
    what tells the two apart.
    """
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=BARELY_PENALISED, fit_intercept=False).fit(
        FEATURES, TARGET
    )

    assert model.intercept == 0.0
    assert model.coefficients["first"] == pytest.approx(
        THROUGH_THE_ORIGIN_FIRST, abs=1e-4
    )
    assert model.coefficients["second"] == pytest.approx(
        THROUGH_THE_ORIGIN_SECOND, abs=1e-4
    )


def test_it_reports_how_the_sweeps_ended(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=0.01).fit(FEATURES, TARGET)

    assert model.converged is True
    assert isinstance(model.iterations_run, int)
    assert model.iterations_run <= model.max_iterations


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    LassoRegression = provided(backend, "LassoRegression")
    model = LassoRegression(penalty=0.01).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) > 0.99


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    LassoRegression = provided(backend, "LassoRegression")

    with pytest.raises(NotFittedError):
        LassoRegression(penalty=0.01).predict(FEATURES)
