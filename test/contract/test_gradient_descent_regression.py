"""The contract every backend's GradientDescentRegression keeps.

A walk rather than a jump, so every assertion here is approximate by nature,
and the tolerance is the one the numpy backend's worked example reaches at
this learning rate and epoch cap. A backend that declines the model skips
with the reason it wrote down.
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


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")
    model = GradientDescentRegression(learning_rate=0.05, max_epochs=5_000)

    assert model.learning_rate == pytest.approx(0.05)
    assert model.max_epochs == 5_000


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")
    model = GradientDescentRegression(learning_rate=0.05, max_epochs=5_000)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")
    model = GradientDescentRegression(learning_rate=0.05, max_epochs=5_000).fit(
        FEATURES, TARGET
    )

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=1e-3)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")
    model = GradientDescentRegression(learning_rate=0.05, max_epochs=5_000).fit(
        FEATURES, TARGET
    )

    assert model.intercept == pytest.approx(1.0, abs=1e-3)
    assert model.coefficients["first"] == pytest.approx(2.0, abs=1e-3)
    assert model.coefficients["second"] == pytest.approx(3.0, abs=1e-3)


def test_it_reports_how_the_walk_ended(backend: ModuleType) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")
    model = GradientDescentRegression(learning_rate=0.05, max_epochs=5_000).fit(
        FEATURES, TARGET
    )

    assert model.converged is True
    assert 0 < model.epochs_run <= 5_000


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")
    model = GradientDescentRegression(learning_rate=0.05, max_epochs=5_000).fit(
        FEATURES, TARGET
    )

    assert model.score(FEATURES, TARGET) > 0.999


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    GradientDescentRegression = provided(backend, "GradientDescentRegression")

    with pytest.raises(NotFittedError):
        GradientDescentRegression().predict(FEATURES)
