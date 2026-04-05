"""The contract every backend's SimpleLinearRegression keeps.

One predictor, one slope, one intercept, and the two are read back by name
rather than out of an array. The numbers are checked against the line the
fixture was built from, not against the other backend.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: y = 2 * x + 1, with a little noise, so the slope and intercept are known.
_INPUTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_TARGETS = (
    2.0 * _INPUTS + 1.0 + np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.02, -0.01])
)


def test_it_is_constructed_without_arguments(backend: ModuleType) -> None:
    SimpleLinearRegression = provided(backend, "SimpleLinearRegression")

    assert not SimpleLinearRegression().is_fitted


def test_it_fits_two_columns_and_returns_itself(backend: ModuleType) -> None:
    SimpleLinearRegression = provided(backend, "SimpleLinearRegression")
    model = SimpleLinearRegression()

    assert model.fit(_INPUTS, _TARGETS) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    SimpleLinearRegression = provided(backend, "SimpleLinearRegression")
    model = SimpleLinearRegression().fit(_INPUTS, _TARGETS)

    predictions = model.predict(_INPUTS)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=0.1)


def test_its_slope_and_intercept_are_read_back_by_name(backend: ModuleType) -> None:
    SimpleLinearRegression = provided(backend, "SimpleLinearRegression")
    model = SimpleLinearRegression().fit(_INPUTS, _TARGETS)

    assert model.slope == pytest.approx(2.0, abs=0.05)
    assert model.intercept == pytest.approx(1.0, abs=0.1)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    SimpleLinearRegression = provided(backend, "SimpleLinearRegression")
    model = SimpleLinearRegression().fit(_INPUTS, _TARGETS)

    assert model.score(_INPUTS, _TARGETS) > 0.99


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    SimpleLinearRegression = provided(backend, "SimpleLinearRegression")

    with pytest.raises(NotFittedError):
        SimpleLinearRegression().predict(_INPUTS)
