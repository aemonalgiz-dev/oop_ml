"""The contract every backend's MultipleLinearRegression keeps.

The fixture is an exact plane, so the answer is known to the last decimal a
solver can reach, and the contract asks for it to a tolerance a closed-form
solve on five rows cannot miss.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: y = 1 + 2 * first + 3 * second, exactly, over the five rows the numpy
#: backend's own worked example uses.
_FIRST = np.array([1.0, 1.0, 2.0, 0.0, 3.0])
_SECOND = np.array([1.0, 2.0, 2.0, 1.0, 0.0])
_TARGETS = 1.0 + 2.0 * _FIRST + 3.0 * _SECOND
FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
TARGET = Feature("target", _TARGETS)

#: Forced through the origin the plane cannot be the one the fixture was
#: built from, and the answer is still exact. X'X is [[15, 7], [7, 10]] and
#: X'y is [58, 50], whose determinant is 101, so the coefficients are
#: 230 / 101 and 344 / 101.
THROUGH_THE_ORIGIN_FIRST = 230.0 / 101.0
THROUGH_THE_ORIGIN_SECOND = 344.0 / 101.0


def test_it_is_constructed_by_the_same_keyword(backend: ModuleType) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")

    assert MultipleLinearRegression(fit_intercept=False).fit_intercept is False


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = MultipleLinearRegression()

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = MultipleLinearRegression().fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=1e-6)


def test_it_predicts_a_row_it_never_saw_from_the_plane(backend: ModuleType) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = MultipleLinearRegression().fit(FEATURES, TARGET)

    prediction = model.predict([Feature("second", [0.0]), Feature("first", [10.0])])

    assert prediction[0] == pytest.approx(21.0, abs=1e-6)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = MultipleLinearRegression().fit(FEATURES, TARGET)

    assert model.intercept == pytest.approx(1.0, abs=1e-6)
    assert model.coefficients["first"] == pytest.approx(2.0, abs=1e-6)
    assert model.coefficients["second"] == pytest.approx(3.0, abs=1e-6)


def test_without_an_intercept_the_plane_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    """The keyword above is read back and never fitted, which is not enough.

    A wrapper that dropped the flag on the way to its solver answers the
    fixture's own plane, 2.0 and 3.0, and reports ``intercept`` as 0.0 while
    doing it, so nothing a caller reads announces the loss. The
    through-the-origin answer is a different pair of numbers and is the only
    thing that separates the two.
    """
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = MultipleLinearRegression(fit_intercept=False).fit(FEATURES, TARGET)

    assert model.intercept == 0.0
    assert model.coefficients["first"] == pytest.approx(
        THROUGH_THE_ORIGIN_FIRST, abs=1e-6
    )
    assert model.coefficients["second"] == pytest.approx(
        THROUGH_THE_ORIGIN_SECOND, abs=1e-6
    )


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = MultipleLinearRegression().fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")

    with pytest.raises(NotFittedError):
        MultipleLinearRegression().predict(FEATURES)
