"""The contract every backend's NewtonLogisticRegression keeps.

The same eight students as the gradient model, because the point of the
Newton model is that it reaches the identical maximum in single-digit
iterations. The contract holds both backends to the known answer and to
having got there in far fewer steps than an ascent would need.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import Probabilities
from oop_ml.core.exceptions import NonBinaryLabelsError, NotFittedError

from .harness import provided

_HOURS = np.array([0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0])
_PASSED = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
FEATURES = [Feature("hours", _HOURS)]
TARGET = Feature("passed", _PASSED)

INTERCEPT = -2.4383
SLOPE = 0.8637
BOUNDARY = 2.823
ODDS_MULTIPLIER = 2.372

#: Quadratic convergence settles in about six steps on any well-posed data.
GENEROUS_ITERATION_CAP = 30

#: The maximum with the intercept held at zero, the same number the gradient
#: model's spec pins, from an independent BFGS minimisation of the log loss
#: in scipy. Newton reaches that maximum by another route, and measured it
#: lands on 0.165246 in both backends. Five times smaller than the free
#: slope, so a backend that quietly fitted an intercept anyway and dropped it
#: on the way out lands nowhere near it.
SLOPE_THROUGH_THE_ORIGIN = 0.1652


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression(max_iterations=7, tolerance=1e-6, threshold=0.3)

    assert model.max_iterations == 7
    assert model.tolerance == pytest.approx(1e-6)
    assert model.threshold == pytest.approx(0.3)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression()

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_label_per_row(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression().fit(FEATURES, TARGET)

    predictions = np.asarray(model.predict(FEATURES))

    assert predictions.shape == (len(_HOURS),)
    assert np.array_equal(predictions[[0, 1, 2]], [0.0, 0.0, 0.0])
    assert np.array_equal(predictions[[5, 6, 7]], [1.0, 1.0, 1.0])


def test_its_probabilities_are_bounded_and_rise_with_hours(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression().fit(FEATURES, TARGET)

    probabilities = model.predict_probability(FEATURES)

    assert isinstance(probabilities, Probabilities)
    values = np.asarray(probabilities)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.diff(values) > 0.0)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression().fit(FEATURES, TARGET)

    assert model.coefficients["hours"] == pytest.approx(SLOPE, abs=1e-3)
    assert model.intercept == pytest.approx(INTERCEPT, abs=1e-3)


def test_it_reads_its_coefficient_as_a_boundary_and_an_odds_multiplier(
    backend: ModuleType,
) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression().fit(FEATURES, TARGET)

    assert model.decision_boundary_at("hours") == pytest.approx(BOUNDARY, abs=1e-2)
    assert model.odds_multiplier_for("hours") == pytest.approx(
        ODDS_MULTIPLIER, abs=1e-2
    )


def test_it_settles_in_a_handful_of_iterations(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression().fit(FEATURES, TARGET)

    assert model.converged is True
    assert 1 <= model.iterations_run <= GENEROUS_ITERATION_CAP


def test_it_refuses_a_target_that_is_not_zero_or_one(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    three_classes = Feature("grade", [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0])

    with pytest.raises(NonBinaryLabelsError):
        NewtonLogisticRegression().fit(FEATURES, three_classes)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression().fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(0.75)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")

    with pytest.raises(NotFittedError):
        NewtonLogisticRegression().predict(FEATURES)


def test_without_an_intercept_the_boundary_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    """The gradient model's spec has this one and the Newton model needs it too.

    The two wrappers build separate engine prototypes, so the flag is passed
    on at two call sites and the other spec covers neither of them here.
    """
    NewtonLogisticRegression = provided(backend, "NewtonLogisticRegression")
    model = NewtonLogisticRegression(fit_intercept=False).fit(FEATURES, TARGET)

    assert model.intercept == 0.0
    assert model.coefficients["hours"] == pytest.approx(
        SLOPE_THROUGH_THE_ORIGIN, abs=1e-3
    )
    assert model.decision_boundary_at("hours") == pytest.approx(0.0)
