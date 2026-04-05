"""The contract every backend's LogisticRegression keeps.

The eight students of the numpy module's worked example. Their maximum
likelihood fit is known to four places, intercept -2.4383 and slope 0.8637,
and every backend reaching the same maximum by its own route has to land
there. The contract also holds both to what a logistic coefficient *means*:
the boundary at ``-intercept / slope`` and the odds multiplier ``exp(slope)``.
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

#: The maximum of the likelihood, from the numpy module's worked example.
INTERCEPT = -2.4383
SLOPE = 0.8637

#: Where p = 0.5, which is -intercept / slope, and exp(slope).
BOUNDARY = 2.823
ODDS_MULTIPLIER = 2.372

#: The maximum with the intercept held at zero, from an independent BFGS
#: minimisation of the log loss in scipy: 0.165246. Five times smaller than
#: the free slope, so a backend that quietly fitted an intercept anyway and
#: dropped it on the way out lands nowhere near it.
SLOPE_THROUGH_THE_ORIGIN = 0.1652

#: Enough epochs for a fixed-step ascent to settle to the tolerance.
MAX_EPOCHS = 100_000


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(threshold=0.4, max_epochs=50, tolerance=1e-6)

    assert model.threshold == pytest.approx(0.4)
    assert model.max_epochs == 50
    assert model.tolerance == pytest.approx(1e-6)
    assert model.fit_intercept is True


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_label_per_row(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    predictions = np.asarray(model.predict(FEATURES))

    assert predictions.shape == (len(_HOURS),)
    assert set(np.unique(predictions)) <= {0.0, 1.0}
    # Below the boundary fails, above it passes; the two rows nearest the
    # boundary are the overlapping ones and are left to the fit.
    assert np.array_equal(predictions[[0, 1, 2]], [0.0, 0.0, 0.0])
    assert np.array_equal(predictions[[5, 6, 7]], [1.0, 1.0, 1.0])


def test_its_probabilities_are_bounded_and_rise_with_hours(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    probabilities = model.predict_probability(FEATURES)

    assert isinstance(probabilities, Probabilities)
    values = np.asarray(probabilities)
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.diff(values) > 0.0)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.coefficients["hours"] == pytest.approx(SLOPE, abs=1e-3)
    assert model.intercept == pytest.approx(INTERCEPT, abs=1e-3)


def test_it_reads_its_coefficient_as_a_boundary_and_an_odds_multiplier(
    backend: ModuleType,
) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.decision_boundary_at("hours") == pytest.approx(BOUNDARY, abs=1e-2)
    assert model.odds_multiplier_for("hours") == pytest.approx(
        ODDS_MULTIPLIER, abs=1e-2
    )


def test_it_reports_that_it_settled(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.converged is True
    assert 1 <= model.epochs_run <= MAX_EPOCHS


def test_it_refuses_a_target_that_is_not_zero_or_one(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    three_classes = Feature("grade", [0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 1.0])

    with pytest.raises(NonBinaryLabelsError):
        LogisticRegression().fit(FEATURES, three_classes)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(max_epochs=MAX_EPOCHS).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(0.75)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")

    with pytest.raises(NotFittedError):
        LogisticRegression().predict(FEATURES)


def test_without_an_intercept_the_boundary_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    LogisticRegression = provided(backend, "LogisticRegression")
    model = LogisticRegression(fit_intercept=False, max_epochs=MAX_EPOCHS).fit(
        FEATURES, TARGET
    )

    assert model.intercept == 0.0
    assert model.coefficients["hours"] == pytest.approx(
        SLOPE_THROUGH_THE_ORIGIN, abs=1e-3
    )
    assert model.decision_boundary_at("hours") == pytest.approx(0.0)
