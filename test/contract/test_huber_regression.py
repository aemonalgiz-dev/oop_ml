"""The contract every backend's HuberRegression keeps.

The reason this model exists is a comparison, so the contract makes it: on a
plane with four of forty targets shifted by thirty, ordinary least squares is
dragged well off and this is not. Both backends are held to recovering the
plane the fixture was built from while the least-squares fit on the same rows
does not.

The threshold is in units of a fitted scale
---------------------------------------------
Which is the one thing here that looks like it needs no translation and does.
A residual's size depends on the target's units, so a fixed threshold would
make the same model behave differently on metres and millimetres; both backends
estimate a scale and measure the threshold in multiples of it, and the contract
reads that scale back.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

_GENERATOR = np.random.default_rng(5)
_ROWS = _GENERATOR.normal(size=(40, 2))
_CLEAN_TARGETS = (
    1.0 + 2.0 * _ROWS[:, 0] - 3.0 * _ROWS[:, 1] + _GENERATOR.normal(scale=0.2, size=40)
)
_SPOILED = _CLEAN_TARGETS.copy()
_SPOILED[[3, 11, 27, 34]] += 30.0

FEATURES = [Feature("first", _ROWS[:, 0]), Feature("second", _ROWS[:, 1])]
CLEAN_TARGET = Feature("target", _CLEAN_TARGETS)
SPOILED_TARGET = Feature("target", _SPOILED)

#: What the plane was built from, before four rows were spoiled.
TRUE_INTERCEPT = 1.0
TRUE_FIRST = 2.0
TRUE_SECOND = -3.0


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    HuberRegression = provided(backend, "HuberRegression")

    model = HuberRegression(threshold=2.0, penalty=0.5)

    assert model.threshold == pytest.approx(2.0)
    assert model.penalty == pytest.approx(0.5)


def test_it_defaults_to_the_usual_threshold_and_no_penalty(
    backend: ModuleType,
) -> None:
    """1.35 is where the fit keeps about 95% of least squares' efficiency on
    data that really is Gaussian. The penalty defaults to nothing, because a
    regression that is not a penalised one should not have one switched on."""
    HuberRegression = provided(backend, "HuberRegression")

    model = HuberRegression()

    assert model.threshold == pytest.approx(1.35)
    assert model.penalty == 0.0


@pytest.mark.parametrize("outside", [1.0, 0.5, -1.0])
def test_it_refuses_a_threshold_at_or_below_one(
    backend: ModuleType, outside: float
) -> None:
    """At one and below there is no room for the quadratic piece to exist, so
    the model would not be the one the name describes."""
    HuberRegression = provided(backend, "HuberRegression")

    with pytest.raises(ValidationError):
        HuberRegression(threshold=outside)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression()

    assert model.fit(FEATURES, SPOILED_TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression().fit(FEATURES, SPOILED_TARGET)

    assert len(model.predict(FEATURES)) == len(_SPOILED)


def test_it_recovers_the_plane_the_outliers_were_added_to(
    backend: ModuleType,
) -> None:
    """The whole point. Four of forty targets are thirty too large, and the fit
    still lands on the plane the other thirty-six sit on."""
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression().fit(FEATURES, SPOILED_TARGET)

    assert model.intercept == pytest.approx(TRUE_INTERCEPT, abs=0.15)
    assert model.coefficients["first"] == pytest.approx(TRUE_FIRST, abs=0.15)
    assert model.coefficients["second"] == pytest.approx(TRUE_SECOND, abs=0.15)


def test_where_least_squares_on_the_same_rows_does_not(backend: ModuleType) -> None:
    """The comparison that makes the test above mean something. Squared error
    lets a row ten times as far off pull a hundred times as hard, so four rows
    out of forty are enough to move the answer."""
    HuberRegression = provided(backend, "HuberRegression")
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")

    robust = HuberRegression().fit(FEATURES, SPOILED_TARGET)
    ordinary = MultipleLinearRegression().fit(FEATURES, SPOILED_TARGET)

    assert abs(ordinary.intercept - TRUE_INTERCEPT) > 2.0
    assert abs(ordinary.coefficients["second"] - TRUE_SECOND) > 1.0
    assert abs(robust.intercept - TRUE_INTERCEPT) < abs(
        ordinary.intercept - TRUE_INTERCEPT
    )


def test_on_clean_data_it_costs_almost_nothing(backend: ModuleType) -> None:
    """The other half of the trade. Robustness is insurance, and the premium at
    the usual threshold is small: on data with no outliers at all the two fits
    are within a hundredth of each other."""
    HuberRegression = provided(backend, "HuberRegression")
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")

    robust = HuberRegression().fit(FEATURES, CLEAN_TARGET)
    ordinary = MultipleLinearRegression().fit(FEATURES, CLEAN_TARGET)

    for name in ("first", "second"):
        assert robust.coefficients[name] == pytest.approx(
            ordinary.coefficients[name], abs=0.05
        )


def test_it_reports_the_scale_the_threshold_is_measured_in(
    backend: ModuleType,
) -> None:
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression().fit(FEATURES, SPOILED_TARGET)

    assert model.scale > 0.0
    assert isinstance(model.scale, float)


def test_it_counts_the_rows_it_held_at_arms_length(backend: ModuleType) -> None:
    """At least the four that were spoiled, and fewer than all of them, or the
    threshold would be doing nothing or everything."""
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression().fit(FEATURES, SPOILED_TARGET)

    assert model.n_outliers >= 4
    assert model.n_outliers < len(_SPOILED)


def test_a_larger_threshold_listens_to_the_outliers_again(
    backend: ModuleType,
) -> None:
    """The dial, and which way it turns. Raising the threshold far enough puts
    every row back inside it, and the fit walks back toward least squares."""
    HuberRegression = provided(backend, "HuberRegression")
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")

    tight = HuberRegression(threshold=1.35).fit(FEATURES, SPOILED_TARGET)
    loose = HuberRegression(threshold=50.0).fit(FEATURES, SPOILED_TARGET)
    ordinary = MultipleLinearRegression().fit(FEATURES, SPOILED_TARGET)

    assert abs(loose.intercept - ordinary.intercept) < abs(
        tight.intercept - ordinary.intercept
    )
    assert loose.n_outliers < tight.n_outliers


def test_it_reports_how_the_passes_ended(backend: ModuleType) -> None:
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression().fit(FEATURES, SPOILED_TARGET)

    assert model.converged is True
    assert isinstance(model.iterations_run, int)
    assert model.iterations_run <= model.max_iterations


def test_without_an_intercept_the_plane_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression(fit_intercept=False).fit(FEATURES, SPOILED_TARGET)

    assert model.intercept == 0.0
    assert model.coefficients["first"] == pytest.approx(TRUE_FIRST, abs=0.5)


def test_it_matches_prediction_columns_by_name(backend: ModuleType) -> None:
    HuberRegression = provided(backend, "HuberRegression")
    model = HuberRegression().fit(FEATURES, SPOILED_TARGET)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.allclose(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    HuberRegression = provided(backend, "HuberRegression")

    with pytest.raises(NotFittedError):
        HuberRegression().predict(FEATURES)


@pytest.mark.parametrize(
    "summary", ["scale", "n_outliers", "iterations_run", "converged", "coefficients"]
)
def test_it_refuses_to_report_its_fit_before_fit(
    backend: ModuleType, summary: str
) -> None:
    HuberRegression = provided(backend, "HuberRegression")

    with pytest.raises(NotFittedError):
        getattr(HuberRegression(), summary)
