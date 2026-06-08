"""Spec for HuberRegression.

Two oracles, neither of which shares anything with the implementation.

:func:`joint_objective` is what the model claims to minimise, written from the
definition in the module docstring, and what is asserted is that no nearby
point scores better -- the same independent check the elastic net's spec uses.

The other is a behaviour rather than a number. Bounded influence means a row
already far outside the threshold cannot pull any harder by moving further, so
:class:`TestBoundedInfluence` drags one outlier ten times as far away and
measures how little the fit moves, against a least-squares fit on the same
rows, which moves proportionally.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import DivergenceError, NotFittedError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.regression.least_squares.multiple_feature_regression import (
    MultipleLinearRegression,
)
from oop_ml.numpy.regression.robust.huber_regression import HuberRegression

_GENERATOR = np.random.default_rng(5)
ROWS = _GENERATOR.normal(size=(40, 2))
CLEAN = (
    1.0 + 2.0 * ROWS[:, 0] - 3.0 * ROWS[:, 1] + _GENERATOR.normal(scale=0.2, size=40)
)
SPOILED = CLEAN.copy()
SPOILED[[3, 11, 27, 34]] += 30.0

FEATURES = [Feature("first", ROWS[:, 0]), Feature("second", ROWS[:, 1])]
CLEAN_TARGET = Feature("target", CLEAN)
SPOILED_TARGET = Feature("target", SPOILED)


def joint_objective(
    weights: FloatArray, scale: float, targets: FloatArray, threshold: float
) -> float:
    """``sum(scale + scale * H2(r / scale))``, straight from the definition.

    ``H2(z)`` is ``z ** 2`` inside the threshold and ``2 t |z| - t ** 2``
    outside. No penalty term, since the specs below all run at a penalty of
    zero.
    """
    design = np.column_stack([np.ones(ROWS.shape[0]), ROWS])
    residual = targets - design @ weights
    standardised = np.abs(residual) / scale

    inside = standardised**2
    outside = 2.0 * threshold * standardised - threshold**2

    return float(
        (scale + scale * np.where(standardised < threshold, inside, outside)).sum()
    )


class TestTheObjectiveIsMinimised:
    @pytest.mark.parametrize("threshold", [1.1, 1.35, 3.0])
    def test_no_nearby_point_scores_better(self, threshold):
        model = HuberRegression(threshold=threshold).fit(FEATURES, SPOILED_TARGET)
        found = np.array(
            [
                model.intercept,
                model.coefficients["first"],
                model.coefficients["second"],
            ]
        )
        here = joint_objective(found, model.scale, SPOILED, threshold)

        generator = np.random.default_rng(23)
        for step in (1e-2, 1e-3):
            for _ in range(60):
                nudged = found + generator.normal(scale=step, size=found.shape)

                assert (
                    joint_objective(nudged, model.scale, SPOILED, threshold)
                    >= here - 1e-9
                )

    @pytest.mark.parametrize("threshold", [1.1, 1.35, 3.0])
    def test_and_no_nearby_scale_does_either(self, threshold):
        """The scale is fitted alongside the coefficients, so it has to be at a
        stationary point of the same objective and not merely somewhere
        plausible."""
        model = HuberRegression(threshold=threshold).fit(FEATURES, SPOILED_TARGET)
        found = np.array(
            [
                model.intercept,
                model.coefficients["first"],
                model.coefficients["second"],
            ]
        )
        here = joint_objective(found, model.scale, SPOILED, threshold)

        for factor in (0.99, 0.999, 1.001, 1.01):
            assert (
                joint_objective(found, model.scale * factor, SPOILED, threshold)
                >= here - 1e-9
            )


class TestBoundedInfluence:
    """The property the whole method is for, measured rather than described."""

    def test_dragging_an_outlier_further_barely_moves_the_fit(self):
        further = SPOILED.copy()
        further[3] += 300.0

        near = HuberRegression().fit(FEATURES, SPOILED_TARGET)
        far = HuberRegression().fit(FEATURES, Feature("target", further))

        assert abs(far.intercept - near.intercept) < 0.05

    def test_where_least_squares_moves_with_it(self):
        """The same 300 added to the same row, and the comparison is the point:
        squared error has no cap on a row's pull, so the answer follows."""
        further = SPOILED.copy()
        further[3] += 300.0

        near = MultipleLinearRegression().fit(FEATURES, SPOILED_TARGET)
        far = MultipleLinearRegression().fit(FEATURES, Feature("target", further))

        assert abs(far.intercept - near.intercept) > 5.0

    def test_the_outlier_count_is_stable_under_that_move(self):
        """A row already outside the threshold is outside it however much
        further out it goes, so nothing about the classification changes."""
        further = SPOILED.copy()
        further[3] += 300.0

        near = HuberRegression().fit(FEATURES, SPOILED_TARGET)
        far = HuberRegression().fit(FEATURES, Feature("target", further))

        assert far.n_outliers == near.n_outliers


class TestTheThresholdIsADial:
    def test_a_huge_threshold_reaches_least_squares(self):
        """With nothing outside it the weights are all one, so the reweighted
        solve is the ordinary one and the fit is least squares exactly."""
        loose = HuberRegression(threshold=1e6).fit(FEATURES, SPOILED_TARGET)
        ordinary = MultipleLinearRegression().fit(FEATURES, SPOILED_TARGET)

        assert loose.n_outliers == 0
        assert loose.intercept == pytest.approx(ordinary.intercept, abs=1e-6)
        assert loose.coefficients["first"] == pytest.approx(
            ordinary.coefficients["first"], abs=1e-6
        )

    def test_a_tighter_threshold_holds_more_rows_at_arms_length(self):
        counts = [
            HuberRegression(threshold=threshold)
            .fit(FEATURES, SPOILED_TARGET)
            .n_outliers
            for threshold in (1.05, 1.35, 2.0, 5.0)
        ]

        assert counts == sorted(counts, reverse=True)


class TestTheScaleGuard:
    """Refused because the arithmetic admits it, not because data reaches it,
    and the difference is measured rather than assumed."""

    def test_it_fires_when_too_many_residuals_are_outside(self):
        """Handed directly to the scale step, because no dataset produces this:
        ``n - t ** 2 * n_outside`` is negative once 30 of 40 rows are outside
        at a threshold of 1.35."""
        model = HuberRegression(threshold=1.35)
        residual = np.concatenate([np.zeros(10), np.full(30, 100.0)])

        with pytest.raises(DivergenceError):
            model._settled_scale(residual, np.abs(residual), 1.0, 40)

    def test_but_ordinary_data_does_not_reach_it(self):
        """The scale is estimated from the same residuals, so it grows until
        most rows are inside again. A target with no plane in it at all, drawn
        from noise at a spread of 100, fits with a minority outside."""
        generator = np.random.default_rng(1)
        scattered = generator.normal(scale=100.0, size=len(CLEAN))

        model = HuberRegression().fit(FEATURES, Feature("target", scattered))

        assert model.n_outliers < len(scattered) / 2
        assert model.scale > 10.0

    def test_a_fit_through_every_point_stops_rather_than_dividing_by_zero(self):
        """Three rows and three parameters leave no residual at all, so the
        scale collapses and there is nothing further to do."""
        exact_rows = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        exact = [
            Feature("first", exact_rows[:, 0]),
            Feature("second", exact_rows[:, 1]),
        ]
        targets = Feature(
            "target", 1.0 + 2.0 * exact_rows[:, 0] - 3.0 * exact_rows[:, 1]
        )

        model = HuberRegression().fit(exact, targets)

        assert model.converged is True
        assert model.coefficients["first"] == pytest.approx(2.0, abs=1e-8)
        assert model.coefficients["second"] == pytest.approx(-3.0, abs=1e-8)


class TestBeforeFit:
    @pytest.mark.parametrize(
        "attribute", ["scale", "n_outliers", "iterations_run", "converged"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(HuberRegression(), attribute)
