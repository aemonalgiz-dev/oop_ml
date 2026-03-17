"""The recorded walk must be the walk, not a re-run that resembles it.

``solver_path`` exists so a convergence plot can be drawn from a real fit. That
is worth nothing unless it is the *same* walk ``_solve`` takes, so the tests
that matter are the agreement tests -- same final weights, same pass count,
same exit, across every solver that shares the frame.

Bit-identical rather than close. Both routes start at zero, apply the same
``_step``, and test the same condition, so any difference at all would mean one
of them is doing something the other is not.

The second thing pinned here is that recording is not fitting: a call to
``solver_path`` must leave ``passes_run`` and ``converged`` reporting what the
model's own fit did. A caller that needs the intermediates that quietly overwrote fitted state would
be worse than no caller that needs the intermediates.
"""

import numpy as np
import pytest

from oop_ml import (
    FeatureSet,
    GradientDescentRegression,
    LogisticRegression,
    NewtonLogisticRegression,
)
from oop_ml.core.observation import Observation
from oop_ml.core.solving.path import SolverPath, SolverStop
from test.fixtures import DISPLACED_PLANE, OVERLAPPING_LABELS


def walked(model, fixture):
    """Fit, then hand back the pieces both routes take."""
    features, target = fixture.input_features, fixture.target_feature
    model.fit(features, target)

    return (
        model,
        model._design_matrix(FeatureSet(features)),
        model._validated_target_column(target),
    )


def gradient_descent():
    return walked(GradientDescentRegression(), DISPLACED_PLANE)


def ascent():
    return walked(LogisticRegression(), OVERLAPPING_LABELS)


def newton():
    return walked(NewtonLogisticRegression(), OVERLAPPING_LABELS)


EVERY_SOLVER = [gradient_descent, ascent, newton]


class TestTheTwoRoutesAgree:
    @pytest.mark.parametrize("build", EVERY_SOLVER)
    def test_the_weights_are_bit_identical(self, build):
        model, design, column = build()

        assert np.array_equal(
            model.solver_path(design, column).result,
            model._solve(design, column),
        )

    @pytest.mark.parametrize("build", EVERY_SOLVER)
    def test_the_pass_count_matches(self, build):
        model, design, column = build()

        model._solve(design, column)

        # _completed_passes is the shared counter; each model renames it in
        # its own terms (epochs_run, iterations_run) and this is the one
        # both routes are actually counting.
        assert model.solver_path(design, column).passes_run == model._completed_passes

    @pytest.mark.parametrize("build", EVERY_SOLVER)
    def test_the_exit_matches(self, build):
        model, design, column = build()

        model._solve(design, column)

        assert model.solver_path(design, column).converged == model.converged

    def test_a_walk_stopped_by_the_pass_limit_says_so(self):
        # The exit that means the coefficients were still moving. Reporting
        # them without saying so is how a diverging fit looks successful.
        model = LogisticRegression(max_epochs=3, tolerance=1e-12)
        model.fit(OVERLAPPING_LABELS.input_features, OVERLAPPING_LABELS.target_feature)
        design = model._design_matrix(FeatureSet(OVERLAPPING_LABELS.input_features))
        column = model._validated_target_column(OVERLAPPING_LABELS.target_feature)

        path = model.solver_path(design, column)

        assert path.stopped_because is SolverStop.PASS_LIMIT_REACHED
        assert not path.converged
        assert path.passes_run == 3


class TestRecordingIsNotFitting:
    def test_it_leaves_the_fitted_state_alone(self):
        model, design, column = ascent()
        passes, converged = model._completed_passes, model.converged

        model.solver_path(design, column)

        assert model._completed_passes == passes
        assert model.converged == converged

    def test_it_leaves_the_coefficients_alone(self):
        model, design, column = ascent()
        before = [model.coefficients[name] for name in ("hours",)]

        model.solver_path(design, column)

        assert [model.coefficients[name] for name in ("hours",)] == before


class TestWhatTheWalkRecords:
    def test_every_pass_is_kept_in_order(self):
        model, design, column = newton()

        path = model.solver_path(design, column)

        assert [step.pass_number for step in path] == list(range(1, len(path) + 1))

    def test_the_first_pass_starts_at_zero(self):
        # Worth seeing rather than assuming; it is where every walk begins.
        model, design, column = newton()

        first = next(iter(model.solver_path(design, column)))

        assert first.weights_before == pytest.approx(
            np.zeros_like(first.weights_before)
        )

    def test_each_pass_lands_where_the_next_one_starts(self):
        # The property that makes it a walk rather than a list of guesses.
        model, design, column = newton()
        steps = list(model.solver_path(design, column))

        for earlier, later in zip(steps, steps[1:], strict=False):
            assert earlier.weights_after == pytest.approx(later.weights_before)

    def test_the_last_pass_lands_on_the_answer(self):
        model, design, column = newton()
        path = model.solver_path(design, column)

        assert list(path)[-1].weights_after == pytest.approx(path.result)

    def test_the_movement_is_what_convergence_is_tested_on(self):
        model, design, column = newton()
        path = model.solver_path(design, column)

        for step in path:
            assert step.largest_movement == pytest.approx(
                float(np.max(np.abs(step.step)))
            )

        assert path.movements[-1] < model.tolerance

    def test_newton_converges_quadratically_where_ascent_does_not(self):
        # The comparison the whole pairing exists to make visible: the same
        # objective, one walk taking single figures and the other thousands.
        newton_path = newton()[0].solver_path(*newton()[1:])
        ascent_path = ascent()[0].solver_path(*ascent()[1:])

        assert newton_path.passes_run < 20
        assert ascent_path.passes_run > 500
        assert newton_path.movements[-1] < ascent_path.movements[-1]


class TestItIsAnObservation:
    def test_result_is_the_answer(self):
        model, design, column = newton()
        path = model.solver_path(design, column)

        assert np.array_equal(path.result, path.final_weights)

    def test_it_satisfies_the_protocol(self):
        model, design, column = newton()
        path = model.solver_path(design, column)

        assert isinstance(path, Observation)
        assert isinstance(path, SolverPath)
        assert len(path) == path.passes_run


class TestTheWalksThatAreNotGradientWalks:
    """Coordinate descent and softmax share the record without sharing a base.

    ``LassoRegression`` sweeps coordinates and ``MultinomialLogisticRegression``
    walks a weight *matrix*, and neither inherits ``IterativeSolver``. They
    produce a ``SolverPath`` anyway, because a sweep and an epoch are the same
    shape of thing -- start somewhere, move, ask whether the movement still
    matters -- and sharing the record is what lets them be compared.
    """

    def test_lasso_sweeps_agree_with_its_solve(self):
        from oop_ml import LassoRegression

        model = LassoRegression(penalty=0.5)
        model.fit(DISPLACED_PLANE.input_features, DISPLACED_PLANE.target_feature)
        design = model._design_matrix(FeatureSet(DISPLACED_PLANE.input_features))
        column = model._validated_target_column(DISPLACED_PLANE.target_feature)

        assert np.array_equal(
            model.solver_path(design, column).result, model._solve(design, column)
        )

    def test_a_lasso_sweep_is_the_unit_that_converges(self):
        from oop_ml import LassoRegression

        model = LassoRegression(penalty=0.5)
        model.fit(DISPLACED_PLANE.input_features, DISPLACED_PLANE.target_feature)
        design = model._design_matrix(FeatureSet(DISPLACED_PLANE.input_features))
        column = model._validated_target_column(DISPLACED_PLANE.target_feature)

        path = model.solver_path(design, column)

        assert path.passes_run == model.iterations_run
        assert path.converged == model.converged

    def test_softmax_keeps_its_weights_as_a_matrix(self):
        # Flattening would make a step unreadable: the point is that every
        # class moves at once.
        from oop_ml import MultinomialLogisticRegression
        from test.fixtures import THREE_CLASSES

        model = MultinomialLogisticRegression(
            learning_rate=1.0, max_epochs=50_000, tolerance=1e-9
        )
        model.fit(THREE_CLASSES.input_features, THREE_CLASSES.target_feature)
        design = model._design_matrix(FeatureSet(THREE_CLASSES.input_features))
        column = THREE_CLASSES.target_feature.column

        path = model.solver_path(design, column)

        assert path.result.ndim == 2
        assert np.array_equal(path.result, model._solve(design, column))
        for step in path:
            assert step.weights_before.shape == path.result.shape
