"""The recorded derivation must be the one the fit performed.

Same requirement as the other pairings: whatever the closed-form ``_solve``
returns, ``normal_equations(...).result`` returns too. Bit-identical, because
both hand the same matrices to the same solver -- any difference would mean one
of them is building something the other is not.

Two things beyond agreement are pinned here.

The penalty matrix is kept separate from ``X.T X`` rather than pre-added,
because its ``[0, 0]`` entry is the intercept exemption, and zeroing that
unconditionally exempts a real predictor whenever ``fit_intercept`` is false.
That was a real bug in this library, and holding the matrix is what lets a
caller check the rule rather than trust it.

The condition number is what the coefficients cannot tell you. A near-collinear
design produces enormous coefficients that cancel, and nothing about the
returned vector says so; the number that says so lives on the matrix.
"""

from typing import Protocol

import numpy as np
import pytest

from oop_ml import (
    Feature,
    FeatureSet,
    MultipleLinearRegression,
    RidgeRegression,
    SimpleLinearRegression,
)
from oop_ml.core.observation import Observation, Stage
from oop_ml.core.solving.normal_equations import LeastSquaresLine, NormalEquations
from test.fixtures import DISPLACED_PLANE


class FeaturesAndTarget(Protocol):
    """What ``solved`` needs of a fixture, structurally.

    ``LinearFixture`` satisfies it, and so does the local one below, without
    either having to inherit from anything.
    """

    @property
    def input_features(self) -> list[Feature]: ...

    @property
    def target_feature(self) -> Feature: ...


class NearlyCollinear:
    """Two predictors that almost repeat each other.

    Built here rather than in fixtures.py because it exists for exactly one
    assertion: that a penalty improves the conditioning of a design where the
    unpenalised system is close to unsolvable.
    """

    @property
    def input_features(self) -> list[Feature]:
        base = np.linspace(0.0, 10.0, 40)
        return [
            Feature("first", base),
            Feature("second", base * 2.0 + 1e-3 * np.arange(40)),
        ]

    @property
    def target_feature(self) -> Feature:
        return Feature("y", np.linspace(1.0, 21.0, 40))


def solved(model, fixture: FeaturesAndTarget = DISPLACED_PLANE):
    model.fit(fixture.input_features, fixture.target_feature)

    return (
        model,
        model._design_matrix(FeatureSet(fixture.input_features)),
        model._validated_target_column(fixture.target_feature),
    )


class TestTheTwoRoutesAgree:
    @pytest.mark.parametrize(
        "model",
        [
            MultipleLinearRegression(),
            MultipleLinearRegression(fit_intercept=False),
            RidgeRegression(penalty=1.0),
            RidgeRegression(penalty=5.0, fit_intercept=False),
        ],
    )
    def test_the_coefficients_are_bit_identical(self, model):
        model, design, column = solved(model)

        assert np.array_equal(
            model.normal_equations(design, column).result,
            model._solve(design, column),
        )

    def test_recording_does_not_refit(self):
        model, design, column = solved(MultipleLinearRegression())
        before = model.coefficients["x1"]

        model.normal_equations(design, column)

        assert model.coefficients["x1"] == before


class TestWhatTheDerivationHolds:
    def test_the_moment_matrix_is_what_it_claims(self):
        model, design, column = solved(MultipleLinearRegression())

        equations = model.normal_equations(design, column)

        assert equations.moment_matrix == pytest.approx(design.values.T @ design.values)
        assert equations.target_moments == pytest.approx(
            design.values.T @ column.values
        )

    def test_the_intercept_column_is_visible(self):
        with_intercept, design, _ = solved(MultipleLinearRegression())
        _, bare, _ = solved(MultipleLinearRegression(fit_intercept=False))

        assert design.values[:, 0] == pytest.approx(np.ones(design.n_rows))
        assert bare.n_columns == design.n_columns - 1

    def test_an_unpenalised_fit_has_no_penalty_matrix(self):
        model, design, column = solved(MultipleLinearRegression())

        equations = model.normal_equations(design, column)

        assert equations.penalty_matrix is None
        assert equations.solved_matrix is equations.moment_matrix

    def test_the_penalty_exempts_the_intercept_only_when_there_is_one(self):
        # The bug this library shipped: zeroing [0, 0] unconditionally exempts
        # a real predictor whenever fit_intercept is false.
        with_intercept, design, column = solved(RidgeRegression(penalty=3.0))
        penalty = with_intercept.normal_equations(design, column).penalty_matrix

        assert penalty is not None
        assert penalty[0, 0] == 0.0
        assert penalty[1, 1] == pytest.approx(3.0)

        bare, bare_design, bare_column = solved(
            RidgeRegression(penalty=3.0, fit_intercept=False)
        )
        bare_penalty = bare.normal_equations(bare_design, bare_column).penalty_matrix

        assert bare_penalty is not None
        assert bare_penalty[0, 0] == pytest.approx(3.0)

    def test_the_solved_matrix_includes_the_penalty(self):
        model, design, column = solved(RidgeRegression(penalty=2.0))
        equations = model.normal_equations(design, column)

        assert equations.penalty_matrix is not None
        assert equations.solved_matrix == pytest.approx(
            equations.moment_matrix + equations.penalty_matrix
        )

    def test_a_penalty_improves_the_conditioning(self):
        # What ridge does numerically, and the reason it survives collinearity
        # where ordinary least squares does not.
        plain, design, column = solved(MultipleLinearRegression(), NearlyCollinear())
        penalised, _, _ = solved(RidgeRegression(penalty=1.0), NearlyCollinear())

        assert (
            penalised.normal_equations(design, column).condition_number
            < plain.normal_equations(design, column).condition_number
        )

    def test_the_stages_come_in_derivation_order(self):
        model, design, column = solved(RidgeRegression(penalty=1.0))

        names = [stage.name for stage in model.normal_equations(design, column)]

        assert names == [
            "design matrix",
            "X.T X",
            "X.T y",
            "penalty",
            "X.T X + penalty",
            "solution",
        ]
        assert all(
            isinstance(stage, Stage) for stage in model.normal_equations(design, column)
        )


class TestTheSingleFeatureLine:
    def test_it_agrees_with_what_fit_stored(self):
        inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
        targets = [2.0, 4.1, 5.9, 8.2, 9.8]
        model = SimpleLinearRegression()
        model.fit(inputs, targets)

        line = model.least_squares_line(inputs, targets)

        assert line.result == (model.slope, model.intercept)

    def test_the_line_passes_through_both_means(self):
        # Why the intercept is what it is, rather than a separate thing learned.
        inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
        targets = [2.0, 4.1, 5.9, 8.2, 9.8]
        line = SimpleLinearRegression().least_squares_line(inputs, targets)

        assert line.slope * line.input_mean + line.intercept == pytest.approx(
            line.target_mean
        )

    def test_the_slope_is_covariation_over_variation(self):
        inputs = [1.0, 2.0, 3.0, 4.0, 5.0]
        targets = [2.0, 4.1, 5.9, 8.2, 9.8]
        line = SimpleLinearRegression().least_squares_line(inputs, targets)

        assert line.slope == pytest.approx(line.covariation / line.input_variation)

    def test_recording_does_not_fit(self):
        model = SimpleLinearRegression()

        model.least_squares_line([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])

        assert not model.is_fitted


class TestTheyAreObservations:
    def test_normal_equations_satisfies_the_protocol(self):
        model, design, column = solved(RidgeRegression(penalty=1.0))
        equations = model.normal_equations(design, column)

        assert isinstance(equations, Observation)
        assert isinstance(equations, NormalEquations)
        assert len(equations) == 6

    def test_the_line_satisfies_the_protocol(self):
        line = SimpleLinearRegression().least_squares_line(
            [1.0, 2.0, 3.0], [2.0, 4.0, 6.5]
        )

        assert isinstance(line, Observation)
        assert isinstance(line, LeastSquaresLine)
        assert len(line) == 6
