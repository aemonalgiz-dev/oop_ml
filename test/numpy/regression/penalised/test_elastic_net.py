"""Spec for ElasticNetRegression.

The oracle here is the objective itself, written out from the definition in
:func:`penalised_objective` and never from the update rule the model uses. A
coordinate descent that agreed with a reference coordinate descent would only
prove the two were written by the same hand; what is asserted instead is that
no nearby point scores better, which is what "this is the minimum" means.

The rest of the file is the two edges and the one behaviour the model exists
for. At ``l1_share = 1`` it must be the lasso and at ``0`` the ridge, both of
which have their own spec beside this one, and in between it must refuse to
hand a shared effect to whichever of two identical columns the sweep happens
to reach first.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import NotFittedError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.regression.penalised.elastic_net_regression import (
    ElasticNetRegression,
)
from oop_ml.numpy.regression.penalised.lasso_regression import LassoRegression
from oop_ml.numpy.regression.penalised.ridge_regression import RidgeRegression
from test.fixtures import EXACT_PLANE, ORIGIN_PLANE

SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]

#: Tight enough that what is being tested is the optimum and not the stopping
#: rule. Identical columns are the slowest case coordinate descent has, and
#: `TestConvergence` measures what the defaults do there instead.
AT_THE_OPTIMUM = {"tolerance": 1e-14, "max_iterations": 200_000}

#: Two columns holding one measurement under two names, and a target built
#: from their sum, so the whole effect is 6 and the only question is the split.
SHARED_COLUMN = np.arange(1.0, 9.0)
TWIN_TARGET = 6.0 * SHARED_COLUMN


def twin_features() -> list[Feature]:
    return [
        Feature("twin_one", SHARED_COLUMN),
        Feature("twin_two", SHARED_COLUMN.copy()),
    ]


def twin_target() -> Feature:
    return Feature("target", TWIN_TARGET)


def penalised_objective(
    weights: FloatArray,
    design: FloatArray,
    targets: FloatArray,
    penalty: float,
    l1_share: float,
) -> float:
    """The quantity the model claims to minimise, straight from the definition.

    Column zero of ``design`` is the intercept's ones column and carries
    neither penalty, which is the only thing this shares with the
    implementation.
    """
    residual = targets - design @ weights
    slopes = weights[1:]

    return float(
        residual @ residual
        + penalty * l1_share * np.abs(slopes).sum()
        + penalty * (1.0 - l1_share) * (slopes**2).sum()
    )


def fitted_model(penalty: float = 1.0, l1_share: float = 0.5) -> ElasticNetRegression:
    return ElasticNetRegression(
        penalty=penalty, l1_share=l1_share, **AT_THE_OPTIMUM
    ).fit(EXACT_PLANE.input_features, EXACT_PLANE.target_feature)


def design_and_targets() -> tuple[FloatArray, FloatArray]:
    columns = [
        np.ones(len(EXACT_PLANE.target_feature.column.values)),
        *(
            np.asarray(feature.column.values, dtype=np.float64)
            for feature in EXACT_PLANE.input_features
        ),
    ]
    return (
        np.column_stack(columns),
        np.asarray(EXACT_PLANE.target_feature.column.values, dtype=np.float64),
    )


class TestConstruction:
    @pytest.mark.parametrize(
        ("field_name", "expected"),
        [
            ("penalty", 1.0),
            ("l1_share", 0.5),
            ("max_iterations", 1_000),
            ("tolerance", 1e-10),
        ],
    )
    def test_defaults(self, field_name, expected):
        assert getattr(ElasticNetRegression(), field_name) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("field_name", "invalid"),
        [
            ("penalty", -1.0),
            ("l1_share", -0.01),
            ("l1_share", 1.01),
            ("max_iterations", 0),
            ("tolerance", 0.0),
        ],
    )
    def test_invalid_settings_are_rejected(self, field_name, invalid):
        with pytest.raises(ValueError):
            ElasticNetRegression(**{field_name: invalid})


class TestBeforeFit:
    @pytest.mark.parametrize(
        "attribute", ["coefficients", "intercept", "iterations_run", "converged"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(ElasticNetRegression(), attribute)


class TestTheObjectiveIsMinimised:
    """The independent check, and the only one here that does not lean on
    another model already being right."""

    @pytest.mark.parametrize("l1_share", SHARES)
    @pytest.mark.parametrize("penalty", [1.0, 4.0, 12.0])
    def test_no_nearby_point_scores_better(self, penalty, l1_share):
        model = fitted_model(penalty, l1_share)
        design, targets = design_and_targets()
        found = np.array(
            [
                model.intercept,
                model.coefficients["x1"],
                model.coefficients["x2"],
            ]
        )
        here = penalised_objective(found, design, targets, penalty, l1_share)

        generator = np.random.default_rng(11)
        for step in (1e-2, 1e-3, 1e-4):
            for _ in range(60):
                nudged = found + generator.normal(scale=step, size=found.shape)
                there = penalised_objective(nudged, design, targets, penalty, l1_share)

                assert there >= here - 1e-12

    @pytest.mark.parametrize("l1_share", SHARES)
    def test_no_penalty_leaves_the_plane_the_fixture_was_built_from(self, l1_share):
        """At zero strength there is nothing for the share to divide, so every
        share has to give the same answer and it has to be the exact plane."""
        model = fitted_model(penalty=0.0, l1_share=l1_share)

        assert model.intercept == pytest.approx(1.0)
        assert model.coefficients["x1"] == pytest.approx(2.0)
        assert model.coefficients["x2"] == pytest.approx(3.0)


class TestTheTwoEdges:
    @pytest.mark.parametrize("penalty", [1.0, 4.0, 12.0])
    def test_the_whole_share_on_l1_is_the_lasso_to_the_last_bit(self, penalty):
        """Not merely close. At ``l1_share = 1`` the added term is multiplied
        by zero and ``penalty * 1.0 / 2`` is ``penalty / 2`` exactly, so the
        two updates are the same arithmetic in the same order."""
        mixed = ElasticNetRegression(penalty=penalty, l1_share=1.0).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )
        pure = LassoRegression(penalty=penalty).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert mixed.intercept == pure.intercept
        assert mixed.coefficients["x1"] == pure.coefficients["x1"]
        assert mixed.coefficients["x2"] == pure.coefficients["x2"]

    @pytest.mark.parametrize("penalty", [1.0, 4.0, 12.0])
    def test_no_share_on_l1_walks_to_ridge_s_closed_form(self, penalty):
        """Approximately, and the tolerance is the interesting part: this is a
        walk being compared against an exact solve, so what is left is the
        walk's own convergence limit rather than a difference of objective."""
        mixed = ElasticNetRegression(
            penalty=penalty, l1_share=0.0, **AT_THE_OPTIMUM
        ).fit(EXACT_PLANE.input_features, EXACT_PLANE.target_feature)
        pure = RidgeRegression(penalty=penalty).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert mixed.intercept == pytest.approx(pure.intercept, abs=1e-11)
        assert mixed.coefficients["x1"] == pytest.approx(
            pure.coefficients["x1"], abs=1e-11
        )
        assert mixed.coefficients["x2"] == pytest.approx(
            pure.coefficients["x2"], abs=1e-11
        )

    @pytest.mark.parametrize("l1_share", [0.25, 0.5, 0.75])
    def test_a_mixture_sits_between_the_two_it_mixes(self, l1_share):
        """Every coefficient of a mixture lies between the ridge answer and
        the lasso answer at the same strength, which is what makes ``l1_share``
        a dial rather than a switch."""
        ridge_like = fitted_model(8.0, 0.0).coefficients["x2"]
        lasso_like = fitted_model(8.0, 1.0).coefficients["x2"]
        mixed = fitted_model(8.0, l1_share).coefficients["x2"]

        assert ridge_like < mixed < lasso_like


class TestTheGroupingEffect:
    """What the mixture buys, on the case that shows it."""

    @pytest.mark.parametrize("l1_share", [0.0, 0.25, 0.5, 0.75])
    def test_any_share_of_l2_splits_two_identical_columns_evenly(self, l1_share):
        model = ElasticNetRegression(
            penalty=4.0, l1_share=l1_share, **AT_THE_OPTIMUM
        ).fit(twin_features(), twin_target())

        assert model.coefficients["twin_one"] == pytest.approx(
            model.coefficients["twin_two"], abs=1e-11
        )

    def test_pure_l1_hands_the_whole_effect_to_the_column_it_reaches_first(self):
        """The failure the mixture repairs, and an implementation fact rather
        than a mathematical one.

        With identical columns the L1 objective is flat along the whole segment
        between the two corners, so every split is equally optimal and the sweep
        settles on the corner it arrives at first. Reverse the two features and
        the answer reverses with them, which is what this asserts and what makes
        a pure lasso the wrong tool for correlated columns.
        """
        forwards = ElasticNetRegression(penalty=4.0, l1_share=1.0).fit(
            twin_features(), twin_target()
        )
        backwards = ElasticNetRegression(penalty=4.0, l1_share=1.0).fit(
            list(reversed(twin_features())), twin_target()
        )

        assert forwards.coefficients["twin_one"] > 5.0
        assert forwards.coefficients["twin_two"] == pytest.approx(0.0, abs=1e-12)
        assert backwards.coefficients["twin_two"] > 5.0
        assert backwards.coefficients["twin_one"] == pytest.approx(0.0, abs=1e-12)


class TestConvergence:
    def test_reports_convergence(self):
        model = fitted_model(penalty=1.0)

        assert model.converged is True
        assert model.iterations_run < model.max_iterations

    def test_identical_columns_are_the_slow_case_and_it_says_so(self):
        """Measured rather than assumed, and the reason
        :data:`AT_THE_OPTIMUM` exists.

        Perfectly collinear columns are where coordinate descent crawls, and
        the less L2 there is holding the split down the slower it goes. At the
        default thousand sweeps a share of 0.75 leaves the two coefficients
        3.2e-04 apart, which is nowhere near the even split they converge to.
        The model does not pretend otherwise.
        """
        model = ElasticNetRegression(penalty=4.0, l1_share=0.75).fit(
            twin_features(), twin_target()
        )

        assert model.converged is False
        assert model.iterations_run == model.max_iterations
        assert model.coefficients["twin_one"] != pytest.approx(
            model.coefficients["twin_two"], abs=1e-8
        )

    def test_stopping_early_is_reported_not_hidden(self):
        model = ElasticNetRegression(penalty=1.0, max_iterations=1).fit(
            EXACT_PLANE.input_features, EXACT_PLANE.target_feature
        )

        assert model.converged is False
        assert model.iterations_run == 1


class TestWithoutIntercept:
    @pytest.mark.parametrize("l1_share", SHARES)
    def test_every_column_is_penalised(self, l1_share):
        """Through the origin there is no ones column to exempt, so column
        zero is an ordinary predictor and shrinks like the rest."""
        model = ElasticNetRegression(
            penalty=10.0, l1_share=l1_share, fit_intercept=False, **AT_THE_OPTIMUM
        ).fit(ORIGIN_PLANE.input_features, ORIGIN_PLANE.target_feature)

        assert model.intercept == 0.0
        assert abs(model.coefficients["x1"]) < 2.0
        assert abs(model.coefficients["x2"]) < 3.0
