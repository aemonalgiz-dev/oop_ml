"""Where the scikit-learn regression wrappers translate, and that the translation holds.

The contract suite checks each backend against a fixture's known answer, and
deliberately never against the other backend. This file is the other half.
Every hyperparameter that changes scale or name on its way to the engine is
pinned here by fitting both backends on one fixture and asking them to agree,
because a wrong scale factor still fits, still scores well, and still passes
a loose contract; only an agreement test at the exact penalty catches it.

The two refusals this backend adds are pinned here too, since they have no
counterpart in the numpy backend and so no place in the contract suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from oop_ml import Feature, scikit
from oop_ml import numpy as reference
from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.kernel.functions import (
    Kernel,
    LinearKernel,
    PolynomialKernel,
    RadialBasisKernel,
    SigmoidKernel,
)

#: y = 1 + 2 * first + 3 * second, exactly. The penalties are the ones the
#: numpy lasso's worked example tabulates, and 12 is the one that selects a
#: feature out.
_FIRST = np.array([1.0, 1.0, 2.0, 0.0, 3.0])
_SECOND = np.array([1.0, 2.0, 2.0, 1.0, 0.0])
PLANE_FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
PLANE_TARGET = Feature("target", 1.0 + 2.0 * _FIRST + 3.0 * _SECOND)

_GENERATOR = np.random.default_rng(7)
_ROWS = _GENERATOR.normal(size=(30, 2))
CURVED_FEATURES = [Feature("left", _ROWS[:, 0]), Feature("right", _ROWS[:, 1])]
CURVED_TARGET = Feature(
    "height", np.sin(_ROWS[:, 0]) + _ROWS[:, 1] ** 2 + 0.05 * _GENERATOR.normal(size=30)
)

#: The same thirty rows under an exact plane, enough rows that a bootstrap
#: resample of them still determines the plane.
WIDE_PLANE_TARGET = Feature("height", 1.0 + 2.0 * _ROWS[:, 0] + 3.0 * _ROWS[:, 1])

#: A height in millimetres beside a trace mass in grams. The two are
#: uncorrelated, at -0.0591, and the centred singular values differ by a
#: factor of 1.770e-07, which is what a plain change of units does to a
#: design matrix. Only the second is what the engine's default rank
#: threshold of 1e-6 reads.
_HEIGHT_MILLIMETRES = np.array(
    [1600.0, 1720.0, 1550.0, 1810.0, 1680.0, 1770.0, 1490.0, 1900.0]
)
_TRACE_GRAMS = np.array([2.0, 1.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0]) * 1e-5
SCALED_APART_FEATURES = [
    Feature("height_millimetres", _HEIGHT_MILLIMETRES),
    Feature("trace_grams", _TRACE_GRAMS),
]
SCALED_APART_TARGET = Feature(
    "score", np.array([3.0, 4.0, 8.0, 9.0, 13.0, 14.0, 18.0, 19.0])
)

#: Forty rows of two uncorrelated columns, at -0.0197, for rescaling one of
#: them by a sweep of factors. The target is an exact plane in the unscaled
#: pair, so a fit that recovers it scores 1.0 whatever the factor.
_WIDE_GENERATOR = np.random.default_rng(20260902)
_WIDE_LEFT = _WIDE_GENERATOR.normal(size=40)
_WIDE_RIGHT = _WIDE_GENERATOR.normal(size=40)
WIDE_SCALE_TARGET = Feature("target", 1.0 + 2.0 * _WIDE_LEFT + 3.0 * _WIDE_RIGHT)


class TestTheLassoPenaltyScale:
    """``alpha = penalty / (2 n)`` is the translation, and it is exact."""

    @pytest.mark.parametrize("penalty", [0.5, 2.0, 8.0, 12.0, 16.0])
    def test_both_backends_reach_the_same_coefficients(self, penalty: float) -> None:
        expected = reference.LassoRegression(penalty=penalty).fit(
            PLANE_FEATURES, PLANE_TARGET
        )
        wrapped = scikit.LassoRegression(penalty=penalty).fit(
            PLANE_FEATURES, PLANE_TARGET
        )

        assert wrapped.intercept == pytest.approx(expected.intercept, abs=1e-6)
        for name in ("first", "second"):
            assert wrapped.coefficients[name] == pytest.approx(
                expected.coefficients[name], abs=1e-6
            )

    def test_the_selection_boundary_lands_in_the_same_place(self) -> None:
        """At 12 the numpy backend zeroes ``second`` and keeps ``first``. A
        translation off by any factor moves that boundary."""
        wrapped = scikit.LassoRegression(penalty=12.0).fit(PLANE_FEATURES, PLANE_TARGET)

        assert wrapped.coefficients["second"] == pytest.approx(0.0, abs=1e-12)
        assert wrapped.coefficients["first"] > 0.0


class TestTheRidgePenaltyScale:
    """``alpha = penalty``, with and without an intercept to exempt."""

    @pytest.mark.parametrize("penalty", [0.0, 1.0, 10.0])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_both_backends_reach_the_same_coefficients(
        self, penalty: float, fit_intercept: bool
    ) -> None:
        expected = reference.RidgeRegression(
            penalty=penalty, fit_intercept=fit_intercept
        ).fit(PLANE_FEATURES, PLANE_TARGET)
        wrapped = scikit.RidgeRegression(
            penalty=penalty, fit_intercept=fit_intercept
        ).fit(PLANE_FEATURES, PLANE_TARGET)

        assert wrapped.intercept == pytest.approx(expected.intercept, abs=1e-9)
        for name in ("first", "second"):
            assert wrapped.coefficients[name] == pytest.approx(
                expected.coefficients[name], abs=1e-9
            )


class TestTheKernelTranslation:
    """Each kernel object maps onto the engine's kernel of the same formula."""

    @pytest.mark.parametrize(
        "kernel",
        [
            LinearKernel(),
            PolynomialKernel(degree=2, gamma=0.5, constant=1.5),
            RadialBasisKernel(gamma=0.7),
            SigmoidKernel(gamma=0.1, constant=0.0),
        ],
        ids=["linear", "polynomial", "rbf", "sigmoid"],
    )
    def test_both_backends_reach_the_same_predictions(self, kernel: Kernel) -> None:
        expected = reference.KernelRidgeRegression(kernel=kernel, penalty=0.1).fit(
            CURVED_FEATURES, CURVED_TARGET
        )
        wrapped = scikit.KernelRidgeRegression(kernel=kernel, penalty=0.1).fit(
            CURVED_FEATURES, CURVED_TARGET
        )

        assert np.allclose(
            np.asarray(wrapped.predict(CURVED_FEATURES)),
            np.asarray(expected.predict(CURVED_FEATURES)),
            atol=1e-8,
        )
        assert np.allclose(wrapped.dual_weights, expected.dual_weights, atol=1e-8)

    def test_a_kernel_the_engine_cannot_be_handed_is_refused(self) -> None:
        class SquaredLinearKernel(Kernel):
            @property
            def description(self) -> str:
                return "(a . b) ** 2"

            def _between_blocks(self, left: RowBlock, right: RowBlock) -> np.ndarray:
                return (left.values @ right.values.T) ** 2

        with pytest.raises(InvalidValuesError):
            scikit.KernelRidgeRegression(kernel=SquaredLinearKernel()).fit(
                CURVED_FEATURES, CURVED_TARGET
            )


class TestTheEnsembleMembers:
    """A bagging engine needs a member that can hand it a scikit-learn prototype."""

    def test_a_member_from_the_numpy_backend_is_refused_at_construction(self) -> None:
        with pytest.raises(InvalidValuesError):
            scikit.BaggingRegressor(base_model=reference.DecisionTreeRegressor())

    def test_a_neighbour_member_keeps_the_rows_it_was_fitted_on(self) -> None:
        model = scikit.BaggingRegressor(
            base_model=scikit.KNearestNeighboursRegressor(n_neighbours=3),
            n_members=4,
            random_seed=0,
        ).fit(CURVED_FEATURES, CURVED_TARGET)

        for member in model.members:
            assert isinstance(member, scikit.KNearestNeighboursRegressor)
            assert member.n_remembered == 30
        assert model.score(CURVED_FEATURES, CURVED_TARGET) > 0.5

    def test_a_linear_member_reads_its_coefficients_back_by_name(self) -> None:
        model = scikit.BaggingRegressor(
            base_model=scikit.RidgeRegression(penalty=0.1),
            n_members=4,
            random_seed=0,
        ).fit(CURVED_FEATURES, WIDE_PLANE_TARGET)

        for member in model.members:
            assert isinstance(member, scikit.RidgeRegression)
            assert member.coefficients["left"] == pytest.approx(2.0, abs=0.3)
            assert member.coefficients["right"] == pytest.approx(3.0, abs=0.3)

    def test_an_adopted_member_carries_the_seed_the_engine_gave_it(self) -> None:
        """The engine seeds each tree from its own stream. A member read back
        reports that seed, not the prototype's, so two members never claim
        one seed while having grown from different draws."""
        model = scikit.RandomForestRegressor(
            n_members=4, max_features=1, random_seed=0
        ).fit(CURVED_FEATURES, CURVED_TARGET)

        seeds = []
        for member in model.members:
            assert isinstance(member, scikit.DecisionTreeRegressor)
            seeds.append(member.random_seed)

        assert len(set(seeds)) == 4
        assert seeds == [tree.random_state for tree in model._engine.estimators_]

    def test_boosting_agrees_with_the_numpy_backend_round_for_round(self) -> None:
        """Squared error, the mean as the start, every round on every row:
        the engine and the numpy backend take the same steps."""
        expected = reference.GradientBoostingRegressor(
            n_rounds=20, learning_rate=0.3, max_depth=2
        ).fit(CURVED_FEATURES, CURVED_TARGET)
        wrapped = scikit.GradientBoostingRegressor(
            n_rounds=20, learning_rate=0.3, max_depth=2
        ).fit(CURVED_FEATURES, CURVED_TARGET)

        assert wrapped.initial_prediction == pytest.approx(expected.initial_prediction)
        assert np.allclose(
            np.asarray(wrapped.predict(CURVED_FEATURES)),
            np.asarray(expected.predict(CURVED_FEATURES)),
            atol=1e-9,
        )


class TestTheCollinearRefusal:
    """The engine solves a deficient design by least norm; the wrapper refuses it.

    The refusal is read off the engine's ``rank_``, which is a rank at a
    threshold rather than the exact one, so the threshold is half of what is
    being asserted here. An exactly deficient design is refused under any
    threshold and cannot see it; the wide-scale pair below is refused only
    under a loose one, and it is the case that tells the two apart.
    """

    def test_an_exact_combination_of_columns_is_refused(self) -> None:
        features = [*PLANE_FEATURES, Feature("total", _FIRST + _SECOND)]

        with pytest.raises(Exception) as caught:
            scikit.MultipleLinearRegression().fit(features, PLANE_TARGET)

        assert type(caught.value).__name__ == "CollinearFeaturesError"

    def test_a_duplicated_column_is_refused(self) -> None:
        """Rank 1 across 2, which no threshold this side of exact repairs."""
        features = [Feature("first", _FIRST), Feature("copy", _FIRST.copy())]

        with pytest.raises(Exception) as caught:
            scikit.MultipleLinearRegression().fit(features, PLANE_TARGET)

        assert type(caught.value).__name__ == "CollinearFeaturesError"

    def test_two_independent_columns_on_different_scales_are_not_refused(self) -> None:
        """A change of units is not a linear dependence, and must not read as one.

        At the engine's default ``tol`` of 1e-6 it is read as one, because
        that number is a floor on a singular value relative to the largest
        and so a statement about the columns' spreads. These two are
        uncorrelated at -0.0591, and the numpy backend fits them at
        r^2 0.975971.
        """
        expected = reference.MultipleLinearRegression().fit(
            SCALED_APART_FEATURES, SCALED_APART_TARGET
        )
        wrapped = scikit.MultipleLinearRegression().fit(
            SCALED_APART_FEATURES, SCALED_APART_TARGET
        )

        assert wrapped.score(SCALED_APART_FEATURES, SCALED_APART_TARGET) == (
            pytest.approx(0.975971, abs=1e-6)
        )
        assert wrapped.intercept == pytest.approx(expected.intercept, rel=1e-9)
        for name in ("height_millimetres", "trace_grams"):
            assert wrapped.coefficients[name] == pytest.approx(
                expected.coefficients[name], rel=1e-9
            )

    @pytest.mark.parametrize("factor", [1e-7, 1e-9, 1e-12])
    def test_a_column_rescaled_far_below_its_neighbour_still_fits(
        self, factor: float
    ) -> None:
        """The same claim swept, since one pair could be a lucky ratio.

        Only the second column is rescaled, so the pair's correlation does
        not move; what moves is the ratio of the two singular values, which
        is the quantity the engine's default threshold is a floor on. All
        three factors sit below that floor and above it the fit is the same
        one.
        """
        features = [
            Feature("left", _WIDE_LEFT),
            Feature("right", _WIDE_RIGHT * factor),
        ]

        expected = reference.MultipleLinearRegression().fit(features, WIDE_SCALE_TARGET)
        wrapped = scikit.MultipleLinearRegression().fit(features, WIDE_SCALE_TARGET)

        assert wrapped.score(features, WIDE_SCALE_TARGET) == pytest.approx(1.0)
        assert wrapped.coefficients["left"] == pytest.approx(
            expected.coefficients["left"], rel=1e-6
        )
