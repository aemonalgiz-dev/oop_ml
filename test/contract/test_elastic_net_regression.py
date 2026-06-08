"""The contract every backend's ElasticNetRegression keeps.

This model sits between two that already have contracts, so the strongest
things to assert are the two edges. At ``l1_share = 1`` it must be that
backend's own lasso and at ``l1_share = 0`` its own ridge, at the same
``penalty`` and on the same rows. Both are checked inside a single backend
rather than across the two, which is what makes them assertions about the
objective rather than about anyone's arithmetic, and both hold: measured, the
lasso edge agrees to exactly zero on both backends and the ridge edge to
2.5e-10 on numpy and 3.1e-12 on scikit, which is each solver's own convergence
limit against a closed form rather than a difference of opinion.

What is deliberately not asserted
----------------------------------
Which of two identical columns a pure L1 fit keeps. The objective is genuinely
flat along the segment between the two corners there, so every split is
equally optimal and the answer comes from the order the solver sweeps in.
Both backends happen to keep the first column, and pinning that here would be
pinning an implementation detail as a promise. What the objective does
determine, once any share of L2 is present, is that the split is *even*, and
that is asserted.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: y = 1 + 2 * first + 3 * second, exactly. The fixture ridge and lasso share.
_FIRST = np.array([1.0, 1.0, 2.0, 0.0, 3.0])
_SECOND = np.array([1.0, 2.0, 2.0, 1.0, 0.0])
_TARGETS = 1.0 + 2.0 * _FIRST + 3.0 * _SECOND
FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
TARGET = Feature("target", _TARGETS)

#: Small enough to shrink almost nothing, and not zero, because at zero the
#: engine warns that coordinate descent is the wrong tool and it is right.
BARELY_PENALISED = 1e-4

#: Forced through the origin at that penalty the answer is the least-squares
#: plane through the origin, whatever the share, since there is no penalty
#: worth speaking of for the share to divide. X'X is [[15, 7], [7, 10]] and
#: X'y is [58, 50], whose determinant is 101.
THROUGH_THE_ORIGIN_FIRST = 230.0 / 101.0
THROUGH_THE_ORIGIN_SECOND = 344.0 / 101.0

#: Two columns holding the identical measurement under different names, with a
#: target built from their sum. The whole effect is 6 and the only question is
#: how the two divide it.
_SHARED = np.arange(1.0, 9.0)
TWINS = [Feature("twin_one", _SHARED), Feature("twin_two", _SHARED.copy())]
TWIN_TARGET = Feature("target", 6.0 * _SHARED)

PENALTIES = (2.0, 8.0)


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")

    model = ElasticNetRegression(penalty=0.5, l1_share=0.25)

    assert model.penalty == pytest.approx(0.5)
    assert model.l1_share == pytest.approx(0.25)


def test_the_share_defaults_to_half(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")

    assert ElasticNetRegression().l1_share == pytest.approx(0.5)


@pytest.mark.parametrize("outside", [-0.1, 1.1])
def test_it_refuses_a_share_that_is_not_a_share(
    backend: ModuleType, outside: float
) -> None:
    """A share below zero or above one is not a proportion of anything, and
    the two penalties would then subtract rather than mix."""
    ElasticNetRegression = provided(backend, "ElasticNetRegression")

    with pytest.raises(ValidationError):
        ElasticNetRegression(l1_share=outside)


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=BARELY_PENALISED)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=BARELY_PENALISED).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=0.05)


def test_its_coefficients_are_addressable_by_feature_name(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=BARELY_PENALISED).fit(FEATURES, TARGET)

    assert model.coefficients["first"] == pytest.approx(2.0, abs=0.05)
    assert model.coefficients["second"] == pytest.approx(3.0, abs=0.05)


@pytest.mark.parametrize("penalty", PENALTIES)
def test_the_whole_share_on_l1_is_this_backend_s_own_lasso(
    backend: ModuleType, penalty: float
) -> None:
    """The upper edge, and the sharper of the two.

    At ``l1_share = 1`` the term this model adds to lasso is multiplied by
    zero, so the two are not merely close but the same fit, and both backends
    measure the gap at exactly zero. The tolerance is looser than that on
    purpose: the claim worth keeping is that they are one fit, not the bit
    pattern a particular engine version reaches it by.
    """
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    LassoRegression = provided(backend, "LassoRegression")

    mixed = ElasticNetRegression(penalty=penalty, l1_share=1.0).fit(FEATURES, TARGET)
    pure = LassoRegression(penalty=penalty).fit(FEATURES, TARGET)

    assert mixed.intercept == pytest.approx(pure.intercept, abs=1e-12)
    for name in ("first", "second"):
        assert mixed.coefficients[name] == pytest.approx(
            pure.coefficients[name], abs=1e-12
        )


@pytest.mark.parametrize("penalty", PENALTIES)
def test_no_share_on_l1_is_this_backend_s_own_ridge(
    backend: ModuleType, penalty: float
) -> None:
    """The lower edge.

    Looser than the upper one, and for a reason worth knowing: ridge has a
    closed form and this does not, so what is being compared is a walk against
    an exact answer. The gap is the solver's own convergence tolerance, 2.5e-10
    on the from-scratch sweep and 3.1e-12 on the engine, and not a
    disagreement about what is being minimised.
    """
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    RidgeRegression = provided(backend, "RidgeRegression")

    mixed = ElasticNetRegression(penalty=penalty, l1_share=0.0).fit(FEATURES, TARGET)
    pure = RidgeRegression(penalty=penalty).fit(FEATURES, TARGET)

    assert mixed.intercept == pytest.approx(pure.intercept, abs=1e-8)
    for name in ("first", "second"):
        assert mixed.coefficients[name] == pytest.approx(
            pure.coefficients[name], abs=1e-8
        )


@pytest.mark.parametrize("l1_share", [0.0, 0.25, 0.5, 0.75])
def test_any_share_of_l2_splits_two_identical_columns_evenly(
    backend: ModuleType, l1_share: float
) -> None:
    """The reason this model exists, and the one claim the objective forces.

    The squared term is strictly convex in the direction that separates two
    identical columns, so their optimum is unique and symmetric. Pure L1 has
    no such term and leaves that direction flat, which is the case below this
    one and the case this spec declines to pin.

    Both solvers are asked for the optimum rather than for their defaults,
    because identical columns are the case coordinate descent is slowest on
    and the smaller the L2 share the slower it gets. At the default tolerance
    the from-scratch sweep hits its cap with the two still 1.5e-08 apart at a
    share of 0.5 and 3.2e-04 apart at 0.75, and says so through ``converged``.
    Asserting the optimum needs the solver pointed at the optimum.
    """
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(
        penalty=4.0, l1_share=l1_share, tolerance=1e-14, max_iterations=200_000
    ).fit(TWINS, TWIN_TARGET)

    assert model.converged is True
    assert model.coefficients["twin_one"] == pytest.approx(
        model.coefficients["twin_two"], abs=1e-11
    )
    assert model.coefficients["twin_one"] > 0.5


def test_a_share_of_one_drops_one_of_the_two_columns(backend: ModuleType) -> None:
    """Selection survives at the mixture's upper edge, and the even split does
    not.

    Which of the two is kept is the solver's choice and is not asserted. What
    is asserted is that one of them is discarded rather than the effect being
    shared, which is exactly what any share of L2 above prevents.

    The discarded one is *not* exactly zero here, which is worth knowing and
    is not a failure of the soft threshold. That operator does return exactly
    zero whenever its argument is inside the threshold, and with identical
    columns the argument lands precisely *on* the threshold once the other
    column has taken the effect, so what is left is the rounding in a
    subtraction of two equal numbers: measured, 6.0e-15 on the from-scratch
    sweep and 4.2e-16 on the engine. A feature dropped from a fixture that is
    not degenerate does reach exactly zero, and the lasso spec asserts that.
    """
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=4.0, l1_share=1.0).fit(TWINS, TWIN_TARGET)

    kept = sorted(model.coefficients[name] for name in ("twin_one", "twin_two"))

    assert kept[0] == pytest.approx(0.0, abs=1e-12)
    assert kept[1] > 5.0


def test_it_matches_prediction_columns_by_name(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=1.0).fit(FEATURES, TARGET)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.allclose(
        np.asarray(model.predict(reversed_order)), np.asarray(model.predict(FEATURES))
    )


def test_without_an_intercept_the_plane_passes_through_the_origin(
    backend: ModuleType,
) -> None:
    """``fit_intercept`` is a keyword this spec otherwise never states, and a
    wrapper that hardcoded it would satisfy every test above."""
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=BARELY_PENALISED, fit_intercept=False).fit(
        FEATURES, TARGET
    )

    assert model.intercept == 0.0
    assert model.coefficients["first"] == pytest.approx(
        THROUGH_THE_ORIGIN_FIRST, abs=1e-4
    )
    assert model.coefficients["second"] == pytest.approx(
        THROUGH_THE_ORIGIN_SECOND, abs=1e-4
    )


def test_it_reports_how_the_sweeps_ended(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=1.0).fit(FEATURES, TARGET)

    assert model.converged is True
    assert isinstance(model.iterations_run, int)
    assert model.iterations_run <= model.max_iterations


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")
    model = ElasticNetRegression(penalty=BARELY_PENALISED).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) > 0.99


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")

    with pytest.raises(NotFittedError):
        ElasticNetRegression().predict(FEATURES)


def test_it_refuses_to_report_its_coefficients_before_fit(backend: ModuleType) -> None:
    ElasticNetRegression = provided(backend, "ElasticNetRegression")

    with pytest.raises(NotFittedError):
        _ = ElasticNetRegression().coefficients
