"""The contract every backend's GradientBoostingRegressor keeps.

On a step with a wide gap, a stump finds the step in one round and every
later round shrinks what is left by ``1 - learning_rate``. Forty rounds at
one half leave an error of ten times two to the minus forty in exact
arithmetic, and a backend is allowed to stop refining once a round's gain is
below its own rounding floor, which is why the prediction is asserted to
1e-4 rather than to the last bit. The starting constant is the target's mean
by definition.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

_POSITIONS = np.array(
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        21.0,
        22.0,
        23.0,
        24.0,
        25.0,
        26.0,
        27.0,
        28.0,
    ]
)
_DISTRACTOR = np.array(
    [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0, 8.0, 9.0, 7.0, 9.0, 3.0]
)
_TARGETS = np.array([0.0] * 8 + [10.0] * 8)
FEATURES = [Feature("position", _POSITIONS), Feature("distractor", _DISTRACTOR)]
TARGET = Feature("level", _TARGETS)

N_ROUNDS = 40
LEARNING_RATE = 0.5
TARGET_MEAN = 5.0

#: One predictor with a lone outlier, for the leaf minimum. A single round at
#: a learning rate of one is the mean plus one unpruned tree on the residuals,
#: so the ensemble predicts what that tree predicts: 20 for the last row when
#: the tree may isolate it, 12.5 when every leaf must hold three rows and the
#: four-row node behind the step cannot be cut at all.
_OUTLIER_POSITIONS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_OUTLIER_TARGETS = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0])
OUTLIER_FEATURES = [Feature("position", _OUTLIER_POSITIONS)]
OUTLIER_TARGET = Feature("level", _OUTLIER_TARGETS)
LAST_ROW = [Feature("position", [8.0])]

#: Two questions, the second asked under the first, for the depth cap.
#: ``height = 10 * (first > 3.5) + 3 * second``, so one round at a learning
#: rate of one is the mean plus one tree on the residuals and the cap decides
#: how much of the second question that tree may ask. Allowed one level it
#: asks about ``first`` only and leaves the ``second`` term unexplained;
#: allowed two it asks both and reproduces every row.
_TWO_STEP_FIRST = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_TWO_STEP_SECOND = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
_TWO_STEP_HEIGHTS = 10.0 * (_TWO_STEP_FIRST > 3.5) + 3.0 * _TWO_STEP_SECOND
TWO_STEP_FEATURES = [
    Feature("first", _TWO_STEP_FIRST),
    Feature("second", _TWO_STEP_SECOND),
]
TWO_STEP_TARGET = Feature("height", _TWO_STEP_HEIGHTS)

#: The stump cuts at ``first > 3.5``, so it predicts 1.0 for the three low
#: rows and 11.8 for the five high ones, leaving 16.8 of the fixture's 235.5
#: total squared deviation unexplained.
SCORE_OF_ONE_STUMP = 1.0 - 16.8 / 235.5


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=1
    )

    assert model.n_rounds == N_ROUNDS
    assert model.learning_rate == pytest.approx(LEARNING_RATE)
    assert model.max_depth == 1


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=1
    )

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=1
    ).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=1e-4)


def test_it_starts_from_the_target_mean_and_adds_one_member_per_round(
    backend: ModuleType,
) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=1
    ).fit(FEATURES, TARGET)

    assert model.initial_prediction == pytest.approx(TARGET_MEAN)
    assert len(model.members) == N_ROUNDS
    assert all(member.max_depth == 1 for member in model.members)
    assert all(member.is_fitted for member in model.members)


@pytest.mark.parametrize(
    ("min_samples_leaf", "prediction_at_the_outlier", "member_leaves"),
    [(1, 20.0, 3), (3, 12.5, 2)],
    ids=["unconstrained", "three_per_leaf"],
)
def test_the_leaf_minimum_reaches_every_round(
    backend: ModuleType,
    min_samples_leaf: int,
    prediction_at_the_outlier: float,
    member_leaves: int,
) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=1, learning_rate=1.0, max_depth=None, min_samples_leaf=min_samples_leaf
    ).fit(OUTLIER_FEATURES, OUTLIER_TARGET)

    assert model.predict(LAST_ROW)[0] == pytest.approx(prediction_at_the_outlier)
    assert model.members[0].n_leaves == member_leaves


@pytest.mark.parametrize(
    ("max_depth", "member_depth", "member_leaves", "score"),
    [(1, 1, 2, SCORE_OF_ONE_STUMP), (2, 2, 4, 1.0)],
    ids=["one_level", "two_levels"],
)
def test_the_depth_cap_reaches_every_round(
    backend: ModuleType,
    max_depth: int,
    member_depth: int,
    member_leaves: int,
    score: float,
) -> None:
    """The test above reads ``member.max_depth`` back, which is a field.

    Every member is rebuilt from this model's own configuration, so that
    number is the one that was asked for whatever tree the engine actually
    grew, and a wrapper handing the engine a different cap still satisfies
    it. The depth the tree reached, and what it costs the fit, are what a
    cap that never arrived cannot fake. Both figures come from the fixture's
    own arithmetic rather than from the other backend.
    """
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=1, learning_rate=1.0, max_depth=max_depth
    ).fit(TWO_STEP_FEATURES, TWO_STEP_TARGET)

    member = model.members[0]

    assert member.depth == member_depth
    assert member.n_leaves == member_leaves
    assert model.score(TWO_STEP_FEATURES, TWO_STEP_TARGET) == pytest.approx(score)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")
    model = GradientBoostingRegressor(
        n_rounds=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=1
    ).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) > 0.999


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    GradientBoostingRegressor = provided(backend, "GradientBoostingRegressor")

    with pytest.raises(NotFittedError):
        GradientBoostingRegressor(n_rounds=N_ROUNDS).predict(FEATURES)
