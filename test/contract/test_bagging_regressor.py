"""The contract every backend's BaggingRegressor keeps.

The step fixture leaves a wide gap between its two levels, so any member
that drew rows from both sides puts its cut somewhere in the gap and routes
every training row correctly under either backend's threshold rule. That is
what lets the averaged prediction be asserted exactly rather than loosely.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError

from .harness import provided

#: Eight low rows, a gap, eight high rows.
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

N_MEMBERS = 10

#: Twelve rows a tree can only answer by memorising, one target per row and
#: no two neighbours alike, so a member fitted on a resample reproduces the
#: rows it drew exactly and answers a neighbour's level for one it missed.
#: That is what makes the pairing of members to samples observable. Measured
#: at six members and seed 7, every member is exact on its own drawn rows and
#: the smallest error any *mismatched* pairing would show is 7.0 on the numpy
#: backend and 14.0 here, against levels spanning 1 to 30.
_SPOTS = np.arange(1.0, 13.0)
_LEVELS = np.array([5.0, 18.0, 2.0, 25.0, 11.0, 30.0, 7.0, 22.0, 1.0, 16.0, 9.0, 28.0])
MEMORISED_FEATURES = [Feature("spot", _SPOTS)]
MEMORISED_TARGET = Feature("level", _LEVELS)
N_PAIRED_MEMBERS = 6


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=3)

    assert model.n_members == N_MEMBERS
    assert model.random_seed == 3


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=0)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=1e-6)


def test_it_holds_one_fitted_member_and_one_resample_each(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert len(model.members) == N_MEMBERS
    assert all(member.is_fitted for member in model.members)
    assert len(model.samples) == N_MEMBERS
    assert all(sample.n_rows == len(_TARGETS) for sample in model.samples)


def test_each_member_is_the_one_fitted_on_the_sample_beside_it(
    backend: ModuleType,
) -> None:
    """Counting the two collections is not the same as pairing them.

    ``members[i]`` has to be the model fitted on ``samples[i]``, because
    ``out_of_bag_estimate`` scores each member against the rows that sample
    says it missed. Two collections read out of one engine and zipped in
    different orders would give every member rows it had actually seen, and
    every assertion above still holds.
    """
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_PAIRED_MEMBERS, random_seed=7).fit(
        MEMORISED_FEATURES, MEMORISED_TARGET
    )

    for member, sample in zip(model.members, model.samples, strict=True):
        answers = np.asarray(member.predict(MEMORISED_FEATURES))
        drawn = sample.drawn
        missed = sample.out_of_bag

        assert np.allclose(answers[drawn], _LEVELS[drawn])
        assert len(missed) > 0
        assert np.max(np.abs(answers[missed] - _LEVELS[missed])) > 1.0


def test_it_bags_the_base_model_it_was_given(backend: ModuleType) -> None:
    """``base_model`` is otherwise named only where a forest refuses one.

    Every fit in this spec runs at the default tree, so an ensemble that
    ignored the field and always bagged a tree would satisfy all of them.
    A linear member is the cheapest thing that cannot be mistaken for one,
    since it answers by name-addressable coefficients rather than by leaves.
    """
    BaggingRegressor = provided(backend, "BaggingRegressor")
    MultipleLinearRegression = provided(backend, "MultipleLinearRegression")
    model = BaggingRegressor(
        base_model=MultipleLinearRegression(), n_members=5, random_seed=0
    ).fit(FEATURES, TARGET)

    assert len(model.members) == 5
    for member in model.members:
        assert isinstance(member, MultipleLinearRegression)
        assert isinstance(member.coefficients["position"], float)
        assert isinstance(member.intercept, float)


def test_the_seed_fixes_every_resample(backend: ModuleType) -> None:
    """Two fits under one seed draw the same rows for every member, and a
    different seed draws differently. Without this, ``random_seed`` is a
    field the fit never reads."""
    BaggingRegressor = provided(backend, "BaggingRegressor")
    first = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)
    second = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)
    other = BaggingRegressor(n_members=N_MEMBERS, random_seed=1).fit(FEATURES, TARGET)

    assert all(
        np.array_equal(one.drawn, another.drawn)
        for one, another in zip(first.samples, second.samples, strict=True)
    )
    assert any(
        not np.array_equal(one.drawn, another.drawn)
        for one, another in zip(first.samples, other.samples, strict=True)
    )


def test_its_importances_are_addressable_by_feature_name(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert model.feature_importances["position"] == pytest.approx(1.0)
    assert model.feature_importances["distractor"] == pytest.approx(0.0)


def test_it_scores_itself_on_the_rows_each_member_missed(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert model.out_of_bag_score() > 0.9


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")
    model = BaggingRegressor(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    BaggingRegressor = provided(backend, "BaggingRegressor")

    with pytest.raises(NotFittedError):
        BaggingRegressor(n_members=N_MEMBERS).predict(FEATURES)
