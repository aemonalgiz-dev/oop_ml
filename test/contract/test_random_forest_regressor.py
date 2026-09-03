"""The contract every backend's RandomForestRegressor keeps.

The same gapped step as the bagging contract, with the forest allowed one
feature per node. Half the time that feature is the distractor, so a member
takes a wrong first turn and recovers below it, and the training rows a member
never drew are the ones it can misroute. The prediction is therefore asserted
to within the step rather than exactly, and the score loosely.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import InvalidValuesError, NotFittedError

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

N_MEMBERS = 20


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    model = RandomForestRegressor(n_members=N_MEMBERS, max_features=1, random_seed=3)

    assert model.n_members == N_MEMBERS
    assert model.max_features == 1
    assert model.random_seed == 3


def test_it_refuses_a_configured_base_model(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")

    with pytest.raises(InvalidValuesError):
        RandomForestRegressor(base_model=DecisionTreeRegressor(max_depth=2))


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    model = RandomForestRegressor(n_members=N_MEMBERS, max_features=1, random_seed=0)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    model = RandomForestRegressor(n_members=N_MEMBERS, max_features=1, random_seed=0)
    model.fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS, atol=5.0)


def test_its_members_are_trees_restricted_as_configured(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    model = RandomForestRegressor(n_members=N_MEMBERS, max_features=1, random_seed=0)
    model.fit(FEATURES, TARGET)

    assert len(model.members) == N_MEMBERS
    assert all(member.max_features == 1 for member in model.members)
    assert len(model.samples) == N_MEMBERS


def test_the_restriction_reaches_the_members(backend: ModuleType) -> None:
    """Every unrestricted member roots on the step, because it is the best
    question on every resample. Allowed one feature per node, a member whose
    root drew the distractor has to ask about the distractor first, and at
    twenty members the chance that none did is one in a million. Reading the
    roots is what separates a forest from bagging wearing the name."""
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    restricted = RandomForestRegressor(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    )
    unrestricted = RandomForestRegressor(n_members=N_MEMBERS, random_seed=0)

    restricted_roots = {
        member.root.split.feature_name
        for member in restricted.fit(FEATURES, TARGET).members
    }
    unrestricted_roots = {
        member.root.split.feature_name
        for member in unrestricted.fit(FEATURES, TARGET).members
    }

    assert restricted_roots == {"position", "distractor"}
    assert unrestricted_roots == {"position"}


def test_its_importances_are_addressable_by_feature_name(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    model = RandomForestRegressor(n_members=N_MEMBERS, max_features=1, random_seed=0)
    model.fit(FEATURES, TARGET)

    importances = model.feature_importances

    assert importances["position"] > importances["distractor"]
    assert importances["position"] + importances["distractor"] == pytest.approx(1.0)


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")
    model = RandomForestRegressor(n_members=N_MEMBERS, max_features=1, random_seed=0)
    model.fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) > 0.8
    assert model.out_of_bag_score() > 0.5


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    RandomForestRegressor = provided(backend, "RandomForestRegressor")

    with pytest.raises(NotFittedError):
        RandomForestRegressor(n_members=N_MEMBERS).predict(FEATURES)
