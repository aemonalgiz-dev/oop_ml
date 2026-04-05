"""The contract every backend's RandomForestClassifier keeps.

The bagging classifier's gap fixture again. Restricting each node to one of
the two features forces half the members to root on the distractor, and
they cannot separate the classes there, so the distractor earns a non-zero
importance where plain bagging gave it none. The forest still predicts the
training rows exactly, because the restriction is drawn afresh at every
node and every member reaches position somewhere along its length.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import ProbabilityMatrix
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
_CLASSES = np.array([0.0] * 8 + [1.0] * 8)
FEATURES = [Feature("position", _POSITIONS), Feature("distractor", _DISTRACTOR)]
TARGET = Feature("side", _CLASSES)

N_MEMBERS = 20


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, max_depth=3, random_seed=3
    )

    assert model.n_members == N_MEMBERS
    assert model.max_features == 1
    assert model.max_depth == 3
    assert model.random_seed == 3


def test_it_refuses_a_configured_base_model(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")

    with pytest.raises(InvalidValuesError):
        RandomForestClassifier(base_model=DecisionTreeClassifier(max_depth=1))


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(n_members=N_MEMBERS, max_features=1, random_seed=0)

    assert model.fit(FEATURES, TARGET) is model
    assert model.n_classes == 2


def test_it_predicts_one_class_per_row(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    ).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_CLASSES)
    assert np.array_equal(np.asarray(predictions), _CLASSES)


def test_its_probability_rows_are_distributions(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    ).fit(FEATURES, TARGET)

    probabilities = model.predict_probabilities(FEATURES)

    assert isinstance(probabilities, ProbabilityMatrix)
    assert probabilities.shape == (len(_CLASSES), 2)
    assert np.allclose(np.asarray(probabilities).sum(axis=1), 1.0)


def test_it_holds_one_fitted_member_and_one_resample_each(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    ).fit(FEATURES, TARGET)

    assert len(model.members) == N_MEMBERS
    assert all(member.is_fitted for member in model.members)
    assert all(member.max_features == 1 for member in model.members)
    assert len(model.samples) == N_MEMBERS


def test_the_restriction_spreads_the_root_across_features(backend: ModuleType) -> None:
    """With one feature per node, some members must root on the distractor,
    which plain bagging never does on this fixture."""
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    ).fit(FEATURES, TARGET)

    assert model.feature_importances["position"] > 0.5
    assert model.feature_importances["distractor"] > 0.0


def test_it_scores_itself_on_the_rows_each_member_missed(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    ).fit(FEATURES, TARGET)

    assert model.out_of_bag_score() > 0.8


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")
    model = RandomForestClassifier(
        n_members=N_MEMBERS, max_features=1, random_seed=0
    ).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    RandomForestClassifier = provided(backend, "RandomForestClassifier")

    with pytest.raises(NotFittedError):
        RandomForestClassifier(n_members=N_MEMBERS).predict(FEATURES)
