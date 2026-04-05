"""The contract every backend's BaggingClassifier keeps.

The gap fixture of the bagging regressor, with the level replaced by a
class. Any member that drew rows from both sides puts its cut in the gap and
routes every training row correctly under either backend's threshold rule,
so the averaged probabilities are one-hot and the predictions exact.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.exceptions import NotFittedError, SingleClassError

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

N_MEMBERS = 10


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=3)

    assert model.n_members == N_MEMBERS
    assert model.random_seed == 3


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0)

    assert model.fit(FEATURES, TARGET) is model
    assert model.n_classes == 2


def test_it_predicts_one_class_per_row(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_CLASSES)
    assert np.array_equal(np.asarray(predictions), _CLASSES)


def test_its_probability_rows_are_distributions(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    probabilities = model.predict_probabilities(FEATURES)

    assert isinstance(probabilities, ProbabilityMatrix)
    assert probabilities.shape == (len(_CLASSES), 2)
    assert np.allclose(
        np.asarray(probabilities), np.column_stack([1 - _CLASSES, _CLASSES])
    )


def test_it_holds_one_fitted_member_and_one_resample_each(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert len(model.members) == N_MEMBERS
    assert all(member.is_fitted for member in model.members)
    assert all(member.n_classes == 2 for member in model.members)
    assert len(model.samples) == N_MEMBERS
    assert all(sample.n_rows == len(_CLASSES) for sample in model.samples)


def test_the_seed_fixes_every_resample(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    first = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)
    second = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)
    other = BaggingClassifier(n_members=N_MEMBERS, random_seed=1).fit(FEATURES, TARGET)

    assert all(
        np.array_equal(one.drawn, another.drawn)
        for one, another in zip(first.samples, second.samples, strict=True)
    )
    assert any(
        not np.array_equal(one.drawn, another.drawn)
        for one, another in zip(first.samples, other.samples, strict=True)
    )


def test_its_importances_are_addressable_by_feature_name(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert model.feature_importances["position"] == pytest.approx(1.0)
    assert model.feature_importances["distractor"] == pytest.approx(0.0)


def test_it_scores_itself_on_the_rows_each_member_missed(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert model.out_of_bag_score() > 0.9


def test_it_refuses_a_target_with_a_gap_in_its_classes(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")

    with pytest.raises(SingleClassError):
        BaggingClassifier(n_members=N_MEMBERS).fit(
            FEATURES, Feature("side", 2.0 * _CLASSES)
        )


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")
    model = BaggingClassifier(n_members=N_MEMBERS, random_seed=0).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    BaggingClassifier = provided(backend, "BaggingClassifier")

    with pytest.raises(NotFittedError):
        BaggingClassifier(n_members=N_MEMBERS).predict(FEATURES)
