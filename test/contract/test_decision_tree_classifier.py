"""The contract every backend's DecisionTreeClassifier keeps.

A step in one feature beside a feature that explains nothing, as for the
regression tree, with the level replaced by a class. The distractor's values
are chosen so no cut on it separates the classes, which keeps the two
backends from breaking a tie differently. A three-class staircase then pins
the second question under the first, and a stated class width is held to
producing a row of that width whichever classes the fit met.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.data.probabilities import ProbabilityMatrix
from oop_ml.core.exceptions import NotFittedError, SingleClassError
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.node import DecisionNode

from .harness import provided

_POSITIONS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_DISTRACTOR = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
_CLASSES = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
FEATURES = [Feature("position", _POSITIONS), Feature("distractor", _DISTRACTOR)]
TARGET = Feature("side", _CLASSES)

STEP_THRESHOLD = 4.5

#: Three classes in three runs, so a pure tree needs two cuts on position.
_STAIRCASE = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0])
STAIRCASE_TARGET = Feature("step", _STAIRCASE)

#: Classes 0 and 2 only, inside a stated width of three.
_GAPPED = np.array([0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
GAPPED_TARGET = Feature("side", _GAPPED)

#: Each criterion beside the impurity it gives the root, which is four rows
#: of each class. Gini there is ``1 - 2 * 0.5**2`` and entropy in bits is
#: 1.0. The pairing is what makes the parametrization discriminating, since
#: every strictly concave measure picks the same cut on this fixture, so the
#: split alone cannot tell the two criteria apart and a backend hard-coding
#: one of them would pass. Both backends measure entropy in bits, so one
#: expected number serves both.
ROOT_IMPURITY_BY_CRITERION = [
    (ClassificationCriterion.GINI, 0.5),
    (ClassificationCriterion.ENTROPY, 1.0),
]

#: A floor either side of the root's gain, with the tree each one leaves.
#: The root's cut removes all of its impurity, so under Gini the gain is
#: 0.5, and the root holds every row, so the engine's scaling of a gain by
#: the node's share of the rows is 1 here and the two backends compare the
#: same number. That is what lets one expected tree serve both; away from
#: the root the scaling makes them part, which the wrapper documents.
TREE_BY_IMPURITY_FLOOR = [(0.4, 2, 1), (0.6, 1, 0)]


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(
        criterion=ClassificationCriterion.ENTROPY, max_depth=1, n_known_classes=4
    )

    assert model.criterion == ClassificationCriterion.ENTROPY
    assert model.max_depth == 1
    assert model.n_known_classes == 4


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1)

    assert model.fit(FEATURES, TARGET) is model
    assert model.n_classes == 2


def test_it_predicts_one_class_per_row(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_CLASSES)
    assert np.array_equal(np.asarray(predictions), _CLASSES)


def test_its_pure_leaves_report_certainty(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1).fit(FEATURES, TARGET)

    probabilities = model.predict_probabilities(FEATURES)

    assert isinstance(probabilities, ProbabilityMatrix)
    expected = np.column_stack([1.0 - _CLASSES, _CLASSES])
    assert np.allclose(np.asarray(probabilities), expected)


@pytest.mark.parametrize(
    ("criterion", "root_impurity"),
    ROOT_IMPURITY_BY_CRITERION,
    ids=[one.value for one, _ in ROOT_IMPURITY_BY_CRITERION],
)
def test_it_asks_the_one_question_that_separates_the_classes(
    backend: ModuleType, criterion: ClassificationCriterion, root_impurity: float
) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(criterion=criterion, max_depth=1).fit(
        FEATURES, TARGET
    )

    root = model.root

    assert isinstance(root, DecisionNode)
    assert root.split.feature_name == "position"
    assert root.split.threshold == pytest.approx(STEP_THRESHOLD)
    assert root.impurity == pytest.approx(root_impurity)
    assert model.depth == 1
    assert model.n_leaves == 2


@pytest.mark.parametrize(
    ("impurity_floor", "n_leaves", "depth"),
    TREE_BY_IMPURITY_FLOOR,
    ids=[f"floor-{floor}" for floor, _, _ in TREE_BY_IMPURITY_FLOOR],
)
def test_the_impurity_floor_decides_whether_the_root_splits_at_all(
    backend: ModuleType, impurity_floor: float, n_leaves: int, depth: int
) -> None:
    """Without this, ``min_impurity_decrease`` is a field the fit never reads.

    A backend that discarded it on the way to its solver grows the same tree
    at every setting, and nothing else in this spec looks at the field.
    """
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(min_impurity_decrease=impurity_floor).fit(
        FEATURES, TARGET
    )

    assert model.n_leaves == n_leaves
    assert model.depth == depth


def test_its_importances_are_addressable_by_feature_name(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1).fit(FEATURES, TARGET)

    assert model.feature_importances["position"] == pytest.approx(1.0)
    assert model.feature_importances["distractor"] == pytest.approx(0.0)


def test_it_grows_a_second_question_under_the_first(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier().fit([FEATURES[0]], STAIRCASE_TARGET)

    assert model.n_classes == 3
    assert model.depth == 2
    assert model.n_leaves == 3
    assert np.array_equal(np.asarray(model.predict([FEATURES[0]])), _STAIRCASE)
    assert model.feature_importances["position"] == pytest.approx(1.0)


def test_a_stated_width_holds_whichever_classes_the_fit_met(
    backend: ModuleType,
) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1, n_known_classes=3).fit(
        FEATURES, GAPPED_TARGET
    )

    probabilities = np.asarray(model.predict_probabilities(FEATURES))

    assert model.n_classes == 3
    assert probabilities.shape == (len(_GAPPED), 3)
    assert np.allclose(probabilities[:, 1], 0.0)
    assert np.array_equal(np.asarray(model.predict(FEATURES)), _GAPPED)


def test_without_a_stated_width_a_gap_is_refused(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")

    with pytest.raises(SingleClassError):
        DecisionTreeClassifier(max_depth=1).fit(FEATURES, GAPPED_TARGET)


def test_it_describes_itself_in_terms_of_its_features(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1).fit(FEATURES, TARGET)

    description = model.describe()

    assert isinstance(description, str)
    assert "position" in description


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")
    model = DecisionTreeClassifier(max_depth=1).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    DecisionTreeClassifier = provided(backend, "DecisionTreeClassifier")

    with pytest.raises(NotFittedError):
        DecisionTreeClassifier(max_depth=1).predict(FEATURES)
