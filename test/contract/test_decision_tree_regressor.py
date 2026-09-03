"""The contract every backend's DecisionTreeRegressor keeps.

A step in one feature beside a feature that explains nothing, so a tree of
depth one has exactly one right question and the fixture says which. The
distractor's values are chosen so that no cut on it separates the two levels,
which keeps the two backends from breaking a tie differently.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import NotFittedError
from oop_ml.core.tree.node import DecisionNode

from .harness import provided

#: The target steps from 0 to 10 between position 4 and position 5.
_POSITIONS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_DISTRACTOR = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
_TARGETS = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0])
FEATURES = [Feature("position", _POSITIONS), Feature("distractor", _DISTRACTOR)]
TARGET = Feature("level", _TARGETS)

#: Halfway between the last low row and the first high one.
STEP_THRESHOLD = 4.5

#: One predictor and a lone outlier at the end, for the stopping rules. Grown
#: freely the tree cuts at 4.5 and then isolates the 20 behind a cut at 7.5.
#: Requiring two rows per leaf forbids that second cut, and the best legal one
#: on (10, 10, 10, 20) is at 6.5, leaving (10, 20) to predict 15. Requiring
#: three forbids every cut of a four-row node, as does requiring five rows to
#: split, so the right child stays a leaf predicting 12.5.
#:
#: The same fixture also fixes where a floor on the gain has to bite, and the
#: two figures it turns on are the fixture's own. The right child
#: (10, 10, 10, 20) has variance 18.75, so no cut inside it can remove more
#: than that, and the root's own cut removes 39.0625. A floor above the
#: first collapses the right child to one leaf and a floor above the second
#: collapses the whole tree. The backends measure the gain on different
#: scales, this library's directly and the engine's weighted by the node's
#: share of the rows, so they part between 9.5 and 19.0 and agree on either
#: side; 25.0 and 50.0 are chosen clear of that band.
_OUTLIER_TARGETS = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0])
OUTLIER_FEATURES = [Feature("position", _POSITIONS)]
OUTLIER_TARGET = Feature("level", _OUTLIER_TARGETS)
LAST_ROW = [Feature("position", [8.0])]

#: Two questions, the second asked under the first, with unequal children so a
#: gain that forgets to weight the children by their rows comes out wrong.
#: ``height = 10 * (first > 3.5) + 3 * second``. Every leaf is pure, so the
#: credit totals ``n * variance(height) = 235.5``; the root removes
#: ``8 * 27.3375 = 218.7`` of it and the two splits on ``second`` the rest,
#: ``3 * 2.0 + 5 * 2.16 = 16.8``.
_FIRST = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_SECOND = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
_HEIGHTS = 10.0 * (_FIRST > 3.5) + 3.0 * _SECOND
TWO_STEP_FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
TWO_STEP_TARGET = Feature("height", _HEIGHTS)
FIRST_SHARE = 218.7 / 235.5
SECOND_SHARE = 16.8 / 235.5


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1, min_samples_leaf=2)

    assert model.max_depth == 1
    assert model.min_samples_leaf == 2


def test_it_fits_features_and_a_target_and_returns_itself(backend: ModuleType) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1)

    assert model.fit(FEATURES, TARGET) is model


def test_it_predicts_one_answer_per_row(backend: ModuleType) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1).fit(FEATURES, TARGET)

    predictions = model.predict(FEATURES)

    assert len(predictions) == len(_TARGETS)
    assert np.allclose(np.asarray(predictions), _TARGETS)


def test_it_asks_the_one_question_that_separates_the_levels(
    backend: ModuleType,
) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1).fit(FEATURES, TARGET)

    root = model.root

    assert isinstance(root, DecisionNode)
    assert root.split.feature_name == "position"
    assert root.split.threshold == pytest.approx(STEP_THRESHOLD)
    assert model.depth == 1
    assert model.n_leaves == 2


def test_its_importances_are_addressable_by_feature_name(backend: ModuleType) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1).fit(FEATURES, TARGET)

    assert model.feature_importances["position"] == pytest.approx(1.0)
    assert model.feature_importances["distractor"] == pytest.approx(0.0)


def test_its_importances_weight_each_split_by_the_rows_it_decided(
    backend: ModuleType,
) -> None:
    """A single split can only ever score 1.0, so it cannot tell a right gain
    from a wrong one. Two levels with unequal children can."""
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor().fit(TWO_STEP_FEATURES, TWO_STEP_TARGET)

    assert model.depth == 2
    assert model.n_leaves == 4
    assert np.allclose(np.asarray(model.predict(TWO_STEP_FEATURES)), _HEIGHTS)
    assert model.feature_importances["first"] == pytest.approx(FIRST_SHARE)
    assert model.feature_importances["second"] == pytest.approx(SECOND_SHARE)


@pytest.mark.parametrize(
    ("configuration", "prediction_at_the_outlier", "n_leaves", "depth"),
    [
        ({}, 20.0, 3, 2),
        ({"min_samples_leaf": 2}, 15.0, 3, 2),
        ({"min_samples_leaf": 3}, 12.5, 2, 1),
        ({"min_samples_split": 5}, 12.5, 2, 1),
        ({"min_impurity_decrease": 25.0}, 12.5, 2, 1),
        ({"min_impurity_decrease": 50.0}, 6.25, 1, 0),
    ],
    ids=[
        "unconstrained",
        "two_per_leaf",
        "three_per_leaf",
        "five_to_split",
        "floor_above_the_second_cut",
        "floor_above_the_root",
    ],
)
def test_its_stopping_rules_change_the_tree_it_grows(
    backend: ModuleType,
    configuration: dict[str, float],
    prediction_at_the_outlier: float,
    n_leaves: int,
    depth: int,
) -> None:
    """Constructing with a rule is not the same as growing under it. Each rule
    here changes what the last row predicts, worked out in the fixture."""
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(**configuration).fit(OUTLIER_FEATURES, OUTLIER_TARGET)

    assert model.predict(LAST_ROW)[0] == pytest.approx(prediction_at_the_outlier)
    assert model.n_leaves == n_leaves
    assert model.depth == depth


def test_it_describes_itself_in_terms_of_its_features(backend: ModuleType) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1).fit(FEATURES, TARGET)

    description = model.describe()

    assert isinstance(description, str)
    assert "position" in description


def test_it_scores_the_fit_it_made(backend: ModuleType) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")
    model = DecisionTreeRegressor(max_depth=1).fit(FEATURES, TARGET)

    assert model.score(FEATURES, TARGET) == pytest.approx(1.0)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    DecisionTreeRegressor = provided(backend, "DecisionTreeRegressor")

    with pytest.raises(NotFittedError):
        DecisionTreeRegressor(max_depth=1).predict(FEATURES)
