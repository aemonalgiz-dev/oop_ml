"""The observed route must record the calculation, not a version of it.

``split_search`` exists so a caller can see how a node decides. That is only
worth anything if it is the *same* decision the fit makes, so the tests that
matter here are the agreement tests: whatever ``_best_split`` returns,
``split_search(...).best`` returns too, across every shape that makes the two
routes take different branches.

This is the arrangement the neighbour models already use, where a threaded
search is justified by a test proving it returns what the serial one does. A
fast path and a slow path with no test between them are two implementations,
not one calculation with two readings.

The rest of the file covers what the record has to contain to be observable at
all: every candidate, in scan order, with the reason each was excluded. A
search that only kept the admitted ones would answer "which won" and not
"against what", and the second question is the one a reader has.
"""

import numpy as np
import pytest

from oop_ml.core.data.column import Column
from oop_ml.core.data.row_block import rows_of
from oop_ml.core.observation import Observation
from oop_ml.core.tree.criterion import ClassificationCriterion
from oop_ml.core.tree.search import SplitRejection, SplitSearch
from oop_ml.core.tree.split import Split
from oop_ml.core.validation import ValueRole
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import EXAM_OUTCOMES, STEP_FUNCTION

EXAM_ROWS = rows_of(
    np.column_stack([feature.values for feature in EXAM_OUTCOMES.input_features]),
    [feature.name for feature in EXAM_OUTCOMES.input_features],
)
EXAM_TARGETS = EXAM_OUTCOMES.class_feature.column
STEP_ROWS = rows_of(
    np.column_stack([feature.values for feature in STEP_FUNCTION.input_features]),
    [feature.name for feature in STEP_FUNCTION.input_features],
)
STEP_TARGETS = STEP_FUNCTION.target_feature.column


def classifier(**overrides) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(**overrides)
    model.fit(EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature)

    return model


def unfitted_classifier(**overrides) -> DecisionTreeClassifier:
    """A classifier with just enough state set to search, without growing.

    ``fit`` cannot be used while ``_grow`` is a stub, and these tests are about
    the search rather than the tree.
    """
    model = DecisionTreeClassifier(**overrides)
    model._feature_names = ("studied", "slept")
    model._n_classes = 2

    return model


def regressor(**overrides) -> DecisionTreeRegressor:
    model = DecisionTreeRegressor(**overrides)
    model._feature_names = ("position",)

    return model


class TestTheTwoRoutesAgree:
    @pytest.mark.parametrize("min_samples_leaf", [1, 2, 4, 6, 7, 8, 9])
    def test_across_every_leaf_size(self, min_samples_leaf):
        model = unfitted_classifier(min_samples_leaf=min_samples_leaf)

        fast = model._best_split(EXAM_ROWS, EXAM_TARGETS)
        recorded = model.split_search(EXAM_ROWS, EXAM_TARGETS).best

        assert _same(fast, recorded)

    @pytest.mark.parametrize("criterion", list(ClassificationCriterion))
    def test_across_every_criterion(self, criterion):
        model = unfitted_classifier(criterion=criterion)

        assert _same(
            model._best_split(EXAM_ROWS, EXAM_TARGETS),
            model.split_search(EXAM_ROWS, EXAM_TARGETS).best,
        )

    @pytest.mark.parametrize("minimum", [0.0, 0.05, 0.2, 0.25, 1.0])
    def test_across_every_impurity_floor(self, minimum):
        model = unfitted_classifier(min_impurity_decrease=minimum)

        assert _same(
            model._best_split(EXAM_ROWS, EXAM_TARGETS),
            model.split_search(EXAM_ROWS, EXAM_TARGETS).best,
        )

    def test_on_the_regressor(self):
        # Different impurity, different scale of gain, same requirement.
        model = regressor()

        assert _same(
            model._best_split(STEP_ROWS, STEP_TARGETS),
            model.split_search(STEP_ROWS, STEP_TARGETS).best,
        )

    def test_when_there_is_nothing_to_find(self):
        # Both must reach None, not one None and one a zero-gain split.
        model = DecisionTreeClassifier()
        model._feature_names = ("flat",)
        model._n_classes = 2
        rows = rows_of(np.full((8, 1), 2.0), ["flat"])
        targets = Column(
            np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
            ValueRole.TARGET_VALUES,
        )

        assert model._best_split(rows, targets) is None
        assert model.split_search(rows, targets).best is None

    def test_on_random_nodes(self):
        # The shapes a fixture cannot cover: ties, constant columns appearing
        # part way through, features that contribute nothing.
        generator = np.random.default_rng(0)

        for _ in range(60):
            n_rows = int(generator.integers(4, 30))
            rows = rows_of(
                generator.integers(0, 4, size=(n_rows, 3)).astype(float),
                ["first", "second", "third"],
            )
            targets = Column(
                generator.integers(0, 2, size=n_rows).astype(float),
                ValueRole.TARGET_VALUES,
            )

            model = DecisionTreeClassifier(
                min_samples_leaf=int(generator.integers(1, 4))
            )
            model._feature_names = ("a", "b", "c")
            model._n_classes = 2

            assert _same(
                model._best_split(rows, targets),
                model.split_search(rows, targets).best,
            )


def _same(first: Split | None, second: Split | None) -> bool:
    if first is None or second is None:
        return first is None and second is None

    return (
        first.feature_index == second.feature_index
        and first.feature_name == second.feature_name
        and first.threshold == pytest.approx(second.threshold)
        and first.gain == pytest.approx(second.gain)
    )


class TestWhatTheRecordHolds:
    def test_every_candidate_is_kept_not_just_the_admitted_ones(self):
        # 13 thresholds on studied, 9 on slept, worked out by hand.
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)

        assert len(search) == 22

    def test_candidates_arrive_in_scan_order(self):
        # Features in fitted order, thresholds ascending -- the order an
        # explanation walks them in, so it cannot be incidental.
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)
        names = [candidate.split.feature_name for candidate in search]

        assert names == ["studied"] * 13 + ["slept"] * 9

        for feature in ("studied", "slept"):
            thresholds = [
                candidate.split.threshold for candidate in search.for_feature(feature)
            ]
            assert thresholds == sorted(thresholds)

    def test_a_rejected_candidate_still_carries_its_gain(self):
        # The interesting case is a split excluded despite scoring well, and
        # that is invisible if the gain is not recorded.
        search = unfitted_classifier(min_samples_leaf=7).split_search(
            EXAM_ROWS, EXAM_TARGETS
        )
        excluded = search.rejected_for(SplitRejection.TOO_FEW_ROWS)

        assert excluded
        assert any(candidate.split.gain > 0.2 for candidate in excluded)

    def test_the_reason_is_recorded_per_candidate(self):
        search = unfitted_classifier(min_samples_leaf=7).split_search(
            EXAM_ROWS, EXAM_TARGETS
        )

        for candidate in search:
            too_small = min(candidate.rows_left, candidate.rows_right) < 7
            assert (candidate.rejection is SplitRejection.TOO_FEW_ROWS) == too_small

    def test_the_row_counts_are_what_the_split_would_produce(self):
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)

        for candidate in search:
            column = EXAM_ROWS.column_at(candidate.split.feature_index)
            expected_left = int((column < candidate.split.threshold).sum())

            assert candidate.rows_left == expected_left
            assert candidate.rows_right == 15 - expected_left

    def test_admitted_is_the_field_the_winner_beat(self):
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)
        best = search.best

        assert best is not None
        assert all(candidate.split.gain <= best.gain for candidate in search.admitted)

    def test_it_can_be_grouped_by_feature(self):
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)

        assert len(search.for_feature("studied")) == 13
        assert len(search.for_feature("slept")) == 9
        assert search.for_feature("nonexistent") == ()


class TestItIsAnObservation:
    def test_result_is_the_answer(self):
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)

        assert _same(search.result, search.best)

    def test_it_satisfies_the_protocol(self):
        # Structural, so generic tooling can walk any explanation without
        # knowing which calculation produced it.
        search = unfitted_classifier().split_search(EXAM_ROWS, EXAM_TARGETS)

        assert isinstance(search, Observation)
        assert isinstance(search, SplitSearch)
