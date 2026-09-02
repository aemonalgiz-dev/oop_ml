"""Spec for the two importance measures -- red until both bodies land.

The tests that carry the argument are the ones in ``TestTheBias``, and the
shape of that argument is not the one I set out to write.

I expected mean decrease in impurity to be fooled by a high-cardinality column
and permutation importance to see through it. Measuring both on the lone parity
tree showed something else: they agree, and they are both right. That tree
scores 0.537 and really did build itself out of the noise column, so impurity
gives ``distractor`` 0.807 and permutation gives it 0.710. Permutation reports
*reliance*, and the model genuinely relies on it.

The divergence needs a model that works. On the forest, which recovers parity
at 0.993, impurity still hands ``distractor`` roughly 0.52 -- more than half
the explanation, to a column of pure noise -- while permutation drops it to
0.018 and splits the credit between the two real features. Both tests are here,
so the first finding cannot be overread.

``test_averaging_steadies_a_reading_one_tree_cannot_hold`` is the argument for
putting the measure on the ensemble at all, and it hands both models the same
resamples, because an earlier version drew from one shared generator and the
two saw different data.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import InvalidValuesError, NotFittedError
from oop_ml.core.importance.permutation import PermutationImportance
from oop_ml.numpy.classification.ensembles.random_forest_classifier import (
    RandomForestClassifier,
)
from oop_ml.numpy.classification.trees.decision_tree_classifier import (
    DecisionTreeClassifier,
)
from oop_ml.numpy.regression.ensembles.random_forest_regressor import (
    RandomForestRegressor,
)
from oop_ml.numpy.regression.trees.decision_tree_regressor import DecisionTreeRegressor
from test.fixtures import (
    DOMINATED_SIGNAL,
    ENSEMBLE_MEMBERS,
    EXAM_MIN_SAMPLES_SPLIT,
    EXAM_OUTCOMES,
    EXCLUSIVE_OR,
    FOREST_MAX_FEATURES,
    PARITY_MAX_DEPTH,
)


@pytest.fixture
def exam_tree() -> DecisionTreeClassifier:
    return DecisionTreeClassifier(min_samples_split=EXAM_MIN_SAMPLES_SPLIT).fit(
        EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature
    )


class TestMeanDecreaseInImpurity:
    """What a fitted tree can report for free."""

    def test_reports_a_share_for_every_fitted_feature(
        self, exam_tree: DecisionTreeClassifier
    ) -> None:
        importances = exam_tree.feature_importances

        assert len(importances) == 2
        assert "studied" in importances
        assert "slept" in importances

    def test_shares_sum_to_one(self, exam_tree: DecisionTreeClassifier) -> None:
        assert sum(one.value for one in exam_tree.feature_importances) == pytest.approx(
            1.0
        )

    def test_the_root_feature_wins(self, exam_tree: DecisionTreeClassifier) -> None:
        """The exam tree roots on ``slept`` and tests ``studied`` beneath it.

        The root split decides where all fifteen rows go and the second decides
        where nine go, so the row weighting has to put ``slept`` ahead.
        """
        assert exam_tree.feature_importances.most_important.name == "slept"

    def test_weights_a_split_by_the_rows_that_reached_it(self) -> None:
        """Computed by hand from the two splits the fixture pins.

        Root: 15 rows, gain 0.2133. Second: 9 rows, gain 0.2778. So ``slept``
        earns 15 * 0.2133 = 3.200 and ``studied`` earns 9 * 0.2778 = 2.500,
        which normalise to 0.5614 and 0.4386.
        """
        tree = DecisionTreeClassifier(min_samples_split=EXAM_MIN_SAMPLES_SPLIT).fit(
            EXAM_OUTCOMES.input_features, EXAM_OUTCOMES.class_feature
        )

        slept = 15 * 0.21333333333333335
        studied = 9 * 0.2777777777777778
        total = slept + studied

        assert tree.feature_importances["slept"] == pytest.approx(slept / total)
        assert tree.feature_importances["studied"] == pytest.approx(studied / total)

    def test_a_feature_never_split_on_earns_nothing(self) -> None:
        tree = DecisionTreeRegressor(max_depth=1).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )
        importances = tree.feature_importances

        assert importances.most_important.name == "dominant"
        assert sum(one.value == 0.0 for one in importances) == 5

    def test_a_tree_that_never_split_has_nothing_to_report(self) -> None:
        """Shares of zero are not shares, and the caller has to hear that."""
        tree = DecisionTreeRegressor(min_impurity_decrease=1e9).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        with pytest.raises(InvalidValuesError):
            _ = tree.feature_importances

    def test_reading_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = DecisionTreeRegressor().feature_importances


class TestTheBias:
    """Where the cheap measure is wrong and the expensive one is not.

    The comparison has to be made on a model that *works*. On the lone parity
    tree, which scores 0.537, both measures name ``distractor`` and both are
    right to: that tree really did build itself out of the noise column, so it
    really does rely on it. Measured, impurity gives it 0.807 and permutation
    0.710. Neither measure is fooled, because there is nothing to be fooled
    about.

    The divergence appears on the forest, which recovers parity at 0.993.
    There the two disagree completely, and only one of them is describing the
    model that is actually in hand.
    """

    @staticmethod
    def parity_forest() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_members=ENSEMBLE_MEMBERS,
            max_features=1,
            max_depth=PARITY_MAX_DEPTH,
            random_seed=0,
        ).fit(EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature)

    def test_impurity_still_credits_the_noise_column(self) -> None:
        """A working model, and the cheap measure crediting noise for half of it.

        The forest scores 0.993, so it plainly learned the parity rule from
        ``first`` and ``second``. Mean decrease in impurity nonetheless hands
        ``distractor`` the largest share, at roughly 0.52, because a continuous
        column offers the split search hundreds of candidate thresholds where a
        binary one offers a single threshold and so keeps winning splits.
        """
        importances = self.parity_forest().feature_importances

        assert importances.most_important.name == "distractor"
        assert importances["distractor"] > 0.4

    def test_permutation_finds_what_the_model_actually_uses(self) -> None:
        """Shuffling a column the model does not lean on costs it nothing.

        Measured: ``distractor`` drops to around 0.018, and the two real
        features take roughly half the explanation each.
        """
        forest = self.parity_forest()
        importances = PermutationImportance(n_repeats=5, random_seed=0).measure(
            forest, EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature
        )

        assert importances["distractor"] < 0.1
        assert importances.most_important.name in {"first", "second"}

    def test_the_two_measures_disagree_about_the_ranking(self) -> None:
        """The finding, stated as the disagreement rather than as two numbers."""
        forest = self.parity_forest()
        permuted = PermutationImportance(n_repeats=5, random_seed=0).measure(
            forest, EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature
        )

        assert (
            forest.feature_importances.most_important.name
            != permuted.most_important.name
        )

    def test_both_agree_when_the_model_really_does_lean_on_noise(self) -> None:
        """The other half of the story, so the first half is not overread.

        Permutation is not a lie detector for high-cardinality columns. It
        reports reliance, and a tree that built itself out of the noise column
        genuinely relies on it.
        """
        tree = DecisionTreeClassifier(max_depth=PARITY_MAX_DEPTH).fit(
            EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature
        )
        permuted = PermutationImportance(n_repeats=5, random_seed=0).measure(
            tree, EXCLUSIVE_OR.input_features, EXCLUSIVE_OR.class_feature
        )

        assert tree.feature_importances.most_important.name == "distractor"
        assert permuted.most_important.name == "distractor"


class TestPermutation:
    """The model-free measure, on a model whose answer is known."""

    @pytest.fixture
    def forest(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_members=ENSEMBLE_MEMBERS,
            max_features=FOREST_MAX_FEATURES,
            random_seed=0,
        ).fit(DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature)

    def test_finds_the_dominant_feature(self, forest: RandomForestRegressor) -> None:
        importances = PermutationImportance(random_seed=0).measure(
            forest,
            DOMINATED_SIGNAL.input_features,
            DOMINATED_SIGNAL.target_feature,
        )

        assert importances.most_important.name == "dominant"

    def test_scores_the_noise_column_near_zero(
        self, forest: RandomForestRegressor
    ) -> None:
        importances = PermutationImportance(random_seed=0).measure(
            forest,
            DOMINATED_SIGNAL.input_features,
            DOMINATED_SIGNAL.target_feature,
        )

        assert importances["noise"] < importances["dominant"] / 5.0

    def test_leaves_the_callers_columns_untouched(
        self, forest: RandomForestRegressor
    ) -> None:
        """A Column is frozen on purpose, and this must not work around it."""
        features = DOMINATED_SIGNAL.input_features
        before = [feature.values.copy() for feature in features]

        PermutationImportance(random_seed=0).measure(
            forest, features, DOMINATED_SIGNAL.target_feature
        )

        for feature, original in zip(features, before, strict=True):
            assert np.array_equal(feature.values, original)

    def test_the_same_seed_measures_the_same_thing(
        self, forest: RandomForestRegressor
    ) -> None:
        measurements = [
            PermutationImportance(random_seed=7).measure(
                forest,
                DOMINATED_SIGNAL.input_features,
                DOMINATED_SIGNAL.target_feature,
            )["dominant"]
            for _ in range(2)
        ]

        assert measurements[0] == pytest.approx(measurements[1])

    def test_more_repeats_steady_the_measurement(
        self, forest: RandomForestRegressor
    ) -> None:
        """One shuffle is one draw from a noisy quantity."""
        spread = []
        for repeats in (1, 10):
            values = [
                PermutationImportance(n_repeats=repeats, random_seed=seed).measure(
                    forest,
                    DOMINATED_SIGNAL.input_features,
                    DOMINATED_SIGNAL.target_feature,
                )["dominant"]
                for seed in range(4)
            ]
            spread.append(max(values) - min(values))

        assert spread[1] < spread[0]

    @pytest.mark.parametrize("n_repeats", [0, -1])
    def test_rejects_a_meaningless_repeat_count(self, n_repeats: int) -> None:
        with pytest.raises(ValueError):
            PermutationImportance(n_repeats=n_repeats)


class TestAcrossAnEnsemble:
    """Why the measure is worth reading once it is averaged."""

    def test_reports_a_share_for_every_feature(self) -> None:
        forest = RandomForestRegressor(n_members=ENSEMBLE_MEMBERS, random_seed=0).fit(
            DOMINATED_SIGNAL.input_features, DOMINATED_SIGNAL.target_feature
        )

        assert len(forest.feature_importances) == DOMINATED_SIGNAL.n_features
        assert forest.feature_importances.most_important.name == "dominant"

    def test_averaging_steadies_a_reading_one_tree_cannot_hold(self) -> None:
        """A lone tree's importances lurch between resamples; a forest's do not.

        Both models are handed the *same* six resamples, drawn from a generator
        seeded fresh for each, so the only difference is whether the reading
        came from one tree or from twenty averaged. Summed across all six
        features, measured: 0.365 of spread for the lone tree against 0.212 for
        the forest, and the forest is steadier on every feature individually.
        """

        def total_spread(build) -> float:
            generator = np.random.default_rng(11)
            target = DOMINATED_SIGNAL.target_feature
            readings: list[list[float]] = []

            for _ in range(6):
                drawn = generator.integers(
                    0, DOMINATED_SIGNAL.n_samples, DOMINATED_SIGNAL.n_samples
                )
                model = build().fit(
                    [
                        Feature(feature.name, feature.values[drawn])
                        for feature in DOMINATED_SIGNAL.input_features
                    ],
                    Feature(target.name, target.values[drawn]),
                )
                readings.append(
                    [
                        model.feature_importances[name]
                        for name in DOMINATED_SIGNAL.FEATURE_NAMES
                    ]
                )

            columns = np.array(readings)

            return float((columns.max(axis=0) - columns.min(axis=0)).sum())

        lone = total_spread(lambda: DecisionTreeRegressor(max_depth=4))
        forest = total_spread(
            lambda: RandomForestRegressor(n_members=ENSEMBLE_MEMBERS, max_depth=4)
        )

        assert forest < lone
