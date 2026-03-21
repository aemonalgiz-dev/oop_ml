"""Spec for the splitters -- red until ``split`` lands on each.

The property that matters for k-fold is coverage: every row lands in exactly one
fold's testing half, and in every other fold's training half. A splitter that
merely returns k plausible-looking partitions can still be wrong about that, so
the tests check it directly rather than checking sizes alone.
"""

import numpy as np
import pytest

from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import EmptyValuesError, TooFewValuesError
from oop_ml.model_selection.splitting import (
    KFold,
    RowShuffler,
    Splits,
    TrainTestSplitter,
)


def make_dataset(n_samples: int = 20) -> Dataset:
    return Dataset(
        [
            Feature("x1", list(range(n_samples))),
            Feature("x2", [value * value for value in range(n_samples)]),
        ],
        Feature("y", [3 * value + 1 for value in range(n_samples)]),
    )


def target_values_of(dataset: Dataset) -> set[float]:
    return set(dataset.target_feature.values.tolist())


class TestRowShuffler:
    def test_without_shuffling_the_order_is_untouched(self):
        order = RowShuffler(shuffle=False).row_order(6)

        np.testing.assert_array_equal(order, [0, 1, 2, 3, 4, 5])

    def test_shuffling_is_a_permutation(self):
        order = RowShuffler(random_seed=0).row_order(20)

        assert sorted(order.tolist()) == list(range(20))

    def test_the_same_seed_gives_the_same_order(self):
        first = RowShuffler(random_seed=7).row_order(20)
        second = RowShuffler(random_seed=7).row_order(20)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_orders(self):
        first = RowShuffler(random_seed=1).row_order(50)
        second = RowShuffler(random_seed=2).row_order(50)

        assert not np.array_equal(first, second)


class TestSplits:
    def test_empty_raises(self):
        with pytest.raises(EmptyValuesError):
            Splits([])


class TestTrainTestSplitter:
    def test_test_fraction_defaults_to_a_quarter(self):
        assert TrainTestSplitter().test_fraction == pytest.approx(0.25)

    @pytest.mark.parametrize("test_fraction", [0.0, 1.0, -0.1, 1.5])
    def test_fraction_outside_the_open_unit_interval_is_rejected(self, test_fraction):
        with pytest.raises(ValueError):
            TrainTestSplitter(test_fraction=test_fraction)

    @pytest.mark.parametrize(
        ("n_samples", "test_fraction", "expected_testing"),
        [(20, 0.25, 5), (20, 0.5, 10), (100, 0.1, 10), (7, 0.5, 4)],
    )
    def test_holds_back_the_requested_share(
        self, n_samples, test_fraction, expected_testing
    ):
        split = TrainTestSplitter(test_fraction=test_fraction, random_seed=0).split(
            make_dataset(n_samples)
        )

        assert split.testing.n_samples == expected_testing
        assert split.training.n_samples == n_samples - expected_testing

    def test_every_row_lands_on_exactly_one_side(self):
        dataset = make_dataset()

        split = TrainTestSplitter(random_seed=0).split(dataset)

        training, testing = (
            target_values_of(split.training),
            target_values_of(split.testing),
        )
        assert training | testing == target_values_of(dataset)
        assert training & testing == set()

    def test_without_shuffling_the_test_block_is_the_tail(self):
        split = TrainTestSplitter(test_fraction=0.25, shuffle=False).split(
            make_dataset(20)
        )

        np.testing.assert_allclose(
            split.testing.input_features[0].values, [15, 16, 17, 18, 19]
        )

    def test_the_same_seed_reproduces_the_split(self):
        first = TrainTestSplitter(random_seed=3).split(make_dataset())
        second = TrainTestSplitter(random_seed=3).split(make_dataset())

        np.testing.assert_allclose(
            first.testing.target_feature.values, second.testing.target_feature.values
        )

    def test_too_few_rows_to_split_raises(self):
        with pytest.raises(TooFewValuesError):
            TrainTestSplitter().split(Dataset([Feature("x1", [1])], Feature("y", [2])))


class TestKFold:
    def test_folds_default_to_five(self):
        assert KFold().n_folds == 5

    @pytest.mark.parametrize("n_folds", [1, 0, -2])
    def test_fewer_than_two_folds_is_rejected(self, n_folds):
        with pytest.raises(ValueError):
            KFold(n_folds=n_folds)

    @pytest.mark.parametrize("n_folds", [2, 3, 5, 10])
    def test_produces_one_split_per_fold(self, n_folds):
        splits = KFold(n_folds=n_folds, random_seed=0).split(make_dataset(20))

        assert splits.n_splits == n_folds
        assert len(splits) == n_folds

    @pytest.mark.parametrize("n_folds", [2, 4, 5])
    def test_every_row_is_tested_exactly_once(self, n_folds):
        dataset = make_dataset(20)

        splits = KFold(n_folds=n_folds, random_seed=0).split(dataset)

        tested = [
            value
            for split in splits
            for value in split.testing.target_feature.values.tolist()
        ]

        assert sorted(tested) == sorted(dataset.target_feature.values.tolist())

    @pytest.mark.parametrize("n_folds", [3, 5])
    def test_training_and_testing_never_overlap(self, n_folds):
        splits = KFold(n_folds=n_folds, random_seed=0).split(make_dataset(20))

        for split in splits:
            assert (
                target_values_of(split.training) & target_values_of(split.testing)
                == set()
            )

    def test_each_fit_sees_the_rest_of_the_data(self):
        splits = KFold(n_folds=5, random_seed=0).split(make_dataset(20))

        for split in splits:
            assert split.training.n_samples == 16
            assert split.testing.n_samples == 4

    def test_uneven_division_differs_by_at_most_one_row(self):
        # 20 rows over 3 folds: 7, 7, 6.
        splits = KFold(n_folds=3, random_seed=0).split(make_dataset(20))

        sizes = sorted(split.testing.n_samples for split in splits)

        assert sizes[-1] - sizes[0] <= 1
        assert sum(sizes) == 20

    def test_the_same_seed_reproduces_the_folds(self):
        first = KFold(n_folds=4, random_seed=11).split(make_dataset())
        second = KFold(n_folds=4, random_seed=11).split(make_dataset())

        for from_first, from_second in zip(first, second, strict=True):
            np.testing.assert_allclose(
                from_first.testing.target_feature.values,
                from_second.testing.target_feature.values,
            )

    def test_more_folds_than_rows_raises(self):
        with pytest.raises(TooFewValuesError):
            KFold(n_folds=10).split(make_dataset(4))
