"""Spec for stratified folding.

The test that carries the argument is
``test_plain_folding_leaves_folds_with_no_positive``. It asserts the *failure*,
on the same data the stratified tests then use, because a fix nobody has seen
break is a fix nobody can judge. Measured there: three of ten folds hold no
positive at all, and recall on such a fold is ``0/0``.

``test_it_cannot_manufacture_a_rare_class`` is the other half. Stratifying
distributes what exists, and four positives dealt round eight folds still
leaves four folds without one. The splitter reports that rather than quietly
doing its best, which is what ``classes_missing_from_a_fold`` is for.
"""

import numpy as np
import pytest

from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.model_selection.splitting import KFold

RARE_CLASS_SHARE = 0.05


def imbalanced(n_rows: int = 200, positives: int = 10, seed: int = 7) -> Dataset:
    """Rows where one class is rare enough for plain folding to lose it."""
    generator = np.random.default_rng(seed)
    classes = np.zeros(n_rows)
    classes[generator.choice(n_rows, positives, replace=False)] = 1.0

    return Dataset(
        [
            Feature("first", generator.normal(size=n_rows)),
            Feature("second", generator.normal(size=n_rows)),
        ],
        Feature("classes", classes),
    )


def positives_per_fold(splits) -> list[int]:
    """How many of the rare class each held-out fold received."""
    return [int(split.testing.target_feature.values.sum()) for split in splits]


class TestWhatPlainFoldingDoes:
    """The failure, asserted before the fix is asserted."""

    @pytest.mark.parametrize("n_folds", [5, 10])
    def test_plain_folding_leaves_folds_with_no_positive(self, n_folds: int) -> None:
        """Shuffling the whole set and cutting lets a rare class clump.

        Measured on these rows: one fold of five and three of ten come back
        empty of positives. Recall there is ``0/0``, so the fold contributes a
        convention rather than a measurement.
        """
        splits = KFold(n_folds=n_folds, stratified=False, random_seed=0).split(
            imbalanced()
        )

        assert splits.classes_missing_from_a_fold() > 0


class TestStratifiedFolding:
    """Every fold a small copy of the whole."""

    @pytest.mark.parametrize("n_folds", [5, 10])
    def test_every_fold_sees_every_class(self, n_folds: int) -> None:
        splits = KFold(n_folds=n_folds, stratified=True, random_seed=0).split(
            imbalanced()
        )

        assert splits.classes_missing_from_a_fold() == 0

    def test_the_rare_class_is_spread_evenly(self) -> None:
        """Ten positives over ten folds is one each, not a distribution."""
        splits = KFold(n_folds=10, stratified=True, random_seed=0).split(imbalanced())

        assert positives_per_fold(splits) == [1] * 10

    def test_each_fold_holds_the_whole_set_s_class_share(self) -> None:
        dataset = imbalanced()
        splits = KFold(n_folds=5, stratified=True, random_seed=0).split(dataset)

        for split in splits:
            held_out = split.testing.target_feature.values

            assert held_out.mean() == pytest.approx(RARE_CLASS_SHARE, abs=0.01)

    def test_every_row_is_held_out_exactly_once(self) -> None:
        """The property that makes an averaged score use all of the data.

        Stratifying changes which fold a row lands in and must not change
        that each row lands in exactly one.
        """
        dataset = imbalanced()
        splits = KFold(n_folds=5, stratified=True, random_seed=0).split(dataset)

        held_out = sum(split.testing.n_samples for split in splits)

        assert held_out == dataset.n_samples

    def test_training_and_testing_stay_disjoint(self) -> None:
        dataset = imbalanced(n_rows=40, positives=8)
        splits = KFold(n_folds=4, stratified=True, random_seed=0).split(dataset)

        for split in splits:
            assert (
                split.training.n_samples + split.testing.n_samples == dataset.n_samples
            )

    def test_the_same_seed_folds_the_same_way(self) -> None:
        dataset = imbalanced()
        folded = [
            positives_per_fold(
                KFold(n_folds=5, stratified=True, random_seed=3).split(dataset)
            )
            for _ in range(2)
        ]

        assert folded[0] == folded[1]

    def test_it_works_for_more_than_two_classes(self) -> None:
        generator = np.random.default_rng(1)
        classes = np.repeat([0.0, 1.0, 2.0], [60, 30, 10])
        dataset = Dataset(
            [Feature("first", generator.normal(size=100))],
            Feature("classes", classes),
        )

        splits = KFold(n_folds=5, stratified=True, random_seed=0).split(dataset)

        assert splits.classes_missing_from_a_fold() == 0


class TestWhatItCannotDo:
    """The limit, reported rather than hidden."""

    def test_it_cannot_manufacture_a_rare_class(self) -> None:
        """Four positives cannot reach eight folds however they are dealt.

        Four of them get one each and four get none, and the count says so.
        Silently returning the best available arrangement would leave a caller
        averaging over folds that could not contribute.
        """
        generator = np.random.default_rng(2)
        dataset = Dataset(
            [Feature("first", generator.normal(size=20))],
            Feature("classes", np.array([1.0] * 4 + [0.0] * 16)),
        )

        assert (
            KFold(n_folds=4, stratified=True, random_seed=0)
            .split(dataset)
            .classes_missing_from_a_fold()
            == 0
        )
        assert (
            KFold(n_folds=8, stratified=True, random_seed=0)
            .split(dataset)
            .classes_missing_from_a_fold()
            == 4
        )

    def test_a_continuous_target_degrades_to_an_even_cut(self) -> None:
        """The worst case for the deal, and the reason it carries an offset.

        The splitter is never told which task it is serving, and stratifying a
        continuous target makes every value its own class of exactly one row.
        Dealing each class from fold zero would then send every row to fold
        zero: measured, all forty, leaving three folds empty enough that
        building a split raised. Carrying the offset across classes turns that
        into an ordinary even cut instead.

        Stratifying still stays off by default, because an even cut of a
        continuous target is what plain folding already does and asking for it
        this way means the caller thought they were getting something else.
        """
        generator = np.random.default_rng(4)
        dataset = Dataset(
            [Feature("first", generator.normal(size=40))],
            Feature("quantity", generator.normal(size=40)),
        )

        splits = KFold(n_folds=4, stratified=True, random_seed=0).split(dataset)

        assert [split.testing.n_samples for split in splits] == [10, 10, 10, 10]

    def test_the_folds_come_out_the_same_size(self) -> None:
        """A consequence of the running offset worth pinning separately.

        Several classes whose counts do not divide evenly by the fold count
        would otherwise each leave their remainder in the low-numbered folds,
        and those remainders add up.
        """
        splits = KFold(n_folds=5, stratified=True, random_seed=0).split(imbalanced())

        assert [split.testing.n_samples for split in splits] == [40] * 5
