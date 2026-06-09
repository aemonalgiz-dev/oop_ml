"""Spec for WeightedTargets, and for what a weight is taken to mean.

The oracle here is not a formula but an identity: a row of weight two counts
the way two copies of that row would. So the weighted measures are checked
against *duplication* rather than against a restatement of their own
arithmetic, which is the one comparison an implementation cannot pass by
agreeing with itself.

The other half is that nothing changed. Every measure at uniform weights has
to reproduce exactly what it produced before weights existed, or five thousand
tests would have been reinterpreting themselves rather than passing.
"""

import numpy as np
import pytest

from oop_ml.core.data.column import Column
from oop_ml.core.exceptions import InvalidValuesError, NonEqualArrayLengthError
from oop_ml.core.tree.impurity import (
    EntropyImpurity,
    GiniImpurity,
    Impurity,
    VarianceImpurity,
)
from oop_ml.core.tree.weights import WeightedTargets
from oop_ml.core.validation import ValueRole

MEASURES = [GiniImpurity(), EntropyImpurity(), VarianceImpurity()]
CLASS_MEASURES = [GiniImpurity(), EntropyImpurity()]


def column(values) -> Column:
    return Column(np.asarray(values, dtype=float), ValueRole.TARGET_VALUES)


def weighted(values, weights=None) -> WeightedTargets:
    return WeightedTargets(column(values), weights)


def duplicated(values, repeats) -> WeightedTargets:
    """The same rows written out as many times as they weigh."""
    return weighted(np.repeat(np.asarray(values, dtype=float), repeats))


class TestWhatAWeightMeans:
    """Counting a row twice and writing it down twice are the same thing."""

    @pytest.mark.parametrize("measure", MEASURES, ids=lambda one: type(one).__name__)
    def test_a_weighted_node_measures_like_the_duplicated_one(self, measure: Impurity):
        values = [0.0, 0.0, 1.0, 1.0, 1.0]
        repeats = [3, 1, 2, 1, 1]

        assert measure.of(weighted(values, repeats)) == pytest.approx(
            measure.of(duplicated(values, repeats)), abs=1e-12
        )

    @pytest.mark.parametrize("measure", MEASURES, ids=lambda one: type(one).__name__)
    def test_and_a_weighted_split_gains_what_the_duplicated_one_gains(
        self, measure: Impurity
    ):
        values = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])
        repeats = np.array([3, 1, 2, 1, 1, 4])
        cut = 3

        by_weight = measure.gain(
            weighted(values, repeats),
            weighted(values[:cut], repeats[:cut]),
            weighted(values[cut:], repeats[cut:]),
        )
        by_copies = measure.gain(
            duplicated(values, repeats),
            duplicated(values[:cut], repeats[:cut]),
            duplicated(values[cut:], repeats[cut:]),
        )

        assert by_weight == pytest.approx(by_copies, abs=1e-12)

    @pytest.mark.parametrize("measure", MEASURES, ids=lambda one: type(one).__name__)
    def test_and_the_whole_swept_sweep_agrees_cut_for_cut(self, measure: Impurity):
        """Not just one cut. The sweep is the optimised route, so it has to
        keep the identity everywhere the slow one does."""
        values = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        repeats = np.array([2, 1, 3, 1, 1, 2, 1])

        swept = measure.gains_at_every_prefix(weighted(values, repeats))
        one_at_a_time = np.array(
            [
                measure.gain(
                    weighted(values, repeats),
                    weighted(values[:cut], repeats[:cut]),
                    weighted(values[cut:], repeats[cut:]),
                )
                for cut in range(1, len(values))
            ]
        )

        assert np.allclose(swept, one_at_a_time, atol=1e-12)

    def test_a_row_of_zero_weight_is_a_row_that_is_not_there(self):
        """The limit of the same rule. Nothing refuses a weight of zero,
        because it is the honest way to say a row should be ignored."""
        ignored = weighted([0.0, 1.0, 1.0, 5.0], [1.0, 1.0, 1.0, 0.0])
        absent = weighted([0.0, 1.0, 1.0])

        assert GiniImpurity().of(ignored) == pytest.approx(GiniImpurity().of(absent))


class TestUniformWeightsChangeNothing:
    @pytest.mark.parametrize("measure", MEASURES, ids=lambda one: type(one).__name__)
    def test_stating_the_weights_matches_leaving_them_out(self, measure: Impurity):
        values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]

        assert measure.of(weighted(values)) == measure.of(
            weighted(values, np.ones(len(values)))
        )

    @pytest.mark.parametrize(
        "measure", CLASS_MEASURES, ids=lambda one: type(one).__name__
    )
    def test_the_hand_worked_numbers_are_where_they_were(self, measure: Impurity):
        """Two classes evenly split: Gini is 0.5 and entropy is exactly 1 bit,
        which is what they were before any of this."""
        even = weighted([0.0, 0.0, 1.0, 1.0])

        expected = 0.5 if isinstance(measure, GiniImpurity) else 1.0

        assert measure.of(even) == pytest.approx(expected)

    def test_variance_is_still_the_mean_squared_distance_from_the_mean(self):
        values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

        assert VarianceImpurity().of(weighted(values)) == pytest.approx(
            float(np.var(values))
        )


class TestTheShares:
    def test_they_are_shares_of_weight_rather_than_of_rows(self):
        """Three rows, one counting for three, so the class shares are 3:1 and
        not 1:2."""
        skewed = weighted([0.0, 1.0, 1.0], [3.0, 0.5, 0.5])

        assert np.allclose(Impurity.target_probabilities(skewed), [0.75, 0.25])

    def test_a_class_carrying_no_weight_is_absent_rather_than_zero(self):
        """Which is what keeps ``log2`` away from zero in the entropy
        measure."""
        one_ignored = weighted([0.0, 0.0, 1.0], [1.0, 1.0, 0.0])

        assert np.allclose(Impurity.target_probabilities(one_ignored), [1.0])
        assert EntropyImpurity().of(one_ignored) == pytest.approx(0.0)


class TestWhatItRefuses:
    def test_one_weight_per_row_or_none(self):
        with pytest.raises(NonEqualArrayLengthError):
            WeightedTargets(column([1.0, 2.0, 3.0]), [1.0, 1.0])

    def test_a_negative_weight_would_have_a_row_count_against_itself(self):
        with pytest.raises(InvalidValuesError):
            WeightedTargets(column([1.0, 2.0]), [1.0, -1.0])

    def test_weights_that_total_nothing_leave_nothing_to_measure(self):
        with pytest.raises(InvalidValuesError):
            WeightedTargets(column([1.0, 2.0]), [0.0, 0.0])

    def test_a_weight_that_is_not_finite_is_refused(self):
        with pytest.raises(InvalidValuesError):
            WeightedTargets(column([1.0, 2.0]), [1.0, np.inf])


class TestSelecting:
    def test_it_takes_the_targets_and_the_weights_together(self):
        """The point of the pairing: one call rather than two selections a
        caller has to keep aligned."""
        pairing = weighted([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
        chosen = np.array([True, False, True, False])

        part = pairing.selecting(chosen)

        assert np.array_equal(part.values, [0.0, 2.0])
        assert np.array_equal(part.weights, [1.0, 3.0])
        assert part.total_weight == pytest.approx(4.0)

    def test_taking_and_dropping_are_the_two_sides_of_one_cut(self):
        pairing = weighted([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])

        assert np.array_equal(pairing.taking(2).values, [0.0, 1.0])
        assert np.array_equal(pairing.dropping(2).values, [2.0, 3.0])
        assert pairing.taking(2).total_weight + pairing.dropping(
            2
        ).total_weight == pytest.approx(pairing.total_weight)

    def test_the_row_count_is_not_the_weight(self):
        pairing = weighted([0.0, 1.0], [3.0, 5.0])

        assert pairing.n_rows == 2
        assert pairing.total_weight == pytest.approx(8.0)
