"""Spec for the distance metrics.

The metric is the model for a neighbour method, so these are not incidental
tests. Four of them are worth reading rather than skimming.

The first is that the metrics disagree about which row is nearest. If a suite
only ever checked shapes, an implementation that quietly computed Euclidean for
all six would pass everything, and the enum would be decoration.

The second is the scaling test. It asserts that an unstandardised column with a
larger unit changes the answer, which is not a bug being pinned but the defining
property of distance-based models -- the one thing a user has to know before
reaching for one.

The third is the oracle. ``EuclideanDistance`` does not compute what it appears
to compute: it expands the square into a matrix multiply, which is a different
sequence of floating-point operations reaching the same place. The oracle here
is therefore written as a plain Python loop over pairs, from the definition,
with no numpy cleverness at all. Its job is to be obviously right rather than
fast, and to have been written without reference to the implementation -- an
oracle that mirrors the code it checks proves only that the code equals itself.

The fourth is the precision test, which pins the exact case that made the fast
route unsafe until it was centred: two points 1e-06 apart with coordinates near
1e06, where the uncentred expansion returns 0.0 and loses the distance
entirely.
"""

import math

import numpy as np
import pytest

from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.distance.calculations import (
    BroadcastDistance,
    CanberraDistance,
    CosineDistance,
    Distance,
    EuclideanDistance,
    HammingDistance,
    MinkowskiDistance,
)
from oop_ml.core.distance.metric import DistanceMetric


def block(values) -> RowBlock:
    """These rows as the block every ``between`` now takes.

    A :class:`~oop_ml.core.data.row_block.RowBlock` rather than a bare array,
    because the orientation is the one thing a caller of a distance can get
    silently wrong: queries down the rows, remembered rows down the other
    argument, and a transposed matrix still multiplies.
    """
    array = np.asarray(values, dtype=float)

    return RowBlock(array, [f"feature_{index}" for index in range(array.shape[1])])


def euclidean_oracle(first_row, second_row):
    """Straight-line distance, written from the definition and nothing else."""
    return math.sqrt(
        sum(
            (first - second) ** 2
            for first, second in zip(first_row, second_row, strict=True)
        )
    )


def pairwise_oracle(query_rows, remembered_rows, of_two_rows):
    """Every pair, one at a time, in plain Python."""
    return np.array(
        [
            [of_two_rows(query, remembered) for remembered in remembered_rows]
            for query in query_rows
        ]
    )


class TestMembers:
    def test_there_are_exactly_six(self):
        assert set(DistanceMetric) == {
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.CHEBYSHEV,
            DistanceMetric.COSINE,
            DistanceMetric.HAMMING,
            DistanceMetric.CANBERRA,
        }

    def test_it_is_a_closed_enum_not_a_string(self):
        # The reason this is not a str parameter: a typo is not reachable.
        assert not hasattr(DistanceMetric, "EUCLIDIAN")

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_every_member_carries_a_calculation(self, metric):
        # A member added without one would fail here rather than at the first
        # call, which is the point of settling it at definition time.
        assert isinstance(metric.calculation, Distance)

    @pytest.mark.parametrize(
        ("metric", "order"),
        [
            (DistanceMetric.MANHATTAN, 1),
            (DistanceMetric.CHEBYSHEV, np.inf),
        ],
    )
    def test_the_p_norms_know_their_own_p(self, metric, order):
        assert isinstance(metric.calculation, MinkowskiDistance)
        assert metric.calculation.order == order


class TestShape:
    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_result_is_queries_by_remembered(self, metric):
        queries = np.zeros((4, 3))
        remembered = np.ones((7, 3))

        assert metric.between(block(queries), block(remembered)).shape == (4, 7)

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_a_single_query_still_returns_a_matrix(self, metric):
        assert metric.between(
            block(np.zeros((1, 2))), block(np.ones((5, 2)))
        ).shape == (1, 5)

    @pytest.mark.parametrize("metric", list(DistanceMetric))
    def test_no_metric_ever_returns_a_negative_distance(self, metric):
        generator = np.random.default_rng(0)
        queries = generator.normal(size=(6, 4))
        remembered = generator.normal(size=(9, 4))

        assert (metric.between(block(queries), block(remembered)) >= 0.0).all()


class TestEuclidean:
    def test_matches_the_formula(self):
        queries = np.array([[0.0, 0.0]])
        remembered = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]])

        distances = DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered))

        assert distances[0] == pytest.approx([5.0, np.sqrt(2.0), 0.0])

    def test_matches_a_plain_python_oracle(self):
        # The implementation expands the square into a matrix multiply, so this
        # is a genuinely different route to the same number, not a restatement.
        generator = np.random.default_rng(11)
        queries = generator.normal(size=(7, 5))
        remembered = generator.normal(size=(13, 5))

        assert DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered)) == (
            pytest.approx(pairwise_oracle(queries, remembered, euclidean_oracle))
        )

    def test_a_row_is_zero_distance_from_itself(self):
        rows = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        distances = DistanceMetric.EUCLIDEAN.between(block(rows), block(rows))

        assert np.diag(distances) == pytest.approx(np.zeros(3))

    def test_is_symmetric(self):
        first = np.array([[1.0, 2.0]])
        second = np.array([[4.0, 6.0]])

        forwards = DistanceMetric.EUCLIDEAN.between(block(first), block(second))
        backwards = DistanceMetric.EUCLIDEAN.between(block(second), block(first))

        assert forwards[0, 0] == pytest.approx(backwards[0, 0])


class TestEuclideanPrecision:
    """The cases that decide whether the matrix-multiply route is allowed."""

    def test_a_tiny_gap_far_from_the_origin_survives(self):
        # Uncentred, the expansion returns 0.0 here -- the whole distance lost.
        # Centring is what makes this pass, so this test is the reason the fast
        # route is the default.
        base = 1.0e06
        queries = np.array([[base, base]])
        remembered = np.array([[base + 1.0e-06, base], [base, base + 3.0e-06]])

        distances = DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered))
        expected = pairwise_oracle(queries, remembered, euclidean_oracle)

        assert distances[0] == pytest.approx(expected[0], rel=1.0e-09)
        assert (distances > 0.0).all()

    def test_never_produces_nan_for_a_point_against_itself(self):
        # sqrt of a small negative is nan, and a nan would travel silently all
        # the way to a mean. The clamp at zero is what stops it.
        base = 1.0e06
        rows = np.array([[base, base], [base + 1.0e-06, base]])

        distances = DistanceMetric.EUCLIDEAN.between(block(rows), block(rows))

        assert not np.isnan(distances).any()
        assert np.diag(distances) == pytest.approx(np.zeros(2))

    def test_agrees_with_broadcasting_to_machine_precision(self):
        generator = np.random.default_rng(3)
        queries = generator.normal(size=(40, 12))
        remembered = generator.normal(size=(60, 12))

        fast = EuclideanDistance().between(block(queries), block(remembered))
        broadcast = MinkowskiDistance(2).between(block(queries), block(remembered))

        assert fast == pytest.approx(broadcast, rel=1.0e-12, abs=1.0e-12)


class TestManhattan:
    def test_matches_the_formula(self):
        queries = np.array([[0.0, 0.0]])
        remembered = np.array([[3.0, 4.0], [1.0, 1.0], [0.0, 0.0]])

        distances = DistanceMetric.MANHATTAN.between(block(queries), block(remembered))

        assert distances[0] == pytest.approx([7.0, 2.0, 0.0])

    def test_never_squares_a_gap(self):
        # The point of the metric: one feature disagreeing wildly costs its own
        # size and not its square, so a single outlying column sways it less.
        queries = np.array([[0.0, 0.0]])

        one_feature = np.array([[10.0, 0.0]])
        assert DistanceMetric.EUCLIDEAN.between(block(queries), block(one_feature))[
            0, 0
        ] == (pytest.approx(10.0))
        assert DistanceMetric.MANHATTAN.between(block(queries), block(one_feature))[
            0, 0
        ] == (pytest.approx(10.0))

        spread = np.array([[6.0, 8.0]])
        assert DistanceMetric.EUCLIDEAN.between(block(queries), block(spread))[
            0, 0
        ] == (pytest.approx(10.0))
        assert DistanceMetric.MANHATTAN.between(block(queries), block(spread))[
            0, 0
        ] == (pytest.approx(14.0))


class TestChebyshev:
    def test_only_the_largest_gap_counts(self):
        queries = np.array([[0.0, 0.0, 0.0]])
        remembered = np.array([[4.0, 2.0, -6.0]])

        assert DistanceMetric.CHEBYSHEV.between(block(queries), block(remembered))[
            0, 0
        ] == (pytest.approx(6.0))

    def test_adding_gaps_in_other_features_changes_nothing(self):
        # This is what "p at infinity" buys, and what makes it the right choice
        # for a tolerance: the worst feature is the whole answer.
        queries = np.array([[0.0, 0.0, 0.0]])
        one_large = np.array([[6.0, 0.0, 0.0]])
        several = np.array([[6.0, 5.0, 5.0]])

        assert DistanceMetric.CHEBYSHEV.between(block(queries), block(one_large))[
            0, 0
        ] == (
            pytest.approx(
                DistanceMetric.CHEBYSHEV.between(block(queries), block(several))[0, 0]
            )
        )

    def test_it_is_the_limit_the_p_norms_approach(self):
        gap = np.array([[4.0, 2.0, -6.0]])
        origin = np.array([[0.0, 0.0, 0.0]])

        rising = [
            MinkowskiDistance(order).between(block(origin), block(gap))[0, 0]
            for order in (1, 2, 4, 10, 30)
        ]

        assert rising == sorted(rising, reverse=True)
        assert rising[-1] == pytest.approx(6.0, abs=1.0e-03)


class TestCosine:
    def test_direction_is_all_that_matters(self):
        queries = np.array([[1.0, 1.0, 1.0]])
        remembered = np.array([[2.0, 2.0, 2.0], [100.0, 100.0, 100.0]])

        distances = DistanceMetric.COSINE.between(block(queries), block(remembered))

        assert distances[0] == pytest.approx([0.0, 0.0])

    def test_the_same_pair_euclidean_calls_distant(self):
        # The contrast worth remembering: these rows are 140 apart in a
        # straight line and identical by angle.
        queries = np.array([[1.0, 1.0]])
        remembered = np.array([[100.0, 100.0]])

        assert (
            DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered))[0, 0]
            > 100.0
        )
        assert DistanceMetric.COSINE.between(block(queries), block(remembered))[
            0, 0
        ] == (pytest.approx(0.0))

    def test_a_right_angle_is_one_and_opposite_is_two(self):
        queries = np.array([[1.0, 0.0]])
        remembered = np.array([[0.0, 1.0], [-1.0, 0.0]])

        assert DistanceMetric.COSINE.between(block(queries), block(remembered))[0] == (
            pytest.approx([1.0, 2.0])
        )

    def test_a_row_against_itself_is_zero_and_never_negative(self):
        # Rounding puts the cosine a hair either side of 1, so the diagonal
        # comes out at a few multiples of machine epsilon rather than at a
        # clean zero. What the clamp guarantees is only the sign: a distance
        # below zero would be nonsense, and would order the neighbours wrongly.
        generator = np.random.default_rng(5)
        rows = generator.normal(size=(20, 8))

        distances = DistanceMetric.COSINE.between(block(rows), block(rows))

        assert np.diag(distances) == pytest.approx(np.zeros(20), abs=1.0e-12)
        assert (distances >= 0.0).all()

    def test_a_zero_row_is_one_away_from_everything(self):
        # No direction at all, so no angle. Reported as a right angle.
        queries = np.array([[0.0, 0.0]])
        remembered = np.array([[1.0, 0.0], [3.0, 4.0]])

        assert DistanceMetric.COSINE.between(block(queries), block(remembered))[0] == (
            pytest.approx([1.0, 1.0])
        )


class TestHamming:
    def test_counts_features_that_differ_as_a_share(self):
        queries = np.array([[1.0, 2.0, 3.0, 4.0]])
        remembered = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 9.0]])

        assert DistanceMetric.HAMMING.between(block(queries), block(remembered))[0] == (
            pytest.approx([0.0, 0.25])
        )

    def test_the_size_of_a_disagreement_is_irrelevant(self):
        # The property that makes it the only metric here fit for labels:
        # class 9 is not further from class 1 than class 2 is.
        queries = np.array([[1.0, 1.0]])
        near = np.array([[2.0, 1.0]])
        far = np.array([[900.0, 1.0]])

        assert DistanceMetric.HAMMING.between(block(queries), block(near))[0, 0] == (
            pytest.approx(
                DistanceMetric.HAMMING.between(block(queries), block(far))[0, 0]
            )
        )

    def test_it_is_a_share_so_it_never_exceeds_one(self):
        queries = np.array([[1.0, 2.0, 3.0]])
        remembered = np.array([[9.0, 9.0, 9.0]])

        assert DistanceMetric.HAMMING.between(block(queries), block(remembered))[
            0, 0
        ] == (pytest.approx(1.0))


class TestCanberra:
    def test_matches_the_formula(self):
        # |1-2|/(1+2) + |1-1|/(1+1) + |1-1|/(1+1) = 1/3
        queries = np.array([[1.0, 1.0, 1.0]])
        remembered = np.array([[2.0, 1.0, 1.0]])

        assert DistanceMetric.CANBERRA.between(block(queries), block(remembered))[
            0, 0
        ] == (pytest.approx(1.0 / 3.0))

    def test_two_zeros_agree_rather_than_dividing_by_zero(self):
        queries = np.array([[0.0, 5.0]])
        remembered = np.array([[0.0, 5.0]])

        distances = DistanceMetric.CANBERRA.between(block(queries), block(remembered))

        assert not np.isnan(distances).any()
        assert distances[0, 0] == pytest.approx(0.0)

    def test_a_feature_contributes_at_most_one_whatever_its_units(self):
        # Why this is the one metric that tolerates unstandardised input.
        queries = np.array([[1.0, 1.0]])
        remembered = np.array([[1.0, 1000000.0]])

        assert (
            DistanceMetric.CANBERRA.between(block(queries), block(remembered))[0, 0]
            <= 2.0
        )

    def test_it_is_most_sensitive_near_zero(self):
        # A doubling costs the same wherever it happens, which is the point on
        # counts and the catch everywhere else.
        queries = np.array([[0.001]])
        remembered = np.array([[0.002]])

        small = DistanceMetric.CANBERRA.between(block(queries), block(remembered))[0, 0]
        large = DistanceMetric.CANBERRA.between(block([[1.0]]), block([[2.0]]))[0, 0]

        assert small == pytest.approx(large)


class TestTheMetricsDisagree:
    def test_euclidean_and_manhattan_choose_different_nearest_rows(self):
        # Without this, an implementation returning Euclidean for both would
        # satisfy every other test in the file.
        # (3, 3) is further along both axes than (5, 0) is along one, so it
        # loses on Manhattan; squaring rewards the difference being spread, so
        # it wins on Euclidean. sqrt(18) = 4.24 against 5, and 6 against 5.
        queries = np.array([[0.0, 0.0]])
        remembered = np.array([[3.0, 3.0], [5.0, 0.0]])

        euclidean = DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered))[
            0
        ]
        manhattan = DistanceMetric.MANHATTAN.between(block(queries), block(remembered))[
            0
        ]

        assert int(np.argmin(euclidean)) == 0
        assert int(np.argmin(manhattan)) == 1

    def test_cosine_and_euclidean_choose_different_nearest_rows(self):
        # (10, 10) points exactly where the query does but is far away;
        # (1, 0) is close but at a right angle.
        queries = np.array([[1.0, 1.0]])
        remembered = np.array([[10.0, 10.0], [1.0, 0.0]])

        euclidean = DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered))[
            0
        ]
        cosine = DistanceMetric.COSINE.between(block(queries), block(remembered))[0]

        assert int(np.argmin(euclidean)) == 1
        assert int(np.argmin(cosine)) == 0

    def test_hamming_and_euclidean_choose_different_nearest_rows(self):
        # One feature off by a mile against every feature off by a hair.
        queries = np.array([[0.0, 0.0, 0.0]])
        remembered = np.array([[50.0, 0.0, 0.0], [0.1, 0.1, 0.1]])

        euclidean = DistanceMetric.EUCLIDEAN.between(block(queries), block(remembered))[
            0
        ]
        hamming = DistanceMetric.HAMMING.between(block(queries), block(remembered))[0]

        assert int(np.argmin(euclidean)) == 1
        assert int(np.argmin(hamming)) == 0


class TestScalingDecidesTheAnswer:
    def test_an_unscaled_column_takes_over(self):
        # Floor area in square metres beside a bathroom count. The area column
        # is not more important; it is only measured in bigger numbers.
        query = np.array([[100.0, 3.0]])
        remembered = np.array([[96.0, 1.0], [118.0, 2.0]])

        raw = DistanceMetric.EUCLIDEAN.between(block(query), block(remembered))[0]
        assert int(np.argmin(raw)) == 0

        def standardise(matrix, centre, spread):
            return (matrix - centre) / spread

        centre = np.array([107.0, 1.5])
        spread = np.array([11.0, 0.5])
        scaled = DistanceMetric.EUCLIDEAN.between(
            block(standardise(query, centre, spread)),
            block(standardise(remembered, centre, spread)),
        )[0]

        assert int(np.argmin(scaled)) == 1

    def test_the_wide_column_dominates_the_squared_distance(self):
        query = np.array([[100.0, 3.0]])
        remembered = np.array([[72.0, 1.0], [140.0, 2.0], [210.0, 3.0]])

        squared_gaps = (query[:, None, :] - remembered[None, :, :]) ** 2
        share = squared_gaps[..., 0].sum() / squared_gaps.sum()

        assert share > 0.99

    def test_canberra_is_the_exception(self):
        # The same two rows, and the metric that divides each gap by the size
        # of its own feature reaches the standardised verdict without help.
        query = np.array([[100.0, 3.0]])
        remembered = np.array([[96.0, 1.0], [118.0, 2.0]])

        canberra = DistanceMetric.CANBERRA.between(block(query), block(remembered))[0]

        assert int(np.argmin(canberra)) == 1


class OneRowAtATimeMinkowski(MinkowskiDistance):
    """Manhattan with a budget so small every block holds a single query."""

    block_bytes = 1


class OneRowAtATimeHamming(HammingDistance):
    """Hamming, likewise forced to the narrowest possible block."""

    block_bytes = 1


class OneRowAtATimeCanberra(CanberraDistance):
    """Canberra, likewise."""

    block_bytes = 1


class TestBlocking:
    """The block loop must change the memory used and nothing else."""

    @pytest.mark.parametrize(
        ("blocked", "whole"),
        [
            (OneRowAtATimeMinkowski(1), MinkowskiDistance(1)),
            (OneRowAtATimeMinkowski(3), MinkowskiDistance(3)),
            (OneRowAtATimeHamming(), HammingDistance()),
            (OneRowAtATimeCanberra(), CanberraDistance()),
        ],
    )
    def test_blocked_and_unblocked_agree_bit_for_bit(self, blocked, whole):
        # Not approx. The block loop reorders nothing and combines nothing, so
        # anything less than exact equality means it changed the arithmetic.
        generator = np.random.default_rng(7)
        queries = generator.normal(size=(23, 4))
        remembered = generator.normal(size=(31, 4))

        assert np.array_equal(
            blocked.between(block(queries), block(remembered)),
            whole.between(block(queries), block(remembered)),
        )

    def test_a_budget_smaller_than_one_query_still_makes_progress(self):
        # max(1, ...) is what stops a zero step from looping forever.
        assert (
            OneRowAtATimeMinkowski(2)._queries_per_block(block(np.ones((1000, 50))))
            == 1
        )

    def test_block_size_shrinks_as_the_remembered_set_grows(self):
        calculation = MinkowskiDistance(2)

        small = calculation._queries_per_block(block(np.ones((100, 5))))
        large = calculation._queries_per_block(block(np.ones((10000, 5))))

        assert small > large

    def test_the_result_is_the_full_matrix_however_it_was_blocked(self):
        blocked = OneRowAtATimeMinkowski(1)

        assert blocked.between(
            block(np.zeros((9, 3))), block(np.ones((4, 3)))
        ).shape == (9, 4)


class TestCustomCalculations:
    def test_minkowski_accepts_an_order_the_enum_does_not_name(self):
        # The reason the calculations are exposed at all: p = 3 is a perfectly
        # ordinary choice and the enum names only three of infinitely many.
        queries = np.array([[0.0, 0.0]])
        remembered = np.array([[3.0, 4.0]])

        expected = (3.0**3 + 4.0**3) ** (1.0 / 3.0)

        assert MinkowskiDistance(3).between(block(queries), block(remembered))[
            0, 0
        ] == (pytest.approx(expected))

    def test_every_calculation_is_a_distance(self):
        for calculation in (
            EuclideanDistance(),
            MinkowskiDistance(2),
            CosineDistance(),
            HammingDistance(),
            CanberraDistance(),
        ):
            assert isinstance(calculation, Distance)

    def test_the_broadcast_ones_share_the_block_loop(self):
        for calculation in (
            MinkowskiDistance(2),
            HammingDistance(),
            CanberraDistance(),
        ):
            assert isinstance(calculation, BroadcastDistance)

    def test_the_matrix_multiply_ones_do_not(self):
        # They have no pairing array to bound, so blocking them would be
        # overhead for nothing.
        assert not isinstance(EuclideanDistance(), BroadcastDistance)
        assert not isinstance(CosineDistance(), BroadcastDistance)
