"""Spec for AgglomerativeClustering.

The oracle is :func:`merged_by_hand`, the merge loop written out from the
definition with the linkage rule passed in as a summary of the member-to-member
block. It is the slow, obvious reading, and it answers about the labels and the
heights both, which is more than the density clusterer's oracle could promise
because merging leaves nothing undetermined except which of two exactly tied
pairs goes first.

Ward is checked separately and against its own formula, because it is the one
rule that is not a summary of the block at all.
"""

import numpy as np
import pytest

from oop_ml.core.clustering.linkage import Linkage
from oop_ml.core.data.feature import Feature
from oop_ml.core.distance.calculations import MinkowskiDistance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import NotFittedError, TooFewValuesError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.clustering.agglomerative import AgglomerativeClustering

#: Two threes and a stray, with every coordinate a round number so that the
#: block of distances between any two groups can be worked out by hand.
ROWS = np.array(
    [
        [0.0, 0.0],
        [0.2, 0.0],
        [0.0, 0.3],
        [5.0, 5.0],
        [5.3, 5.0],
        [5.0, 5.4],
        [10.0, 0.0],
    ]
)
FEATURES = [Feature("left", ROWS[:, 0]), Feature("right", ROWS[:, 1])]

_GENERATOR = np.random.default_rng(17)
_SCATTER = _GENERATOR.normal(size=(24, 3)) + np.repeat(
    np.array([[0.0, 0.0, 0.0], [6.0, 0.0, 0.0], [0.0, 6.0, 3.0]]), 8, axis=0
)
#: Three loose clumps in three columns, unrounded, so the oracle is comparing
#: against numbers nobody chose.
SCATTER_FEATURES = [
    Feature(f"column_{index}", _SCATTER[:, index]) for index in range(3)
]

SUMMARIES = {
    Linkage.SINGLE: np.min,
    Linkage.COMPLETE: np.max,
    Linkage.AVERAGE: np.mean,
}


def merged_by_hand(
    rows: FloatArray, n_clusters: int, summary
) -> tuple[FloatArray, list[float]]:
    """The definition: join the nearest pair of groups until enough are left.

    ``summary`` is what the distance between two groups is taken to be, applied
    to the block of member-to-member distances. Deliberately written without
    any of the implementation's bookkeeping.
    """
    gaps = rows[:, None, :] - rows[None, :, :]
    pairwise = np.sqrt((gaps * gaps).sum(axis=2))

    members = [[index] for index in range(rows.shape[0])]
    heights: list[float] = []

    while len(members) > n_clusters:
        best = float("inf")
        where = (0, 1)
        for one in range(len(members)):
            for other in range(one + 1, len(members)):
                value = float(summary(pairwise[np.ix_(members[one], members[other])]))
                if value < best:
                    best, where = value, (one, other)

        one, other = where
        heights.append(best)
        members[one] = members[one] + members[other]
        members.pop(other)

    labels = np.empty(rows.shape[0])
    for number, group in enumerate(members):
        labels[group] = float(number)

    return labels, heights


def same_partition(left, right) -> bool:
    first, second = np.asarray(left), np.asarray(right)

    return all(
        (first[one] == first[other]) == (second[one] == second[other])
        for one in range(len(first))
        for other in range(one + 1, len(first))
    )


class TestAgainstTheDefinition:
    @pytest.mark.parametrize("linkage", list(SUMMARIES))
    @pytest.mark.parametrize("n_clusters", [2, 3, 5])
    def test_labels_and_heights_both_match(self, linkage, n_clusters):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(
            SCATTER_FEATURES
        )
        labels, heights = merged_by_hand(_SCATTER, n_clusters, SUMMARIES[linkage])

        assert same_partition(model.labels, labels)
        assert np.allclose(np.asarray(model.merge_distances), heights, atol=1e-12)

    def test_the_fixture_is_not_trivially_grouped(self):
        """A guard on the guard: if every row ended alone, or all together, the
        comparison above would hold for an implementation that never merged."""
        model = AgglomerativeClustering(n_clusters=3, linkage=Linkage.AVERAGE).fit(
            SCATTER_FEATURES
        )

        sizes = [int((np.asarray(model.labels) == group).sum()) for group in range(3)]

        assert min(sizes) >= 2
        assert len(np.asarray(model.merge_distances)) == len(_SCATTER) - 3


class TestTheThreeBlockRules:
    """Each is a summary of the member-to-member distances, and on a fixture
    this small the number can be written down."""

    def test_single_is_the_nearest_pair(self):
        """The third merge joins row 2 at (0, 0.3) to the group {0, 1}. Its
        distance to (0, 0) is 0.3 and to (0.2, 0) is sqrt(0.13); the nearest is
        0.3."""
        model = AgglomerativeClustering(n_clusters=3, linkage=Linkage.SINGLE).fit(
            FEATURES
        )

        assert model.merge_distances[2] == pytest.approx(0.3)

    def test_complete_is_the_furthest_pair(self):
        """The same merge under complete linkage takes the furthest instead,
        which is sqrt(0.2 ** 2 + 0.3 ** 2) = 0.360555."""
        model = AgglomerativeClustering(n_clusters=3, linkage=Linkage.COMPLETE).fit(
            FEATURES
        )

        assert model.merge_distances[2] == pytest.approx(np.hypot(0.2, 0.3))

    def test_average_is_the_mean_of_the_two(self):
        model = AgglomerativeClustering(n_clusters=3, linkage=Linkage.AVERAGE).fit(
            FEATURES
        )

        assert model.merge_distances[2] == pytest.approx(
            (0.3 + float(np.hypot(0.2, 0.3))) / 2
        )

    def test_single_never_exceeds_complete_at_the_same_step(self):
        """True by construction of min against max, and worth pinning because
        it is the one relation between the three that always holds."""
        nearest = AgglomerativeClustering(n_clusters=3, linkage=Linkage.SINGLE).fit(
            SCATTER_FEATURES
        )
        furthest = AgglomerativeClustering(n_clusters=3, linkage=Linkage.COMPLETE).fit(
            SCATTER_FEATURES
        )

        assert (
            np.asarray(nearest.merge_distances).max()
            <= np.asarray(furthest.merge_distances).max() + 1e-12
        )


class TestWard:
    """Not a summary of the block, which is why it gets its own oracle."""

    def test_it_is_the_weighted_distance_between_the_group_means(self):
        """``sqrt(2 |A| |B| / (|A| + |B|)) * ||mean_A - mean_B||``. On the third
        merge, {0, 1} has mean (0.1, 0) and {2} is (0, 0.3), so the gap is
        sqrt(0.1) and the weight is sqrt(4 / 3)."""
        model = AgglomerativeClustering(n_clusters=3, linkage=Linkage.WARD).fit(
            FEATURES
        )

        gap = float(np.hypot(0.1, 0.3))
        weight = float(np.sqrt(2 * 1 * 2 / 3))

        assert model.merge_distances[2] == pytest.approx(gap * weight)

    def test_the_weight_is_what_makes_it_reluctant_to_join_two_large_groups(self):
        """Two groups whose means are the same distance apart cost more to
        merge the more rows they hold, because the objective counts every row
        that moves."""
        small = np.sqrt(2 * 1 * 1 / 2)
        large = np.sqrt(2 * 4 * 4 / 8)

        assert large > small

    def test_it_is_refused_with_a_metric_it_is_not_defined_for(self):
        with pytest.raises(ValueError):
            AgglomerativeClustering(
                linkage=Linkage.WARD, metric=DistanceMetric.CHEBYSHEV
            )


class TestTheMetricIsFreeForTheOtherThree:
    def test_a_distance_the_enum_does_not_name_is_accepted(self):
        """The numpy backend computes the pairwise block itself, so any
        ``Distance`` serves. The scikit wrapper refuses this, since its engine
        takes a metric by name; that divergence is pinned in the translation
        tests."""
        model = AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=MinkowskiDistance(3)
        ).fit(FEATURES)

        assert same_partition(model.labels, [0, 0, 0, 1, 1, 1, 2])

    def test_two_metrics_can_disagree_about_the_heights(self):
        euclidean = AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=DistanceMetric.EUCLIDEAN
        ).fit(FEATURES)
        manhattan = AgglomerativeClustering(
            n_clusters=3, linkage=Linkage.AVERAGE, metric=DistanceMetric.MANHATTAN
        ).fit(FEATURES)

        assert not np.allclose(
            np.asarray(euclidean.merge_distances),
            np.asarray(manhattan.merge_distances),
        )


class TestTheEndsOfTheRange:
    def test_asking_for_one_group_merges_everything(self):
        model = AgglomerativeClustering(n_clusters=1).fit(FEATURES)

        assert np.all(np.asarray(model.labels) == 0.0)
        assert len(np.asarray(model.merge_distances)) == len(ROWS) - 1

    def test_asking_for_as_many_groups_as_rows_merges_nothing(self):
        model = AgglomerativeClustering(n_clusters=len(ROWS)).fit(FEATURES)

        assert len(np.asarray(model.merge_distances)) == 0
        assert len(set(np.asarray(model.labels).tolist())) == len(ROWS)

    def test_more_groups_than_rows_is_refused(self):
        with pytest.raises(TooFewValuesError):
            AgglomerativeClustering(n_clusters=len(ROWS) + 1).fit(FEATURES)


class TestBeforeFit:
    @pytest.mark.parametrize(
        "attribute", ["labels", "merge_distances", "feature_names"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(AgglomerativeClustering(), attribute)
