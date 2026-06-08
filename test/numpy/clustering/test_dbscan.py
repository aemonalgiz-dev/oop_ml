"""Spec for DBSCAN.

The oracle is :func:`textbook_grouping`, a plain-Python reading of the
definition: count each row's neighbours to find the core points, then take the
transitive closure of "core and within the radius of each other". It is written
without any of the implementation's array work, and it deliberately answers
only about *core* points, because those and their grouping are the whole of what
the algorithm determines. Where a border point goes is a convention, and this
library's is tested on its own terms further down.

What the rest of the file is for is the two properties the module docstring
claims. That a group is whatever connected dense region the data contains,
which is asserted on a chain whose ends are three times the radius apart, and
that the metric is a real choice here where it is not for k-means, which is
asserted on one fixture that finds two groups under one metric and none under
another.
"""

import numpy as np
import pytest

from oop_ml.core.data.feature import Feature
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import NonEqualArrayLengthError, NotFittedError
from oop_ml.core.types import FloatArray
from oop_ml.numpy.clustering.dbscan import DBSCAN

NOISE = -1.0

#: Four points 0.9 apart in a line. Neighbouring pairs are inside a radius of
#: 1.0 and the two ends are 2.7 apart, so anything that groups them together is
#: doing it through the middle rather than directly.
CHAIN_POSITIONS = np.array([0.0, 0.9, 1.8, 2.7])
CHAIN = [
    Feature("position", CHAIN_POSITIONS),
    Feature("height", np.zeros(len(CHAIN_POSITIONS))),
]

#: Two pairs, each pair a diagonal step apart and the pairs five apart in the
#: second column. At a radius of 1.5 the diagonal step is inside a Euclidean
#: ball, at 1.273, and outside a Manhattan one, at 1.8.
CORNER = [
    Feature("left", np.array([0.0, 0.9, 0.0, 0.9])),
    Feature("right", np.array([0.0, 0.9, 5.0, 5.9])),
]

_GENERATOR = np.random.default_rng(31)
_BLOBS = np.vstack(
    [
        _GENERATOR.normal(loc=[0.0, 0.0], scale=0.6, size=(30, 2)),
        _GENERATOR.normal(loc=[5.0, 0.0], scale=0.6, size=(25, 2)),
        _GENERATOR.normal(loc=[2.5, 5.0], scale=0.6, size=(20, 2)),
        _GENERATOR.uniform(low=-6.0, high=11.0, size=(12, 2)),
    ]
)
#: Three blobs with a dozen points scattered across the whole area, so there is
#: something for the model to call noise as well as something to group.
BLOB_FEATURES = [Feature("left", _BLOBS[:, 0]), Feature("right", _BLOBS[:, 1])]


def textbook_grouping(
    rows: FloatArray, radius: float, minimum: int
) -> tuple[list[int], dict[int, int]]:
    """The definition, in plain Python, answering only about core points.

    Returns the core rows' positions and a mapping from each of them to a group
    number, where the groups are the transitive closure of "both core and
    within the radius of each other". Border points are deliberately not
    labelled here, since the algorithm does not determine where they go.
    """
    n_rows = rows.shape[0]

    def near(one: int, other: int) -> bool:
        gap = rows[one] - rows[other]
        return bool(float(np.sqrt(float(gap @ gap))) <= radius)

    core = [
        index
        for index in range(n_rows)
        if sum(1 for other in range(n_rows) if near(index, other)) >= minimum
    ]

    groups: dict[int, int] = {}
    next_group = 0
    for start in core:
        if start in groups:
            continue
        groups[start] = next_group
        waiting = [start]
        while waiting:
            current = waiting.pop()
            for other in core:
                if other not in groups and near(current, other):
                    groups[other] = next_group
                    waiting.append(other)
        next_group += 1

    return core, groups


def grouped_alike(first: dict[int, int], second: dict[int, int]) -> bool:
    """Whether two group numberings put the same members together."""
    keys = sorted(first)

    return all(
        (first[one] == first[other]) == (second[one] == second[other])
        for position, one in enumerate(keys)
        for other in keys[position + 1 :]
    )


class TestAgainstTheDefinition:
    @pytest.mark.parametrize(
        ("radius", "minimum"), [(0.8, 4), (1.0, 5), (1.5, 5), (2.0, 8)]
    )
    def test_the_core_points_and_their_groups_match(self, radius, minimum):
        model = DBSCAN(radius=radius, min_neighbourhood_size=minimum).fit(BLOB_FEATURES)
        core, groups = textbook_grouping(_BLOBS, radius, minimum)

        found = np.asarray(model.core_indices).tolist()
        assert found == core

        mine = {index: int(np.asarray(model.labels)[index]) for index in found}
        assert grouped_alike(mine, groups)

    def test_the_fixture_has_something_of_each_kind_to_find(self):
        """A guard on the guard. If every row were core, or every row noise,
        the comparison above would hold for an implementation that had got the
        interesting part wrong."""
        model = DBSCAN(radius=1.0, min_neighbourhood_size=5).fit(BLOB_FEATURES)

        labels = np.asarray(model.labels)
        core = set(np.asarray(model.core_indices).tolist())
        border = [
            index
            for index in range(len(labels))
            if labels[index] != NOISE and index not in core
        ]

        assert model.n_clusters >= 2
        assert model.n_noise > 0
        assert len(border) > 0


class TestAGroupIsWhateverIsConnected:
    def test_a_chain_is_one_group_though_its_ends_are_far_apart(self):
        """Reachability is transitive, so the ends do not have to see each
        other. This is the whole difference from a centre-based clusterer,
        which would have to put a boundary somewhere along the chain."""
        model = DBSCAN(radius=1.0, min_neighbourhood_size=2).fit(CHAIN)

        assert model.n_clusters == 1
        assert np.all(np.asarray(model.labels) == 0.0)
        assert abs(CHAIN_POSITIONS[0] - CHAIN_POSITIONS[-1]) > 2.0

    def test_breaking_the_chain_breaks_the_group(self):
        broken = [
            Feature("position", np.array([0.0, 0.9, 4.0, 4.9])),
            Feature("height", np.zeros(4)),
        ]

        model = DBSCAN(radius=1.0, min_neighbourhood_size=2).fit(broken)

        assert model.n_clusters == 2

    def test_rows_that_are_all_the_same_point_are_one_group(self):
        identical = [Feature("left", np.zeros(5)), Feature("right", np.zeros(5))]

        model = DBSCAN(radius=0.5, min_neighbourhood_size=5).fit(identical)

        assert model.n_clusters == 1
        assert model.n_noise == 0


class TestTheMetricIsARealChoiceHere:
    """Which it is not for k-means, whose update step is a mean and so is tied
    to Euclidean distance. This model only ever asks whether one row is within
    the radius of another."""

    def test_two_metrics_disagree_on_the_same_radius(self):
        euclidean = DBSCAN(
            radius=1.5, min_neighbourhood_size=2, metric=DistanceMetric.EUCLIDEAN
        ).fit(CORNER)
        manhattan = DBSCAN(
            radius=1.5, min_neighbourhood_size=2, metric=DistanceMetric.MANHATTAN
        ).fit(CORNER)

        assert euclidean.n_clusters == 2
        assert manhattan.n_clusters == 0
        assert manhattan.n_noise == 4

    def test_and_the_reason_is_arithmetic_a_reader_can_check(self):
        diagonal = np.array([0.9, 0.9])

        assert float(np.sqrt(diagonal @ diagonal)) == pytest.approx(1.2728, abs=1e-4)
        assert float(np.abs(diagonal).sum()) == pytest.approx(1.8)


class TestTheBorderConvention:
    """A border point within reach of two groups is not decided by the
    algorithm. This library decides it by the nearest core point, and where
    that ties, by the lower core index."""

    def test_a_border_point_can_reach_two_groups_at_once(self):
        """The fixture is built so that it does: two dense clumps whose
        outposts sit 1.2 apart, which is beyond the radius so they stay
        separate, with a row exactly between them at 0.6 from each."""
        positions = np.concatenate(
            [np.linspace(-0.05, 0.05, 8), [1.0], np.linspace(2.95, 3.05, 8), [2.2, 1.6]]
        )
        features = [
            Feature("position", positions),
            Feature("height", np.zeros(len(positions))),
        ]

        model = DBSCAN(radius=1.05, min_neighbourhood_size=5).fit(features)

        labels = np.asarray(model.labels)
        core_positions = np.asarray(model.core_points.values)[:, 0]
        core_labels = labels[np.asarray(model.core_indices)]
        gaps = np.abs(core_positions - positions[-1])
        reachable = sorted(set(core_labels[gaps <= 1.05].tolist()))

        assert model.n_clusters == 2
        assert len(reachable) == 2
        assert labels[-1] in reachable

    def test_an_equal_tie_goes_to_the_lower_core_index(self):
        """Deterministic rather than right: the two claims are equally good,
        and what matters is that the same input always gets the same answer."""
        positions = np.concatenate(
            [np.linspace(-0.05, 0.05, 8), [1.0], np.linspace(2.95, 3.05, 8), [2.2, 1.6]]
        )
        features = [
            Feature("position", positions),
            Feature("height", np.zeros(len(positions))),
        ]

        first = DBSCAN(radius=1.05, min_neighbourhood_size=5).fit(features)
        again = DBSCAN(radius=1.05, min_neighbourhood_size=5).fit(features)

        assert np.array_equal(np.asarray(first.labels), np.asarray(again.labels))
        assert first.labels[-1] == 0.0


class TestBeforeFitAndBadInput:
    @pytest.mark.parametrize(
        "attribute", ["labels", "n_clusters", "n_noise", "core_indices", "core_points"]
    )
    def test_learned_attributes_raise_before_fit(self, attribute):
        with pytest.raises(NotFittedError):
            getattr(DBSCAN(), attribute)

    def test_ragged_columns_are_refused(self):
        with pytest.raises(NonEqualArrayLengthError):
            DBSCAN().fit(
                [Feature("left", [1.0, 2.0, 3.0]), Feature("right", [0.0, 0.0])]
            )

    @pytest.mark.parametrize(
        ("field_name", "invalid"),
        [("radius", 0.0), ("radius", -1.0), ("min_neighbourhood_size", 0)],
    )
    def test_a_setting_that_describes_no_neighbourhood_is_refused(
        self, field_name, invalid
    ):
        with pytest.raises(ValueError):
            DBSCAN(**{field_name: invalid})
