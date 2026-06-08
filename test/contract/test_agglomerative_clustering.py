"""The contract every backend's AgglomerativeClustering keeps.

The third clusterer, and the one whose character is a choice rather than a
number. What "nearest" means for two *groups* is not implied by what it means
for two rows, so ``linkage`` is the modelling decision and the four values
behave differently enough that the contract tests them apart rather than
together.

What can be asserted exactly, and what cannot
-----------------------------------------------
The *heights* the merges happen at are determined: each step takes the smallest
remaining group-to-group distance, and that sequence is the same however ties
among equal pairs are broken. Measured across all four linkages the two
backends agree on them to 1.1e-14. Which pair merged is not determined when two
pairs tie, so no tree of merges is exposed and none is asserted -- the same
position the density clusterer takes on its border points.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature, Linkage
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import NotFittedError, TooFewValuesError
from oop_ml.core.types import FloatArray, IndexArray

from .harness import provided

#: Three rows about the origin, three about (5, 5), and one far out at (10, 0).
#: Small enough that every merge height can be checked by hand, and arranged so
#: that the seventh row is a group of its own at three.
_LEFT = np.array([0.0, 0.2, 0.0, 5.0, 5.3, 5.0, 10.0])
_RIGHT = np.array([0.0, 0.0, 0.3, 5.0, 5.0, 5.4, 0.0])
FEATURES = [Feature("left", _LEFT), Feature("right", _RIGHT)]
THREE_GROUPS = np.array([0, 0, 0, 1, 1, 1, 2])

#: The first two merges are the same under every linkage, because the pairs
#: they join are singletons and every rule agrees about two single rows: the
#: nearest pair anywhere is (0, 0) and (0.2, 0) at 0.2, then (5, 5) and
#: (5.3, 5) at 0.3.
FIRST_TWO_HEIGHTS = [0.2, 0.3]

#: A long thin pair of arcs, which single linkage follows and complete linkage
#: cuts across.
_ARC = np.linspace(0.0, np.pi, 40)
_CHAIN_ROWS = np.vstack(
    [
        np.column_stack([np.cos(_ARC) * 4.0, np.sin(_ARC) * 4.0]),
        np.column_stack([4.0 - np.cos(_ARC) * 4.0, 2.0 - np.sin(_ARC) * 4.0]),
    ]
)
CHAIN_FEATURES = [
    Feature("left", _CHAIN_ROWS[:, 0]),
    Feature("right", _CHAIN_ROWS[:, 1]),
]
CHAIN_GROUPS = np.array([0] * 40 + [1] * 40)


def same_partition(
    left: Sequence[float] | FloatArray, right: Sequence[int] | IndexArray
) -> bool:
    """Whether two labellings group the rows identically, ignoring the numbers."""
    first = np.asarray(left)
    second = np.asarray(right)

    return all(
        (first[one] == first[other]) == (second[one] == second[other])
        for one in range(len(first))
        for other in range(one + 1, len(first))
    )


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    model = AgglomerativeClustering(n_clusters=4, linkage=Linkage.AVERAGE)

    assert model.n_clusters == 4
    assert model.linkage is Linkage.AVERAGE


def test_it_defaults_to_ward_over_euclidean(backend: ModuleType) -> None:
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    model = AgglomerativeClustering()

    assert model.linkage is Linkage.WARD
    assert model.metric is DistanceMetric.EUCLIDEAN
    assert model.n_clusters == 2


def test_ward_refuses_a_metric_it_is_not_defined_for(backend: ModuleType) -> None:
    """Ward merges whichever pair adds least to the spread about a group mean,
    and a mean minimises squared Euclidean distance and no other. Refused at
    construction on both backends, which is where a wrong pairing is cheapest
    to catch."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    with pytest.raises(ValidationError):
        AgglomerativeClustering(linkage=Linkage.WARD, metric=DistanceMetric.MANHATTAN)


def test_another_linkage_keeps_that_metric(backend: ModuleType) -> None:
    """The refusal is about ward specifically, not about the metric. The other
    three read the block of member-to-member distances and nothing else, so any
    metric serves them."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    model = AgglomerativeClustering(
        linkage=Linkage.AVERAGE, metric=DistanceMetric.MANHATTAN
    ).fit(FEATURES)

    assert model.n_clusters == 2


@pytest.mark.parametrize("linkage", list(Linkage))
def test_it_finds_the_three_groups_the_fixture_was_built_with(
    backend: ModuleType, linkage: Linkage
) -> None:
    """Every rule agrees here, because the groups are far enough apart that no
    choice of what group-distance means can reach across."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(FEATURES)

    assert same_partition(model.labels, THREE_GROUPS)


@pytest.mark.parametrize("linkage", list(Linkage))
def test_the_merge_heights_ascend_and_start_where_they_must(
    backend: ModuleType, linkage: Linkage
) -> None:
    """One height per merge, in the order made, and the first two are the same
    under every rule because they join single rows and every rule agrees about
    two of those."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage).fit(FEATURES)

    heights = np.asarray(model.merge_distances)

    assert heights.shape == (len(_LEFT) - 3,)
    assert np.all(np.diff(heights) >= -1e-12)
    assert np.allclose(heights[:2], FIRST_TWO_HEIGHTS)


def test_asking_for_fewer_groups_keeps_the_merges_already_made(
    backend: ModuleType,
) -> None:
    """The property no other clusterer here has. A coarser fit is the same fit
    carried further, so its heights start with the finer fit's heights rather
    than being a different sequence."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    finer = AgglomerativeClustering(n_clusters=5).fit(FEATURES)
    coarser = AgglomerativeClustering(n_clusters=2).fit(FEATURES)

    assert np.allclose(
        np.asarray(coarser.merge_distances)[: len(np.asarray(finer.merge_distances))],
        np.asarray(finer.merge_distances),
    )


def test_single_linkage_follows_a_chain_where_complete_cuts_across_it(
    backend: ModuleType,
) -> None:
    """The clearest difference the choice makes. Two long thin arcs are one
    connected run each, which single linkage follows because it asks only
    whether the groups touch anywhere; complete linkage asks whether every pair
    is close, which a long arc never satisfies, so it cuts the arcs into
    compact pieces instead."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    chaining = AgglomerativeClustering(n_clusters=2, linkage=Linkage.SINGLE).fit(
        CHAIN_FEATURES
    )
    compact = AgglomerativeClustering(n_clusters=2, linkage=Linkage.COMPLETE).fit(
        CHAIN_FEATURES
    )

    assert same_partition(chaining.labels, CHAIN_GROUPS)
    assert not same_partition(compact.labels, CHAIN_GROUPS)


def test_every_row_is_placed_in_a_group(backend: ModuleType) -> None:
    """Unlike the density clusterer, which may decline. Merging starts from one
    group per row and only ever joins them, so nothing can be left out."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3).fit(FEATURES)

    labels = np.asarray(model.labels)

    assert labels.shape == (len(_LEFT),)
    assert set(labels.tolist()) == {0.0, 1.0, 2.0}


def test_predicting_the_training_rows_reproduces_the_labels(
    backend: ModuleType,
) -> None:
    """A hierarchical clustering has no centres, so a fitted one is the rows it
    saw and their labels, and a query takes the label of the nearest. On the
    training rows that is the row itself, at distance zero."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3).fit(FEATURES)

    assert np.array_equal(np.asarray(model.predict(FEATURES)), np.asarray(model.labels))


def test_a_row_it_never_saw_joins_its_nearest_neighbour_s_group(
    backend: ModuleType,
) -> None:
    """The engine this backend wraps has no ``predict`` at all, so this rule is
    the library's own and both backends follow it."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3).fit(FEATURES)

    unseen = [
        Feature("left", np.array([0.1, 5.1, 9.9])),
        Feature("right", np.array([0.1, 5.1, 0.1])),
    ]
    answered = np.asarray(model.predict(unseen))

    assert answered[0] == model.labels[0]
    assert answered[1] == model.labels[3]
    assert answered[2] == model.labels[6]


def test_fit_predict_agrees_with_the_labels(backend: ModuleType) -> None:
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3)

    answered = model.fit_predict(FEATURES)

    assert np.array_equal(np.asarray(answered), np.asarray(model.labels))


def test_it_refuses_more_groups_than_there_are_rows(backend: ModuleType) -> None:
    """Every row starts alone and merging only reduces the count, so there is
    no way to reach more groups than rows."""
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    three_rows = [
        Feature("left", np.array([0.0, 1.0, 2.0])),
        Feature("right", np.array([0.0, 1.0, 2.0])),
    ]

    with pytest.raises(TooFewValuesError):
        AgglomerativeClustering(n_clusters=4).fit(three_rows)


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")
    model = AgglomerativeClustering(n_clusters=3).fit(FEATURES)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.labels)
    )


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    with pytest.raises(NotFittedError):
        AgglomerativeClustering().predict(FEATURES)


@pytest.mark.parametrize("summary", ["labels", "merge_distances", "feature_names"])
def test_it_refuses_to_report_its_fit_before_fit(
    backend: ModuleType, summary: str
) -> None:
    AgglomerativeClustering = provided(backend, "AgglomerativeClustering")

    with pytest.raises(NotFittedError):
        getattr(AgglomerativeClustering(), summary)
