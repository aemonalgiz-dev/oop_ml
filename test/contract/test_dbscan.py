"""The contract every backend's DBSCAN keeps.

The second clusterer here and the first model of any kind that answers two
questions nothing before it could. How many groups are there, which it works
out rather than being told; and does this row belong to any of them at all,
which every other model here is obliged to answer yes to.

What is asserted, and what is not
----------------------------------
Which rows are core, and how the core points group, are determined by the
radius and the neighbourhood size, so both are asserted exactly. Where a
*border* point goes is not determined by the algorithm at all: a row within
reach of core points from two groups has two equally good claims on it, and
the usual implementations settle it by the order the rows arrived in. Both
backends here follow one written-down convention instead, the nearest core
point, so that a border point does have one answer; the contract asserts the
convention on a fixture where it is unambiguous rather than on one where it
would be pinning an arbitrary choice.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from oop_ml import Feature
from oop_ml.core.exceptions import (
    NonEqualArrayLengthError,
    NotFittedError,
)
from oop_ml.core.types import FloatArray, IndexArray

from .harness import provided

#: Eight points on a line, arranged so that every row's kind can be worked out
#: by hand at radius 1 and a neighbourhood of 3. Three at 0, 0.4, 0.8 are each
#: within reach of the other two, so all three are core and they form a group.
#: 1.7 reaches only 0.8, so it is a border point of that group. Three at 5,
#: 5.4, 5.8 do the same thing far enough away to be a second group. 10 reaches
#: nothing and is noise.
_POSITIONS = np.array([0.0, 0.4, 0.8, 1.7, 5.0, 5.4, 5.8, 10.0])
FEATURES = [
    Feature("position", _POSITIONS),
    Feature("height", np.zeros(len(_POSITIONS))),
]
RADIUS = 1.0
NEIGHBOURHOOD = 3
EXPECTED_LABELS = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0]
EXPECTED_CORE_INDICES = [0, 1, 2, 4, 5, 6]
NOISE = -1.0

_GENERATOR = np.random.default_rng(4)
_ARC = np.linspace(0.0, np.pi, 60)
_FIRST_CRESCENT = np.column_stack(
    [np.cos(_ARC) * 4.0, np.sin(_ARC) * 4.0]
) + _GENERATOR.normal(scale=0.25, size=(60, 2))
_SECOND_CRESCENT = np.column_stack(
    [4.0 - np.cos(_ARC) * 4.0, 2.0 - np.sin(_ARC) * 4.0]
) + _GENERATOR.normal(scale=0.25, size=(60, 2))
_CRESCENT_ROWS = np.vstack([_FIRST_CRESCENT, _SECOND_CRESCENT])

#: Two interleaving crescents. No straight line separates them and their
#: centres of mass nearly coincide, which is what makes them the standard
#: demonstration that a centre-based clusterer is the wrong tool.
CRESCENT_FEATURES = [
    Feature("left", _CRESCENT_ROWS[:, 0]),
    Feature("right", _CRESCENT_ROWS[:, 1]),
]
CRESCENT_GROUPS = np.array([0] * 60 + [1] * 60)


def same_partition(
    left: Sequence[float] | FloatArray, right: Sequence[int] | IndexArray
) -> bool:
    """Whether two labellings group the rows identically, ignoring the numbers.

    A clusterer's label 0 means "the first group this fit happened to number
    0", so what has to match is which rows share a label, not what the label
    is.
    """
    first = np.asarray(left)
    second = np.asarray(right)

    return all(
        (first[one] == first[other]) == (second[one] == second[other])
        for one in range(len(first))
        for other in range(one + 1, len(first))
    )


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    DBSCAN = provided(backend, "DBSCAN")

    model = DBSCAN(radius=2.5, min_neighbourhood_size=7)

    assert model.radius == pytest.approx(2.5)
    assert model.min_neighbourhood_size == 7


def test_it_is_never_told_how_many_groups_to_find(backend: ModuleType) -> None:
    """The difference from every other clusterer here, and it is structural
    rather than a default: there is no such field to set."""
    DBSCAN = provided(backend, "DBSCAN")

    assert "n_clusters" not in DBSCAN.model_fields

    with pytest.raises(ValidationError):
        DBSCAN(n_clusters=3)


@pytest.mark.parametrize(
    ("field_name", "invalid"),
    [("radius", 0.0), ("radius", -1.0), ("min_neighbourhood_size", 0)],
)
def test_it_refuses_a_setting_that_describes_no_neighbourhood(
    backend: ModuleType,
    field_name: str,
    invalid: float,
) -> None:
    DBSCAN = provided(backend, "DBSCAN")

    with pytest.raises(ValidationError):
        DBSCAN(**{field_name: invalid})


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD)

    assert model.fit(FEATURES) is model


def test_it_finds_the_groups_the_fixture_was_built_with(backend: ModuleType) -> None:
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    assert model.n_clusters == 2
    assert np.array_equal(np.asarray(model.labels), EXPECTED_LABELS)


def test_it_names_the_rows_dense_enough_to_hold_a_group_together(
    backend: ModuleType,
) -> None:
    """The three at 0, 0.4 and 0.8 each reach the other two; the one at 1.7
    reaches only one of them, so it joins their group without being able to
    extend it."""
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    assert np.array_equal(np.asarray(model.core_indices), EXPECTED_CORE_INDICES)
    assert model.labels[3] == 0.0


def test_a_row_in_a_sparse_region_belongs_to_nothing(backend: ModuleType) -> None:
    """The first answer in this library that is not a group. Every other model
    here must name one; this one may decline, and the row at 10 is genuinely
    not a member of anything."""
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    assert model.labels[-1] == NOISE
    assert model.n_noise == 1


def test_it_separates_two_crescents_no_straight_line_could(
    backend: ModuleType,
) -> None:
    """The reason to reach for this over k-means, on the standard case. A
    centre-based clusterer cuts along straight lines, and there is no straight
    line between two shapes that curve around each other."""
    DBSCAN = provided(backend, "DBSCAN")
    KMeans = provided(backend, "KMeans")

    density = DBSCAN(radius=1.2, min_neighbourhood_size=5).fit(CRESCENT_FEATURES)
    centres = KMeans(n_clusters=2, random_seed=0).fit(CRESCENT_FEATURES)

    assert density.n_clusters == 2
    assert same_partition(density.labels, CRESCENT_GROUPS)
    assert not same_partition(
        np.asarray(centres.predict(CRESCENT_FEATURES)), CRESCENT_GROUPS
    )


def test_predicting_the_training_rows_reproduces_the_labels(
    backend: ModuleType,
) -> None:
    """Which is what makes ``fit_predict`` mean what it says, and is not free:
    it holds because the labels were produced by the rule ``predict`` uses,
    rather than by whichever expansion happened to reach a border point."""
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    assert np.array_equal(np.asarray(model.predict(FEATURES)), np.asarray(model.labels))


def test_fit_predict_agrees_with_the_labels(backend: ModuleType) -> None:
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD)

    answered = model.fit_predict(FEATURES)

    assert np.array_equal(np.asarray(answered), np.asarray(model.labels))


def test_a_row_it_never_saw_is_placed_or_declined(backend: ModuleType) -> None:
    """A fitted clustering is a set of labelled core points, so an unseen row
    asks the same question a border point does. The engine this backend wraps
    offers no such method at all."""
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    unseen = [
        Feature("position", np.array([0.2, 5.2, 40.0])),
        Feature("height", np.array([0.0, 0.0, 0.0])),
    ]

    assert np.array_equal(np.asarray(model.predict(unseen)), [0.0, 1.0, NOISE])


def test_too_small_a_radius_leaves_everything_noise(backend: ModuleType) -> None:
    """One of the two visible failures of a badly chosen radius, and the reason
    ``n_clusters`` and ``n_noise`` are both worth reading after a fit."""
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=0.01, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    assert model.n_clusters == 0
    assert model.n_noise == len(_POSITIONS)
    assert np.all(np.asarray(model.labels) == NOISE)


def test_too_large_a_radius_fuses_the_groups_into_one(backend: ModuleType) -> None:
    """The other one. Nothing raises either way, which is why both numbers are
    exposed rather than only the labels."""
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=20.0, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    assert model.n_clusters == 1
    assert model.n_noise == 0


def test_it_matches_query_columns_by_name(backend: ModuleType) -> None:
    DBSCAN = provided(backend, "DBSCAN")
    model = DBSCAN(radius=RADIUS, min_neighbourhood_size=NEIGHBOURHOOD).fit(FEATURES)

    reversed_order = [FEATURES[1], FEATURES[0]]

    assert np.array_equal(
        np.asarray(model.predict(reversed_order)), np.asarray(model.labels)
    )


def test_it_refuses_columns_of_different_lengths(backend: ModuleType) -> None:
    DBSCAN = provided(backend, "DBSCAN")
    ragged = [Feature("position", [1.0, 2.0, 3.0]), Feature("height", [0.0, 0.0])]

    with pytest.raises(NonEqualArrayLengthError):
        DBSCAN().fit(ragged)


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    DBSCAN = provided(backend, "DBSCAN")

    with pytest.raises(NotFittedError):
        DBSCAN().predict(FEATURES)


@pytest.mark.parametrize(
    "summary", ["labels", "n_clusters", "n_noise", "core_indices", "core_points"]
)
def test_it_refuses_to_report_its_fit_before_fit(
    backend: ModuleType,
    summary: str,
) -> None:
    DBSCAN = provided(backend, "DBSCAN")

    with pytest.raises(NotFittedError):
        getattr(DBSCAN(), summary)
