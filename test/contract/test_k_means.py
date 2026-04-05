"""The contract every backend's KMeans keeps.

A cluster label means nothing across fits. A run that puts rows 0-3 together
and rows 4-7 together has found the same structure whether it numbered them
(0, 1) or (1, 0), and two backends seeding from different random streams will
number them differently on purpose. So nothing here asserts a label; what is
compared is the *partition*, which rows share a label, exactly as the numpy
spec does.

The fixture is three blobs whose grouping is not a judgement call. Rows within
a blob sit 1.0 apart and the blobs' centres 10 apart, so any correct
implementation recovers the same partition from any seeding. The centres are
round numbers and every row sits exactly 0.5 from its own centre, so the
inertia is 12 * 0.25 and is written down rather than computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from oop_ml import Feature
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NotFittedError,
    TooFewValuesError,
)

from .harness import provided

#: Three blobs of four rows each, centred at (1, 1), (11, 1) and (6, 11).
_FIRST = [0.5, 1.5, 1.0, 1.0, 10.5, 11.5, 11.0, 11.0, 5.5, 6.5, 6.0, 6.0]
_SECOND = [1.0, 1.0, 0.5, 1.5, 1.0, 1.0, 0.5, 1.5, 11.0, 11.0, 10.5, 11.5]
FEATURES = [Feature("first", _FIRST), Feature("second", _SECOND)]
TRUE_GROUPS = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
CENTRES = sorted([(1.0, 1.0), (11.0, 1.0), (6.0, 11.0)])
INERTIA = 3.0

#: One query beside each blob, and the blob it belongs to.
QUERIES = [Feature("first", [1.2, 10.8, 6.1]), Feature("second", [0.9, 1.2, 10.9])]
QUERY_GROUPS = [0, 1, 2]

#: Two concentric rings of twelve points, at radius 1 and radius 5.
#:
#: The blobs above cannot carry the two seeding tests, because their grouping
#: is not a judgement call and that is the whole point of them: measured at
#: three groups, two fits agree at an inertia of 3.0 and one restart already
#: reaches what twenty reach, with the seed and the restart count discarded
#: exactly as without them. Both assertions would then hold against a wrapper
#: that read neither field, which is the trap the numpy spec records. The
#: rings are ambiguous instead, since no arrangement of four or five round
#: groups fits two rings, so where a seeding starts decides which local
#: minimum it settles in.
_ANGLES = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
_UNIT = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])
_RING_BLOCK = np.vstack([_UNIT, 5.0 * _UNIT])
RINGS = [Feature("first", _RING_BLOCK[:, 0]), Feature("second", _RING_BLOCK[:, 1])]

#: The best arrangement of the rings into five groups, which twenty restarts
#: reach from seed 4 on both backends where one restart stops at 76.823748.
BEST_FIVE_GROUP_INERTIA = 63.196613


def same_partition(
    left: Sequence[int] | np.ndarray, right: Sequence[int] | np.ndarray
) -> bool:
    """Whether two labellings group the rows identically, ignoring the numbers.

    Two rows agree when they are together in both labellings or apart in
    both, so comparing every pair settles it without a matching problem.
    """
    left_labels = np.asarray(left)
    right_labels = np.asarray(right)

    together_left = left_labels[:, None] == left_labels[None, :]
    together_right = right_labels[:, None] == right_labels[None, :]

    return bool(np.array_equal(together_left, together_right))


def centres_of(model: Any) -> list[tuple[float, float]]:
    """The learned centres, sorted, since group order carries no meaning."""
    return sorted(
        (
            round(centroid.coordinate_for("first"), 6),
            round(centroid.coordinate_for("second"), 6),
        )
        for centroid in model.centroids
    )


def test_it_is_constructed_by_the_same_keywords(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(
        n_clusters=3,
        n_initialisations=4,
        max_iterations=50,
        tolerance=1e-06,
        random_seed=7,
    )

    assert model.n_clusters == 3
    assert model.n_initialisations == 4
    assert model.max_iterations == 50
    assert model.tolerance == pytest.approx(1e-06)
    assert model.random_seed == 7


def test_it_fits_features_and_returns_itself(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0)

    assert model.fit(FEATURES) is model


def test_it_predicts_one_whole_label_per_row(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    labels = np.asarray(model.predict(FEATURES))

    assert labels.shape == (len(_FIRST),)
    assert np.array_equal(labels, np.floor(labels))
    assert labels.min() >= 0 and labels.max() < 3


def test_fit_predict_recovers_the_generating_partition(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")

    labels = KMeans(n_clusters=3, random_seed=0).fit_predict(FEATURES)

    assert same_partition(np.asarray(labels).astype(int), TRUE_GROUPS)


def test_the_clustering_labels_agree_with_predict(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    assert np.array_equal(
        model.clustering.labels, np.asarray(model.predict(FEATURES)).astype(int)
    )


def test_its_centroids_are_addressable_by_name(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    assert [centroid.name for centroid in model.centroids] == [
        "cluster_1",
        "cluster_2",
        "cluster_3",
    ]
    assert model.centroids["cluster_1"].feature_names == ("first", "second")
    assert isinstance(model.centroids["cluster_1"].coordinate_for("first"), float)


def test_it_finds_the_centres_of_the_blobs(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    assert centres_of(model) == pytest.approx(CENTRES)


def test_it_reports_the_inertia_and_the_sizes(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    assert model.inertia == pytest.approx(INERTIA)
    assert sorted(model.clustering.sizes) == [4, 4, 4]
    assert not model.clustering.has_an_empty_cluster


def test_it_settles_inside_the_iteration_ceiling(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, max_iterations=50, random_seed=0).fit(FEATURES)

    assert 0 < model.iterations_run <= 50


def test_a_new_row_falls_to_its_nearest_centre(backend: ModuleType) -> None:
    """The queries sit beside the blobs, so their labels must match the
    training rows of those blobs, whatever numbers the fit chose."""
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    training_labels = np.asarray(model.predict(FEATURES)).astype(int)
    query_labels = np.asarray(model.predict(QUERIES)).astype(int)

    for query_label, group in zip(query_labels, QUERY_GROUPS, strict=True):
        assert query_label == training_labels[TRUE_GROUPS.index(group)]


def test_column_order_does_not_matter(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    assert np.array_equal(
        np.asarray(model.predict(FEATURES)), np.asarray(model.predict(FEATURES[::-1]))
    )


def test_the_seed_fixes_the_fit(backend: ModuleType) -> None:
    """Without this, ``random_seed`` is a field the fit never reads.

    Nothing here asserts a number, because the two backends draw their
    seedings from different streams and the wrappers say so. On the rings at
    four groups the numpy backend answers 102.072142 and the scikit backend
    107.924171, both stably. What is asserted is that one seed names one fit,
    which is the whole content of the field.

    Measured with the scikit wrapper's ``random_state`` discarded, two fits of
    this configuration landed on 107.924171 and 110.957977, so the assertion
    goes red on a wrapper that accepts the seed and drops it.
    """
    KMeans = provided(backend, "KMeans")

    first = KMeans(n_clusters=4, random_seed=11).fit(RINGS)
    second = KMeans(n_clusters=4, random_seed=11).fit(RINGS)

    assert first.inertia == second.inertia
    assert centres_of(first) == centres_of(second)
    assert np.array_equal(first.clustering.labels, second.clustering.labels)


def test_more_restarts_find_a_better_arrangement(backend: ModuleType) -> None:
    """Without this, ``n_initialisations`` is a field the fit never reads.

    Strictly better, never merely no worse. Restarts that share one seeding
    produce the identical fit, so ``<=`` holds trivially against a wrapper
    that runs one restart whatever it was told, and it is the ``<`` that
    discriminates.

    The configuration is chosen rather than assumed. At five groups on the
    rings, one restart from seed 4 reaches 76.823748 on both backends and
    twenty reach 63.196613, a margin of 13.63. Seeds 0 and 3 are no use here,
    since the scikit backend's first restart already lands on the best
    arrangement and ties, and seed 5 is no use for the numpy backend for the
    same reason; seeds 1, 2, 4, 6 and 7 discriminate on both. With the scikit
    wrapper's ``n_init`` forced to one, twenty restarts answer 76.823748 and
    the strict comparison goes red.
    """
    KMeans = provided(backend, "KMeans")

    one = KMeans(n_clusters=5, n_initialisations=1, random_seed=4).fit(RINGS)
    many = KMeans(n_clusters=5, n_initialisations=20, random_seed=4).fit(RINGS)

    assert many.inertia < one.inertia
    assert many.inertia == pytest.approx(BEST_FIVE_GROUP_INERTIA)


def test_writing_into_the_learned_centres_does_not_move_them(
    backend: ModuleType,
) -> None:
    """A fitted model's answers do not change because a caller wrote into an
    array it handed out. Three routes reach the same coordinates, and each is
    a writeable float array, so each is written into here.

    ``coordinate_for`` is the assertion that discriminates, since the
    per-centroid buffer is the one handed out through a copy. With that copy
    removed, a write through ``coordinates`` puts 999.0 into the fitted centre
    and this test goes red on both backends.
    """
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    before = np.asarray(model.predict(FEATURES)).astype(int)

    model.centroids.positions[:] = 999.0
    model.clustering.centroids.positions[:] = 999.0
    model.centroids["cluster_1"].coordinates[:] = 999.0

    after = np.asarray(model.predict(FEATURES)).astype(int)

    assert np.array_equal(before, after)
    assert centres_of(model) == pytest.approx(CENTRES)
    assert model.inertia == pytest.approx(INERTIA)


def test_a_refused_refit_leaves_the_earlier_fit_intact(backend: ModuleType) -> None:
    """Compute into locals, assign at the end, checked rather than intended.

    A refit that raises must leave the model as the last successful fit left
    it, rather than half replaced or unfitted. Nothing else in the suite
    exercises a failure part way through a second fit.
    """
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)
    before = np.asarray(model.predict(FEATURES)).astype(int)

    with pytest.raises(TooFewValuesError):
        model.fit([Feature("first", [1.0, 2.0]), Feature("second", [3.0, 4.0])])

    assert model.is_fitted
    assert centres_of(model) == pytest.approx(CENTRES)
    assert model.inertia == pytest.approx(INERTIA)
    assert np.array_equal(np.asarray(model.predict(FEATURES)).astype(int), before)


def test_it_refuses_more_clusters_than_rows(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")

    with pytest.raises(TooFewValuesError):
        KMeans(n_clusters=13).fit(FEATURES)


def test_it_refuses_a_query_over_the_wrong_features(backend: ModuleType) -> None:
    KMeans = provided(backend, "KMeans")
    model = KMeans(n_clusters=3, random_seed=0).fit(FEATURES)

    with pytest.raises(InvalidValuesError):
        model.predict([FEATURES[0]])


def test_it_refuses_to_predict_before_fit_in_the_library_s_own_words(
    backend: ModuleType,
) -> None:
    KMeans = provided(backend, "KMeans")

    with pytest.raises(NotFittedError):
        KMeans(n_clusters=3).predict(FEATURES)

    with pytest.raises(NotFittedError):
        _ = KMeans(n_clusters=3).centroids
