"""Spec for k-means -- red until the three Lloyd steps land.

Almost nothing here asserts a label. Cluster numbering is an artefact of which
centre the seeding happened to place first, so two correct fits of the same
rows can agree completely about the grouping and disagree about every number.
``same_partition`` is what the assertions go through instead: it compares which
rows share a label, not what the label is.

``THREE_BLOBS`` is separated far enough that the answer is not a judgement call
-- rows within a blob sit 1.0 apart while the blobs' centres are 10 apart -- so
any correct implementation recovers exactly that partition from any seeding.
The centres are round numbers for the same reason, and the inertia is 3.0
because all twelve rows sit exactly 0.5 from their own centre.

``TestWhatItCannotFind`` is the other half, and it asserts a *failure* on
purpose. Concentric rings are obvious to a person and outside what "closest
centre" can express at any k. A spec that only showed the cases that work would
be advertising rather than documenting.
"""

import numpy as np
import pytest

from oop_ml.clustering.k_means import KMeans
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NonUniqueFeaturesError,
    NotFittedError,
    TooFewValuesError,
)
from test.fixtures import (
    CONCENTRIC_RINGS,
    THREE_BLOBS,
    THREE_BLOBS_CENTRES,
    THREE_BLOBS_INERTIA,
    THREE_BLOBS_SIZES,
    same_partition,
)


def fitted(n_clusters: int = 3, **overrides) -> KMeans:
    """A model fitted to the three well-separated blobs."""
    return KMeans(n_clusters=n_clusters, random_seed=0, **overrides).fit(
        THREE_BLOBS.input_features
    )


def centres_found(model: KMeans) -> list[tuple[float, float]]:
    """The learned centres, sorted so they can be compared to known ones.

    Sorted because the order the groups came out in carries no meaning, and an
    assertion that depended on it would be asserting the seeding.
    """
    return sorted(
        (
            round(centroid.coordinate_for("first"), 6),
            round(centroid.coordinate_for("second"), 6),
        )
        for centroid in model.centroids
    )


class TestWhatItFinds:
    """The grouping, on data whose grouping is not in doubt."""

    def test_recovers_the_generating_partition(self) -> None:
        labels = fitted().clustering.labels

        assert same_partition(labels, THREE_BLOBS.true_groups)

    def test_finds_the_centres_of_the_blobs(self) -> None:
        """The centres are round numbers, so this is checkable by hand."""
        assert centres_found(fitted()) == pytest.approx(sorted(THREE_BLOBS_CENTRES))

    def test_every_blob_keeps_its_four_rows(self) -> None:
        assert sorted(fitted().clustering.sizes) == sorted(THREE_BLOBS_SIZES)

    def test_inertia_is_the_total_squared_distance(self) -> None:
        """Twelve rows, each exactly 0.5 from its own centre: 12 * 0.25."""
        assert fitted().inertia == pytest.approx(THREE_BLOBS_INERTIA)

    def test_no_group_comes_out_empty(self) -> None:
        assert not fitted().clustering.has_an_empty_cluster

    def test_names_the_groups_by_position(self) -> None:
        assert [centroid.name for centroid in fitted().centroids] == [
            "cluster_1",
            "cluster_2",
            "cluster_3",
        ]

    def test_centres_are_addressable_by_feature_name(self) -> None:
        """What a centroid is for: a sentence rather than two subscripts."""
        centroid = fitted().centroids["cluster_1"]

        assert isinstance(centroid.coordinate_for("first"), float)
        assert centroid.feature_names == ("first", "second")


class TestConvergence:
    """That the loop stops, and stops because it settled."""

    def test_it_converges_well_inside_the_iteration_ceiling(self) -> None:
        """Separated blobs settle in a handful of passes, not three hundred."""
        model = fitted()

        assert 0 < model.iterations_run < 20

    def test_inertia_never_rises_as_k_grows(self) -> None:
        """Both Lloyd steps lower the objective, so more groups cannot do worse.

        This is also why inertia cannot choose k: it reaches zero when every
        row is its own group.
        """
        scores = [fitted(n_clusters=count).inertia for count in (1, 2, 3, 4)]

        assert scores == sorted(scores, reverse=True)

    def test_one_cluster_puts_its_centre_at_the_overall_mean(self) -> None:
        """The mean minimises the sum of squared distances, which is the proof."""
        model = fitted(n_clusters=1)
        rows = np.column_stack(
            [feature.values for feature in THREE_BLOBS.input_features]
        )

        assert centres_found(model)[0] == pytest.approx(
            tuple(np.mean(rows, axis=0)), abs=1e-06
        )

    def test_k_equal_to_n_samples_reaches_zero_inertia(self) -> None:
        """Every row its own centre, which is the degenerate end of the scale."""
        model = fitted(n_clusters=THREE_BLOBS.n_samples)

        assert model.inertia == pytest.approx(0.0, abs=1e-12)


class TestSeeding:
    """Restarts, reproducibility, and what k-means++ is for."""

    def test_the_same_seed_finds_the_same_grouping(self) -> None:
        first = fitted().clustering.labels
        second = fitted().clustering.labels

        assert np.array_equal(first, second)

    def test_different_seeds_still_recover_the_partition(self) -> None:
        """Separated blobs leave no room for a bad local minimum to survive."""
        for seed in range(5):
            model = KMeans(n_clusters=3, random_seed=seed).fit(
                THREE_BLOBS.input_features
            )

            assert same_partition(model.clustering.labels, THREE_BLOBS.true_groups)

    def test_restarts_keep_the_best_not_the_last(self) -> None:
        """Ten restarts cannot score worse than one, since the best is kept."""
        one = KMeans(n_clusters=3, n_initialisations=1, random_seed=4).fit(
            THREE_BLOBS.input_features
        )
        ten = KMeans(n_clusters=3, n_initialisations=10, random_seed=4).fit(
            THREE_BLOBS.input_features
        )

        assert ten.inertia <= one.inertia + 1e-12

    def test_each_restart_gets_its_own_seed(self) -> None:
        """Otherwise ten restarts are one restart reported as the best of ten.

        The comparison is **strict**, and that is the whole test. Restarts that
        all share the ensemble's single seed produce the identical fit every
        time, so ``many.inertia == one.inertia`` and any ``<=`` assertion
        passes while the bug sits there untouched. Measured on these rings at
        five clusters: one restart reaches 97.0961 and twenty reach 63.1966, so
        equality here means the seeding never varied.
        """
        one = KMeans(n_clusters=5, n_initialisations=1, random_seed=3).fit(
            CONCENTRIC_RINGS.input_features
        )
        many = KMeans(n_clusters=5, n_initialisations=20, random_seed=3).fit(
            CONCENTRIC_RINGS.input_features
        )

        assert many.inertia < one.inertia


class TestPredicting:
    """Labelling rows, including rows the fit never saw."""

    def test_predicting_the_training_rows_matches_the_fit(self) -> None:
        model = fitted()

        assert np.array_equal(
            np.asarray(model.predict(THREE_BLOBS.input_features)).astype(int),
            model.clustering.labels,
        )

    def test_a_new_row_falls_to_its_nearest_centre(self) -> None:
        """Nothing is relearned: the centres stay where ``fit`` left them."""
        model = fitted()
        near_first_blob = [Feature("first", [1.1]), Feature("second", [0.9])]

        predicted = int(np.asarray(model.predict(near_first_blob))[0])
        expected = int(model.clustering.labels[0])

        assert predicted == expected

    def test_column_order_does_not_matter(self) -> None:
        model = fitted()
        first, second = THREE_BLOBS.input_features

        assert np.array_equal(
            np.asarray(model.predict([first, second])),
            np.asarray(model.predict([second, first])),
        )

    def test_fit_predict_matches_fitting_then_predicting(self) -> None:
        together = KMeans(n_clusters=3, random_seed=0).fit_predict(
            THREE_BLOBS.input_features
        )
        apart = fitted().predict(THREE_BLOBS.input_features)

        assert np.array_equal(np.asarray(together), np.asarray(apart))

    def test_labels_are_whole_positions_inside_the_range(self) -> None:
        values = np.asarray(fitted().predict(THREE_BLOBS.input_features))

        assert np.array_equal(values, np.floor(values))
        assert values.min() >= 0
        assert values.max() <= 2


class TestWhatItCannotFind:
    """The limit, asserted rather than left for a user to discover."""

    def test_concentric_rings_defeat_it_at_two_clusters(self) -> None:
        """No pair of centres carves a ring out of a ring.

        The structure here is obvious to a person and outside what "closest
        centre" can express, at this k or any other. k-means cuts the picture
        in half instead, which is the best a pair of centres can do.
        """
        model = KMeans(n_clusters=2, random_seed=0).fit(CONCENTRIC_RINGS.input_features)

        assert not same_partition(model.clustering.labels, CONCENTRIC_RINGS.true_groups)

    def test_it_still_produces_a_usable_grouping_on_them(self) -> None:
        """Failing to find the rings is not failing to run."""
        model = KMeans(n_clusters=2, random_seed=0).fit(CONCENTRIC_RINGS.input_features)

        assert model.clustering.n_samples == CONCENTRIC_RINGS.n_samples
        assert not model.clustering.has_an_empty_cluster


class TestWhatItRefuses:
    """Guards, each raising from the MLLibError hierarchy."""

    def test_reading_the_clustering_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            _ = KMeans(n_clusters=2).clustering

    def test_predicting_before_fitting_raises(self) -> None:
        with pytest.raises(NotFittedError):
            KMeans(n_clusters=2).predict(THREE_BLOBS.input_features)

    def test_more_clusters_than_rows_raises(self) -> None:
        """A group would have to be empty, so there is nothing to fit."""
        with pytest.raises(TooFewValuesError):
            KMeans(n_clusters=5).fit(
                [Feature("first", [1.0, 2.0]), Feature("second", [3.0, 4.0])]
            )

    def test_duplicate_feature_names_are_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            KMeans(n_clusters=2).fit(
                [Feature("same", [1.0, 2.0, 3.0]), Feature("same", [4.0, 5.0, 6.0])]
            )

    def test_predicting_without_every_fitted_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().predict([THREE_BLOBS.input_features[0]])

    def test_predicting_with_an_unknown_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            fitted().predict(
                [
                    *THREE_BLOBS.input_features,
                    Feature("extra", [1.0] * THREE_BLOBS.n_samples),
                ]
            )
