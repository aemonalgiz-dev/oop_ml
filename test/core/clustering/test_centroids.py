"""Spec for what a clustering learned, and what a clustering is.

Green already, since these are value objects rather than concepts. Two of their
guards are worth exercising anyway.

``test_a_non_finite_coordinate_is_rejected`` is the one that matters. Averaging
an empty group divides by zero, and numpy answers with ``nan`` and a warning
rather than an error. A ``nan`` centre makes every later distance ``nan``, every
comparison against it false, and every ``argmin`` return 0 -- so the fit
collapses to a single cluster while reporting a full set of centres. The
constructor refuses the ``nan`` at the moment it is created instead.

``TestNoOrdering`` records a deliberate absence. Principal components are ranked
and their container enforces it; clusters are not, and inventing an order would
be asserting a fact the mathematics does not contain.
"""

import numpy as np
import pytest

from oop_ml.core.clustering.centroids import Centroid, Centroids
from oop_ml.core.clustering.clustering import Clustering, InitialisationAttempt
from oop_ml.core.data.row_block import rows_of
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)

NAMES = ("first", "second")


def centroid(name: str, *coordinates: float, names: tuple[str, ...] = NAMES):
    """One named point."""
    return Centroid(name, np.array(coordinates, dtype=np.float64), names)


def two_centres() -> Centroids:
    """Centres at (0, 0) and (3, 4), which are 5 apart."""
    return Centroids([centroid("cluster_1", 0.0, 0.0), centroid("cluster_2", 3.0, 4.0)])


def block(*rows: tuple[float, float]):
    """Rows over the two fixture features."""
    return rows_of(np.array(rows, dtype=np.float64), NAMES)


class TestOneCentre:
    """A named point, and the distances it can measure."""

    def test_coordinates_are_addressable_by_feature_name(self) -> None:
        one = centroid("cluster_1", 148.0, 2.0)

        assert one.coordinate_for("first") == pytest.approx(148.0)
        assert one.coordinate_for("second") == pytest.approx(2.0)

    def test_asking_for_an_unknown_feature_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            centroid("cluster_1", 1.0, 2.0).coordinate_for("unseen")

    def test_squared_distance_is_squared(self) -> None:
        """(3, 4) is 5 away, so the squared distance is 25, not 5."""
        distances = centroid("cluster_1", 0.0, 0.0).squared_distance_to(
            block((3.0, 4.0))
        )

        assert distances == pytest.approx([25.0])

    def test_measures_every_row_at_once(self) -> None:
        distances = centroid("cluster_1", 0.0, 0.0).squared_distance_to(
            block((3.0, 4.0), (1.0, 0.0), (0.0, 0.0))
        )

        assert distances == pytest.approx([25.0, 1.0, 0.0])

    def test_rows_over_the_wrong_features_are_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            centroid("cluster_1", 0.0, 0.0).squared_distance_to(
                rows_of(np.array([[1.0, 2.0]]), ("first", "different"))
            )

    def test_a_non_finite_coordinate_is_rejected(self) -> None:
        """What averaging an empty group produces, caught where it is made.

        numpy returns nan with a warning rather than raising, and a nan centre
        poisons every distance downstream while looking like a centre.
        """
        with pytest.raises(InvalidValuesError):
            centroid("cluster_1", float("nan"), 0.0)

    def test_the_wrong_number_of_coordinates_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            centroid("cluster_1", 1.0, 2.0, 3.0)

    def test_a_blank_name_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            centroid("   ", 1.0, 2.0)


class TestTheGroupOfCentres:
    """Reading the centres, and what the container refuses."""

    def test_positions_are_one_group_per_row(self) -> None:
        """The orientation the assignment step wants."""
        positions = two_centres().positions

        assert positions.shape == (2, 2)
        assert positions[1] == pytest.approx([3.0, 4.0])

    def test_distances_are_rows_by_clusters(self) -> None:
        """Entry [i, k] is row i against group k, so argmin along axis 1."""
        distances = two_centres().squared_distances_to(block((0.0, 0.0), (3.0, 4.0)))

        assert distances.shape == (2, 2)
        assert distances[0] == pytest.approx([0.0, 25.0])
        assert distances[1] == pytest.approx([25.0, 0.0])

    def test_addressable_by_name(self) -> None:
        assert two_centres()["cluster_2"].coordinate_for("first") == pytest.approx(3.0)

    def test_asking_for_an_unknown_cluster_raises(self) -> None:
        with pytest.raises(InvalidValuesError):
            _ = two_centres()["cluster_9"]

    def test_no_centres_is_rejected(self) -> None:
        with pytest.raises(EmptyValuesError):
            Centroids([])

    def test_a_duplicate_name_is_rejected(self) -> None:
        with pytest.raises(NonUniqueFeaturesError):
            Centroids(
                [centroid("cluster_1", 0.0, 0.0), centroid("cluster_1", 1.0, 1.0)]
            )

    def test_centres_in_different_features_are_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            Centroids(
                [
                    centroid("cluster_1", 0.0, 0.0),
                    centroid("cluster_2", 1.0, 1.0, names=("first", "different")),
                ]
            )


class TestNoOrdering:
    """A deliberate absence, recorded so it is not read as an oversight."""

    def test_centres_need_not_be_in_any_particular_order(self) -> None:
        """Clusters are not ranked, and nothing about them says which is first.

        PrincipalComponents refuses an out-of-order set because components
        genuinely are ranked by variance. Cluster numbering is an artefact of
        which centre the seeding placed first, so there is no order to enforce
        and enforcing one would invent a fact.
        """
        far_then_near = Centroids(
            [centroid("cluster_1", 100.0, 100.0), centroid("cluster_2", 0.0, 0.0)]
        )

        assert far_then_near.n_clusters == 2


class TestClustering:
    """Labels, sizes, and the number being minimised."""

    def result(self, labels: list[int], inertia: float = 3.0) -> Clustering:
        return Clustering(np.array(labels), two_centres(), inertia)

    def test_reports_its_sizes_in_group_order(self) -> None:
        assert self.result([0, 0, 0, 1]).sizes == (3, 1)

    def test_an_empty_group_is_reported_rather_than_hidden(self) -> None:
        """A centre no row is nearest to is a finding, not a detail."""
        result = self.result([0, 0, 0, 0])

        assert result.sizes == (4, 0)
        assert result.has_an_empty_cluster

    def test_finds_the_rows_in_a_named_group(self) -> None:
        assert list(self.result([0, 1, 0, 1]).rows_in("cluster_2")) == [1, 3]

    def test_labels_become_predictions_for_the_rest_of_the_library(self) -> None:
        predictions = self.result([0, 1, 0, 1]).predictions

        assert np.asarray(predictions) == pytest.approx([0.0, 1.0, 0.0, 1.0])

    def test_counts_its_rows_and_groups(self) -> None:
        result = self.result([0, 1, 0])

        assert result.n_samples == 3
        assert result.n_clusters == 2
        assert len(result) == 3

    def test_a_label_outside_the_groups_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            self.result([0, 1, 2])

    def test_a_negative_inertia_is_rejected(self) -> None:
        """It is a sum of squares, so a negative one is a calculation error."""
        with pytest.raises(InvalidValuesError):
            self.result([0, 1], inertia=-1.0)


class TestInitialisationAttempt:
    """One restart, and how restarts are compared."""

    def attempt(self, inertia: float, iterations: int = 5) -> InitialisationAttempt:
        return InitialisationAttempt(
            Clustering(np.array([0, 1]), two_centres(), inertia), iterations
        )

    def test_lower_inertia_wins(self) -> None:
        assert self.attempt(3.0).beats(self.attempt(4.0))
        assert not self.attempt(4.0).beats(self.attempt(3.0))

    def test_a_tie_goes_to_the_incumbent(self) -> None:
        """Strict comparison, so iteration order cannot decide the fit.

        Two restarts finding equivalent groupings come back differing in the
        last bits, and swapping on a tie would make the reported result depend
        on which restart ran first.
        """
        assert not self.attempt(3.0).beats(self.attempt(3.0))

    def test_carries_what_it_cost(self) -> None:
        assert self.attempt(3.0, iterations=7).iterations_run == 7

    def test_a_negative_pass_count_is_rejected(self) -> None:
        with pytest.raises(InvalidValuesError):
            self.attempt(3.0, iterations=-1)
