"""What a clustering *is*: labels, centres, and the number being minimised.

A fitted clusterer has three things to say and they are usually returned as
three separate values a caller has to hold in step -- the label per row, the
centres, and the inertia. :class:`Clustering` is the pairing, and it exists for
the reason every pairing here does: a function reaching for
``return labels, centres, inertia`` is an object nobody has written yet.

Inertia, and why it cannot choose ``k`` for you
-----------------------------------------------
Inertia is the sum over rows of the squared distance from each row to its own
group's centre. It is exactly what k-means minimises, so it is the honest
measure of how well a given fit did -- against other fits *at the same ``k``*.

Across different ``k`` it is useless, and not merely unreliable. Inertia falls
monotonically as ``k`` grows, and at ``k == n_samples`` every row is its own
centre and inertia is 0. So "pick the ``k`` with the lowest inertia" always
answers "one cluster per row", which is not a grouping of anything.

What people do instead is look for the elbow: plot inertia against ``k`` and
find where the curve stops dropping steeply. That is a judgement, not a
calculation, and this library does not pretend otherwise --
:attr:`Clustering.inertia` is reported and nothing here selects on it.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.clustering.centroids import Centroids
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.exceptions import InvalidValuesError
from oop_ml.core.types import IndexArray


class Clustering:
    """A grouping of rows: which group each row is in, and where the groups are.

    Parameters
    ----------
    labels:
        One whole group position per row, each in ``0 .. n_clusters - 1``.
    centroids:
        The groups the labels refer to.
    inertia:
        The total squared distance from every row to its own group's centre.
        Supplied rather than recomputed, because the fit already had every
        distance in hand at the moment it assigned the labels.

    Raises
    ------
    InvalidValuesError
        If a label is not a whole position inside the centroids' range, or if
        ``inertia`` is negative.
    """

    __slots__ = ("_centroids", "_inertia", "_labels")

    def __init__(
        self, labels: IndexArray, centroids: Centroids, inertia: float
    ) -> None:
        as_array = np.asarray(labels)

        if as_array.ndim != 1:
            raise InvalidValuesError(
                f"labels must be one per row, so one dimension; got shape "
                f"{as_array.shape}"
            )

        # Before the range check and the intp cast: astype would silently
        # truncate 1.5 to 1, reassigning the row instead of refusing it.
        if as_array.size and np.any(as_array != np.floor(as_array)):
            raise InvalidValuesError(
                "labels must be whole group positions; got a fractional value"
            )

        if as_array.size and (
            as_array.min() < 0 or as_array.max() >= centroids.n_clusters
        ):
            raise InvalidValuesError(
                f"labels must fall in 0 .. {centroids.n_clusters - 1}; got "
                f"{as_array.min()} .. {as_array.max()}"
            )

        if inertia < 0.0:
            raise InvalidValuesError(
                f"inertia is a sum of squared distances and cannot be negative; "
                f"got {inertia}"
            )

        self._labels = as_array.astype(np.intp)
        self._centroids = centroids
        self._inertia = float(inertia)

    @property
    def labels(self) -> IndexArray:
        """Which group each row is in, as whole positions."""
        return self._labels.copy()

    @property
    def predictions(self) -> Predictions:
        """The labels as the library's one-answer-per-row type.

        Floats rather than positions, because that is what every ``predict``
        here hands back and what the evaluations read. The values are still
        whole numbers.
        """
        return Predictions.already_checked(self._labels.astype(np.float64))

    @property
    def centroids(self) -> Centroids:
        """Where the groups are."""
        return self._centroids

    @property
    def inertia(self) -> float:
        """Total squared distance from every row to its own group's centre.

        Lower is a better fit *at this ``k``*. See the module docstring for why
        comparing it across different ``k`` answers a question nobody asked.
        """
        return self._inertia

    @property
    def n_clusters(self) -> int:
        """How many groups the fit produced."""
        return self._centroids.n_clusters

    @property
    def n_samples(self) -> int:
        """How many rows were grouped."""
        return int(self._labels.size)

    @property
    def sizes(self) -> tuple[int, ...]:
        """How many rows landed in each group, in group order.

        A zero here is worth looking at. An empty group means a centre that no
        row is nearest to, which is a sign the initialisation placed it badly
        or that ``k`` is larger than the data supports.
        """
        return tuple(
            int(count) for count in np.bincount(self._labels, minlength=self.n_clusters)
        )

    @property
    def has_an_empty_cluster(self) -> bool:
        """Whether any group ended up with no rows at all."""
        return 0 in self.sizes

    def rows_in(self, name: str) -> IndexArray:
        """The positions of the rows belonging to one named group.

        Raises
        ------
        InvalidValuesError
            If no group has that name.
        """
        for position, centroid in enumerate(self._centroids):
            if centroid.name == name:
                return np.flatnonzero(self._labels == position)

        raise InvalidValuesError(
            f"unknown cluster {name!r}; this holds "
            f"{[centroid.name for centroid in self._centroids]}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"Clustering(n_clusters={self.n_clusters}, "
            f"n_samples={self.n_samples}, inertia={self._inertia:.4f})"
        )


class InitialisationAttempt:
    """One restart's result: what it found, and how long it took to settle.

    k-means runs the whole algorithm several times from different seedings and
    keeps the best. Each run has two things to report, and a method reaching for
    ``return clustering, iterations`` is this object not yet written -- the
    caller would then have to remember which came first.

    Parameters
    ----------
    clustering:
        The grouping this restart settled on.
    iterations_run:
        How many assign/update passes it took. Reported rather than discarded
        because it is the cheapest signal that ``max_iterations`` was the
        stopping condition rather than convergence.

    Raises
    ------
    InvalidValuesError
        If ``iterations_run`` is negative.
    """

    __slots__ = ("_clustering", "_iterations_run")

    def __init__(self, clustering: Clustering, iterations_run: int) -> None:
        if iterations_run < 0:
            raise InvalidValuesError(
                f"a restart cannot run a negative number of passes; got "
                f"{iterations_run}"
            )

        self._clustering = clustering
        self._iterations_run = int(iterations_run)

    @property
    def clustering(self) -> Clustering:
        """The grouping this restart settled on."""
        return self._clustering

    @property
    def iterations_run(self) -> int:
        """How many assign/update passes it took."""
        return self._iterations_run

    @property
    def inertia(self) -> float:
        """What this restart scored, which is how restarts are compared."""
        return self._clustering.inertia

    def beats(self, other: InitialisationAttempt) -> bool:
        """Whether this restart found a better grouping than ``other``.

        Lower inertia wins, and a tie goes to the incumbent -- the same strict
        comparison :meth:`~oop_ml.core.tree.split.Split.beats` uses, and for the
        same reason. Two restarts finding genuinely equivalent groupings come
        back differing in the last bits, and swapping on a tie would make the
        reported fit depend on iteration order.
        """
        return self.inertia < other.inertia

    def __repr__(self) -> str:
        return (
            f"InitialisationAttempt(inertia={self.inertia:.4f}, "
            f"iterations_run={self._iterations_run})"
        )
