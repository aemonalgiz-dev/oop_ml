"""What a clustering learns: a point per group, bound to the features it lives in.

A fitted k-means holds a ``(n_clusters, n_features)`` array of centres. As a
bare array that is a table addressed by two positions, neither of which is
named: row 2 is a cluster only by convention, and column 1 is a feature only
because of the order somebody assembled the input in.

:class:`Centroid` is one centre as an object -- a named point whose coordinates
are bound to the feature names they are coordinates *in*, and which knows how to
measure its own distance to a block of rows. :class:`Centroids` is the group.

A centroid is not a coefficient, though the structure is identical
-------------------------------------------------------------------
Both are named floats, and this library already reuses
:class:`~oop_ml.core.data.coefficients.Coefficients` for a principal component's
loadings. That reuse was honest, because a loading really is a weight applied to
a feature. A centroid coordinate is not a weight applied to anything -- it is a
*location*, a value the feature itself takes at the centre of the group. It
answers "what is a typical member like", not "how much does this feature
count". Hence its own class, and hence ``distance_to`` living on it, which is
behaviour no coefficient should have.

Why there is no ordering invariant here
---------------------------------------
:class:`~oop_ml.core.decomposition.components.PrincipalComponents` refuses an
out-of-order set, because components are genuinely ranked and an unreversed sort
is a silent bug. Clusters have no such order. Nothing makes cluster 0 more
important than cluster 2, the numbering is an artefact of which centre the
initialisation happened to place first, and two fits of the same data can agree
completely about the grouping while disagreeing about every label. Imposing an
order would invent a fact the mathematics does not contain.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.core.types import FloatArray


class Centroid:
    """One group's centre: a named point in the space the features span.

    Parameters
    ----------
    name:
        What this group is called. Named by position (``cluster_1`` and so on)
        for the same reason a principal component is: a group discovered
        without a target has no name of its own, and any name a fit invented
        would be a claim about what the group *means* that nothing in the data
        supports.
    coordinates:
        One value per feature, bound to that feature's name. This is a location
        in feature space, in the features' own units.
    feature_names:
        The features the coordinates are coordinates in, in order.

    Raises
    ------
    InvalidValuesError
        If ``name`` is blank, if the coordinates are not finite, or if there is
        not exactly one coordinate per feature name.
    """

    __slots__ = ("_coordinates", "_feature_names", "_name")

    def __init__(
        self, name: str, coordinates: FloatArray, feature_names: Sequence[str]
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("Centroid name must be a non-empty string")

        as_array = np.asarray(coordinates, dtype=np.float64)

        if as_array.ndim != 1:
            raise InvalidValuesError(
                f"{name} must be one point, so one dimension; got shape "
                f"{as_array.shape}"
            )

        if as_array.size != len(feature_names):
            raise InvalidValuesError(
                f"{name} has {as_array.size} coordinates against "
                f"{len(feature_names)} features"
            )

        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError(
                f"{name} holds a non-finite coordinate, which usually means an "
                f"empty group was averaged"
            )

        self._name = name.strip()
        self._coordinates = as_array
        self._feature_names = tuple(feature_names)

    @property
    def name(self) -> str:
        """What this group is called."""
        return self._name

    @property
    def coordinates(self) -> FloatArray:
        """The centre as a plain vector, in feature order."""
        return self._coordinates.copy()

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features this point is a point in, in order."""
        return self._feature_names

    @property
    def n_features(self) -> int:
        """How many dimensions this point has."""
        return self._coordinates.size

    def coordinate_for(self, name: str) -> float:
        """What this group's centre looks like along one feature.

        The reason coordinates are bound to names: "the centre of cluster 2 has
        floor area 148" is a sentence, and ``centres[2][0]`` is not.

        Raises
        ------
        InvalidValuesError
            If ``name`` is not one of this centroid's features.
        """
        if name not in self._feature_names:
            raise InvalidValuesError(
                f"unknown feature {name!r}; this centroid lives in "
                f"{list(self._feature_names)}"
            )

        return float(self._coordinates[self._feature_names.index(name)])

    def squared_distance_to(self, rows: RowBlock) -> FloatArray:
        """Squared Euclidean distance from this centre to every row.

        Squared rather than plain, and that is not an optimisation. k-means
        minimises the sum of *squared* distances, so the squared quantity is the
        objective itself; taking a square root here would mean taking it back
        out again everywhere the number is used. Ordering is unaffected either
        way, since squaring is monotonic on non-negative numbers.

        Parameters
        ----------
        rows:
            A block whose columns are this centroid's features, in its order.

        Returns
        -------
        FloatArray
            One distance per row, shape ``(n_rows,)``.

        Raises
        ------
        InvalidValuesError
            If the block's features are not this centroid's, in order.
        """
        if rows.feature_names != self._feature_names:
            raise InvalidValuesError(
                f"this centroid lives in {list(self._feature_names)}; got rows "
                f"over {list(rows.feature_names)}"
            )

        gaps = rows.values - self._coordinates

        return np.sum(gaps * gaps, axis=1)

    def __repr__(self) -> str:
        return f"Centroid({self._name!r}, n_features={self.n_features})"


class Centroids:
    """Every group's centre, addressable by name.

    Parameters
    ----------
    centroids:
        One per group. All must live in the same features, in the same order.

    Raises
    ------
    EmptyValuesError
        If no centroids are supplied.
    NonUniqueFeaturesError
        If two centroids share a name.
    InvalidValuesError
        If the centroids do not all live in the same features.
    """

    __slots__ = ("_centroids",)

    def __init__(self, centroids: Sequence[Centroid]) -> None:
        if not centroids:
            raise EmptyValuesError("a clustering needs at least one centroid")

        self._centroids = tuple(centroids)

        names = [centroid.name for centroid in self._centroids]
        if len(set(names)) != len(names):
            raise NonUniqueFeaturesError(f"centroid names must be unique; got {names}")

        expected = self._centroids[0].feature_names
        for centroid in self._centroids[1:]:
            if centroid.feature_names != expected:
                raise InvalidValuesError(
                    f"{centroid.name} lives in {list(centroid.feature_names)}, but "
                    f"{self._centroids[0].name} lives in {list(expected)}"
                )

    @property
    def n_clusters(self) -> int:
        """How many groups this holds."""
        return len(self._centroids)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features every centre lives in, in order."""
        return self._centroids[0].feature_names

    @property
    def n_features(self) -> int:
        """How many dimensions the centres have."""
        return len(self.feature_names)

    @property
    def positions(self) -> FloatArray:
        """The centres as a matrix, one group per row.

        Shape ``(n_clusters, n_features)``, which is the orientation the
        assignment step wants when it compares every row against every centre.
        """
        return np.array(
            [centroid.coordinates for centroid in self._centroids], dtype=np.float64
        )

    def squared_distances_to(self, rows: RowBlock) -> FloatArray:
        """Squared distance from every row to every centre.

        Returns
        -------
        FloatArray
            Shape ``(n_rows, n_clusters)``. Entry ``[i, k]`` is how far row
            ``i`` sits from group ``k``, so the assignment is an ``argmin``
            along axis 1.
        """
        return np.column_stack(
            [centroid.squared_distance_to(rows) for centroid in self._centroids]
        )

    def value_for(self, name: str) -> Centroid:
        """The centroid called ``name``.

        Raises
        ------
        InvalidValuesError
            If no group has that name.
        """
        for centroid in self._centroids:
            if centroid.name == name:
                return centroid

        raise InvalidValuesError(
            f"unknown cluster {name!r}; this holds "
            f"{[centroid.name for centroid in self._centroids]}"
        )

    def __getitem__(self, name: str) -> Centroid:
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return any(centroid.name == name for centroid in self._centroids)

    def __iter__(self) -> Iterator[Centroid]:
        return iter(self._centroids)

    def __len__(self) -> int:
        return self.n_clusters

    def __repr__(self) -> str:
        return f"Centroids(n_clusters={self.n_clusters}, n_features={self.n_features})"
