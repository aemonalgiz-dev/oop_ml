"""Find groups by finding dense regions, and let the count fall out of the data.

Every other clusterer here is told how many groups to find. This one is told
how *crowded* a group has to be, and how many groups that turns out to imply is
an answer rather than a question. Two consequences follow, and they are the
whole reason to reach for it.

A group can be any shape
-------------------------
k-means assigns each row to the nearest centre, so its groups are the cells of
a Voronoi diagram: convex, and separated by straight lines. That is exactly
right for round clumps and exactly wrong for two interleaving crescents, which
have no straight line between them and whose centres are nearly the same point.
Density asks a local question instead -- is this row in a crowd, and is that
crowd joined to this one -- so a group is whatever connected dense region the
data actually contains, however it bends.

A row may belong to nothing
----------------------------
k-means places every row, because a nearest centre always exists. Here a row in
a sparse region is *noise*, labelled
:data:`~oop_ml.core.clustering.density.NOISE_LABEL`, which is the first answer
in this library that is not a group. That is not a
gap in the method; an outlier genuinely is not a member of anything, and being
forced to name the least distant group is how a single stray point drags a
centre off its cluster in k-means.

The three kinds of row
-----------------------
Everything follows from one count. Take a row's *neighbourhood*: every row
within ``radius`` of it, itself included, since a row is always within zero of
itself.

- **Core.** The neighbourhood holds at least ``min_neighbourhood_size`` rows.
  This row is inside a crowd and can hold a group together.
- **Border.** Not core, but within ``radius`` of one that is. Swept into that
  core point's group without being able to extend it any further.
- **Noise.** Neither.

A group is then a set of core points connected through each other: start at any
core point, take every core point within ``radius``, take every core point
within ``radius`` of *those*, and keep going. Connectivity is transitive, so
the groups are the connected components of the graph on core points, and
nothing about that depends on the order the rows arrived in.

Where the order does matter, and why it is settled elsewhere
--------------------------------------------------------------
Border points. A border point within ``radius`` of core points from two
different groups has two equally good claims on it, and the usual
implementations hand it to whichever expansion reached it first. The rule this
library uses instead is the nearest core point, written down once in
:mod:`oop_ml.core.clustering.density` so that both backends and any later query
follow it. See that module for the argument.

Choosing the two numbers
-------------------------
``min_neighbourhood_size`` is the easier one: a common starting point is twice
the number of columns, and raising it makes the model more willing to call
things noise. ``radius`` is the hard one and it is what the method trades away
for not having to name ``k``. Too small and everything is noise; too large and
separate groups fuse into one. Both failures are visible, which is the saving
grace: fit twice and look at ``n_clusters`` and ``n_noise``.

Both are also distances, so both depend on the units. Standardise the columns
first, exactly as for the neighbour models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Clusterer
from oop_ml.core.clustering.density import NOISE_LABEL, labels_from_core_points
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.distance.calculations import Distance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.types import FloatArray, IndexArray


class DBSCAN(Clusterer[Sequence[Feature]]):
    """Group rows by density, discovering how many groups there are.

    Parameters
    ----------
    radius:
        How near two rows have to be to count as neighbours. The number this
        method trades for not having to be told ``k``, and the one worth
        sweeping: too small leaves everything noise, too large fuses separate
        groups into one.
    min_neighbourhood_size:
        How many rows have to lie within ``radius`` of a row, itself included,
        before it is dense enough to hold a group together. Raising it makes
        the model quicker to call a row noise.
    metric:
        What near means. Unlike
        :class:`~oop_ml.numpy.clustering.k_means.KMeans`, which is tied to
        Euclidean distance because its update step is a mean, this model only
        ever asks whether one row is within ``radius`` of another, and any of
        the six metrics answers that as well as another.

    Raises
    ------
    pydantic.ValidationError
        If ``radius`` is not positive or the neighbourhood size is below one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    radius: float = Field(default=0.5, gt=0.0)
    min_neighbourhood_size: int = Field(default=5, ge=1)
    metric: DistanceMetric | Distance = DistanceMetric.EUCLIDEAN

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_core_points",
        "_core_labels",
        "_core_indices",
        "_labels",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _core_points: RowBlock | None = PrivateAttr(default=None)
    _core_labels: FloatArray | None = PrivateAttr(default=None)
    _core_indices: IndexArray | None = PrivateAttr(default=None)
    _labels: FloatArray | None = PrivateAttr(default=None)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The columns this model was fitted on, in order."""
        self._check_fitted()
        assert self._feature_names is not None
        return self._feature_names

    @property
    def labels(self) -> FloatArray:
        """The group each training row was put in, or ``-1`` for noise."""
        self._check_fitted()
        assert self._labels is not None
        return self._labels

    @property
    def n_clusters(self) -> int:
        """How many groups the fit found.

        Learned rather than configured, which is the difference from every
        other clusterer here. Zero is a real answer: it means no row had
        enough company to hold a group together at this radius.
        """
        self._check_fitted()
        assert self._core_labels is not None
        return int(np.unique(self._core_labels).size) if self._core_labels.size else 0

    @property
    def n_noise(self) -> int:
        """How many training rows were placed in no group at all.

        Worth reading beside ``n_clusters`` whenever ``radius`` is being
        chosen, since the two failures of a bad radius look different here:
        too small drives this number toward every row, too large drives
        ``n_clusters`` toward one.
        """
        self._check_fitted()
        assert self._labels is not None
        return int(np.count_nonzero(self._labels == NOISE_LABEL))

    @property
    def core_indices(self) -> IndexArray:
        """Which training rows were dense enough to hold a group together.

        Positions into the rows ``fit`` was given, ascending.
        """
        self._check_fitted()
        assert self._core_indices is not None
        return self._core_indices

    @property
    def core_points(self) -> RowBlock:
        """The core rows themselves, which are the whole of the fitted model.

        Everything else a fitted clustering can answer follows from these and
        their labels: a border point is near one of them, and noise is near
        none of them.
        """
        self._check_fitted()
        assert self._core_points is not None
        return self._core_points

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Find the dense regions, join them up, and label every row.

        Three passes, and only the middle one is interesting. Count each row's
        neighbours to find the core points; walk the core points to find which
        of them are connected; then label every row by the nearest core point
        it can reach.

        Parameters
        ----------
        input_values:
            One or more columns, all the same length.

        Returns
        -------
        Self
            This model, so calls can chain.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If the features disagree in length.
        """
        feature_set = FeatureSet(input_values)
        names = tuple(feature.name for feature in feature_set)
        rows = rows_of(feature_set.feature_matrix, names)

        distances = self.metric.between(rows, rows)
        # A row is within zero of itself, so its own entry is always inside the
        # radius and the count includes it. That is the convention the size is
        # named for, and it is the one scikit-learn's min_samples uses too.
        within_radius = distances <= self.radius
        is_core = within_radius.sum(axis=1) >= self.min_neighbourhood_size

        core_indices = np.flatnonzero(is_core)
        core_labels = self._connected_components(
            within_radius[np.ix_(core_indices, core_indices)]
        )
        core_points = rows_of(rows.values[core_indices], names)

        labels = labels_from_core_points(
            rows, core_points, core_labels, self.radius, self.metric
        )

        self._feature_names = names
        self._core_points = core_points
        self._core_labels = core_labels
        self._core_indices = core_indices
        self._labels = labels

        self._mark_fitted()
        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The group each row belongs to, or ``-1`` if it belongs to none.

        Rows the fit never saw are allowed, and answering them is the same
        question a border point asks: which core point is nearest, and is it
        near enough. A fitted model is a set of labelled core points, so
        nothing else is needed.

        On the training rows this reproduces ``labels`` exactly, because it is
        the same rule that produced them.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._feature_names is not None
        assert self._core_points is not None
        assert self._core_labels is not None

        matched = FeatureSet.matching(self._feature_names, input_values)
        queries = rows_of(matched.feature_matrix, self._feature_names)

        return Predictions.already_checked(
            labels_from_core_points(
                queries,
                self._core_points,
                self._core_labels,
                self.radius,
                self.metric,
            )
        )

    @staticmethod
    def _connected_components(adjacency: np.ndarray) -> FloatArray:
        """Number each core point by which connected group it sits in.

        A breadth-first walk from every core point not yet reached. Written as
        a plain walk rather than as a matrix power or a union-find, because
        what it has to be is obviously the definition: a group is everything
        reachable from a starting point through neighbours, and reachability is
        transitive.

        Parameters
        ----------
        adjacency:
            ``(n_core, n_core)`` of booleans, true where two core points are
            within the radius of each other.

        Returns
        -------
        FloatArray
            ``(n_core,)`` labels running ``0.0`` upward, in the order the walk
            first reached each group.
        """
        n_core = adjacency.shape[0]
        labels = np.full(n_core, NOISE_LABEL, dtype=np.float64)
        next_label = 0.0

        for start in range(n_core):
            if labels[start] != NOISE_LABEL:
                continue

            labels[start] = next_label
            waiting = [start]
            while waiting:
                current = waiting.pop()
                for neighbour in np.flatnonzero(adjacency[current]):
                    if labels[neighbour] == NOISE_LABEL:
                        labels[neighbour] = next_label
                        waiting.append(int(neighbour))

            next_label += 1.0

        return labels
