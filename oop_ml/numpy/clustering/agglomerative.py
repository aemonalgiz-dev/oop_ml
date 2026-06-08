"""Build the groups from the bottom up, by repeatedly merging the nearest pair.

The third clusterer here and the one with the simplest rule. Start with every
row alone in a group of one. Find the two groups that are nearest each other
and join them. Repeat. Stop when the number of groups left is the number asked
for.

Nothing about that says what "nearest" means for two *groups*, and that is the
whole of the method's character rather than a detail. A group is not a point,
so the distance between two of them is a choice; the four available are in
:class:`~oop_ml.core.clustering.linkage.Linkage`, and they behave differently
enough that picking one is the modelling decision. Single linkage follows
chains and finds long thin shapes; complete linkage insists every pair be close
and makes round ones; average sits between them; ward merges whichever pair
adds least to the total spread and is the hierarchical relative of k-means.

What it gives that neither other clusterer does
------------------------------------------------
A whole family of answers rather than one. Every merge happens at some height,
and the sequence of heights is the same whatever number of groups is asked
for -- ask for two and the fit stops earlier, ask for five and it stops later,
but the merges up to that point are identical. So ``merge_distances`` describes
every grouping the data admits at once, and a large gap between consecutive
heights is the data saying that this is where it would rather be cut.

That is also the answer to a question k-means cannot be asked. k-means has to
be told ``k`` and inertia falls monotonically with it, so inertia cannot choose;
here the heights are on a comparable scale to each other and a jump between two
of them means something.

Why it is quadratic in memory and cubic in time
------------------------------------------------
Every step needs the distance between every surviving pair of groups, and there
are ``n`` groups to begin with, so the pairwise block is ``n^2`` and the search
for its minimum runs once per merge. Written plainly that is O(n^3), which is
what this does. The usual repairs are a priority queue over the candidate pairs
and the Lance-Williams recurrence, which updates a merged group's distances
from the two it came from rather than recomputing them; both are optimisations
of this loop rather than different algorithms, and neither is here yet.

Where the answer is not determined
------------------------------------
Two pairs exactly tied for nearest. The merge taken first is then whichever the
search reached first, and on the next step the other pair may no longer be a
pair at all. The *heights* survive that -- the sequence of minimum distances is
the same either way -- which is why ``merge_distances`` is exposed and the tree
of which-merged-with-which is not. Pinning the tree would be pinning an
arbitrary choice, the same argument the density clusterer makes about its
border points.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from oop_ml.core.base.estimator import Clusterer
from oop_ml.core.clustering.assignment import labels_from_nearest
from oop_ml.core.clustering.linkage import Linkage
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.predictions import Predictions
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.distance.calculations import Distance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import TooFewValuesError
from oop_ml.core.types import FloatArray


class AgglomerativeClustering(Clusterer[Sequence[Feature]]):
    """Group rows by merging the nearest pair of groups until enough are left.

    Parameters
    ----------
    n_clusters:
        How many groups to stop at. Unlike the density clusterer this has to be
        given, and unlike k-means the fit that produces it also produces every
        coarser grouping on the way, which ``merge_distances`` reports.
    linkage:
        What the distance between two groups is taken to be, and the decision
        that gives the method its character. See
        :class:`~oop_ml.core.clustering.linkage.Linkage`.
    metric:
        What near means for two rows. Any of the six, except under ``WARD``
        linkage, which measures spread about a mean and so is defined for
        Euclidean distance alone.

    Raises
    ------
    pydantic.ValidationError
        If ``n_clusters`` is below one, or ``WARD`` is paired with a metric
        that is not Euclidean.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_clusters: int = Field(default=2, ge=1)
    linkage: Linkage = Linkage.WARD
    metric: DistanceMetric | Distance = DistanceMetric.EUCLIDEAN

    LEARNED_STATE: ClassVar[tuple[str, ...]] = (
        "_feature_names",
        "_remembered_rows",
        "_labels",
        "_merge_distances",
    )

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _remembered_rows: RowBlock | None = PrivateAttr(default=None)
    _labels: FloatArray | None = PrivateAttr(default=None)
    _merge_distances: FloatArray | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_ward_has_the_metric_it_needs(self) -> Self:
        """Ward measures spread about a mean, and a mean minimises squared
        Euclidean distance specifically, so the pairing is refused at
        construction rather than producing a number that is not the one the
        rule describes. The engine refuses the same pairing."""
        if self.linkage is Linkage.WARD and self.metric is not DistanceMetric.EUCLIDEAN:
            raise ValueError(
                "ward linkage merges whichever pair adds least to the spread "
                "about a group mean, and a mean minimises squared Euclidean "
                "distance and no other, so it is defined for "
                "DistanceMetric.EUCLIDEAN alone. Choose another linkage to keep "
                "this metric, or Euclidean to keep ward"
            )

        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The columns this model was fitted on, in order."""
        self._check_fitted()
        assert self._feature_names is not None
        return self._feature_names

    @property
    def labels(self) -> FloatArray:
        """The group each training row was put in, ``0 .. n_clusters - 1``."""
        self._check_fitted()
        assert self._labels is not None
        return self._labels

    @property
    def merge_distances(self) -> FloatArray:
        """The height of every merge the fit made, in the order it made them.

        Ascending, since each step takes the nearest pair left, and one entry
        shorter than the number of rows by ``n_clusters``. The gaps are the
        useful part: a jump between two consecutive heights says the data would
        rather have been cut there.
        """
        self._check_fitted()
        assert self._merge_distances is not None
        return self._merge_distances

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Merge the nearest pair of groups until ``n_clusters`` are left.

        Parameters
        ----------
        input_values:
            One or more columns, all the same length, and at least as many rows
            as groups asked for.

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
        TooFewValuesError
            If there are fewer rows than groups asked for, since a group cannot
            be empty and merging only ever reduces the count.
        """
        feature_set = FeatureSet(input_values)
        names = tuple(feature.name for feature in feature_set)
        rows = rows_of(feature_set.feature_matrix, names)
        n_rows = rows.n_rows

        if n_rows < self.n_clusters:
            raise TooFewValuesError(
                f"{self.n_clusters} groups were asked for and {n_rows} row(s) "
                "were supplied. Every row starts in a group of its own and "
                "merging only reduces the count, so there is no way to reach "
                "that many"
            )

        members, heights = self._merged_until_enough_remain(rows)

        labels = np.empty(n_rows, dtype=np.float64)
        for number, group in enumerate(members):
            labels[group] = float(number)

        self._feature_names = names
        self._remembered_rows = rows
        self._labels = labels
        self._merge_distances = np.asarray(heights, dtype=np.float64)

        self._mark_fitted()
        return self

    def predict(self, input_values: Sequence[Feature]) -> Predictions:
        """The group each row belongs to, by its nearest training row.

        A hierarchical clustering has no centres and no shape of any kind, so
        what a fitted one holds is the rows it saw and their labels, and an
        unseen row joins whichever of them is closest. See
        :mod:`oop_ml.core.clustering.assignment` for why that rule rather than
        a nearest-centre one.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._feature_names is not None
        assert self._remembered_rows is not None
        assert self._labels is not None

        matched = FeatureSet.matching(self._feature_names, input_values)
        queries = rows_of(matched.feature_matrix, self._feature_names)

        return Predictions.already_checked(
            labels_from_nearest(
                queries, self._remembered_rows, self._labels, self.metric
            )
        )

    def _merged_until_enough_remain(
        self, rows: RowBlock
    ) -> tuple[list[list[int]], list[float]]:
        """The merge loop, and the whole of the algorithm.

        Returns the surviving groups as lists of row positions, and the height
        of each merge in the order it was made. Written as the definition
        rather than as a priority queue, so that what it does is legible; see
        the module docstring for what that costs.
        """
        pairwise = self.metric.between(rows, rows)
        members: list[list[int]] = [[index] for index in range(rows.n_rows)]
        heights: list[float] = []

        while len(members) > self.n_clusters:
            nearest_height = float("inf")
            nearest_pair = (0, 1)

            for first in range(len(members)):
                for second in range(first + 1, len(members)):
                    height = self._between_groups(
                        rows, pairwise, members[first], members[second]
                    )
                    if height < nearest_height:
                        nearest_height = height
                        nearest_pair = (first, second)

            first, second = nearest_pair
            heights.append(nearest_height)
            members[first] = members[first] + members[second]
            members.pop(second)

        return members, heights

    def _between_groups(
        self,
        rows: RowBlock,
        pairwise: FloatArray,
        first: list[int],
        second: list[int],
    ) -> float:
        """How far apart two groups are, under this model's linkage.

        Three of the four rules are a summary of the block of member-to-member
        distances, which is why they work under any metric. Ward is the one
        that is not: it asks how much the total spread would grow, which is a
        statement about means, so it reads the rows themselves.
        """
        if self.linkage.reads_the_pairwise_block:
            block = pairwise[np.ix_(first, second)]

            if self.linkage is Linkage.SINGLE:
                return float(block.min())
            if self.linkage is Linkage.COMPLETE:
                return float(block.max())
            return float(block.mean())

        return self._ward_between(rows, first, second)

    @staticmethod
    def _ward_between(rows: RowBlock, first: list[int], second: list[int]) -> float:
        """``sqrt(2 |A| |B| / (|A| + |B|)) * ||mean_A - mean_B||``.

        The increase in total within-group spread that merging the two would
        cause, in the form the usual implementations report it. The weight in
        front is what makes ward reluctant to merge two *large* groups even
        when their means are close: joining them moves many rows, and the
        objective counts every one.
        """
        values = rows.values
        gap = values[first].mean(axis=0) - values[second].mean(axis=0)
        weight = 2.0 * len(first) * len(second) / (len(first) + len(second))

        return float(np.sqrt(weight * float(gap @ gap)))
