"""The frame for models that answer by looking at whoever is nearby.

The first thing in this library that is not linear in anything, and it needed
its own frame because it does not fit the one the others share. There is no
design matrix to build, no intercept to split off, and no coefficients to pair
with names -- there are no coefficients at all. ``LinearModel`` describes how a
fitted answer is assembled from weights, and a neighbour model has no weights
to assemble.

What replaces fitting
---------------------
Nothing. ``fit`` validates its inputs and remembers them, which is why these
are called *non-parametric*: there is no fixed set of parameters standing in
for the data, so the data itself is the model and every decision is deferred to
``predict``. That inverts the usual cost -- fitting is instant and predicting
is the expensive half -- and it means the memory a fitted model occupies grows
with the training set rather than staying constant.

``k`` is the whole of the bias-variance dial
--------------------------------------------
There is only one hyperparameter that matters and it does the work a penalty
does elsewhere. Measured on two hundred points over two predictors:

======  ===============  ==============
``k``   train accuracy   test accuracy
======  ===============  ==============
1               1.0000          0.7370
3               0.8450          0.7830
5               0.8250          0.7825
15              0.8150          0.8010
51              0.8100          0.8085
199             0.5250          0.4945
======  ===============  ==============

At ``k = 1`` every training row is its own nearest neighbour, so training
accuracy is exactly 1.0 and means nothing: the model has memorised rather than
learned, and the test column is where that shows. As ``k`` grows the answer
smooths, the two columns converge, and at ``k = n`` every query gets the same
answer -- the global majority or the global mean -- which is the most biased
model there is.

The two tasks differ in one line
--------------------------------
Finding the neighbours is identical whether the target is a label or a
quantity. What differs is what you do with the ones you found: vote for
classification, average for regression. That single line is
:meth:`NeighbourModel._combine`, and it is the only thing a concrete neighbour
model has to supply.
"""

from __future__ import annotations

import os
from abc import abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.data.column import Column
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.distance.calculations import Distance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.exceptions import TooFewValuesError
from oop_ml.core.neighbours.search import NeighbourQuery, NeighbourSearch
from oop_ml.core.types import FloatArray, IndexArray

# Below this many (query, remembered) pairs, splitting the work across threads
# costs more than it saves -- a pool takes a millisecond or two to start, and
# the crossover measured between 300000 and 600000 pairs.
PARALLEL_PAIR_THRESHOLD = 500_000

# Eight rather than every core: BLAS is already threading the matrix multiply
# underneath, so the two layers compete for the same cores. Measured on
# twenty-four, the curve is flat from six workers upward and turns back down at
# twenty-four, so a cap well below the core count is the robust choice.
MAX_PARALLEL_WORKERS = 8


class NeighbourModel(Fittable):
    """A model that remembers its training rows and consults the nearest ones.

    Parameters
    ----------
    n_neighbours:
        How many neighbours vote on each answer. The bias-variance dial: 1
        memorises, and a value approaching the training size returns the same
        answer for every query.
    metric:
        What "near" means. Usually one of the six named in
        :class:`~oop_ml.core.distance.metric.DistanceMetric`; any
        :class:`~oop_ml.core.distance.calculations.Distance` is accepted, so a
        p-norm the enum does not name is ``MinkowskiDistance(3)`` and a metric
        the library has never heard of is a subclass. Standardise the features
        before trusting any of them, ``CANBERRA`` aside.
    """

    n_neighbours: int = Field(default=5, gt=0)
    metric: DistanceMetric | Distance = DistanceMetric.EUCLIDEAN

    _feature_names: tuple[str, ...] | None = PrivateAttr(default=None)
    _remembered_rows: RowBlock | None = PrivateAttr(default=None)
    _remembered_targets: FloatArray | None = PrivateAttr(default=None)

    @property
    def n_remembered(self) -> int:
        """How many training rows the model is carrying.

        Worth exposing because it is the model's size in a way no other
        estimator here has one: a linear model occupies the same memory
        whatever it was fitted on, and this does not.

        Raises
        ------
        NotFittedError
            If accessed before ``fit``.
        """
        self._check_fitted()
        assert self._remembered_rows is not None
        return self._remembered_rows.n_rows

    @abstractmethod
    def _combine(self, neighbour_targets: FloatArray) -> FloatArray:
        """Turn each query's neighbours' targets into one answer.

        The single line that separates a neighbour regressor from a neighbour
        classifier, and the only thing a subclass supplies.

        Parameters
        ----------
        neighbour_targets:
            ``(n_queries, n_neighbours)``. Row ``i`` holds the target values of
            the ``k`` rows nearest to query ``i``, nearest first.

        Returns
        -------
        FloatArray
            One value per query.
        """

    def _nearest_within(self, query_rows: RowBlock) -> IndexArray:
        """The nearest remembered rows to one block of queries.

        Notes
        -----
        A full sort of every distance is more work than the question needs.
        Only the ``k`` smallest matter and their order among themselves matters
        only for tie-breaking, so partitioning at ``k`` and sorting just that
        slice is the cheaper route -- O(n) against O(n log n) per query, which
        is the difference between usable and not once the training set is
        large.
        """
        assert self._remembered_rows is not None

        # 1. every query against every remembered row
        distances = self.metric.between(query_rows, self._remembered_rows)

        # 2. split around a pivot until position k-1 holds its final value.
        #    Everything left of it is then one of the k smallest -- as a set,
        #    in no particular order, which is all the partition promises.
        partitioned = np.argpartition(distances, self.n_neighbours - 1, axis=1)

        # 3. keep that clump
        nearest = partitioned[:, : self.n_neighbours]

        # 4. sort the clump so they come back nearest-first. numpy happens to
        #    leave them ordered already, but its contract says the order is
        #    undefined, and the classifier's tie-break depends on this being
        #    true rather than usually true.
        #
        #    take_along_axis rather than plain indexing: each row needs its own
        #    columns, and distances[nearest] would pick rows instead.
        ordering = np.argsort(np.take_along_axis(distances, nearest, axis=1), axis=1)

        return np.take_along_axis(nearest, ordering, axis=1)

    def _neighbour_indices(self, query_rows: RowBlock) -> IndexArray:
        """Which remembered rows are nearest to each query row.

        Parameters
        ----------
        query_rows:
            ``(n_queries, n_features)``, already matched to the fitted feature
            order.

        Returns
        -------
        IndexArray
            ``(n_queries, n_neighbours)`` of positions into the remembered
            rows, nearest first.

        Notes
        -----
        Queries are independent of one another, so this splits them into blocks
        and runs the blocks on threads. That is worth doing because of *where*
        the time goes. Measured at 20000 remembered rows by 500 queries over 20
        features, the work divides almost evenly -- 0.064s building the
        distance matrix against 0.059s selecting from it -- and only the first
        half was already parallel, because BLAS threads the matrix multiply on
        its own while ``argpartition`` runs on one core. Roughly half the
        calculation was therefore using one core of twenty-four.

        Threads rather than processes because numpy releases the interpreter
        lock for exactly these operations, so this is real parallelism with no
        pickling and no copies -- each worker reads the same remembered rows.
        Measured 2.5x to 3.4x quicker end to end.

        Below :data:`PARALLEL_PAIR_THRESHOLD` it stays on one thread. Starting
        a pool costs a millisecond or two, which is free against a hundred and
        a disaster against one: at 500 queries by 500 remembered rows the
        threaded route measured *nine times slower*. The threshold counts pairs
        rather than either dimension alone, since the work is the product.

        The result is identical either way, not merely equivalent. Blocks are
        written back into the positions they came from, and no query's answer
        depends on any other query, so the arithmetic each row receives is
        unchanged by how the rows were grouped.
        """
        assert self._remembered_rows is not None

        n_queries = query_rows.n_rows
        pairs = n_queries * self._remembered_rows.n_rows
        workers = min(MAX_PARALLEL_WORKERS, os.cpu_count() or 1)

        if pairs < PARALLEL_PAIR_THRESHOLD or workers < 2 or n_queries < 2:
            return self._nearest_within(query_rows)

        block_size = max(1, -(-n_queries // workers))
        starts = range(0, n_queries, block_size)

        def nearest_for_block(start: int) -> tuple[int, IndexArray]:
            return start, self._nearest_within(
                query_rows.rows_between(start, start + block_size)
            )

        indices = np.empty((n_queries, self.n_neighbours), dtype=np.intp)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start, block in pool.map(nearest_for_block, starts):
                indices[start : start + block.shape[0]] = block

        return indices

    def neighbour_search(self, input_values: Sequence[Feature]) -> NeighbourSearch:
        """Every distance behind each query, rather than only the winners.

        The observed route beside :meth:`_neighbour_indices`. Same metric,
        same selection, same tie-breaking -- it keeps each query's whole
        distance vector instead of selecting from it and dropping it.

        Deliberately the expensive shape: one distance vector per query is
        precisely the allocation the efficient route avoids. Sized for
        looking at, not for twenty thousand remembered rows.

        Runs on one thread whatever the size, because the block loop exists to
        divide work rather than to be watched, and dividing it here would
        interleave the record.

        Returns
        -------
        NeighbourSearch
            Iterable over the queries. ``search.result`` is the same array
            :meth:`_neighbour_indices` returns, and a test asserts that.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        query_rows = self._matched_rows(input_values)

        assert self._remembered_rows is not None
        assert self._remembered_targets is not None

        distances = self.metric.between(query_rows, self._remembered_rows)
        chosen = self._nearest_within(query_rows)

        return NeighbourSearch(
            [
                NeighbourQuery(
                    query_rows.row(position),
                    distances[position],
                    chosen[position],
                    self._remembered_targets[chosen[position]],
                )
                for position in range(query_rows.n_rows)
            ]
        )

    def _remember(
        self, input_values: Sequence[Feature], target_values: Feature
    ) -> Self:
        """Validate the inputs and keep them. This is the whole of fitting.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonUniqueFeaturesError
            If two features share a name.
        NonEqualArrayLengthError
            If any feature's length differs from the target's.
        TooFewValuesError
            If there are fewer rows than neighbours asked for. Averaging over
            more neighbours than exist is not a degraded answer, it is a
            different question.
        """
        feature_set = FeatureSet(input_values)
        feature_set.check_aligned_with(target_values)

        if feature_set.n_samples < self.n_neighbours:
            raise TooFewValuesError(
                f"{self.n_neighbours} neighbours were asked for and only "
                f"{feature_set.n_samples} rows were supplied"
            )

        self._feature_names = tuple(feature.name for feature in feature_set)
        self._remembered_rows = rows_of(
            feature_set.feature_matrix, [one.name for one in feature_set]
        )
        self._remembered_targets = target_values.column.values

        self._mark_fitted()
        return self

    def _matched_rows(self, input_values: Sequence[Feature]) -> RowBlock:
        """The query rows, in the column order the fit saw.

        Matched by name rather than position, the same contract every other
        model here follows.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonEqualArrayLengthError
            If the supplied features disagree in length.
        InvalidValuesError
            If the supplied feature names do not match those seen in ``fit``.
        """
        self._check_fitted()
        assert self._feature_names is not None

        matched = FeatureSet.matching(self._feature_names, input_values)

        return rows_of(matched.feature_matrix, self._feature_names)

    def _neighbour_targets(self, input_values: Sequence[Feature]) -> FloatArray:
        """The target values of each query's nearest neighbours.

        Everything ``predict`` needs on either task, so that a concrete model
        is only ever :meth:`_combine` away from an answer.

        Returns
        -------
        FloatArray
            ``(n_queries, n_neighbours)``, nearest first.
        """
        # _matched_rows carries the fitted-state guard, so it has to run before
        # any assert about what the fit stored -- otherwise an unfitted call
        # dies on an AssertionError rather than saying NotFittedError.
        query_rows = self._matched_rows(input_values)

        assert self._remembered_targets is not None

        return self._remembered_targets[self._neighbour_indices(query_rows)]

    def _validated_target(self, target_values: Feature) -> Column:
        """The target as a column, checked against whatever the task requires.

        The seam between the two neighbour models, mirroring the one
        :class:`~oop_ml.core.base.linear_model.LinearModel` uses. The default
        is the regression answer: a column is already numeric and finite, and
        averaging asks nothing further of it.
        """
        return target_values.column
