"""What each query saw, not just which rows it settled on.

``_neighbour_indices`` computes a distance from every query to every remembered
row and returns five numbers per query. The discarded part is most of the
lesson: that the sixth-nearest was barely further than the fifth, that under a
different metric the ranking reorders, that in two hundred dimensions the
nearest row is only 28% closer than the farthest and "the k nearest" has become
a random subset wearing a disguise.

None of that is visible from the indices. So the search is modelled: every
query keeps its full distance vector, the ranking that came out of it, and the
targets it ended up averaging or voting over.

Deliberately the expensive shape. A distance vector per query is exactly the
allocation the efficient path works to avoid -- it holds one matrix, selects
from it and drops it. This route keeps everything, which is affordable on data
sized for looking at and is not what an application should call.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.types import FloatArray, IndexArray


class NeighbourQuery:
    """One query row, and every remembered row ranked against it.

    Parameters
    ----------
    row:
        The query, in the fitted column order.
    distances:
        ``(n_remembered,)``, this query to every remembered row, in the order
        the rows were remembered.
    chosen_indices:
        ``(k,)`` positions into the remembered rows, nearest first.
    chosen_targets:
        ``(k,)`` targets of those rows, in the same order. What ``_combine``
        is handed.
    """

    __slots__ = ("_chosen_indices", "_chosen_targets", "_distances", "_row")

    def __init__(
        self,
        row: FloatArray,
        distances: FloatArray,
        chosen_indices: IndexArray,
        chosen_targets: FloatArray,
    ) -> None:
        self._row = row
        self._distances = distances
        self._chosen_indices = chosen_indices
        self._chosen_targets = chosen_targets

    @property
    def row(self) -> FloatArray:
        """The query itself."""
        return self._row

    @property
    def distances(self) -> FloatArray:
        """Distance to every remembered row, in remembered order."""
        return self._distances

    @property
    def chosen_indices(self) -> IndexArray:
        """Which rows were chosen, nearest first."""
        return self._chosen_indices

    @property
    def chosen_distances(self) -> FloatArray:
        """How far the chosen rows were, nearest first."""
        return self._distances[self._chosen_indices]

    @property
    def chosen_targets(self) -> FloatArray:
        """The targets that were combined into this query's answer."""
        return self._chosen_targets

    @property
    def nearest_distance(self) -> float:
        """Distance to the closest remembered row."""
        return float(self._distances.min())

    @property
    def farthest_distance(self) -> float:
        """Distance to the most distant remembered row.

        Worth having beside the nearest: their ratio is what collapses as
        dimensions grow, and it collapsing is why a neighbour model stops
        working long before it stops running.
        """
        return float(self._distances.max())

    @property
    def first_rejected_distance(self) -> float | None:
        """How far the row that just missed the cut was, or ``None``.

        The number that says whether ``k`` was a real choice or an arbitrary
        one. If the k+1th row is barely further than the kth, the boundary
        fell between two rows with equal claim.
        """
        if self._chosen_indices.size >= self._distances.size:
            return None

        return float(np.sort(self._distances)[self._chosen_indices.size])

    def __repr__(self) -> str:
        return (
            f"NeighbourQuery(nearest {self.nearest_distance:.4f}, "
            f"{self._chosen_indices.size} chosen)"
        )


class NeighbourSearch:
    """Every query of one prediction, each with its full ranking.

    Iterate it to walk the queries in the order they were supplied.

    Parameters
    ----------
    queries:
        One entry per query row, in order.
    """

    __slots__ = ("_queries",)

    def __init__(self, queries: Sequence[NeighbourQuery]) -> None:
        self._queries = tuple(queries)

    @property
    def result(self) -> IndexArray:
        """What the efficient path would have returned.

        ``(n_queries, k)`` of positions into the remembered rows, nearest
        first -- the same array ``_neighbour_indices`` produces.
        """
        return np.array(
            [query.chosen_indices for query in self._queries], dtype=np.intp
        ).reshape(len(self._queries), -1)

    @property
    def chosen_targets(self) -> FloatArray:
        """``(n_queries, k)`` of the targets each query combined."""
        return np.array(
            [query.chosen_targets for query in self._queries], dtype=np.float64
        ).reshape(len(self._queries), -1)

    def __iter__(self) -> Iterator[NeighbourQuery]:
        return iter(self._queries)

    def __len__(self) -> int:
        return len(self._queries)

    def __repr__(self) -> str:
        return f"NeighbourSearch({len(self._queries)} queries)"
