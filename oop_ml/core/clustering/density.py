"""What a density-based clustering does with a row once the cores are known.

Here, and shared, because it is the one part of DBSCAN that the algorithm
itself leaves undecided, so the two backends must decide it the same way or
they will disagree about rows neither of them is wrong about.

Which rows are core points, and how the core points group, are both determined
by the radius and the neighbourhood size: a core point's group is the set of
core points reachable from it through other core points, and connectivity is
not a matter of opinion. What is *not* determined is where a border point
goes. A border point is one that is close enough to some core point to be
swept into a cluster but not dense enough to hold one together, and if it is
within the radius of core points from two different clusters then both claims
are equally good. The usual implementations resolve that by whichever cluster's
expansion reached the point first, which makes the answer depend on the order
the rows arrived in.

The rule here is the nearest core point within the radius, ties going to the
lower core index. It is a convention rather than a discovery, the same way
lasso's choice between two identical columns is, and it is written down once so
that both backends and a caller's later queries all follow it.

That it is also the rule for *new* rows is the useful part. A fitted clustering
is a set of core points with labels; asking where an unseen row belongs is the
same question a border point asks, so one function answers both and a fitted
model can label data it has never seen. scikit-learn's own DBSCAN offers no
such method at all.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.distance.calculations import Distance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.types import FloatArray

NOISE_LABEL = -1.0
"""The answer for a row that belongs to no cluster.

The first label in this library that is not a group. Every other model here
answers with a group for every row, because every other model is obliged to;
a density-based clustering is the one that may decline, and a row in a sparse
region is genuinely not a member of anything rather than a member of whichever
group is least far away.

Negative on purpose, so that it can never be mistaken for a group: the groups
run from zero upward, and any code summing or indexing by label meets an
obviously wrong number rather than a plausible one.
"""


def labels_from_core_points(
    query_rows: RowBlock,
    core_points: RowBlock,
    core_labels: FloatArray,
    radius: float,
    metric: DistanceMetric | Distance,
) -> FloatArray:
    """Label each query by the nearest core point it can reach.

    Parameters
    ----------
    query_rows:
        ``(n_queries, n_features)``, the rows being asked about.
    core_points:
        ``(n_core, n_features)``, the rows the fit found dense enough to hold
        a cluster together.
    core_labels:
        ``(n_core,)``, which cluster each of those belongs to.
    radius:
        How near a core point has to be for the query to join its cluster.
    metric:
        What near means.

    Returns
    -------
    FloatArray
        One label per query, ``0.0 .. n_clusters - 1`` or
        :data:`NOISE_LABEL`.
    """
    if core_points.n_rows == 0:
        # Every row was too isolated to hold a cluster together, so there are
        # no clusters and nothing for a query to join. Answering all-noise is
        # the honest reading rather than an edge case being tidied away.
        return np.full(query_rows.n_rows, NOISE_LABEL, dtype=np.float64)

    distances = metric.between(query_rows, core_points)

    # argmin returns the *first* smallest, so an exact tie between two core
    # points goes to the lower index. That is what makes the convention above
    # a rule rather than whatever the arithmetic happened to do.
    nearest = np.argmin(distances, axis=1)
    reach = distances[np.arange(distances.shape[0]), nearest]

    return np.where(reach <= radius, core_labels[nearest], NOISE_LABEL)
