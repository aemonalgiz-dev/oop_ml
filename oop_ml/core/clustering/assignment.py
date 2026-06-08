"""Labelling a row a clustering never saw, when the clustering has no shape.

k-means can answer this from its centres alone, and does. A hierarchical
clustering has no centres and no parametric form of any kind: what it produces
is a labelling of the rows it was given, and nothing about that labelling
extends to a new row by itself.

The rule here is the nearest labelled row, which is the one answer that needs
no extra assumption. It says a row belongs wherever its closest known
neighbour belongs, and that is true of *every* linkage rather than only the
centre-like ones -- a nearest-centre rule would quietly impose complete
linkage's roundness on a single-linkage fit that had deliberately found a long
thin group.

Two consequences worth knowing. On the training rows it reproduces the fit's
own labels exactly, since a row is at distance zero from itself, which is what
lets ``fit_predict`` mean what it says. And it never answers "none": unlike
:mod:`~oop_ml.core.clustering.density`, a hierarchical clustering places every
row by construction, so there is no noise for a query to fall into either.

Shared because both backends need it and neither may import the other, and
because scikit-learn's ``AgglomerativeClustering`` has no ``predict`` at all --
only ``fit_predict`` -- so this is not a translation of anything the engine
does. It is the rule this library adds so that a clusterer keeps the frame's
promise of working on new data.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.data.row_block import RowBlock
from oop_ml.core.distance.calculations import Distance
from oop_ml.core.distance.metric import DistanceMetric
from oop_ml.core.types import FloatArray


def labels_from_nearest(
    query_rows: RowBlock,
    remembered_rows: RowBlock,
    remembered_labels: FloatArray,
    metric: DistanceMetric | Distance,
) -> FloatArray:
    """Give each query the label of the nearest remembered row.

    Parameters
    ----------
    query_rows:
        ``(n_queries, n_features)``, the rows being asked about.
    remembered_rows:
        ``(n_remembered, n_features)``, the rows the fit was given.
    remembered_labels:
        ``(n_remembered,)``, the group each of those was put in.
    metric:
        What near means. The same one the fit used, or the answer describes a
        different clustering from the one that was made.

    Returns
    -------
    FloatArray
        One label per query.
    """
    distances = metric.between(query_rows, remembered_rows)

    # argmin takes the first smallest, so a query equidistant from two
    # remembered rows goes to the lower index. Ties matter less here than in a
    # density clustering, since a query sitting exactly between two groups is
    # not a case the method leaves undetermined -- it is simply on a boundary.
    return remembered_labels[np.argmin(distances, axis=1)]
