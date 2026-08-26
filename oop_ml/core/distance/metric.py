"""Which notion of "near" to use, which for a neighbour model is the model.

Every other estimator in the library learns something: coefficients, scalings,
a set of terms. A neighbour model learns nothing at all. ``fit`` remembers the
rows and every decision moves to ``predict``, so the only modelling choice left
is what "near" means -- and that makes the metric the counterpart of the
objective function elsewhere, not a configuration detail.

A closed enum rather than a string, for the reason the rest of the library uses
enums: ``metric="euclidian"`` is a runtime surprise several layers down, and
``DistanceMetric.EUCLIDIAN`` does not exist at all.

Two consequences worth knowing before choosing
----------------------------------------------
**Scaling stops being optional.** Distance sums over the features, so a column
measured in thousands drowns one measured in units. On house data with floor
area in square metres beside a bathroom count, the area contributed 99.917% of
the squared distance -- the metric was reading one column and ignoring the
other, and the nearest row by raw distance was a different house than the
nearest by standardised distance. Everywhere else in this library standardising
is a convenience; here it is part of being correct. ``CANBERRA`` is the one
exception, and it buys that by being most sensitive near zero instead.

**Distance stops discriminating as dimensions grow.** Sample a thousand uniform
points and ask how much further the farthest is than the nearest:

===========  ================================
dimensions   (farthest - nearest) / nearest
===========  ================================
1            4390.95
2              42.52
5               7.11
10              2.61
50              0.68
200             0.28
===========  ================================

At two hundred features the nearest point is only 28% closer than the farthest,
so "the k nearest" is very nearly a random subset wearing a disguise. That is
the curse of dimensionality, and it is a property of distance itself rather
than of any implementation of it. ``COSINE`` suffers least, which is part of
why text data is usually handled with angles.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from oop_ml.core.distance.calculations import (
    CanberraDistance,
    CosineDistance,
    Distance,
    EuclideanDistance,
    HammingDistance,
    MinkowskiDistance,
)
from oop_ml.core.types import FloatArray


class DistanceMetric(StrEnum):
    """Which notion of "near" a neighbour model should use.

    Six of them, and the fastest way to see what separates them is to fix one
    row at ``(1, 1, 1)`` and vary the other:

    =============  =========  =========  =========  ======  =======  ========
    other row      euclidean  manhattan  chebyshev  cosine  hamming  canberra
    =============  =========  =========  =========  ======  =======  ========
    ``(2, 1, 1)``     1.0000     1.0000     1.0000  0.0572   0.3333    0.3333
    ``(2, 2, 2)``     1.7321     3.0000     1.0000  0.0000   1.0000    1.0000
    ``(4, 1, 1)``     3.0000     3.0000     3.0000  0.1835   0.3333    0.6000
    ``(3, 3, 3)``     3.4641     6.0000     2.0000  0.0000   1.0000    1.5000
    =============  =========  =========  =========  ======  =======  ========

    Read row two against row four. ``(2, 2, 2)`` and ``(3, 3, 3)`` both point
    exactly where ``(1, 1, 1)`` points, so ``COSINE`` calls them identical and
    returns 0 for both, while ``HAMMING`` sees three features out of three
    disagreeing and returns 1 -- the same pairs of rows, as close as possible
    and as far as possible, depending only on which question is being asked.
    Neither is wrong. They are different questions.

    Read row one against row three to watch the p-norms separate. Tripling a
    single feature's gap triples Manhattan and Chebyshev, but Euclidean moves
    by less than three because squaring already charged that gap for sitting in
    one place, and Canberra moves less still because the division holds every
    feature to the same ceiling however far apart the values are.

    The p-norm family
    -----------------
    ``EUCLIDEAN``, ``MANHATTAN`` and ``CHEBYSHEV`` are not separate formulas.
    All three are p-norms of the gap vector::

        ||gap||_p  =  ( sum |gap_i| ** p ) ** (1 / p)

    and only ``p`` changes. Raising it shifts weight onto whichever single
    feature disagrees most. On a gap of ``(4, 2, -6)``:

    =====  ==========  ===========================================
    ``p``  ``||.||``
    =====  ==========  ===========================================
    1        12.0000   Manhattan: every gap counts its own size
    1.5       8.6692
    2         7.4833   Euclidean: straight-line distance
    4         6.2927
    10        6.0103   converging on 6, the largest single gap
    inf       6.0000   Chebyshev: only the largest gap counts
    =====  ==========  ===========================================

    Attributes
    ----------
    EUCLIDEAN:
        ``p = 2``. Straight-line distance, and what most people mean by it.
        The default, and the right one whenever the features are comparable
        quantities and have been standardised.
    MANHATTAN:
        ``p = 1``. Named for walking a street grid, where the corner cannot be
        cut. A large gap in one feature costs its own size rather than its
        square, which makes it the steadier choice on data with outliers or
        many dimensions.
    CHEBYSHEV:
        ``p`` at infinity. Only the single worst feature counts, and the rest
        are ignored entirely. Use it where a row is unacceptable if it is far
        off on *any* feature -- tolerances, bounding boxes, grids where a king
        moves one square in any direction.
    COSINE:
        The angle between two rows, ignoring how long they are. The usual
        choice for text and other count data, where a long document should
        resemble a short one on the same subject. Not a true metric: it
        violates the triangle inequality.
    HAMMING:
        The share of features on which the two rows disagree. The only one
        that treats values as labels rather than quantities, and therefore the
        only one that makes sense on categorical codes.
    CANBERRA:
        Manhattan with each gap divided by the size of the values involved, so
        every feature contributes between 0 and 1 whatever its units. The one
        option that tolerates unstandardised input.
    """

    # Annotated but never assigned, so the enum does not read it as a member.
    _calculation: Distance

    EUCLIDEAN = ("euclidean", EuclideanDistance())
    MANHATTAN = ("manhattan", MinkowskiDistance(1))
    CHEBYSHEV = ("chebyshev", MinkowskiDistance(np.inf))
    COSINE = ("cosine", CosineDistance())
    HAMMING = ("hamming", HammingDistance())
    CANBERRA = ("canberra", CanberraDistance())

    def __new__(cls, value: str, calculation: Distance) -> DistanceMetric:
        """Bind each member to the object that knows how to compute it."""
        member = str.__new__(cls, value)
        member._value_ = value
        member._calculation = calculation

        return member

    @property
    def calculation(self) -> Distance:
        """The object that computes this metric.

        Exposed because the calculation is a real part of the design rather
        than an implementation detail -- ``MinkowskiDistance`` accepts any
        ``p``, and this enum names only the three worth having a spelling for.
        """
        return self._calculation

    def between(
        self, query_rows: FloatArray, remembered_rows: FloatArray
    ) -> FloatArray:
        """Distance from every query row to every remembered row.

        Parameters
        ----------
        query_rows:
            ``(n_queries, n_features)``, the rows being asked about.
        remembered_rows:
            ``(n_remembered, n_features)``, the rows ``fit`` kept.

        Returns
        -------
        FloatArray
            ``(n_queries, n_remembered)``. Entry ``(i, j)`` is the distance
            from query ``i`` to remembered row ``j``.

        Notes
        -----
        Delegates to :attr:`calculation`. There is deliberately no branch here:
        which formula runs is settled when the member is defined, so a metric
        cannot be added without also being given a way to compute it.
        """
        return self._calculation.between(query_rows, remembered_rows)
