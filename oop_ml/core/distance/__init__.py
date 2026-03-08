"""What "near" means, split from what is done with the rows that are near.

:mod:`~oop_ml.core.distance.metric` is the vocabulary a user chooses from --
a closed enum of six, so a misspelling cannot reach runtime.
:mod:`~oop_ml.core.distance.calculations` is how each of them is actually
computed, one class per formula.

Two modules rather than one because the split is real. Four of the six metrics
are computed by pairing every query with every remembered row and reducing over
the features; the other two collapse to a matrix multiply that never builds
that pairing at all. Keeping the calculations separate from the enum lets each
take the route that suits it, and lets ``MinkowskiDistance`` accept any ``p``
while the enum names only the three worth having a spelling for.

This sits in ``core`` rather than beside the neighbour models because distance
is not a neighbour-model idea. Clustering, anomaly detection and any spatial
index would want the same six, and none of them are neighbour models.
"""
