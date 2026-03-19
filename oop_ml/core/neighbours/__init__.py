"""What a neighbour query looked like, beyond the rows it settled on.

:mod:`~oop_ml.core.neighbours.search` records the full ranking behind each
prediction -- every distance, not only the k that survived it.

Separate from ``core.distance``, which answers how far apart two rows are and
knows nothing about models, and from ``core.base``, because none of this is
inherited. A search is a fact about one prediction.
"""
