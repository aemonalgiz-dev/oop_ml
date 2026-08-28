"""Every member's answer, before the ensemble has an answer of its own.

This is the array that caused the most trouble in building the ensembles, and
for a reason the annotation could not express: it is ``(n_members, n_queries)``
for a regressor and ``(n_members, n_queries, n_classes)`` for a classifier.
``FloatArray`` said neither.

Two bugs came out of that in one sitting. ``_combine`` reduced the wrong axis
because the writer was thinking of the regression shape; and the out-of-bag
estimate sliced a single query as ``predictions[missed, row]``, collapsing it
to one dimension, where ``_combine`` needed the query axis kept. The second is
the reason ``for_query`` exists: asking for a query by name gets the width-one
slice right, and there is nothing left to get wrong.

The class axis riding along behind the query axis is what lets one body serve
both tasks. Nothing here needs to know which task it is in.
"""

from __future__ import annotations

import numpy as np

from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.core.types import FloatArray, IndexArray


class MemberPredictions:
    """What every member said about every query, uncombined.

    Parameters
    ----------
    values:
        Members on axis 0 and queries on axis 1. A regressor's members answer
        with one number per query, so the array is two-dimensional; a
        classifier's answer with one number per class per query, so it is
        three-dimensional and the class axis comes last.

    Raises
    ------
    EmptyValuesError
        If there are no members or no queries.
    InvalidValuesError
        If the array is neither two- nor three-dimensional.
    """

    __slots__ = ("_values",)

    def __init__(self, values: FloatArray) -> None:
        if values.ndim not in (2, 3):
            raise InvalidValuesError(
                "Member predictions are (n_members, n_queries) or "
                f"(n_members, n_queries, n_classes), got {values.ndim} axes"
            )
        if values.shape[0] == 0:
            raise EmptyValuesError("At least one member is required")
        if values.shape[1] == 0:
            raise EmptyValuesError("At least one query is required")

        self._values = values

    @property
    def values(self) -> FloatArray:
        """The array itself, for the reduction ``_combine`` performs."""
        return self._values

    @property
    def n_members(self) -> int:
        """How many members answered."""
        return int(self._values.shape[0])

    @property
    def n_queries(self) -> int:
        """How many rows were asked about."""
        return int(self._values.shape[1])

    @property
    def is_per_class(self) -> bool:
        """Whether each member answered with a distribution rather than a value.

        The one question a caller might genuinely need to ask, answered by
        the object rather than by counting axes at the call site.
        """
        return self._values.ndim == 3

    def for_query(self, query: int, members: IndexArray) -> MemberPredictions:
        """What ``members`` said about one query, kept as a width-one batch.

        The query axis stays. Indexing a single query with a scalar drops it,
        which turns a regressor's ``(k, n_queries)`` into ``(k,)`` and makes
        ``_combine`` average across members and queries at once -- a mistake
        that produces a number of the right type and the wrong value.

        Raises
        ------
        InvalidValuesError
            If the query is out of range, or no members are supplied.
        """
        if not 0 <= query < self.n_queries:
            raise InvalidValuesError(
                f"query {query} is outside a batch of {self.n_queries}"
            )
        if members.size == 0:
            raise InvalidValuesError(f"No members were given a say about query {query}")

        return MemberPredictions(self._values[members, query : query + 1])

    def __len__(self) -> int:
        return self.n_members

    def __repr__(self) -> str:
        shape = "x".join(str(one) for one in self._values.shape)

        return f"MemberPredictions({shape})"


def predictions_of(answers: list[FloatArray]) -> MemberPredictions:
    """Stack one answer per member into the batch ``_combine`` takes."""
    return MemberPredictions(np.array(answers, dtype=np.float64))
