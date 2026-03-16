"""One question a node asks, bound to the name of the feature it asks about.

A split is four things that only mean something together -- which column, which
threshold, what that bought, and what the column is called -- so it is a class
rather than a tuple returned from a search. The caller should never have to
know that position 2 was the threshold.

The name is carried rather than looked up later because it is what makes a
fitted tree readable. ``slept < 6.25`` is a sentence about the data;
``feature 1 < 6.25`` is a sentence about an array, and by the time anyone reads
it the array is gone.
"""

from __future__ import annotations

from oop_ml.core.types import FloatArray, MaskArray


class Split:
    """A threshold on one feature, and the impurity drop it achieved.

    Parameters
    ----------
    feature_index:
        Which column of the fitted design the question is about.
    feature_name:
        What that column is called, so the tree can describe itself.
    threshold:
        Rows below this go left, rows at or above it go right. Strictly below,
        so that a threshold equal to some row's value does not send that row
        both ways.
    gain:
        How much impurity this split removed, as measured by whichever
        criterion chose it.
    """

    __slots__ = ("_feature_index", "_feature_name", "_gain", "_threshold")

    def __init__(
        self,
        feature_index: int,
        feature_name: str,
        threshold: float,
        gain: float,
    ) -> None:
        self._feature_index = feature_index
        self._feature_name = feature_name
        self._threshold = threshold
        self._gain = gain

    @property
    def feature_index(self) -> int:
        """Which column the question is about."""
        return self._feature_index

    @property
    def feature_name(self) -> str:
        """What that column is called."""
        return self._feature_name

    @property
    def threshold(self) -> float:
        """The value rows are compared against."""
        return self._threshold

    @property
    def gain(self) -> float:
        """How much impurity this split removed."""
        return self._gain

    def sends_left(self, rows: FloatArray) -> MaskArray:
        """Which of these rows this split sends to the left child.

        Parameters
        ----------
        rows:
            ``(n_rows, n_features)``, in the fitted column order.

        Returns
        -------
        MaskArray
            ``(n_rows,)``, true where the row goes left.
        """
        return rows[:, self._feature_index] < self._threshold

    def __repr__(self) -> str:
        return (
            f"Split({self._feature_name} < {self._threshold:g}, gain={self._gain:.4f})"
        )
