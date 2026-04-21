"""The rows a margin model kept, and how much weight each one carries.

A support vector machine's fitted self is not a weight per feature, because
the space its kernel implies has no named coordinates. It is a subset of the
training rows with a multiplier each, and that is what both backends hand
back, so the type carrying it belongs here rather than under either of them.

The threshold is part of the vocabulary rather than a detail of one solver.
Every practical method leaves tiny non-zero multipliers on rows that are not
really support vectors, so "kept" has to mean "above some floor" and the floor
has to be the same number on both sides or the two backends would disagree
about how many vectors a fit has while agreeing about every prediction.
"""

from __future__ import annotations

from collections.abc import (
    Iterator,
    Sequence,
)

from oop_ml.core.exceptions import InvalidValuesError

SUPPORT_VECTOR_THRESHOLD = 1e-08
"""Above this multiplier, a training row counts as a support vector.

Exactly zero is what the mathematics says and not what a floating-point solver
returns; a point comfortably outside the margin lands at 1e-15 rather than 0.
Counting those as support vectors would report every training row as one and
lose the property that makes the model small.
"""


class SupportVector:
    """One training row that touches the margin, and how hard it pulls.

    Parameters
    ----------
    position:
        Which training row this was, so a caller can go back to it.
    multiplier:
        The dual variable. Strictly positive by construction -- a row with a
        zero multiplier is not a support vector, it is a row the boundary does
        not depend on.
    label:
        The row's class as ``-1`` or ``+1``, which is the encoding the dual is
        written in.

    Raises
    ------
    InvalidValuesError
        If the multiplier is not positive, or the label is not -1 or +1.
    """

    __slots__ = ("_label", "_multiplier", "_position")

    def __init__(self, position: int, multiplier: float, label: float) -> None:
        if multiplier <= 0.0:
            raise InvalidValuesError(
                f"a support vector has a positive multiplier; got {multiplier}. "
                f"A row with a zero multiplier is not one."
            )

        if label not in (-1.0, 1.0):
            raise InvalidValuesError(
                f"the dual is written in -1/+1 labels; got {label}"
            )

        self._position = int(position)
        self._multiplier = float(multiplier)
        self._label = float(label)

    @property
    def position(self) -> int:
        """Which training row this was."""
        return self._position

    @property
    def multiplier(self) -> float:
        """How hard this row pulls on the boundary."""
        return self._multiplier

    @property
    def label(self) -> float:
        """This row's class, as -1 or +1."""
        return self._label

    def is_at_the_cap(self, capacity: float) -> bool:
        """Whether this row's multiplier hit ``C``.

        A row at the cap is one the soft margin gave up on -- it sits inside
        the corridor or on the wrong side of the boundary, and the only thing
        stopping it pulling harder is the cap itself. Counting them is the
        quickest read on whether ``C`` is too small for the data.
        """
        return self._multiplier >= capacity * (1.0 - 1e-09)

    def __repr__(self) -> str:
        return (
            f"SupportVector(row={self._position}, "
            f"multiplier={self._multiplier:.4f}, label={self._label:+.0f})"
        )


class SupportVectors:
    """The rows the boundary actually depends on.

    Deleting every training row that is not in here and refitting gives the
    same boundary, which is the sense in which these *are* the model.

    Raises
    ------
    InvalidValuesError
        If two entries name the same training row.
    """

    __slots__ = ("_n_training_rows", "_vectors")

    def __init__(self, vectors: Sequence[SupportVector], n_training_rows: int) -> None:
        positions = [vector.position for vector in vectors]

        if len(set(positions)) != len(positions):
            raise InvalidValuesError(
                f"a training row can appear once; got positions {positions}"
            )

        self._vectors = tuple(vectors)
        self._n_training_rows = int(n_training_rows)

    @property
    def n_vectors(self) -> int:
        """How many rows the boundary depends on."""
        return len(self._vectors)

    @property
    def n_training_rows(self) -> int:
        """How many rows the fit saw."""
        return self._n_training_rows

    @property
    def share_of_training_rows(self) -> float:
        """What fraction of the training set survived into the model.

        The number that separates this from kernel ridge, which keeps all of
        them. A share near 1 means the fit found no structure and every point
        is touching the margin, which usually means ``C`` is far too large or
        the kernel far too flexible.
        """
        if self._n_training_rows == 0:
            return 0.0

        return self.n_vectors / self._n_training_rows

    def positions(self) -> list[int]:
        """Which training rows these were."""
        return [vector.position for vector in self._vectors]

    def n_at_the_cap(self, capacity: float) -> int:
        """How many multipliers hit ``C``, so how many the margin gave up on."""
        return sum(1 for vector in self._vectors if vector.is_at_the_cap(capacity))

    def __iter__(self) -> Iterator[SupportVector]:
        return iter(self._vectors)

    def __len__(self) -> int:
        return self.n_vectors

    def __repr__(self) -> str:
        return (
            f"SupportVectors({self.n_vectors} of {self._n_training_rows}, "
            f"{self.share_of_training_rows:.1%})"
        )
