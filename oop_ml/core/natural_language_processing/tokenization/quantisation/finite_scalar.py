"""Finite scalar quantisation: a vector to an id, with no codebook to learn.

The same question, a different answer to where the codebook comes from
----------------------------------------------------------------------
:mod:`~oop_ml.core.natural_language_processing.tokenization.quantisation.codebook`
turns a vector into the id of its nearest code, and the codes have to come from
somewhere: k-means, or a VQ-VAE moving them with a gradient. The second is
notoriously hard to train, for three reasons that are all about the codebook. A
code moves only when an input is assigned to it, so a code that starts far
from the data is never chosen and never moves, and the book *collapses* onto
the few codes that are. The encoder must be held near the codes it is snapped
to with a commitment loss, or it drifts and the assignments churn. And the
snapping has no gradient, so one is faked by passing the decoder's gradient
straight through to the encoder.

Mentzer, Minnen, Agustsson and Tschannen (2023) noticed that a codebook need
not be learned at all. Fix it: bound each coordinate of the encoder's output to
``[-1, 1]`` with ``tanh`` and round it to one of ``L`` evenly spaced points.
With ``levels = (L_0, ..., L_{d-1})`` the implicit codebook is the grid of all
``L_0 * ... * L_{d-1}`` combinations, ``1000`` codes for the paper's
``(8, 5, 5, 5)``, and nothing about it is learned, so there is no collapse to
guard against and no commitment loss to tune. The one remaining trick, the
straight-through gradient, is the encoder's business and not this class's.

It answers the question :class:`~.codebook.CodebookQuantizer` answers, a
vector to an id and back, and differs in exactly one respect: the codebook is
a fixed grid rather than a learned table. That is why it lives beside the
codebook lookup, and why its answer is the same
:class:`~.codebook.CodeAssignment`.

The bounding, and the even-level offset
---------------------------------------
For a dimension with ``L`` levels the half-width is ``h = (L - 1) / 2`` and
the bounded value is ``tanh(z) * h``, in ``[-h, h]``. For odd ``L`` that
interval rounded to the nearest integer gives exactly ``L`` positions,
``-h .. h``: five levels are ``-2, -1, 0, 1, 2``. For even ``L`` it does not.
Eight levels give ``h = 3.5``, and ``tanh(z) * 3.5`` rounds to ``-3 .. 3`` in
the interior, seven positions, and to ``+-4`` at saturation, nine in all. The
paper's remedy is an offset of one half for even ``L``:
``round(tanh(z) * h - 0.5)`` lies in ``-L/2 .. L/2 - 1``, which is ``L``
positions, ``-4 .. 3`` for eight. The offset is added back when a position is
mapped to its coordinate, ``(position + offset) / h``, so the grid is symmetric
in ``[-1, 1]`` either way: eight levels sit at ``-1, -5/7, -3/7, -1/7, 1/7,
3/7, 5/7, 1``, and ``L`` levels at ``-1 + 2 j / (L - 1)`` for ``j`` from ``0``
to ``L - 1``. The spec checks this directly: for every ``L`` from 2 to 9 a
sweep over inputs produces exactly ``L`` distinct ids.

The id, in mixed radix
----------------------
A position ``p`` becomes the digit ``j = p + L // 2`` in ``0 .. L - 1``, and
the digits are one number in mixed radix with **dimension 0 least
significant**::

    id = j_0 + L_0 * (j_1 + L_1 * (j_2 + ...))

Worked, at the paper's ``levels = (8, 5, 5, 5)`` on the input
``z = (1.0, -0.5, 0.2, 3.0)``::

    d  L  h    tanh(z)     tanh(z) h - offset   position  digit  coordinate
    0  8  3.5   0.761594    2.165580              2         6      5/7
    1  5  2    -0.462117   -0.924234             -1         1     -1/2
    2  5  2     0.197375    0.394751              0         2      0
    3  5  2     0.995055    1.990110              2         4      1

    id = 6 + 8 (1 + 5 (2 + 5 * 4)) = 6 + 8 * 111 = 894

The reconstruction is ``(5/7, -1/2, 0, 1)``. Saturation is the last level:
``z = (1000, 1000, 1000, 1000)`` has ``tanh(z) = 1.0`` exactly in float64 and
lands on digits ``(7, 4, 4, 4)``, id ``999``, the last of the thousand; its
negation lands on id ``0``.

The rounding tie
----------------
Rounding is numpy's ``rint``, half to even. A tie needs ``tanh(z) h - offset``
to land exactly on a half-integer, which random inputs never do and the zero
vector under an even ``L`` always does: ``tanh(0) * 3.5 - 0.5 = -0.5`` rounds
to position ``0`` rather than ``-1``, so the zero vector at ``(8, 5, 5, 5)`` is
digits ``(4, 2, 2, 2)``, id ``500``, with a first coordinate of ``1/7`` rather
than ``-1/7``. Stated once, like every tie rule here. Under odd levels only,
zero is a grid point and the zero vector reconstructs exactly, with distortion
``0.0``.

Distortion is measured against the bounded input
------------------------------------------------
:class:`~.codebook.CodeAssignment` reports the mean squared distance from an
input to what it became. Here the input is unbounded and the grid lives in
``[-1, 1]``, so the distance is taken from ``tanh(z)``, the bounded input, to
the grid point in the same coordinates. Against the raw input it would be
dominated by how far outside the box ``z`` sat and say nothing about the grid;
against the bounded input, a saturating input lands on a corner exactly and
reports ``0.0``. On the worked example it is ``0.042654671`` to nine places.

Two routes to one id
--------------------
Rounding each coordinate *is* the nearest point of the grid, because the
squared Euclidean distance to a grid point is a sum over coordinates of terms
that are each smallest at that coordinate's nearest level. So :attr:`codebook`
materialises the grid in id order as a real :class:`~.codebook.Codebook`, and a
:class:`~.codebook.CodebookQuantizer` over it, handed ``tanh(z)``, assigns the
same ids and the same reconstruction, ties aside; the spec says so on random
inputs. Materialising costs ``O(n_codes * dimension)`` and is for inspection.
The quantizer itself never needs the table, which is the point of the method.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.exceptions import EmptyValuesError
from oop_ml.core.natural_language_processing.tokenization.quantisation.codebook import (
    CodeAssignment,
    Codebook,
    checked_code_id,
    checked_vectors,
)
from oop_ml.core.types import FloatArray, IndexArray


class FiniteScalarQuantizer(BaseModel):
    """Round each bounded coordinate to one of a few levels; the id is the digits.

    Configured, never fitted, since the grid is fixed by ``levels`` and there is
    nothing to learn about it.

    Parameters
    ----------
    levels:
        How many levels each coordinate is rounded to, one entry per input
        dimension. At least one dimension, every level at least 2, since a
        dimension with one level carries no information.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    levels: tuple[Annotated[int, Field(ge=2)], ...] = Field(min_length=1)

    @property
    def dimension(self) -> int:
        """How many coordinates every input must have."""
        return len(self.levels)

    @property
    def n_codes(self) -> int:
        """The size of the implicit codebook, the product of the levels."""
        return math.prod(self.levels)

    @property
    def codebook(self) -> Codebook:
        """Every grid point in id order, as the codebook it implicitly is.

        ``O(n_codes * dimension)`` to build. For inspection and for checking
        that a nearest-code lookup over it agrees with :meth:`quantize`; the
        quantizer never needs it.
        """
        digits = np.array(
            [self.digits_of(code_id) for code_id in range(self.n_codes)],
            dtype=np.intp,
        )
        return Codebook(self._coordinates_of_digits(digits))

    def quantize(self, vectors: FloatArray) -> CodeAssignment:
        """Bound each row with ``tanh``, round each coordinate, compose the id.

        The reconstruction is the grid point in ``[-1, 1]`` coordinates, and the
        distortion is measured against the bounded input ``tanh(z)``, not the
        raw one. See the module docstring for both choices.

        Raises
        ------
        InvalidValuesError
            If ``vectors`` is not numeric, not two-dimensional, or not finite.
        EmptyValuesError
            If there are no rows.
        ShapeMismatchError
            If the rows are not ``dimension`` wide.
        """
        inputs = checked_vectors(vectors, self.dimension)
        bounded = np.tanh(inputs)
        digits = self._digits_of_bounded(bounded)
        coordinates = self._coordinates_of_digits(digits)

        gaps = bounded - coordinates
        squared_gaps = np.sum(gaps * gaps, axis=1)

        return CodeAssignment(
            self._ids_of_digits(digits), coordinates, float(np.mean(squared_gaps))
        )

    def dequantize(self, code_ids: Sequence[int]) -> FloatArray:
        """The grid points the ids name, ``(n_ids, dimension)``, a fresh array.

        Raises
        ------
        EmptyValuesError
            If there are no ids.
        InvalidValuesError
            If an id is not a whole number.
        UnknownTokenError
            If an id is outside ``0 .. n_codes - 1``.
        """
        if len(code_ids) == 0:
            raise EmptyValuesError("at least one code id is required")

        digits = np.array(
            [self.digits_of(code_id) for code_id in code_ids], dtype=np.intp
        )
        return self._coordinates_of_digits(digits)

    def digits_of(self, code_id: int) -> tuple[int, ...]:
        """The mixed-radix digits of one id, dimension 0 first.

        The inverse of composing the id: repeated ``divmod`` by each level in
        turn, so digit ``d`` lies in ``0 .. levels[d] - 1``.

        Raises
        ------
        InvalidValuesError
            If ``code_id`` is not a whole number.
        UnknownTokenError
            If ``code_id`` is outside ``0 .. n_codes - 1``.
        """
        remainder = checked_code_id(code_id, self.n_codes)

        digits: list[int] = []
        for level in self.levels:
            remainder, digit = divmod(remainder, level)
            digits.append(digit)

        return tuple(digits)

    def _levels_array(self) -> IndexArray:
        return np.array(self.levels, dtype=np.intp)

    def _half_widths(self) -> FloatArray:
        """``(L - 1) / 2`` per dimension: the scale that puts levels on integers."""
        return (self._levels_array() - 1) / 2.0

    def _offsets(self) -> FloatArray:
        """One half for an even level count, zero for odd. See the module docstring."""
        return np.where(self._levels_array() % 2 == 0, 0.5, 0.0)

    def _lowest_positions(self) -> IndexArray:
        """``-(L // 2)`` per dimension: the position digit 0 stands for."""
        return -(self._levels_array() // 2)

    def _digits_of_bounded(self, bounded: FloatArray) -> IndexArray:
        """Round bounded coordinates to positions, then shift them to digits."""
        positions = np.rint(bounded * self._half_widths() - self._offsets())
        return (positions - self._lowest_positions()).astype(np.intp)

    def _coordinates_of_digits(self, digits: IndexArray) -> FloatArray:
        """The grid point in ``[-1, 1]`` coordinates for each row of digits."""
        positions = digits + self._lowest_positions()
        return (positions + self._offsets()) / self._half_widths()

    def _ids_of_digits(self, digits: IndexArray) -> IndexArray:
        """Compose each row of digits into one id, dimension 0 least significant."""
        ids = np.zeros(digits.shape[0], dtype=np.intp)
        weight = 1
        for dimension, level in enumerate(self.levels):
            ids += digits[:, dimension] * weight
            weight *= level
        return ids
