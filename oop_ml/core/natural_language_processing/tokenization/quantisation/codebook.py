"""Vector quantisation's lookup: a vector to the id of its nearest code, and back.

The tokenization problem, asked of vectors
------------------------------------------
Every tokenizer in this package turns a string into whole numbers a model can
read. An image patch, a slice of audio or a frame of speech features is not a
string but a vector, and the same question is asked of it: which of a fixed set
of units is this one. The answer is *vector quantisation*. A :class:`Codebook`
holds ``n_codes`` vectors, each owning the id that is its position, and a
vector is encoded as the id of the code nearest to it. That id is the token, the
code's vector is what the token means, and the gap between the input and the
code it was rounded to is the price of saying it in ``n_codes`` words.

That is the id half of a VQ-VAE (van den Oord, Vinyals and Kavukcuoglu, 2017)
and of VQGAN, where an encoder's output at each spatial position is snapped to
its nearest code and the image becomes a grid of ids. It is also exactly what
the "speech units" of HuBERT and wav2vec 2.0 are: k-means is run over frames of
speech features, and a frame's unit is the id of the centroid nearest to it. A
centroid table is a codebook.

Where the codebook comes from
-----------------------------
Not from here. A codebook is *learned* -- by k-means, whose centroids are one
(``Codebook.from_centroids(kmeans.centroids)`` for
:class:`~oop_ml.numpy.clustering.k_means.KMeans`), or by a VQ-VAE's training
loop, which moves the codes with a gradient. This module holds the lookup and
nothing that learns, deliberately. ``oop_ml.core.natural_language_processing`` is
shared vocabulary in the sense ``test/test_layering.py`` checks for it and for
``core``: written against ``core``, importing no backend, so a k-means of its own
would either duplicate one or reach into ``oop_ml.numpy``. The link to k-means
is therefore a classmethod on the codebook reading
:class:`~oop_ml.core.clustering.centroids.Centroids`, which is shared.

So :class:`CodebookQuantizer` is configured rather than fitted, for the reason
the byte-level tokenizers are plain ``Tokenizer`` rather than
``LearnedTokenizer``: a guard that says "fit me first" on an object with
nothing to fit would be a small lie. Nor is it a ``Tokenizer`` at all, since its
input is an array and not a text, and the template's text checks would refuse
every input it exists for.

Worked, on three codes in the plane
-----------------------------------
Codes ``0 = (0, 0)``, ``1 = (4, 0)``, ``2 = (0, 3)``. Four inputs, with the
squared distance to each code and the id chosen::

    input     to 0  to 1  to 2      id   squared gap
    (1, 0)       1     9    10  ->   0        1
    (3, 1)      10     2    13  ->   1        2
    (1, 2)       5    13     2  ->   2        2
    (2, 0)       4     4    13  ->   0        4      a tie with code 1

The distortion is the mean of the squared gaps, ``(1 + 2 + 2 + 4) / 4 = 2.25``:
the squared Euclidean distance from an input to the code it became, summed over
coordinates and averaged over rows. It is the quantity a VQ-VAE calls its
quantisation error, and k-means' inertia divided by the row count.

The tie rule
------------
``(2, 0)`` sits exactly between codes 0 and 1. The tie goes to the lower id,
which is the code that comes first in the book and not the smaller vector:
reorder the same three codes so that ``(4, 0)`` is code 0 and the same input
goes to it. Arbitrary, and stated once, like
:meth:`~oop_ml.core.tree.split.Split.beats`.

Why two identical codes are refused
-----------------------------------
Two codes at one point are two ids for one vector. The tie rule would silently
hand every input near them to the lower id and leave the other an id that is
never produced and means the same thing. That is the vocabulary's own duplicate
rule, so it raises the vocabulary's own error,
:class:`~oop_ml.core.exceptions.NonUniqueTokensError`. Equality is ``==`` on
the coordinates, so ``0.0`` and ``-0.0`` are one point. A *collapsed* codebook,
in which most codes are never chosen because an encoder learned to lean on a
few, is the VQ-VAE failure this refusal does not and cannot catch: that is a
fact about the inputs, not about the book.

What it costs
-------------
The distances are computed as written, one pass per code, each pass a
subtraction and a sum over the rows: ``O(n_rows * n_codes * dimension)``, and
exact. The usual repair is the expansion ``|a|^2 - 2 a.b + |b|^2`` as one
matrix multiply, which is what
:meth:`~oop_ml.core.clustering.centroids.Centroids.squared_distances_to` does,
and it is exact only because it shifts rows and centres to a common origin
first. The plain form needs no such care, which is why it is the one here.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from oop_ml.core.clustering.centroids import Centroids
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueTokensError,
    ShapeMismatchError,
    UnknownTokenError,
)
from oop_ml.core.types import FloatArray, IndexArray, array_for_protocol


def _as_float_block(values: object) -> FloatArray:
    """``values`` as a two-dimensional float64 array, one vector per row.

    Raises
    ------
    InvalidValuesError
        If ``values`` is not numeric, or is not two-dimensional.
    """
    try:
        block = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(
            f"vectors must be numeric, got {type(values).__name__}"
        ) from error

    if block.ndim != 2:
        raise InvalidValuesError(
            f"vectors form a two-dimensional block, one vector per row; got "
            f"shape {block.shape}. A single vector is a block of one row"
        )

    return block


def checked_vectors(vectors: FloatArray, dimension: int) -> FloatArray:
    """The one place a block of vectors is confirmed fit to quantise.

    Parameters
    ----------
    vectors:
        ``(n_rows, dimension)``.
    dimension:
        The width every row must have, which is the codes' own.

    Raises
    ------
    InvalidValuesError
        If ``vectors`` is not numeric, not two-dimensional, or not finite.
    EmptyValuesError
        If there are no rows.
    ShapeMismatchError
        If the rows are not ``dimension`` wide.
    """
    block = _as_float_block(vectors)

    if block.shape[0] == 0:
        raise EmptyValuesError("at least one vector is required")

    if block.shape[1] != dimension:
        raise ShapeMismatchError(
            f"vectors have {block.shape[1]} coordinates but the codes have {dimension}"
        )

    if not np.all(np.isfinite(block)):
        raise InvalidValuesError("every coordinate must be finite")

    return block


def checked_code_id(code_id: int, n_codes: int) -> int:
    """``code_id`` as a Python int that names one of ``n_codes`` codes.

    Raises
    ------
    InvalidValuesError
        If ``code_id`` is not a whole number.
    UnknownTokenError
        If no code has that id. The vocabulary's error, because a codebook is
        the vocabulary of a vector tokenizer and an id past its end is the same
        mistake as an id past a
        :class:`~oop_ml.core.natural_language_processing.tokenization.vocabulary.Vocabulary`'s.
    """
    try:
        position = operator.index(code_id)
    except TypeError as error:
        raise InvalidValuesError(
            f"a code id is a whole number, got {code_id!r}"
        ) from error

    if not 0 <= position < n_codes:
        raise UnknownTokenError(
            f"no code has id {position}; ids run from 0 to {n_codes - 1}"
        )

    return position


class Codebook:
    """A fixed set of code vectors, each addressable by its position.

    The vocabulary of a vector tokenizer: code ``i`` is whichever vector the
    block held in row ``i``, so there is no separate id table to drift.

    Parameters
    ----------
    vectors:
        ``(n_codes, dimension)``, every entry finite, at least one code, at
        least one coordinate, no two codes equal. Copied and frozen, so the
        caller keeps their own array and nothing handed back out can be
        written through.

    Raises
    ------
    EmptyValuesError
        If there are no codes.
    InvalidValuesError
        If the block is not numeric, not two-dimensional, zero wide, or holds
        a non-finite coordinate.
    NonUniqueTokensError
        If two codes are the same vector.
    """

    __slots__ = ("_vectors",)

    def __init__(self, vectors: FloatArray) -> None:
        block = _as_float_block(vectors)
        n_codes, dimension = block.shape

        if n_codes == 0:
            raise EmptyValuesError("a codebook needs at least one code")

        if dimension == 0:
            raise InvalidValuesError(
                "a code needs at least one coordinate; got codes of width 0"
            )

        if not np.all(np.isfinite(block)):
            raise InvalidValuesError("every coordinate of every code must be finite")

        # Keyed on Python floats so that equality is ``==``: 0.0 and -0.0 hash
        # and compare alike, and NaN has already been refused.
        positions_by_point: dict[tuple[float, ...], int] = {}
        for position, code in enumerate(block):
            point = tuple(code.tolist())
            if point in positions_by_point:
                raise NonUniqueTokensError(
                    f"codes {positions_by_point[point]} and {position} are the same "
                    f"vector, which would be two ids for one point"
                )
            positions_by_point[point] = position

        frozen = block.copy()
        frozen.setflags(write=False)
        self._vectors = frozen

    @classmethod
    def from_centroids(cls, centroids: Centroids) -> Codebook:
        """The centres of a clustering as a codebook, cluster ``k`` as code ``k``.

        The HuBERT construction: k-means over speech features, and a frame's
        unit is the id of its nearest centroid.
        """
        return cls(centroids.positions)

    @property
    def vectors(self) -> FloatArray:
        """``(n_codes, dimension)``, read-only, code ``i`` in row ``i``."""
        return self._vectors

    @property
    def n_codes(self) -> int:
        """How many codes there are, which is the size of the id space."""
        return int(self._vectors.shape[0])

    @property
    def dimension(self) -> int:
        """How many coordinates each code has, which every input must match."""
        return int(self._vectors.shape[1])

    @property
    def shape(self) -> tuple[int, ...]:
        """The wrapped array's shape, so a caller can assert on it directly."""
        return self._vectors.shape

    def vector_of(self, code_id: int) -> FloatArray:
        """The code at position ``code_id``, as a read-only vector.

        Raises
        ------
        InvalidValuesError
            If ``code_id`` is not a whole number.
        UnknownTokenError
            If no code has that id.
        """
        return self._vectors[checked_code_id(code_id, self.n_codes)]

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter.

        See :func:`~oop_ml.core.types.array_for_protocol` for the contract and
        the corruption it exists to prevent.
        """
        return array_for_protocol(self._vectors, dtype, copy)

    def __getitem__(self, code_id: int) -> FloatArray:
        """The code for ``code_id``, so that ``codebook[3]`` reads well."""
        return self.vector_of(code_id)

    def __iter__(self) -> Iterator[FloatArray]:
        """Iterate the codes in id order, each a read-only vector."""
        return iter(self._vectors)

    def __len__(self) -> int:
        return self.n_codes

    def __eq__(self, other: object) -> bool:
        """One verdict, like a vocabulary's, and not one answer per entry.

        A codebook is the vocabulary of a vector tokenizer and is a field of
        :class:`CodebookQuantizer`, and pydantic compares fields with ``==``
        expecting a truth value. The element-wise answer
        :class:`~oop_ml.core.data.predictions.Predictions` gives is right for a
        result and wrong for a configuration: two quantizers would raise on
        comparison rather than compare.
        """
        if not isinstance(other, Codebook):
            return NotImplemented
        return bool(np.array_equal(self._vectors, other._vectors))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Codebook(n_codes={self.n_codes}, dimension={self.dimension})"


class CodeAssignment:
    """What a block of vectors became: ids, the codes they name, and the cost.

    Three things that mean nothing apart, so they travel together: which code
    each row was assigned, the vector that code stands for, and how far the
    rows had to move to be said in codes at all.

    Parameters
    ----------
    code_ids:
        ``(n_rows,)`` whole numbers, one per input row, none negative.
    reconstruction:
        ``(n_rows, dimension)``, row ``i`` the vector of code ``code_ids[i]``.
        Finite.
    distortion:
        The mean over rows of the squared Euclidean distance from an input to
        its code. Finite and non-negative.

    Raises
    ------
    EmptyValuesError
        If there are no rows.
    InvalidValuesError
        If the ids are not whole numbers, not one-dimensional, or negative; if
        the reconstruction is not a finite two-dimensional block at least one
        coordinate wide; or if the distortion is not a finite non-negative
        number.
    NonEqualArrayLengthError
        If there is not one reconstructed row per id.
    """

    __slots__ = ("_code_ids", "_distortion", "_reconstruction")

    def __init__(
        self, code_ids: IndexArray, reconstruction: FloatArray, distortion: float
    ) -> None:
        ids = np.asarray(code_ids)

        if ids.size == 0:
            raise EmptyValuesError("an assignment needs at least one row")

        if not np.issubdtype(ids.dtype, np.integer):
            raise InvalidValuesError(
                f"code ids are whole numbers, got dtype {ids.dtype}"
            )

        if ids.ndim != 1:
            raise InvalidValuesError(
                f"code ids are one per row, so one-dimensional; got shape {ids.shape}"
            )

        if np.any(ids < 0):
            raise InvalidValuesError("a code id is a position and cannot be negative")

        block = _as_float_block(reconstruction)

        if block.shape[0] != ids.size:
            raise NonEqualArrayLengthError(
                f"{ids.size} code ids against {block.shape[0]} reconstructed rows"
            )

        if block.shape[1] == 0:
            raise InvalidValuesError(
                "a reconstructed vector needs at least one coordinate; got width 0"
            )

        if not np.all(np.isfinite(block)):
            raise InvalidValuesError("every reconstructed coordinate must be finite")

        try:
            distortion_value = float(distortion)
        except (TypeError, ValueError) as error:
            raise InvalidValuesError(
                f"a distortion is a number, got {distortion!r}"
            ) from error

        if not math.isfinite(distortion_value) or distortion_value < 0.0:
            raise InvalidValuesError(
                f"a distortion is a mean of squared distances, so finite and "
                f"non-negative; got {distortion_value}"
            )

        frozen_ids = ids.astype(np.intp)
        frozen_ids.setflags(write=False)
        frozen_block = block.copy()
        frozen_block.setflags(write=False)

        self._code_ids = frozen_ids
        self._reconstruction = frozen_block
        self._distortion = distortion_value

    @property
    def code_ids(self) -> IndexArray:
        """``(n_rows,)``, read-only, for the arithmetic."""
        return self._code_ids

    @property
    def ids(self) -> tuple[int, ...]:
        """The ids as Python ints, in row order: what a model reads.

        The same shape
        :attr:`~oop_ml.core.natural_language_processing.tokenization.encoding.Encoding.ids`
        has, so a run of speech units and a run of subword ids look alike
        downstream.
        """
        return tuple(int(code_id) for code_id in self._code_ids)

    @property
    def reconstruction(self) -> FloatArray:
        """``(n_rows, dimension)``, read-only: each row's code as a vector."""
        return self._reconstruction

    @property
    def distortion(self) -> float:
        """Mean squared Euclidean distance from the inputs to their codes."""
        return self._distortion

    @property
    def n_rows(self) -> int:
        """How many vectors were assigned."""
        return int(self._code_ids.size)

    @property
    def dimension(self) -> int:
        """How many coordinates each reconstructed vector has."""
        return int(self._reconstruction.shape[1])

    def __len__(self) -> int:
        return self.n_rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeAssignment):
            return NotImplemented
        return (
            bool(np.array_equal(self._code_ids, other._code_ids))
            and bool(np.array_equal(self._reconstruction, other._reconstruction))
            and self._distortion == other._distortion
        )

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return (
            f"CodeAssignment(n_rows={self.n_rows}, dimension={self.dimension}, "
            f"distortion={self._distortion})"
        )


class CodebookQuantizer(BaseModel):
    """Nearest-code lookup against a codebook learned elsewhere.

    Configured, never fitted, and not a ``Tokenizer``: see the module docstring
    for both. The codebook is a field so that a quantizer is validated at
    construction like every other configured object here, and so that two
    quantizers compare by their books.

    Parameters
    ----------
    codebook:
        The codes to snap to. Learned by k-means
        (:meth:`Codebook.from_centroids`) or by a VQ-VAE, never here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    codebook: Codebook

    @property
    def n_codes(self) -> int:
        """How many ids this quantizer can produce."""
        return self.codebook.n_codes

    @property
    def dimension(self) -> int:
        """How many coordinates every input must have."""
        return self.codebook.dimension

    def quantize(self, vectors: FloatArray) -> CodeAssignment:
        """Each row's nearest code by Euclidean distance, ties to the lower id.

        Parameters
        ----------
        vectors:
            ``(n_rows, dimension)``, finite.

        Raises
        ------
        InvalidValuesError
            If ``vectors`` is not numeric, not two-dimensional, or not finite.
        EmptyValuesError
            If there are no rows.
        ShapeMismatchError
            If the rows are not ``dimension`` wide.
        """
        inputs = checked_vectors(vectors, self.codebook.dimension)
        codes = self.codebook.vectors
        n_rows = inputs.shape[0]

        squared_distances = np.empty((n_rows, codes.shape[0]), dtype=np.float64)
        for code_id, code in enumerate(codes):
            gaps = inputs - code
            squared_distances[:, code_id] = np.sum(gaps * gaps, axis=1)

        # argmin takes the first minimum, which is the tie rule: the lower id.
        nearest = np.argmin(squared_distances, axis=1)
        smallest = squared_distances[np.arange(n_rows), nearest]

        return CodeAssignment(nearest, codes[nearest], float(np.mean(smallest)))

    def dequantize(self, code_ids: Sequence[int]) -> FloatArray:
        """The code vectors the ids name, ``(n_ids, dimension)``, a fresh array.

        Raises
        ------
        EmptyValuesError
            If there are no ids.
        InvalidValuesError
            If an id is not a whole number.
        UnknownTokenError
            If an id names no code.
        """
        if len(code_ids) == 0:
            raise EmptyValuesError("at least one code id is required")

        return np.array(
            [self.codebook.vector_of(code_id) for code_id in code_ids],
            dtype=np.float64,
        )
