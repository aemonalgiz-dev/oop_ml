"""A table of vectors read by number, which is the layer that was impossible.

What is actually new here
-------------------------
Every layer before this one computes its answer. This one looks it up. There is
a table with one row per position in a vocabulary, a sequence arrives naming
positions, and the answer is those rows stacked in the order they were named.
The parameters are the table itself, so what is learned is *what each position
means* rather than a function of anything.

That is the only way a symbol gets into a network at all. A word, a token, a
category has no magnitude and no order -- position 4 is not twice position 2 and
is not between 3 and 5 -- so handing the number itself to a dense layer states
three false things about it at once. A lookup states none of them: the rows begin
unrelated and any relationship between two positions is a thing the fit
discovered rather than a thing the numbering imposed.

Why this layer could not be written until recently
---------------------------------------------------
This package recorded the hole in its own words. Every layer coerced what it was
handed to float64 and scanned it for infinities, which is exactly right for
measurements and checks nothing whatever for positions. A position of ``2.9999``
truncates to 2, a position of ``-1`` wraps round to the last row of the table,
and both are finite float64 values that no scan objects to. The layer answers a
row nobody asked for and nothing is raised.

:meth:`~oop_ml.core.network.layer.Layer._read_block` is now an overridable
method for precisely this case, and :meth:`Embedding._read_block` is the one
override in the library. It refuses a position that is not a whole number and a
position outside the table, by name, and the two checks it replaces are the two
that could not have caught either. The shape checks in
:meth:`~oop_ml.core.network.layer.Layer.respond_to` still run afterwards on what
it returns, so overriding the coercion takes over the coercion and nothing else.

Why the step count is fixed at construction
--------------------------------------------
The same trade :mod:`~oop_ml.core.network.recurrent` states and for the same
reason. This package's promise is that a network which cannot work is refused in
integer comparisons before a row is read, so a sequence layer has to know how
many steps it reads. Sequences of unequal length are padded or cut to one length
before they arrive here, where an implementation discovering the count from the
block would take them as they came.

Why the gradient is unusual, and how
-------------------------------------
Three things about it are worth saying out loud, because none of them is what a
dense layer does.

**Only the rows that were looked up have any slope at all.** A table of fifty
thousand positions over a batch naming two hundred of them has a gradient that is
zero on 99.6% of its rows, and that is the truth about the calculation rather
than a sparsity trick: a row nothing read contributed nothing to the loss and is
owed nothing back.

**A position named twice collects a contribution from each appearance.** The same
row was read at two different steps, so the chain rule gives one term per
appearance and the slope is their sum. So the backward pass accumulates into a
block of zeros -- ``np.add.at``, which adds at repeated indices where ordinary
fancy-index assignment keeps only the last write. That is the same accumulation
:mod:`~oop_ml.core.network.convolution` records for a shared kernel and
:mod:`~oop_ml.core.network.recurrent` for shared weights across steps, and it
fails the same quiet way: assigning rather than adding reports a fraction of the
slope, in the right direction, with the right shape, and trains.

**The bias block is zeros, and that is honest rather than a fudge.**
:class:`~oop_ml.core.network.gradient.LayerGradient` pairs a weight block with a
bias block of matching height, and an embedding has no biases -- a lookup adds
nothing to the row it found. Zeros are the true slopes of the parameters this
layer does not have, they move nothing at any learning rate, and
:meth:`Embedding.stepped_by` reads only the weight half. The alternative is a
second gradient type for one layer, which is the cost
:mod:`~oop_ml.core.network.gradient` already measured and declined. Note that
``gradient=None`` would be the *wrong* answer here, and means something else:
flattening and pooling say ``None`` because they have nothing to learn, and this
layer has the largest parameter block in most networks that contain it.

Why nothing is passed down
---------------------------
A position is a name, not a measurement. There is no derivative of "row 7" with
respect to anything, because there is no continuum for a slope to live on -- the
positions are exactly the discrete objects this layer exists to turn into
something continuous. So :meth:`Embedding.correction_for` answers with a block of
zeros arranged like the sequence it read.

Zeros rather than ``None``, because
:class:`~oop_ml.core.network.gradient.LayerCorrection` requires a block: every
layer sits on something. An embedding sits on nothing, since it is the bottom of
any stack that contains it, and a stack whose walk reaches this layer and then
continues downward is a stack that put a layer beneath the one reading its
symbols. The zeros are what such a layer would correctly be told.

Where the table starts
-----------------------
Normal draws scaled by ``sqrt(1 / n_dimensions)``, which makes the expected
squared length of a row exactly ``n_dimensions * (1 / n_dimensions) = 1``
whatever the width. So widening the table does not make every symbol arrive
louder at the layer above, which is the same fan-in argument
:mod:`~oop_ml.core.network.convolution` makes for its kernels, read from the
other side: a convolution scales by what each output *sums*, and a lookup sums
nothing, so what there is to control is the length of the row it hands on.
"""

from __future__ import annotations

import operator
from math import sqrt

import numpy as np

from oop_ml.core.exceptions import InvalidValuesError, ShapeMismatchError
from oop_ml.core.network.blocks import as_block
from oop_ml.core.network.gradient import LayerCorrection, LayerGradient
from oop_ml.core.network.layer import Layer, LayerResponse
from oop_ml.core.network.purpose import PassPurpose
from oop_ml.core.network.shape import LayerShape
from oop_ml.core.types import FloatArray


def _checked_setting(value: int, least: int, role: str) -> int:
    """Read one configured whole number, refusing anything below ``least``.

    Any whole number is accepted rather than only a builtin ``int``, since a
    vocabulary size read off a numpy computation is a ``numpy.int64`` and is a
    count by every standard except ``isinstance``. ``bool`` is excluded ahead of
    that, because ``True`` indexes as 1 and would otherwise slip through as a
    table of one row by accident.

    Parameters
    ----------
    value:
        The configured number.
    least:
        The smallest value that means anything for this setting.
    role:
        The parameter's name, for the message.

    Returns
    -------
    int
        The same number, as a builtin ``int``.

    Raises
    ------
    InvalidValuesError
        If the value is a bool, is not a whole number, or is below ``least``.
    """
    if isinstance(value, bool):
        raise InvalidValuesError(f"{role} must be a whole number, not a bool")
    try:
        whole = operator.index(value)
    except TypeError:
        raise InvalidValuesError(f"{role} must be a whole number") from None
    if whole < least:
        raise InvalidValuesError(f"{role} must be at least {least}, got {whole}")
    return whole


class Embedding(Layer):
    """A learned vector per position, answered by lookup rather than arithmetic.

    Parameters
    ----------
    n_positions:
        How many positions the vocabulary holds, which is how many rows the
        table has. A position outside ``[0, n_positions)`` is refused rather
        than wrapped.
    n_dimensions:
        How wide each vector is, which is how much a position is allowed to
        mean.
    n_steps:
        How many positions one sequence names. Fixed here rather than read off
        the block, for the reason the module docstring gives.
    random_seed:
        Which draw the table starts from. Fixed by default, so two layers built
        the same way are the same layer and a spec can rely on it.

    Raises
    ------
    InvalidValuesError
        If any of the three counts is not a whole number of at least one.

    Notes
    -----
    Reads ``(n_steps,)`` and answers ``(n_steps, n_dimensions)``, so it is the
    one layer here whose answer has an extent its input does not.
    """

    __slots__ = ("_n_dimensions", "_n_positions", "_n_steps", "_shape", "_table")

    def __init__(
        self,
        n_positions: int,
        n_dimensions: int,
        n_steps: int,
        random_seed: int = 0,
    ) -> None:
        checked_positions = _checked_setting(n_positions, 1, "n_positions")
        checked_dimensions = _checked_setting(n_dimensions, 1, "n_dimensions")
        checked_steps = _checked_setting(n_steps, 1, "n_steps")

        generator = np.random.default_rng(random_seed)
        table = generator.normal(size=(checked_positions, checked_dimensions)) * sqrt(
            1.0 / checked_dimensions
        )

        self._configure(
            shape=LayerShape(
                n_inputs=(checked_steps,),
                n_outputs=(checked_steps, checked_dimensions),
            ),
            table=table,
        )

    def _configure(self, shape: LayerShape, table: FloatArray) -> None:
        """Fill every slot from an already validated configuration.

        Separated from ``__init__`` for the reason
        :meth:`~oop_ml.core.network.convolution.Conv2d._configure` gives: there
        are two ways to arrive at a layer and only one of them draws random
        numbers. A training step builds the next layer from this one's geometry
        and a corrected table, and rerunning the constructor would throw a draw
        away and misreport where those numbers came from.

        Parameters
        ----------
        shape:
            What the layer reads and answers with, already computed.
        table:
            ``(n_positions, n_dimensions)``, copied and frozen here.
        """
        self._shape = shape
        self._n_steps = shape.reads[0]
        self._n_dimensions = shape.answers[1]

        block = np.array(table, dtype=np.float64, copy=True)
        block.setflags(write=False)
        self._table = block
        self._n_positions = int(block.shape[0])

    @property
    def shape(self) -> LayerShape:
        """``(n_steps,)`` in, ``(n_steps, n_dimensions)`` out.

        The one layer in the package whose answer carries an extent its input
        does not, which is what turning a name into a quantity looks like as a
        shape.
        """
        return self._shape

    @property
    def table(self) -> FloatArray:
        """``(n_positions, n_dimensions)``, frozen, one row per position.

        The learned parameters, in the arrangement they are used in. Frozen for
        the reason every learned array here is: a caller who edited it would
        change a layer that has already answered.
        """
        return self._table

    @property
    def n_positions(self) -> int:
        """How many positions the vocabulary holds."""
        return self._n_positions

    @property
    def n_dimensions(self) -> int:
        """How wide each position's vector is."""
        return self._n_dimensions

    @property
    def n_steps(self) -> int:
        """How many positions one sequence names."""
        return self._n_steps

    def with_parameters(self, table: FloatArray) -> Embedding:
        """A new layer of the same geometry, carrying this table.

        Training does not mutate. A step builds the next layer from this one's
        geometry and the corrected numbers, which is what keeps the table frozen
        and every answer reproducible.

        Parameters
        ----------
        table:
            ``(n_positions, n_dimensions)``.

        Returns
        -------
        Embedding
            A new layer, same geometry, carrying these vectors.

        Raises
        ------
        ShapeMismatchError
            If the block does not match this layer's table.
        InvalidValuesError
            If the block cannot be read as a float array.
        """
        block = as_block(table, "an embedding's table")

        expected = (self._n_positions, self._n_dimensions)
        if block.shape != expected:
            raise ShapeMismatchError(
                f"this layer's table is {expected}, got {block.shape}"
            )

        replacement = Embedding.__new__(Embedding)
        replacement._configure(shape=self._shape, table=block)
        return replacement

    def _read_block(self, inputs: object) -> FloatArray:
        """Read a block of positions, refusing anything that is not one.

        The one override of this method in the library, and the reason it is a
        method at all. The inherited rule coerces to float64 and scans for
        non-finite entries, which for positions checks nothing that can go wrong
        and misses both things that can: ``2.9999`` truncates to 2 and ``-1``
        wraps to the last row, and neither is non-finite.

        So the two checks are the two that mean something here. Every entry is a
        whole number, and every entry names a row this table actually has. Both
        refuse rather than round or clip, because a caller who meant position
        two and wrote 2.9999 has a bug upstream and rounding it hides the bug
        while answering plausibly.

        The finiteness scan is kept, ahead of the other two, so that a ``nan``
        is refused as the non-finite value it is rather than as a fractional
        position. Both refusals are correct; the first is the one that names
        what went wrong.

        Parameters
        ----------
        inputs:
            Whatever the caller handed in, commonly a block of Python integers.

        Returns
        -------
        FloatArray
            The positions as float64, which is what
            :meth:`~oop_ml.core.network.layer.Layer.respond_to` then checks the
            arrangement of and what :meth:`_response_for` indexes with.

        Raises
        ------
        InvalidValuesError
            If the block cannot be read as a float array, holds a non-finite
            entry, holds a value that is not a whole number, or names a position
            outside ``[0, n_positions)``.
        """
        block = as_block(inputs, "positions")

        if not np.isfinite(block).all():
            raise InvalidValuesError("a position must be finite")
        flat = block.reshape(-1)
        fractional = flat[flat != np.floor(flat)]
        if fractional.size > 0:
            raise InvalidValuesError(
                "a position names a row of the table, so it is a whole number; "
                f"got {float(fractional[0])}"
            )
        if block.size > 0 and (block.min() < 0.0 or block.max() >= self._n_positions):
            raise InvalidValuesError(
                f"this table holds positions 0 to {self._n_positions - 1}, and the "
                f"block names {float(block.min())} to {float(block.max())}"
            )

        return block

    def _response_for(self, inputs: FloatArray, purpose: PassPurpose) -> LayerResponse:
        """The lookup, given positions already known to be whole and in range.

        Parameters
        ----------
        inputs:
            ``(n_rows, n_steps)``, validated by :meth:`_read_block` and then by
            :meth:`~oop_ml.core.network.layer.Layer.respond_to`. Nothing here
            re-checks either.
        purpose:
            Ignored. A table is read the same way whatever the pass is for.

        Returns
        -------
        LayerResponse
            ``outputs`` is ``(n_rows, n_steps, n_dimensions)``, the named rows
            stacked in the order they were named. ``scores`` is the same block,
            because nothing is bent here and so there is no pre-activation value
            distinct from the answer, which is the honesty
            :class:`~oop_ml.core.network.flatten.Flatten` already practises.

        Notes
        -----
        The cast to ``intp`` is safe only because :meth:`_read_block` has
        already established that every entry is a whole number in range. Written
        the other way round -- cast first, check afterwards -- the cast is the
        thing that destroys the evidence, which is the whole failure this layer
        was recorded as unable to avoid.
        """
        positions = inputs.astype(np.intp)
        outputs = self._table[positions]

        return LayerResponse.already_checked(
            inputs=inputs, scores=outputs, outputs=outputs
        )

    def correction_for(
        self, response: LayerResponse, arriving: FloatArray
    ) -> LayerCorrection:
        """Accumulate each appearance's blame into the row it was read from.

        Parameters
        ----------
        response:
            What this layer did on the way up. Its ``inputs`` are the positions,
            which is all the backward step needs -- the answer was a copy of the
            table and carries nothing the table does not.
        arriving:
            ``(n_rows, n_steps, n_dimensions)``, the slope of the loss at this
            layer's answer.

        Returns
        -------
        LayerCorrection
            ``gradient`` carries a ``(n_positions, n_dimensions)`` weight block,
            zero on every row nothing read, and a bias block of zeros because an
            embedding has no biases. ``passed_down`` is a block of zeros
            arranged like the sequence of positions that was read.

        Raises
        ------
        ShapeMismatchError
            If the response did not come from a layer of this shape, or the
            arriving block does not describe this layer's answer for the rows
            the response holds.
        InvalidValuesError
            If the arriving block cannot be read as a float array.

        Notes
        -----
        ``np.add.at`` rather than ``gradient[positions] += arriving``. The two
        differ exactly when a position appears twice, which is the ordinary case
        rather than an edge one: fancy-index assignment resolves the repeat by
        keeping the last write, so the row would be credited with one appearance
        out of however many it had. The result trains, descends, and is a
        fraction of the gradient of the thing being minimised.
        """
        checked = self._checked_arriving(response, arriving)
        positions = response.inputs.astype(np.intp)
        table_slopes = np.zeros_like(self._table)
        np.add.at(table_slopes, positions, checked)
        return LayerCorrection(
            passed_down=np.zeros_like(response.inputs),
            gradient=LayerGradient(
                weights=table_slopes, biases=np.zeros(self._n_positions)
            ),
        )

    def stepped_by(
        self, gradient: LayerGradient | None, learning_rate: float
    ) -> Embedding:
        """A new layer whose table has been moved against its slope.

        Downhill is against the slope, so this subtracts. Only the rows that
        were read have a non-zero slope, so only they move, which is the whole
        of what "learning an embedding" means arithmetically.

        Parameters
        ----------
        gradient:
            What the table wants to change by. The bias half is ignored, since
            this layer has no biases and the block exists only so that
            :class:`~oop_ml.core.network.gradient.LayerGradient` holds.
        learning_rate:
            How far to move.

        Returns
        -------
        Embedding
            A new layer, same geometry, moved table.

        Raises
        ------
        ShapeMismatchError
            If no gradient is supplied, since this layer has parameters and has
            nothing to answer with when asked to step by nothing, or if the
            gradient does not describe this layer's table.
        """
        if gradient is None:
            raise ShapeMismatchError(
                "an embedding has a table and needs a gradient to step by"
            )

        expected = (self._n_positions, self._n_dimensions)
        if gradient.weights.shape != expected:
            raise ShapeMismatchError(
                f"this layer's table is {expected}, and the gradient describes "
                f"{gradient.weights.shape}"
            )

        return self.with_parameters(self._table - learning_rate * gradient.weights)

    def __repr__(self) -> str:
        return (
            f"Embedding(n_positions={self._n_positions!r}, "
            f"n_dimensions={self._n_dimensions!r}, "
            f"n_steps={self._n_steps!r})"
        )
