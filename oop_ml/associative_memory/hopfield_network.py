"""Store patterns in a symmetric weight matrix and recall one from a fragment.

The learning rule is local, and that is the whole reason this exists
--------------------------------------------------------------------
Every other network in this library learns by backpropagation. A weight sitting
deep in a stack is changed by a signal that began at the loss, several layers
away, and travelled back through everything in between; the two units the
weight actually joins have very little to do with the size of its correction.

Hebbian storage is the opposite, and it is the concept this family is here to
teach. The change to the weight joining unit ``i`` and unit ``j`` is a function
of the value at unit ``i`` and the value at unit ``j``, and of nothing else
whatsoever:

    weights[i][j]  +=  pattern[i] * pattern[j] / n_units

Two units that agree inside a stored pattern are wired together positively, two
that disagree negatively, and no third party is consulted. Nothing is
propagated, there is no target, there is no loss, and there is no gradient. A
synapse can read the two neurons it joins and cannot read a number computed at
the far end of the brain, so a local rule is the only kind a piece of tissue
could plausibly implement, and this is the cleanest one there is.

What the model is
-----------------
Units hold **bipolar** values, each either ``-1`` or ``+1``. A state of the
network is one value per unit, and storage is the sum of the outer products of
the patterns to be remembered, with the diagonal forced to zero:

    weights  =  (1 / n_units) * sum over patterns of outer(pattern, pattern)
    weights[i][i]  =  0

Recall starts from whatever state you hand it and repeatedly updates units:

    state[i]  =  sign(sum over j of weights[i][j] * state[j])

with the convention that a sum of exactly zero leaves that unit alone. Run that
until nothing changes and the state has fallen into a **fixed point**, which for
a lightly loaded network is one of the patterns that were stored. That is what
"associative" means here. A classifier is addressed by a key and answers with a
label; this is addressed by a corrupted version of the content itself and
answers with the content.

Everything the network knows is in the weights. The patterns are not kept, and
cannot be read back except by asking the weights to settle.

Why this is not a ``ConvergentFit``
-----------------------------------
Every other iterative model here inherits
:class:`~oop_ml.core.base.convergent_fit.ConvergentFit`, and this one
deliberately does not. Its ``fit`` is one shot: read the patterns, form one
matrix, stop. There is no walk, no tolerance and nothing to converge. The thing
that iterates is **recall**, which happens at transform time, once per pattern
presented, and which can settle for one pattern and hit its pass limit for
another in the same call.

So a ``converged`` attribute on this model would be describing an event that did
not happen during its fit. Whether a settling converged is a fact about one
recall, and it lives on :class:`RecallWalk`, which is the object that records
one.

Asynchronous update is guaranteed to settle, and this is a theorem
------------------------------------------------------------------
Define the energy of a state:

    energy(state)  =  -0.5 * state . weights . state

Take one unit ``i`` and write ``weighted_sum_i = sum over j of w[i][j] s[j]``.
Because the matrix is symmetric and ``w[i][i]`` is zero, every term of the
energy that mentions ``s_i`` collects into ``-s_i * weighted_sum_i``, and
``weighted_sum_i`` itself does not mention ``s_i``. So flipping that one unit
changes the energy by exactly

    change  =  -(new - old) * weighted_sum_i  =  -2 * |weighted_sum_i|

because the rule only flips the unit when ``sign(weighted_sum_i)`` disagrees
with the value it currently holds. Every flip that actually happens therefore
lowers the energy **strictly**, whenever the weighted sum is not zero.

The state space has ``2 ** n_units`` members and no more. An energy that
strictly decreases can never revisit a state, so the walk cannot cycle, and a
finite set with no repeats must run out. What it runs out at is a state where no
single unit wants to move, which is a local minimum of the energy and a fixed
point of the update. That is the whole proof, and it is unusually short for a
convergence result about a network.

Note what it rests on. Symmetry, so the two halves of the quadratic form combine
into one factor, and a zero diagonal, so ``weighted_sum_i`` is free of ``s_i``.
Break either and the argument is gone.

Synchronous update is not guaranteed to settle
-----------------------------------------------
Update every unit at once from the same old state and the argument above does
not apply, because two units can each move in a direction that would have
lowered the energy on its own while together they raise it. The result is a
period-two oscillation that runs forever.

This is not a corner case that needs a contrived matrix. Store the single
two-unit pattern ``(+1, -1)``. The weights are ``[[0, -0.5], [-0.5, 0]]``, and
presenting ``(+1, +1)``:

===============  =======================  ==============================
route            what happens             where it ends
===============  =======================  ==============================
synchronous      ``(+1, +1)`` becomes     never, measured over any number
                 ``(-1, -1)`` becomes     of passes
                 ``(+1, +1)`` ...
asynchronous     one unit moves, the      the first sweep, at a stored
                 other is then already    pattern (``(+1, -1)`` or
                 content                  ``(-1, +1)``, depending on
                                          which unit was visited first)
===============  =======================  ==============================

Both routes are offered, as :class:`UpdateRule`, because refusing to implement
the broken one would hide the single most instructive fact about this model.
The default is asynchronous, and the enum member for the other says plainly what
it does.

The zero diagonal is load-bearing, and its failure is silent
-------------------------------------------------------------
Leave ``weights[i][i]`` at its Hebbian value and the network stops recalling
anything. The reason is worth stating carefully, because the obvious reason is
wrong.

The diagonal does **not** change the energy landscape. ``s_i * s_i`` is 1 for
every bipolar value, so a self weight contributes the same constant to every
state and reorders nothing. What it changes is the *update*. The weighted sum
the rule reads becomes ``w[i][i] * s_i + sum over j != i``, and the first term
always agrees with the sign the unit already has. Make it large enough and the
unit can never move, so every state is a fixed point and recall returns whatever
it was handed.

Measured on the sixteen-unit fixture in the spec, whose largest absolute row sum
away from the diagonal is 1.3125, a self weight of 3.0 makes a one-flip probe
come straight back with its flip intact. Nothing raises, nothing is infinite,
and the model reports a perfectly ordinary fit.

The negation of a stored pattern is stored too
-----------------------------------------------
Energy is quadratic in the state, so ``energy(-state)`` equals ``energy(state)``
exactly, and every weighted sum reverses along with the state. If a pattern is a
fixed point then so is its negation, at the identical energy. This is not a bug
and no implementation can avoid it: it follows from the form of the energy.
Present a probe more than half wrong and the negation is what comes back.

Spurious minima
---------------
Fixed points that were never stored also exist. The reliable family is the odd
mixture: for three stored patterns, ``sign(first + second + third)`` is stable
whenever the three are not too correlated. Measured on the spec's three
orthogonal sixteen-unit patterns, that mixture settles to itself under thirty
different unit orderings, is neither a stored pattern nor a negation of one, and
sits at energy -4.5 where each real pattern sits at -6.5. Spurious states are
minima; they are just shallower ones.

Capacity, measured rather than quoted
--------------------------------------
The number in the literature is ``0.138 * n_units`` patterns, and it is the
load at which errorless recall breaks down as the network grows without bound.
At any size you can actually run, the collapse is gradual rather than sharp.

Measured here: 100 units, 20 independently drawn sets of random bipolar
patterns per load, presenting each stored pattern back to the network unchanged
and asking whether it comes back exactly. The second column repeats the
experiment with 10 of the 100 units flipped first.

=========  =============  ===========================
load       exact probe    probe with 10 units flipped
=========  =============  ===========================
0.05       1.000          1.000
0.08       1.000          0.994
0.10       0.960          0.945
0.12       0.887          0.829
0.14       0.775          0.689
0.16       0.684          0.584
0.18       0.503          0.372
0.20       0.372          0.258
0.25       0.154          0.058
0.30       0.042          0.007
0.40       0.001          0.000
=========  =============  ===========================

So 0.138 is a fair marker for where it starts going badly and a poor
description of what happens at 100 units, where recall is already imperfect at
0.10 and still working half the time at 0.18. The honest summary is that the
capacity is linear in the number of units with a constant near a seventh, and
that the failure is a gradual erosion rather than a cliff.

Correlated patterns do worse than random ones, which is the practical trap. The
crosstalk between stored patterns is what eats the basins, and patterns that
resemble each other produce more of it.

What this is not
----------------
It is not a classifier, because it has no labels and no target. It is not a
clusterer, because it answers with a state rather than a group. It is a
:class:`~oop_ml.core.base.estimator.Transformer`, because it learns from inputs
alone and rewrites inputs in terms of what it learned, which is exactly what
recall does to a corrupted pattern.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import ClassVar, Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.column import Column, ColumnSource
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import (
    InvalidValuesError,
    NonUniqueFeaturesError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray, IndexArray, array_for_protocol
from oop_ml.core.validation import ValueRole

SYMMETRY_TOLERANCE = 1e-12
"""How far ``weights[i][j]`` may sit from ``weights[j][i]`` and still pass.

A sum of outer products of bipolar vectors is exactly symmetric in float64,
because every entry is a whole number of agreements divided by one common
count, so in practice this is never reached. It is a tolerance rather than an
equality only so that a matrix assembled by some other arithmetic is not
refused over a last-bit difference, and it is tight enough that a genuinely
asymmetric matrix, which would differ by whole units, is still refused.
"""


def check_is_bipolar(values: FloatArray, described_as: str) -> None:
    """Raise unless every value is exactly ``-1`` or exactly ``+1``.

    The one implementation of this library's bipolar rule, so that a pattern
    and a unit column cannot drift apart about what bipolar means. A guard
    function rather than a method, for the same reason
    :func:`~oop_ml.core.validation.check_is_binary` is one: it inspects a
    single array and produces nothing.

    Parameters
    ----------
    values:
        The array to inspect.
    described_as:
        What to call the array in the message, and nothing else. It selects no
        behaviour, which is why it is an ordinary string where an argument that
        decided something would have to be a closed enum.

    Raises
    ------
    InvalidValuesError
        If any value is neither ``-1`` nor ``+1``.
    """
    offending = values[(values != -1.0) & (values != 1.0)]

    if offending.size:
        found = ", ".join(str(value) for value in np.unique(offending)[:5])
        raise InvalidValuesError(
            f"{described_as} must hold only -1 and 1, and holds {found}"
        )


class UpdateRule(StrEnum):
    """How many units a recall pass is allowed to move at a time.

    A closed enum rather than a boolean, because the two members do not stand
    in a default/override relationship: they are different algorithms with
    different guarantees, and the caller should have to name the one they mean.

    Attributes
    ----------
    ASYNCHRONOUS:
        One unit at a time, each reading the state the previous unit left
        behind. **Guaranteed to settle.** Every flip strictly lowers the
        energy and the state space is finite, so the walk cannot cycle. This is
        the default and the only member with a convergence proof behind it.
    SYNCHRONOUS:
        Every unit at once, all reading the same old state. **Not guaranteed to
        settle**, and the failure is a period-two oscillation that runs until
        the pass limit stops it, not a slow convergence. Two units can each
        move in a direction that would have lowered the energy alone while
        together they raise it. Offered because the oscillation is the most
        instructive single fact about this model, and refusing to implement it
        would hide that; not offered as a reasonable default.
    """

    ASYNCHRONOUS = "asynchronous"
    SYNCHRONOUS = "synchronous"


class RecallStop(StrEnum):
    """Which of the two exits a settling took.

    Both are recorded because they mean opposite things about the state that
    came back. A settled recall reached a fixed point and its answer is the
    network's memory; a recall that ran out of passes was still moving, and
    under :attr:`UpdateRule.SYNCHRONOUS` it may have been moving between the
    same two states forever.

    Attributes
    ----------
    SETTLED:
        A whole pass went by without a single unit changing.
    PASS_LIMIT_REACHED:
        ``max_passes`` ran out first, and the state is wherever the walk
        happened to be.
    """

    SETTLED = "settled"
    PASS_LIMIT_REACHED = "pass limit reached"


class BipolarPattern:
    """One state of the network: every unit exactly ``-1`` or exactly ``+1``.

    A Hopfield state that is not bipolar is not a state this model can be in.
    The update rule answers with a sign and the storage rule multiplies pairs of
    values, so a 0.5 somewhere is neither refused nor handled by the
    mathematics, it simply produces a network describing a different problem.
    Making the case unrepresentable rather than handled is the same move
    :class:`~oop_ml.core.data.column.Column` makes for emptiness.

    Built on ``Column``, in the way
    :class:`~oop_ml.core.data.feature.Feature` is, so the coercion, the
    finiteness check, the copy and the freeze all happen in the one place this
    library does them, and only the bipolar rule is added here.

    Parameters
    ----------
    values:
        One value per unit, each ``-1`` or ``+1``. In a fitted network these
        are in :attr:`HopfieldNetwork.unit_names` order.

    Raises
    ------
    InvalidValuesError
        If the values are not a finite one-dimensional array, or if any of them
        is neither ``-1`` nor ``+1``.
    EmptyValuesError
        If there are no units.
    """

    __slots__ = ("_column",)

    def __init__(self, values: ColumnSource) -> None:
        column = Column.of(values, ValueRole.INPUT_VALUES)
        check_is_bipolar(column.values, "a bipolar pattern")
        self._column = column

    @classmethod
    def already_checked(cls, values: FloatArray) -> BipolarPattern:
        """Wrap an array this library itself produced, without re-checking it.

        The counterpart of ``Column.selecting`` and
        ``Predictions.already_checked``. A settling loop writes nothing but
        ``-1`` and ``+1`` into its working buffer, so re-running the bipolar
        check on the result re-establishes what cannot have changed, and a
        recall of any length would pay for it once per pass recorded.

        **This trusts its caller.** Anything from outside the library goes
        through the ordinary constructor.

        Parameters
        ----------
        values:
            A ``(n_units,)`` float array holding only ``-1`` and ``+1``. It is
            frozen in place, so the caller must not write to it afterwards.
        """
        pattern = cls.__new__(cls)
        pattern._column = Column.selecting(values, ValueRole.INPUT_VALUES)

        return pattern

    @property
    def values(self) -> FloatArray:
        """The state as a read-only ``float64`` array, one entry per unit."""
        return self._column.values

    @property
    def column(self) -> Column:
        """The state as a column, which is what makes this a ``HasColumn``."""
        return self._column

    @property
    def n_units(self) -> int:
        """How many units this state covers."""
        return self._column.n_samples

    def flipped(self) -> BipolarPattern:
        """The negation of this state, every unit reversed.

        Worth a method rather than a caller's minus sign, because the negation
        is not an arbitrary transformation here. It is the other pattern the
        network stored without being asked to: the energy is quadratic, so a
        state and its negation sit at exactly the same energy, and one is a
        fixed point precisely when the other is.
        """
        return BipolarPattern.already_checked(-self._column.values)

    def agreements_with(self, other: BipolarPattern) -> int:
        """How many units hold the same value in both states.

        The natural distance between two bipolar states, counted rather than
        measured. ``n_units`` means identical and ``0`` means one is the
        negation of the other, so the number says which attractor a recall
        landed in as well as how far off it was.

        Raises
        ------
        InvalidValuesError
            If the two states cover different numbers of units.
        """
        if other.n_units != self.n_units:
            raise InvalidValuesError(
                f"cannot compare a {self.n_units} unit state with a "
                f"{other.n_units} unit one"
            )

        return int(np.count_nonzero(self._column.values == other.values))

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter.

        See :func:`~oop_ml.core.types.array_for_protocol` for the contract and
        the aliasing it exists to prevent.
        """
        return array_for_protocol(self._column.values, dtype, copy)

    def __eq__(self, other: object) -> bool:
        """Whether the two states are the same state, as one verdict.

        :class:`~oop_ml.core.data.predictions.Predictions` compares element by
        element and argues that anything numpy treats as an array must, and
        this deliberately does not follow it. The question asked of a recalled
        state is "is this the pattern that was stored", which is one question
        about a whole state rather than sixteen questions about units, and a
        row of booleans is not an answer to it.
        :class:`~oop_ml.core.data.feature.Feature` already reads a named vector
        the same way. The element-by-element comparison is still one
        ``np.asarray`` away for anyone who wants it.
        """
        if not isinstance(other, BipolarPattern):
            return NotImplemented

        return np.array_equal(self._column.values, other.values)

    def __hash__(self) -> int:
        # The values are frozen and are whole numbers, so the bytes are a
        # stable content hash for the object's life, and -0.0 cannot occur.
        return hash(self._column.values.tobytes())

    def __len__(self) -> int:
        return self.n_units

    def __repr__(self) -> str:
        return f"BipolarPattern(n_units={self.n_units})"


class HebbianWeights:
    """A symmetric matrix of connection strengths, addressable by unit name.

    Everything a fitted network knows. As a bare array it is a table addressed
    by two positions, neither of which is named, which is exactly the habit
    this library exists to avoid at the point a reader looks.

    What is checked here, and what deliberately is not
    --------------------------------------------------
    Square, finite and symmetric are checked, because all three are structural
    and all three are silent when broken. An asymmetric matrix still settles
    sometimes, still answers, and has no convergence argument behind it at all.

    The **zero diagonal is not checked**, and that is a decision rather than an
    oversight. A self connection is what the storage rule is responsible for
    removing, and refusing it here would move the lesson out of the rule and
    into the container, leaving nothing able to demonstrate what a self
    connection does. :attr:`has_self_connections` reports it instead.

    Parameters
    ----------
    values:
        ``(n_units, n_units)``. Copied and frozen, so a constructed matrix
        cannot change under its holder.
    unit_names:
        One per row, in row order, and unique. The same names index the
        columns, since the matrix is symmetric.

    Raises
    ------
    InvalidValuesError
        If the array is not a square two-dimensional matrix, if it holds a
        non-finite value, if it is not symmetric, or if there is not exactly one
        name per row.
    NonUniqueFeaturesError
        If two units share a name.
    """

    __slots__ = ("_position_of", "_unit_names", "_values")

    def __init__(self, values: FloatArray, unit_names: Sequence[str]) -> None:
        as_array = np.array(values, dtype=np.float64)

        if as_array.ndim != 2 or as_array.shape[0] != as_array.shape[1]:
            raise InvalidValuesError(
                f"a weight matrix is square and two-dimensional, got shape "
                f"{as_array.shape}"
            )

        if not np.all(np.isfinite(as_array)):
            raise InvalidValuesError("a weight matrix holds only finite values")

        if not np.allclose(as_array, as_array.T, rtol=0.0, atol=SYMMETRY_TOLERANCE):
            raise InvalidValuesError(
                "a weight matrix must be symmetric; without that the energy "
                "argument that makes settling guaranteed does not hold"
            )

        if as_array.shape[0] != len(unit_names):
            raise InvalidValuesError(
                f"{len(unit_names)} unit names for {as_array.shape[0]} rows"
            )

        position_of: dict[str, int] = {}
        for position, name in enumerate(unit_names):
            if name in position_of:
                raise NonUniqueFeaturesError(f"duplicate unit name: {name!r}")
            position_of[name] = position

        as_array.setflags(write=False)

        self._values = as_array
        self._unit_names = tuple(unit_names)
        self._position_of = position_of

    @property
    def values(self) -> FloatArray:
        """The matrix as a read-only array, for the kernels that want it."""
        return self._values

    @property
    def unit_names(self) -> tuple[str, ...]:
        """The units, in the order the rows and columns are in."""
        return self._unit_names

    @property
    def n_units(self) -> int:
        """How many units this connects."""
        return len(self._unit_names)

    @property
    def has_self_connections(self) -> bool:
        """Whether any unit is wired to itself.

        Should always be ``False`` for weights a storage rule produced. See the
        module docstring for what a ``True`` here does to recall, and why
        nothing else about the model looks wrong when it happens.
        """
        return bool(np.any(self._values.diagonal() != 0.0))

    def weight_between(self, first_unit: str, second_unit: str) -> float:
        """The strength of the connection joining two named units.

        The reason the names are carried: "unit ``top_left`` and unit
        ``bottom_right`` agreed in every stored pattern, so their weight is
        0.1875" is a sentence, and ``weights[0][15]`` is not.

        Raises
        ------
        InvalidValuesError
            If either name is not one of this matrix's units.
        """
        return float(
            self._values[
                self._position_for(first_unit), self._position_for(second_unit)
            ]
        )

    def weighted_sums_for(self, state: BipolarPattern) -> FloatArray:
        """What each unit's connections are telling it, given this state.

        One number per unit, ``sum over j of weights[i][j] * state[j]``. The
        update rule is the sign of this, and the literature calls it the local
        field. Because the diagonal is zero, a unit's own value contributes
        nothing to its own sum, which is precisely the fact the settling proof
        needs.

        Raises
        ------
        InvalidValuesError
            If the state does not cover exactly these units.
        """
        self.check_covers(state)

        return self._values @ state.values

    def energy_of(self, state: BipolarPattern) -> float:
        """``-0.5 * state . weights . state``, the quantity recall lowers.

        Low energy means many pairs of units agree with the sign of the weight
        joining them, which is what "this state resembles the stored patterns"
        amounts to. Asynchronous recall walks downhill in this number and stops
        at a local minimum.

        Quadratic in the state, so ``energy_of(state.flipped())`` equals
        ``energy_of(state)`` exactly, which is why every stored pattern brings
        its negation along.

        Raises
        ------
        InvalidValuesError
            If the state does not cover exactly these units.
        """
        self.check_covers(state)

        return float(-0.5 * state.values @ self._values @ state.values)

    def check_covers(self, state: BipolarPattern) -> None:
        """Raise unless the state has one value per unit of this matrix.

        Raises
        ------
        InvalidValuesError
            If the widths disagree.
        """
        if state.n_units != self.n_units:
            raise InvalidValuesError(
                f"these weights join {self.n_units} units; got a state over "
                f"{state.n_units}"
            )

    def _position_for(self, unit_name: str) -> int:
        """Where the named unit sits, raising if it is not one of them."""
        if unit_name not in self._position_of:
            known = ", ".join(self._unit_names)
            raise InvalidValuesError(f"no unit named {unit_name!r}. Units: {known}")

        return self._position_of[unit_name]

    def __array__(self, dtype=None, copy=None) -> FloatArray:
        """Hand numpy this wrapper, honouring the copy parameter."""
        return array_for_protocol(self._values, dtype, copy)

    def __len__(self) -> int:
        return self.n_units

    def __repr__(self) -> str:
        return f"HebbianWeights(n_units={self.n_units})"


class RecallPass:
    """One sweep of the update rule over the units.

    Parameters
    ----------
    pass_number:
        Counting from 1, matching what :attr:`RecallWalk.passes_run` reports.
    state_before:
        The state this sweep started from. The first sweep starts at the probe,
        which is worth seeing rather than assuming.
    state_after:
        Where it left the network.
    energy_before:
        The energy of ``state_before``. Carried rather than derived so that one
        pass reads on its own, without its neighbours.
    energy_after:
        The energy of ``state_after``. Never above ``energy_before`` under
        :attr:`UpdateRule.ASYNCHRONOUS`; under
        :attr:`UpdateRule.SYNCHRONOUS` it can be, which is the whole difference
        between the two rules made into a number.
    """

    __slots__ = (
        "_energy_after",
        "_energy_before",
        "_pass_number",
        "_state_after",
        "_state_before",
    )

    def __init__(
        self,
        pass_number: int,
        state_before: BipolarPattern,
        state_after: BipolarPattern,
        energy_before: float,
        energy_after: float,
    ) -> None:
        self._pass_number = pass_number
        self._state_before = state_before
        self._state_after = state_after
        self._energy_before = energy_before
        self._energy_after = energy_after

    @property
    def pass_number(self) -> int:
        """Which sweep this was, counting from 1."""
        return self._pass_number

    @property
    def state_before(self) -> BipolarPattern:
        """The state this sweep started from."""
        return self._state_before

    @property
    def state_after(self) -> BipolarPattern:
        """The state this sweep ended at."""
        return self._state_after

    @property
    def energy_before(self) -> float:
        """The energy this sweep started at."""
        return self._energy_before

    @property
    def energy_after(self) -> float:
        """The energy this sweep ended at."""
        return self._energy_after

    @property
    def n_units_changed(self) -> int:
        """How many units this sweep moved.

        Zero is the stopping condition: a sweep that moves nothing has found a
        state no single unit disagrees with, which is a fixed point.
        """
        return self._state_before.n_units - self._state_before.agreements_with(
            self._state_after
        )

    @property
    def changed_anything(self) -> bool:
        """Whether this sweep moved a unit at all."""
        return self.n_units_changed > 0

    def __repr__(self) -> str:
        return (
            f"RecallPass({self._pass_number}, {self.n_units_changed} units moved, "
            f"energy {self._energy_after:.4f})"
        )


class RecallWalk:
    """Every sweep one recall took, and the state it left the network in.

    The observed twin of :meth:`HopfieldNetwork.recall`, in the sense
    :mod:`oop_ml.core.observation` sets out. ``recall`` keeps a single working
    buffer and returns the answer; this keeps every state the network passed
    through, which is what a settling animation, a convergence plot or an
    argument about the two update rules actually needs.

    Iterate it to walk the sweeps in order.

    Parameters
    ----------
    passes:
        The sweeps, in order. Empty is impossible, since even a probe that is
        already a fixed point costs one sweep to discover that.
    initial_energy:
        The energy of the probe, before anything moved. Held separately because
        it is the only energy no pass reports as its own ``energy_after``.
    stopped_because:
        Which exit the settling took.
    """

    __slots__ = ("_initial_energy", "_passes", "_stopped_because")

    def __init__(
        self,
        passes: Sequence[RecallPass],
        initial_energy: float,
        stopped_because: RecallStop,
    ) -> None:
        if not passes:
            raise InvalidValuesError("a recall walk records at least one pass")

        self._passes = tuple(passes)
        self._initial_energy = initial_energy
        self._stopped_because = stopped_because

    @property
    def result(self) -> BipolarPattern:
        """What the efficient route would have returned: the settled state."""
        return self._passes[-1].state_after

    @property
    def settled_state(self) -> BipolarPattern:
        """Where the network ended up, under the name a caller will reach for."""
        return self.result

    @property
    def probe(self) -> BipolarPattern:
        """The state the recall was handed."""
        return self._passes[0].state_before

    @property
    def stopped_because(self) -> RecallStop:
        """Which exit the settling took."""
        return self._stopped_because

    @property
    def settled(self) -> bool:
        """Whether a whole sweep went by without a unit moving.

        ``False`` means the pass limit stopped it. Under
        :attr:`UpdateRule.SYNCHRONOUS` that is the ordinary outcome of an
        oscillation rather than evidence that a few more passes would have
        helped.
        """
        return self._stopped_because is RecallStop.SETTLED

    @property
    def passes_run(self) -> int:
        """How many sweeps were taken, including the one that moved nothing."""
        return len(self._passes)

    @property
    def initial_energy(self) -> float:
        """The energy of the probe, before any unit moved."""
        return self._initial_energy

    @property
    def energies(self) -> FloatArray:
        """The energy after each sweep, in order.

        The series worth plotting. Under asynchronous update it is
        non-increasing, by the argument in the module docstring, and it is flat
        from the first sweep that changes nothing onward. Under synchronous
        update on an oscillating probe it alternates between two values and
        never stops.
        """
        return np.array(
            [recall_pass.energy_after for recall_pass in self._passes],
            dtype=np.float64,
        )

    def __iter__(self) -> Iterator[RecallPass]:
        return iter(self._passes)

    def __len__(self) -> int:
        return len(self._passes)

    def __repr__(self) -> str:
        return (
            f"RecallWalk({self.passes_run} passes, {self._stopped_because.value}, "
            f"energy {self.energies[-1]:.4f})"
        )


class HopfieldNetwork(Transformer[Sequence[Feature]]):
    """Remember bipolar patterns in a weight matrix and settle back into them.

    Rows are patterns and columns are units, which is the same arrangement
    every other model here uses and means something different because of it. A
    :class:`~oop_ml.core.data.feature.Feature` is one *unit*, holding the value
    that unit takes in each pattern, and a row across all the features is one
    whole pattern to be remembered.

    Parameters
    ----------
    max_passes:
        A ceiling on the sweeps one recall may take. Under
        :attr:`UpdateRule.ASYNCHRONOUS` settling is guaranteed, so this is a
        guard against a pathological case rather than the normal stopping
        condition. Under :attr:`UpdateRule.SYNCHRONOUS` it is the only thing
        that ever stops an oscillation, so it is load-bearing there.
    update_rule:
        One unit at a time or all of them at once. See :class:`UpdateRule`; the
        default is the one with a convergence proof.
    random_seed:
        Fixes the order the units are visited in during an asynchronous recall.
        A fresh order is drawn for each sweep, and a fresh generator is built
        for each recall, so a seeded network answers the same question the same
        way every time. Left as ``None`` the order varies between calls, which
        is legitimate and occasionally visible: a probe sitting equally close
        to two attractors really can settle into either, and the order is what
        decides.

    Raises
    ------
    pydantic.ValidationError
        If ``max_passes`` is below 1. Field bounds are pydantic's to enforce,
        so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    max_passes: int = Field(default=20, ge=1)
    update_rule: UpdateRule = UpdateRule.ASYNCHRONOUS
    random_seed: int | None = None

    LEARNED_STATE: ClassVar[tuple[str, ...]] = ("_weights", "_n_stored_patterns")

    _weights: HebbianWeights | None = PrivateAttr(default=None)
    _n_stored_patterns: int | None = PrivateAttr(default=None)

    @property
    def weights(self) -> HebbianWeights:
        """Everything this network learned.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._weights is not None
        return self._weights

    @property
    def unit_names(self) -> tuple[str, ...]:
        """The units, in the order a :class:`BipolarPattern` must be in.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.weights.unit_names

    @property
    def n_units(self) -> int:
        """How many units the network has.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.weights.n_units

    @property
    def n_stored_patterns(self) -> int:
        """How many patterns were stored.

        A count, and deliberately not the patterns themselves. A Hopfield
        network does not keep what it was shown; the memory is the weights, and
        an attribute holding the originals would let a caller read back
        something the model genuinely no longer has.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._n_stored_patterns is not None
        return self._n_stored_patterns

    @property
    def load(self) -> float:
        """Patterns stored per unit, which is the number that predicts failure.

        The literature's threshold is 0.138 and it is an asymptotic one. See
        the module docstring for what was measured at 100 units, where recall
        is already imperfect at 0.10 and still working half the time at 0.18.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self.n_stored_patterns / self.n_units

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Store the patterns in ``input_values``.

        One shot and no iteration: the patterns are read, one matrix is formed,
        and that is the whole of learning here. The plumbing around
        :meth:`_stored_weights` is the validation, the naming, and the ordering
        rule this library follows everywhere, which is that ``_stored_weights``
        does not set ``_fitted`` and this method does, last, after the weights
        are stored.

        Parameters
        ----------
        input_values:
            One feature per unit, each holding that unit's value in every
            pattern. Every value must be ``-1`` or ``+1``, and there must be at
            least two units, since a lone unit has nothing to be connected to.

        Returns
        -------
        Self

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        NonEqualArrayLengthError
            If the features are different lengths.
        NonUniqueFeaturesError
            If two features share a name.
        TooFewValuesError
            If there are fewer than two units.
        InvalidValuesError
            If any value is neither ``-1`` nor ``+1``.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_features < 2:
            raise TooFewValuesError(
                f"a network needs at least two units to have a connection; got "
                f"{feature_set.n_features}"
            )

        self._check_units_are_bipolar(feature_set)

        unit_names = tuple(feature.name for feature in feature_set)
        rows = rows_of(
            np.column_stack([feature.values for feature in feature_set]), unit_names
        )
        patterns = self._patterns_of(rows)

        # Nothing is committed until every step has succeeded, so a failed
        # refit leaves the previous fit intact rather than half replaced.
        weights = HebbianWeights(self._stored_weights(patterns), unit_names)

        self._weights = weights
        self._n_stored_patterns = len(patterns)
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Recall each supplied pattern and hand back what the network settled on.

        The named route, and the one a caller reaches for. Each row is settled
        independently, so a call may settle some rows and hit the pass limit on
        others. If which of those happened matters, take the rows through
        :meth:`patterns_in` and :meth:`recall_walk` instead, where each recall
        reports its own exit.

        Every fitted unit must be present, by name rather than position. A
        missing one leaves its row of the weight matrix with nothing to read,
        and an extra one has no row at all.

        Parameters
        ----------
        input_values:
            One feature per fitted unit, in any order, each holding that unit's
            value in every pattern to be recalled. Bipolar, like the patterns
            stored.

        Returns
        -------
        list[Feature]
            One feature per unit, named as the fit named them and in the fit's
            order, holding the settled value of that unit for each supplied
            pattern.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or if any
            value is neither ``-1`` nor ``+1``.
        """
        patterns = self.patterns_in(input_values)

        return self.features_from([self.recall(pattern) for pattern in patterns])

    def recall(self, pattern: BipolarPattern) -> BipolarPattern:
        """Settle from ``pattern`` into the nearest thing the network remembers.

        The efficient route. One working buffer is updated in place, sweep after
        sweep, until a whole sweep moves nothing or ``max_passes`` runs out.
        Nothing about the trajectory is kept; :meth:`recall_walk` is the route
        that keeps it.

        Where it lands is a fixed point of the update rule. For a lightly loaded
        network that is one of the stored patterns or its negation. It may also
        be a spurious minimum, which is a state nobody stored that no single
        unit disagrees with, and the model has no way to tell you which it
        found, since it kept no copy of what it stored.

        Parameters
        ----------
        pattern:
            The probe, over the fitted units in :attr:`unit_names` order.

        Returns
        -------
        BipolarPattern
            The state the network settled on, or the state it had reached when
            the pass limit stopped it.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the probe does not cover exactly the fitted units.
        """
        self._check_fitted()
        weights = self.weights
        weights.check_covers(pattern)

        generator = np.random.default_rng(self.random_seed)
        state = np.array(pattern.values, dtype=np.float64)

        for _ in range(self.max_passes):
            if not self._one_pass(state, weights, generator):
                break

        return BipolarPattern.already_checked(state)

    def recall_walk(self, pattern: BipolarPattern) -> RecallWalk:
        """Settle from ``pattern``, keeping every sweep and every energy.

        The observed twin of :meth:`recall`, and the two share the update rule
        itself rather than reimplementing it, so the only difference between
        them is what is written down. The spec asserts they agree, which is the
        rule for every observed pair in this library.

        Free to be slower and to allocate freely, because it is called
        deliberately, on a pattern sized for looking at.

        Parameters
        ----------
        pattern:
            The probe, over the fitted units in :attr:`unit_names` order.

        Returns
        -------
        RecallWalk
            One :class:`RecallPass` per sweep, the energy before and after each,
            the exit that was taken, and the settled state as ``result``.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the probe does not cover exactly the fitted units.
        """
        self._check_fitted()
        weights = self.weights
        weights.check_covers(pattern)

        generator = np.random.default_rng(self.random_seed)
        state = np.array(pattern.values, dtype=np.float64)

        initial_energy = weights.energy_of(pattern)
        energy_before = initial_energy
        state_before = pattern

        passes: list[RecallPass] = []
        stopped_because = RecallStop.PASS_LIMIT_REACHED

        for pass_number in range(1, self.max_passes + 1):
            moved = self._one_pass(state, weights, generator)
            state_after = BipolarPattern.already_checked(state.copy())
            energy_after = weights.energy_of(state_after)

            passes.append(
                RecallPass(
                    pass_number,
                    state_before,
                    state_after,
                    energy_before,
                    energy_after,
                )
            )

            state_before = state_after
            energy_before = energy_after

            if not moved:
                stopped_because = RecallStop.SETTLED
                break

        return RecallWalk(passes, initial_energy, stopped_because)

    def energy_of(self, state: BipolarPattern) -> float:
        """``-0.5 * state . weights . state``, the quantity recall lowers.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the state does not cover exactly the fitted units.
        """
        return self.weights.energy_of(state)

    def is_fixed_point(self, state: BipolarPattern) -> bool:
        """Whether no single unit disagrees with the state it is in.

        The definition of a memory in this model, and it is checkable in one
        pass rather than by settling. A unit is content when its own value has
        the same sign as the weighted sum arriving at it, or when that sum is
        exactly zero, which the update rule treats as leaving the unit alone. So
        the whole state is a fixed point when every product of value and
        weighted sum is non-negative.

        That is a statement about *every* update order at once, which is what
        makes it stronger than one settling run: if no unit would move, then no
        sequence of units moves either.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the state does not cover exactly the fitted units.
        """
        weighted_sums = self.weights.weighted_sums_for(state)

        return bool(np.all(state.values * weighted_sums >= 0.0))

    def patterns_in(self, input_values: Sequence[Feature]) -> list[BipolarPattern]:
        """The rows of ``input_values`` as patterns, in fitted unit order.

        The boundary between the named world outside and the positional one
        inside. A weight matrix is indexed by position on both axes, so a
        :class:`BipolarPattern` is a plain vector of units; this is what puts a
        caller's features into the order that vector has to be in, and refuses
        the call if they are not the fitted units.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones, or if any
            value is neither ``-1`` nor ``+1``.
        """
        self._check_fitted()
        self._check_features_match(input_values)

        feature_set = FeatureSet.matching(self.unit_names, list(input_values))
        self._check_units_are_bipolar(feature_set)

        rows = rows_of(
            np.column_stack([feature.values for feature in feature_set]),
            self.unit_names,
        )

        return self._patterns_of(rows)

    def features_from(self, patterns: Sequence[BipolarPattern]) -> list[Feature]:
        """The inverse of :meth:`patterns_in`: patterns back to named units.

        One feature per fitted unit, in fitted order, each holding that unit's
        value across the supplied patterns.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If no patterns are supplied, or if any of them does not cover
            exactly the fitted units.
        """
        self._check_fitted()

        if not patterns:
            raise InvalidValuesError("at least one pattern is required")

        for pattern in patterns:
            self.weights.check_covers(pattern)

        values = np.array([pattern.values for pattern in patterns], dtype=np.float64)

        return [
            Feature(name, values[:, position])
            for position, name in enumerate(self.unit_names)
        ]

    def _stored_weights(self, patterns: Sequence[BipolarPattern]) -> FloatArray:
        """Turn the patterns to be remembered into the weight matrix.

        The concept, and the only part of this class that is the learning rule
        rather than the bookkeeping.

        Parameters
        ----------
        patterns:
            The patterns to store, each already validated as bipolar and each
            covering the same units in the same order. At least one, and the
            width is ``patterns[0].n_units``.

        Returns
        -------
        FloatArray
            ``(n_units, n_units)``, symmetric, with every diagonal entry zero.
            Do not set ``_fitted`` here; ``fit`` does that, afterwards.

        Notes
        -----
        **The rule.** Hebb's proposal is that two units which are active
        together should be wired together. In bipolar terms "together" means
        agreeing, so for a single pattern the weight joining units ``i`` and
        ``j`` is the product of their values, which is ``+1`` when they agree
        and ``-1`` when they do not. For several patterns the contributions add,
        and the sum is divided by the number of units:

            weights[i][j]  =  (1 / n_units)
                              * sum over patterns of pattern[i] * pattern[j]

        Then, for every ``i``:

            weights[i][i]  =  0

        **Why the ``1 / n_units``.** It makes the weighted sum a unit reads,
        ``sum over j of weights[i][j] * state[j]``, an average rather than a
        total, so it stays the same size as the network grows. It changes no
        sign anywhere and therefore changes no answer, since the update rule
        reads only the sign. It scales every energy by the same factor too, so
        comparisons between states are untouched. It is a choice of units, and
        the conventional one.

        **Why the zeroed diagonal, which is the part to get right.** Two
        separate things go wrong without it, and only one of them is obvious.

        The one that is not obvious first. The energy is
        ``-0.5 * state . weights . state``, and a diagonal entry contributes
        ``-0.5 * weights[i][i] * state[i] * state[i]``. Every bipolar value
        squares to 1, so that term is the same constant in every state: the
        diagonal changes the energy of everything equally and reorders nothing.
        The landscape is untouched, which is exactly why the bug is quiet.

        What it does change is the update. The weighted sum a unit reads becomes
        ``weights[i][i] * state[i] + sum over j != i``, and the first term
        always agrees with the sign the unit currently holds, so it is a vote
        for staying put that grows with the self weight. Make it large enough
        relative to the rest of the row and the unit can never move at all, at
        which point every state is a fixed point and recall hands back whatever
        it was given. The network still fits, still transforms, and remembers
        nothing.

        It also breaks the settling proof, which needs the weighted sum to be
        free of ``state[i]`` in order to say that one flip changes the energy by
        ``-2 * |weighted_sum|``.

        **Symmetry comes for free**, since ``pattern[i] * pattern[j]`` and
        ``pattern[j] * pattern[i]`` are the same number, and
        :class:`HebbianWeights` checks it on the way in. It is not decoration:
        the settling proof needs it, so an asymmetric matrix would be a network
        with no guarantee that recall ever stops.

        **Do not read ``self.n_units`` or ``self.weights`` here.** Both are
        guarded by ``_check_fitted``, and this runs during ``fit``, before the
        model is marked fitted, so both would raise ``NotFittedError`` from
        inside the method whose job is to make them available. The width you
        want is ``patterns[0].n_units``. This trap has been walked into five
        times in this repository already.

        A plain double loop over the units is a perfectly good answer here.
        """
        raise NotImplementedError

    def _one_pass(
        self,
        state: FloatArray,
        weights: HebbianWeights,
        generator: np.random.Generator,
    ) -> bool:
        """Run one sweep of the configured update rule over ``state`` in place.

        Shared by :meth:`recall` and :meth:`recall_walk`, which is what keeps
        the two routes one calculation observed two ways rather than two
        implementations of it. What differs between them is only what is
        written down around this.

        Returns
        -------
        bool
            Whether any unit changed. ``False`` is the stopping condition, and
            it means the state is a fixed point of the rule.
        """
        if self.update_rule is UpdateRule.SYNCHRONOUS:
            return self._synchronous_pass(state, weights)

        return self._asynchronous_pass(state, weights, generator)

    @staticmethod
    def _asynchronous_pass(
        state: FloatArray,
        weights: HebbianWeights,
        generator: np.random.Generator,
    ) -> bool:
        """Visit the units one at a time, in a random order, updating in place.

        Each unit reads the state the previous unit left behind, which is what
        the settling proof assumes and what makes every flip strictly lower the
        energy. A weighted sum of exactly zero leaves the unit alone, which is
        the convention that keeps a tie from being a coin toss.
        """
        matrix = weights.values
        moved = False

        for unit in HopfieldNetwork._visiting_order(state.size, generator):
            weighted_sum = float(matrix[unit] @ state)

            if weighted_sum > 0.0:
                updated = 1.0
            elif weighted_sum < 0.0:
                updated = -1.0
            else:
                updated = state[unit]

            if updated != state[unit]:
                state[unit] = updated
                moved = True

        return moved

    @staticmethod
    def _synchronous_pass(state: FloatArray, weights: HebbianWeights) -> bool:
        """Update every unit at once from the state they all start in.

        No unit sees any other unit's move, which is precisely what the
        settling proof forbids, and why this can oscillate with period two
        forever. See :class:`UpdateRule`.
        """
        weighted_sums = weights.values @ state
        updated = np.where(weighted_sums != 0.0, np.sign(weighted_sums), state)
        moved = not np.array_equal(updated, state)
        state[:] = updated

        return moved

    @staticmethod
    def _visiting_order(n_units: int, generator: np.random.Generator) -> IndexArray:
        """The order this sweep visits the units in.

        A fresh permutation per sweep, drawn from the recall's own generator,
        so a seeded network answers reproducibly while the order is still not
        the same every sweep. Visiting the units in their fitted order every
        time is also a valid asynchronous rule, and a worse one: it lets the
        answer depend on how the caller happened to name the columns.
        """
        return generator.permutation(n_units)

    @staticmethod
    def _patterns_of(rows: RowBlock) -> list[BipolarPattern]:
        """Each row of the block as a pattern, already checked as bipolar.

        The rows come from features this class has already checked, so the
        patterns skip the copy and the re-validation that would re-establish
        what cannot have changed on the way through a column stack.
        """
        return [
            BipolarPattern.already_checked(np.array(row, dtype=np.float64))
            for row in rows
        ]

    @staticmethod
    def _check_units_are_bipolar(feature_set: FeatureSet) -> None:
        """Raise unless every unit holds only ``-1`` and ``+1``.

        Checked per feature rather than per pattern so that the error names the
        unit at fault, which is the column a caller can go and look at.
        """
        for feature in feature_set:
            check_is_bipolar(feature.values, f"unit {feature.name!r}")

    def _check_features_match(self, input_values: Sequence[Feature]) -> None:
        """Raise unless the supplied features are exactly the fitted units."""
        supplied = {feature.name for feature in input_values}
        fitted = set(self.unit_names)

        if supplied != fitted:
            raise InvalidValuesError(
                f"expected exactly the fitted units {sorted(fitted)}; "
                f"got {sorted(supplied)}"
            )

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"HopfieldNetwork(update_rule={self.update_rule.value}, unfitted)"

        return (
            f"HopfieldNetwork(n_units={self.n_units}, "
            f"n_stored_patterns={self.n_stored_patterns}, load={self.load:.4f})"
        )
