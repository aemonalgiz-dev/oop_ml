"""What a Markov chain learns and answers with, each a table bound to state names.

Why three objects rather than one array
---------------------------------------
A transition table held as a bare ``(n, n)`` array leaves three facts for the
caller to carry: which row is which state, that rows are where a step starts
and columns where it ends, and that every row sums to one. Each object here
carries those facts itself, so ``table["sunny"]["rainy"]`` reads the way the
question is asked and a table whose rows do not sum to one cannot exist.

The counts are kept apart from the table because the table is the counts plus
a decision, the smoothing, and a caller comparing two smoothings wants the one
thing that did not change between them. A fitted chain stores only the counts,
and derives the table from them and its own smoothing whenever it is asked.

The three refusals that matter
------------------------------
A :class:`TransitionCounts` refuses anything that is not a whole number of at
least zero, because a count of 2.5 or of -1 is not an observation. A
:class:`TransitionMatrix` refuses a row that does not sum to one, which includes
a row of zeros: a state with nowhere to go is not a state of a Markov chain.
And a :class:`StateDistribution` refuses numbers that do not sum to one, to
within the rounding a matrix power leaves behind.

Why a tolerance on the sums
---------------------------
Dividing a row by its total and adding the quotients back up does not return
exactly one in float64, and a thousand steps of a matrix power accumulate the
same rounding. :data:`SUM_TOLERANCE` is 1e-9, which admits that rounding with
room to spare and still refuses a row that sums to 0.999, which is a bug.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt
from numpy.typing import DTypeLike

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    ShapeMismatchError,
    TooFewValuesError,
)
from oop_ml.core.types import FloatArray, array_for_protocol

SUM_TOLERANCE = 1e-9
"""How far from one a sum of probabilities may be and still count as one."""


def checked_states(states: Sequence[str]) -> tuple[str, ...]:
    """Read a sequence of state names, refusing anything that cannot name one.

    Raises
    ------
    InvalidValuesError
        If ``states`` is a single string, since a string is a sequence of
        one-character strings and would otherwise name its letters; or if a
        name is not a non-empty string; or if a name repeats.
    EmptyValuesError
        If there are no states at all.
    """
    if isinstance(states, str):
        raise InvalidValuesError(
            f"the states are a sequence of names, not one string; got {states!r}"
        )

    names = tuple(states)
    if not names:
        raise EmptyValuesError("a table over states needs at least one state")

    for name in names:
        if not isinstance(name, str) or not name:
            raise InvalidValuesError(
                f"a state is named by a non-empty string, got {name!r}"
            )

    if len(set(names)) != len(names):
        repeated = sorted({name for name in names if names.count(name) > 1})
        raise InvalidValuesError(f"each state is named once; {repeated} repeat")

    return names


def _position_in(positions: dict[str, int], state: object) -> int:
    """Where one named state sits, or a refusal naming the states that exist.

    Raises
    ------
    InvalidValuesError
        If ``state`` is not one of the known names.
    """
    if not isinstance(state, str) or state not in positions:
        raise InvalidValuesError(
            f"{state!r} is not a state here; the states are {list(positions)}"
        )
    return positions[state]


def _as_float_block(values: object, role: str) -> FloatArray:
    """A private float64 copy of ``values``, refused in the library's words."""
    try:
        return np.array(values, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as error:
        raise InvalidValuesError(
            f"{role} could not be read as numbers: {error}"
        ) from None


class StateDistribution:
    """How probable each named state is, the probabilities summing to one.

    What a chain answers when asked where a walk will be: after some number of
    steps, in the long run, or one step on from a given state.

    Parameters
    ----------
    states:
        The names, in the order the probabilities are given.
    probabilities:
        One per state, each at least zero, summing to one within
        :data:`SUM_TOLERANCE`.

    Raises
    ------
    EmptyValuesError
        If there are no states.
    InvalidValuesError
        If a name is repeated or is not a non-empty string, if ``states`` is a
        single string, or if the probabilities are not finite, are negative, or
        do not sum to one.
    ShapeMismatchError
        If there is not exactly one probability per state.
    """

    __slots__ = ("_positions", "_states", "_values")

    def __init__(self, states: Sequence[str], probabilities: object) -> None:
        names = checked_states(states)
        values = _as_float_block(probabilities, "a distribution's probabilities")

        if values.shape != (len(names),):
            raise ShapeMismatchError(
                f"a distribution over {len(names)} states needs one probability "
                f"per state, got an array shaped {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise InvalidValuesError("a probability must be finite")
        if np.any(values < 0.0):
            raise InvalidValuesError(
                f"a probability cannot be negative, got {float(np.min(values))}"
            )
        total = float(np.sum(values))
        if abs(total - 1.0) > SUM_TOLERANCE:
            raise InvalidValuesError(
                f"the probabilities of a distribution sum to one, these sum to {total}"
            )

        values.setflags(write=False)
        self._states = names
        self._positions = {name: position for position, name in enumerate(names)}
        self._values = values

    @classmethod
    def concentrated_on(cls, states: Sequence[str], state: str) -> StateDistribution:
        """Certainty of one state, which is where a walk from that state begins.

        Raises
        ------
        InvalidValuesError
            If ``state`` is not among ``states``.
        """
        names = checked_states(states)
        position = _position_in(
            {name: index for index, name in enumerate(names)}, state
        )
        values = np.zeros(len(names), dtype=np.float64)
        values[position] = 1.0
        return cls(names, values)

    @property
    def states(self) -> tuple[str, ...]:
        """The state names, in the order the probabilities are held."""
        return self._states

    @property
    def n_states(self) -> int:
        """How many states the distribution is over."""
        return len(self._states)

    @property
    def values(self) -> FloatArray:
        """The probabilities, frozen, in the order of :attr:`states`."""
        return self._values

    def probability_of(self, state: str) -> float:
        """How probable one named state is.

        Raises
        ------
        InvalidValuesError
            If ``state`` is not one of this distribution's states.
        """
        return float(self._values[_position_in(self._positions, state)])

    def in_order_of(self, states: Sequence[str]) -> StateDistribution:
        """The same distribution, with its states put in another order.

        Raises
        ------
        InvalidValuesError
            If ``states`` does not name exactly this distribution's states.
        """
        names = checked_states(states)
        if set(names) != set(self._states):
            raise InvalidValuesError(
                f"a distribution over {list(self._states)} cannot be read over "
                f"{list(names)}"
            )
        return StateDistribution(
            names, self._values[[self._positions[name] for name in names]]
        )

    def __getitem__(self, state: str) -> float:
        return self.probability_of(state)

    def __len__(self) -> int:
        return len(self._states)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __repr__(self) -> str:
        return f"StateDistribution(states={self._states!r})"


class TransitionCounts:
    """How many times each state was seen to follow each other, by name.

    Row ``i`` column ``j`` counts the steps observed from state ``i`` to state
    ``j``. Nothing here has been divided or smoothed; it is the evidence, and
    :meth:`transition_matrix` is what a chain makes of it.

    Parameters
    ----------
    states:
        The names, in the order the rows and columns are given.
    counts:
        ``(n_states, n_states)``, whole numbers of at least zero.

    Raises
    ------
    EmptyValuesError
        If there are no states.
    InvalidValuesError
        If a name is repeated or is not a non-empty string, or if a count is
        not finite, is negative, or is not a whole number.
    ShapeMismatchError
        If the counts are not one row and one column per state.
    """

    __slots__ = ("_positions", "_states", "_values")

    def __init__(self, states: Sequence[str], counts: object) -> None:
        names = checked_states(states)
        values = _as_float_block(counts, "transition counts")

        if values.shape != (len(names), len(names)):
            raise ShapeMismatchError(
                f"counts over {len(names)} states are one row and one column per "
                f"state, got an array shaped {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise InvalidValuesError("a count must be finite")
        if np.any(values < 0.0):
            raise InvalidValuesError(
                f"a count cannot be negative, got {float(np.min(values))}"
            )
        if np.any(values != np.floor(values)):
            raise InvalidValuesError("a count of transitions is a whole number")

        whole: npt.NDArray[np.int64] = values.astype(np.int64)
        whole.setflags(write=False)
        self._states = names
        self._positions = {name: position for position, name in enumerate(names)}
        self._values = whole

    @property
    def states(self) -> tuple[str, ...]:
        """The state names, in the order the rows and columns are held."""
        return self._states

    @property
    def n_states(self) -> int:
        """How many states were counted."""
        return len(self._states)

    @property
    def values(self) -> npt.NDArray[np.int64]:
        """The counts, frozen, rows the source and columns the destination."""
        return self._values

    @property
    def n_transitions(self) -> int:
        """How many steps were counted in total."""
        return int(np.sum(self._values))

    @property
    def never_left(self) -> tuple[str, ...]:
        """The states no counted step began from, in state order.

        Each has a row of zeros, which is zero over zero once divided.
        """
        totals = np.sum(self._values, axis=1)
        return tuple(
            name for name, total in zip(self._states, totals, strict=True) if total == 0
        )

    def count_of(self, source: str, target: str) -> int:
        """How many steps went from ``source`` to ``target``.

        Raises
        ------
        InvalidValuesError
            If either is not one of the counted states.
        """
        return int(
            self._values[
                _position_in(self._positions, source),
                _position_in(self._positions, target),
            ]
        )

    def leaving(self, state: str) -> int:
        """How many counted steps began from ``state``.

        Raises
        ------
        InvalidValuesError
            If ``state`` is not one of the counted states.
        """
        return int(np.sum(self._values[_position_in(self._positions, state)]))

    def transition_matrix(self, smoothing: float) -> TransitionMatrix:
        """Each row divided by its total, after adding ``smoothing`` to every cell.

        ``(count + smoothing) / (row total + smoothing * n_states)``, which is
        additive smoothing over the states counted here. At zero it is the
        plain relative frequency.

        Raises
        ------
        InvalidValuesError
            If ``smoothing`` is negative or not finite.
        TooFewValuesError
            If ``smoothing`` is zero and some state was never left, since its
            row is zero over zero. The message names the states.
        """
        amount = float(smoothing)
        if not np.isfinite(amount) or amount < 0.0:
            raise InvalidValuesError(
                f"a smoothing count is finite and at least zero, got {amount}"
            )

        stranded = self.never_left
        if amount == 0.0 and stranded:
            raise TooFewValuesError(
                f"{list(stranded)} never began a counted step, so each has a row "
                "of transitions that is zero over zero. Give a smoothing above "
                "zero, or count sequences that leave them"
            )

        smoothed = self._values.astype(np.float64) + amount
        return TransitionMatrix(
            self._states, smoothed / np.sum(smoothed, axis=1, keepdims=True)
        )

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray[np.int64]:
        return array_for_protocol(self._values, dtype, copy)  # type: ignore[arg-type,return-value]

    def __repr__(self) -> str:
        return (
            f"TransitionCounts(states={self._states!r}, "
            f"n_transitions={self.n_transitions!r})"
        )


class TransitionMatrix:
    """Where each state goes next, with what probability, by name.

    Row ``i`` is the distribution of the state that follows state ``i``, so
    every row sums to one and ``table["sunny"]["rainy"]`` is the probability of
    rain tomorrow given sun today.

    Parameters
    ----------
    states:
        The names, in the order the rows and columns are given.
    probabilities:
        ``(n_states, n_states)``, each entry at least zero and each row summing
        to one within :data:`SUM_TOLERANCE`.

    Raises
    ------
    EmptyValuesError
        If there are no states.
    InvalidValuesError
        If a name is repeated or is not a non-empty string, or if an entry is
        not finite or is negative, or a row does not sum to one.
    ShapeMismatchError
        If the table is not one row and one column per state.
    """

    __slots__ = ("_positions", "_states", "_values")

    def __init__(self, states: Sequence[str], probabilities: object) -> None:
        names = checked_states(states)
        values = _as_float_block(probabilities, "a transition matrix")

        if values.shape != (len(names), len(names)):
            raise ShapeMismatchError(
                f"a transition matrix over {len(names)} states is one row and one "
                f"column per state, got an array shaped {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise InvalidValuesError("a transition probability must be finite")
        if np.any(values < 0.0):
            raise InvalidValuesError(
                "a transition probability cannot be negative, got "
                f"{float(np.min(values))}"
            )
        totals = np.sum(values, axis=1)
        wrong = [
            name
            for name, total in zip(names, totals, strict=True)
            if abs(float(total) - 1.0) > SUM_TOLERANCE
        ]
        if wrong:
            raise InvalidValuesError(
                f"every row of a transition matrix sums to one; the rows for "
                f"{wrong} do not"
            )

        values.setflags(write=False)
        self._states = names
        self._positions = {name: position for position, name in enumerate(names)}
        self._values = values

    @property
    def states(self) -> tuple[str, ...]:
        """The state names, in the order the rows and columns are held."""
        return self._states

    @property
    def n_states(self) -> int:
        """How many states the table is over."""
        return len(self._states)

    @property
    def values(self) -> FloatArray:
        """The probabilities, frozen, rows the source and columns the destination."""
        return self._values

    def probability_of(self, source: str, target: str) -> float:
        """The probability that ``target`` follows ``source``.

        Raises
        ------
        InvalidValuesError
            If either is not one of the table's states.
        """
        return float(
            self._values[
                _position_in(self._positions, source),
                _position_in(self._positions, target),
            ]
        )

    def leaving(self, state: str) -> StateDistribution:
        """The distribution of the state that follows ``state``.

        Raises
        ------
        InvalidValuesError
            If ``state`` is not one of the table's states.
        """
        return StateDistribution(
            self._states, self._values[_position_in(self._positions, state)]
        )

    def __getitem__(self, state: str) -> StateDistribution:
        return self.leaving(state)

    def __iter__(self) -> Iterator[StateDistribution]:
        """Each row as a distribution, in state order."""
        return (StateDistribution(self._states, row) for row in self._values)

    def __len__(self) -> int:
        return len(self._states)

    def __array__(
        self, dtype: DTypeLike | None = None, copy: bool | None = None
    ) -> FloatArray:
        return array_for_protocol(self._values, dtype, copy)

    def __repr__(self) -> str:
        return f"TransitionMatrix(states={self._states!r})"
