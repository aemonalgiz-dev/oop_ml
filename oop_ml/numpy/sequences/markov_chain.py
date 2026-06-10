"""A model of sequences of named states that remembers only the current one.

What it assumes
---------------
A Markov chain says that where a sequence goes next depends on where it is now
and on nothing before that. Tomorrow's weather depends on today's, and not on
last week's once today is known. That one assumption turns a model of whole
sequences into a table: for every state, the probability of each state that can
follow it.

Fitting is counting and dividing. Every adjacent pair in every training
sequence adds one to the cell for that step, and each row of counts divided by
its total is that state's row of the table. Counting stops at the end of each
sequence, so the last state of one and the first of the next are never counted
as a step, since nobody observed one.

Why the input is a sequence of sequences, and a string is refused
-----------------------------------------------------------------
A string is a sequence of one-character strings. So ``fit("sunny rainy")``
would learn a chain over letters and spaces, and ``fit(["sunny", "rainy"])``,
the likelier mistake, would learn two sequences of letters. Both run, both
answer plausible probabilities, and neither is what was meant. Both are refused
by name, which is the rule the tokenization package's
:class:`~oop_ml.core.natural_language_processing.tokenization.corpus.Corpus`
already keeps for the same trap.

A state that is never left
--------------------------
Every sequence ends somewhere, and if the state it ends on never begins a step
anywhere in the training data, that state's row of counts is all zeros. Its row
of the table is then zero over zero. There are two established ways to paper
over that, a row of ``1 / n_states`` or a row sending the state to itself with
certainty, and both invent an observation nobody made. The second also changes
the chain's structure, since it turns the state into a trap that every long
walk eventually falls into.

So at a smoothing of zero the fit is refused, with
:class:`~oop_ml.core.exceptions.TooFewValuesError` naming the states. The
remedy is a smoothing above zero, which adds that count to every cell of every
row before dividing. That gives the never-left state an even row, and says so
in the hyperparameter a caller chose rather than in a convention hidden here.

The stationary distribution, and its three cases
------------------------------------------------
A stationary distribution is one that a step leaves unchanged: ``pi P = pi``.
Whether a chain has one, how many it has, and whether a walk ends there are
three separate questions, and each fixture in the spec is built to separate
them.

**Irreducible and aperiodic.** Every state can reach every other and the walk
does not cycle in lockstep. There is exactly one stationary distribution and a
walk from anywhere converges to it. This is the case that the textbook
statement covers, and every chain with a smoothing above zero is in it, since
every cell is then positive.

**Periodic.** The coin that always turns, heads to tails to heads, has exactly
one stationary distribution, ``(0.5, 0.5)``, and a walk from heads never
settles on it: it is certainly heads after an even number of steps and
certainly tails after an odd one. This method returns the distribution, since
it exists and is unique. What it means there is the share of time the walk
spends in each state over a long run, which is a weaker statement than where
the walk ends up, and the one that survives periodicity.

**Several closed groups.** Two sequences that never share a state give two
groups a walk can enter and never leave. Each has its own stationary
distribution, and every mixture of the two is stationary as well, so "the"
stationary distribution does not name anything. This is refused with
:class:`~oop_ml.core.exceptions.NonUniqueStationaryDistributionError`, whose
message names the groups.

The refusal is about several closed groups and not about reducibility, and the
difference is a real case. A chain that leaves ``start`` for ``sink`` and never
returns cannot get back to ``start``, so it is reducible, and it still has
exactly one stationary distribution, all of it on ``sink``. Only one of its
groups is closed. The number of closed groups is read off which cells of the
table are positive, which is integer reasoning about reachability, rather than
off a numerical rank that would have to choose a threshold.

Why solve rather than iterate
-----------------------------
Walking many steps and reading where the walk ended finds the distribution only
in the first case, and slowly when the chain mixes slowly. It never finds it on
a periodic chain at all. The linear system has the answer in every case where
there is one: ``pi (P - I) = 0`` has a one-dimensional solution space exactly
when there is one closed group, and replacing one of its equations, which are
dependent since each row of ``P`` sums to one, by ``sum(pi) = 1`` pins the
scale. The spec checks the solve against two thousand steps of a plain loop on
a chain where both are defined.

Where a state holds no share at all, nothing in the arithmetic stops the solve
landing a rounding error either side of zero, so negative values are clipped to
zero before the distribution is built. Measured on the draining fixture and on
two other chains with states that drain away, the solve returned exactly zero
for every such state, as ``-0.0``, so the clip has so far had nothing to do. It
stays because a probability cannot be negative and the distribution refuses one.

What a sequence's probability is conditioned on
-----------------------------------------------
:meth:`MarkovChain.log_probability_of` scores every step of a sequence and not
its first state, which is given. Where sequences start is a second distribution
this model does not learn: a chain is its table of steps, and a model of starts
would be a separate count over a separate quantity. A sequence
taking a step the table gives no probability to scores minus infinity, which is
what the logarithm of zero is, rather than a large negative number a caller
might average by accident.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import ClassVar, Self

import numpy as np
from pydantic import Field, PrivateAttr

from oop_ml.core.base.estimator import Fittable
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueStationaryDistributionError,
    NotFittedError,
)
from oop_ml.core.sequences.transitions import (
    StateDistribution,
    TransitionCounts,
    TransitionMatrix,
)


def checked_sequence(sequence: Sequence[str], role: str) -> tuple[str, ...]:
    """One sequence of state names, refusing the one-string trap.

    Raises
    ------
    InvalidValuesError
        If ``sequence`` is a single string, or holds anything that is not a
        non-empty string.
    EmptyValuesError
        If it holds no states.
    """
    if isinstance(sequence, str):
        raise InvalidValuesError(
            f"{role} is a sequence of state names, not one string; got "
            f"{sequence!r}, which would be read as its letters"
        )

    states = tuple(sequence)
    if not states:
        raise EmptyValuesError(f"{role} holds no states")

    for state in states:
        if not isinstance(state, str) or not state:
            raise InvalidValuesError(
                f"a state is named by a non-empty string; {role} holds {state!r}"
            )

    return states


def checked_sequences(
    sequences: Sequence[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Every training sequence, each checked by :func:`checked_sequence`.

    Raises
    ------
    InvalidValuesError
        If ``sequences`` is a single string, or any one sequence is.
    EmptyValuesError
        If there are no sequences, or any one is empty.
    """
    if isinstance(sequences, str):
        raise InvalidValuesError(
            "a chain is fitted on a sequence of sequences of state names, not "
            f"one string; got {sequences!r}"
        )

    checked = tuple(
        checked_sequence(sequence, f"the sequence at position {position}")
        for position, sequence in enumerate(sequences)
    )
    if not checked:
        raise EmptyValuesError("a chain needs at least one sequence to count")

    return checked


def whole_number_of_at_least(value: object, minimum: int, role: str) -> int:
    """Read a count, refusing a bool, a fraction, or one below ``minimum``.

    ``operator.index`` rather than ``isinstance(value, int)``, so a
    ``numpy.int64`` is accepted, and ``bool`` refused ahead of it, since
    ``True`` indexes as 1.

    Raises
    ------
    InvalidValuesError
        If the value is not a whole number of at least ``minimum``.
    """
    if isinstance(value, bool):
        raise InvalidValuesError(f"{role} must be a whole number, not a bool")
    try:
        whole = operator.index(value)  # type: ignore[arg-type]
    except TypeError:
        raise InvalidValuesError(
            f"{role} must be a whole number, got {value!r}"
        ) from None
    if whole < minimum:
        raise InvalidValuesError(f"{role} must be at least {minimum}, got {whole}")
    return whole


def closed_groups(reachable_in_one_step: np.ndarray) -> list[tuple[int, ...]]:
    """The groups of states a walk can enter and never leave, by position.

    Parameters
    ----------
    reachable_in_one_step:
        ``(n, n)`` booleans, true where the table gives a step a probability
        above zero.

    Returns
    -------
    list[tuple[int, ...]]
        One tuple of positions per closed group, each in position order, the
        groups ordered by their first member.

    Notes
    -----
    Reachability is the transitive closure of the one-step relation, found by
    repeated boolean squaring until nothing changes. Two states communicate
    when each reaches the other, which partitions the states into groups, and a
    group is closed when nothing outside it is reachable from inside it.
    """
    n_states = reachable_in_one_step.shape[0]
    reachable = reachable_in_one_step | np.eye(n_states, dtype=bool)

    while True:
        as_counts = reachable.astype(np.int64)
        extended = reachable | ((as_counts @ as_counts) > 0)
        if np.array_equal(extended, reachable):
            break
        reachable = extended

    communicating = reachable & reachable.T

    groups: list[tuple[int, ...]] = []
    assigned: set[int] = set()
    for position in range(n_states):
        if position in assigned:
            continue
        group = tuple(int(member) for member in np.flatnonzero(communicating[position]))
        assigned.update(group)
        outside = np.ones(n_states, dtype=bool)
        outside[list(group)] = False
        if not np.any(reachable[list(group)][:, outside]):
            groups.append(group)

    return groups


class MarkovChain(Fittable):
    """Learns which named state follows which, and answers questions about walks.

    Parameters
    ----------
    smoothing:
        A count added to every cell of the table before each row is divided by
        its total, at least zero and finite. At zero the table is the observed
        relative frequency, a step never observed has probability zero, and a
        state never left is refused at ``fit``. Above zero every step has some
        probability and every chain has one stationary distribution.

    Notes
    -----
    A sibling of the transformer and the clusterer rather than a special case of
    either. It takes no target and it rewrites no columns and labels no rows;
    what it learns is a distribution over sequences, which it answers questions
    about. See the module docstring for the three cases of the stationary
    distribution and for the never-left state.

    Only the counts are stored. The table is derived from them and from
    ``smoothing`` whenever it is asked for, so there is one source of truth for
    what was learned and a saved chain holds exactly that.
    """

    smoothing: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    LEARNED_STATE: ClassVar[tuple[str, ...]] = ("_counts",)
    """The counted steps, from which the table and everything else follows."""

    _counts: TransitionCounts | None = PrivateAttr(default=None)

    def fit(self, sequences: Sequence[Sequence[str]]) -> Self:
        """Count every step in every sequence.

        The states are numbered in order of first appearance, reading each
        sequence in turn, which is the order every answer is given in.

        Nothing is committed until the counts are known to make a table, so a
        refused refit leaves the previous fit intact.

        Parameters
        ----------
        sequences:
            A sequence of sequences of state names. A sequence of one state is
            allowed; it names a state and counts no step.

        Returns
        -------
        Self
            This chain, so calls can chain.

        Raises
        ------
        InvalidValuesError
            If ``sequences`` or any sequence in it is a single string, or a
            state is not a non-empty string.
        EmptyValuesError
            If there are no sequences, or one of them is empty.
        TooFewValuesError
            If ``smoothing`` is zero and some state never begins a step.
        """
        checked = checked_sequences(sequences)

        positions: dict[str, int] = {}
        for sequence in checked:
            for state in sequence:
                positions.setdefault(state, len(positions))

        sources = [positions[state] for sequence in checked for state in sequence[:-1]]
        targets = [positions[state] for sequence in checked for state in sequence[1:]]
        counts = np.zeros((len(positions), len(positions)), dtype=np.int64)
        np.add.at(counts, (sources, targets), 1)

        learned = TransitionCounts(tuple(positions), counts)
        learned.transition_matrix(self.smoothing)

        self._counts = learned
        self._mark_fitted()

        return self

    def _fitted_counts(self) -> TransitionCounts:
        """The counts, after the guard that says whether there are any.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        if self._counts is None:
            raise NotFittedError(f"{type(self).__name__} has counted nothing")
        return self._counts

    @property
    def states(self) -> tuple[str, ...]:
        """Every state the fit saw, in order of first appearance.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._fitted_counts().states

    @property
    def n_states(self) -> int:
        """How many states the fit saw.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._fitted_counts().n_states

    @property
    def transition_counts(self) -> TransitionCounts:
        """How many times each step was observed, before any smoothing.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._fitted_counts()

    @property
    def transitions(self) -> TransitionMatrix:
        """The table: for each state, the distribution of the state after it.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        return self._fitted_counts().transition_matrix(self.smoothing)

    def probability_of(self, source: str, target: str) -> float:
        """The probability that ``target`` follows ``source`` in one step.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If either state was not seen during ``fit``.
        """
        return self.transitions.probability_of(source, target)

    def distribution_after(
        self, start: str | StateDistribution, n_steps: int
    ) -> StateDistribution:
        """Where a walk will be after ``n_steps`` steps, as a distribution.

        The starting distribution times the table raised to the ``n_steps``
        power. Zero steps is the start itself.

        Parameters
        ----------
        start:
            A state name, meaning certainty of that state, or a distribution
            over exactly this chain's states, matched by name so its order does
            not matter.
        n_steps:
            A whole number of at least zero.

        Returns
        -------
        StateDistribution
            Over this chain's states, in the chain's order.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``n_steps`` is not a whole number of at least zero, if ``start``
            names a state the fit never saw, or if a starting distribution is
            over other states.
        """
        table = self.transitions
        steps = whole_number_of_at_least(n_steps, 0, "a step count")
        starting = self._starting_distribution(start, table.states)

        walked = starting.values @ np.linalg.matrix_power(table.values, steps)

        return StateDistribution(table.states, walked)

    def stationary_distribution(self) -> StateDistribution:
        """The distribution one step leaves unchanged, where there is exactly one.

        On an aperiodic chain it is where every walk ends up. On a periodic one
        it is the long-run share of time spent in each state, and no walk ends
        there. See the module docstring for both.

        Returns
        -------
        StateDistribution
            Over this chain's states, in the chain's order.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        NonUniqueStationaryDistributionError
            If the chain holds more than one closed group of states, each with
            its own stationary distribution. The message names the groups.
        """
        table = self.transitions
        groups = closed_groups(table.values > 0.0)

        if len(groups) > 1:
            named = [
                tuple(table.states[position] for position in group) for group in groups
            ]
            raise NonUniqueStationaryDistributionError(
                f"this chain has {len(groups)} closed groups of states, "
                f"{', '.join(repr(group) for group in named)}. A walk that enters "
                "one never leaves it, so each has its own stationary distribution "
                "and every mixture of them is stationary too; there is no single "
                "one to give. A smoothing above zero joins them"
            )

        n_states = table.n_states
        system = table.values.T - np.eye(n_states)
        system[-1, :] = 1.0
        right_hand_side = np.zeros(n_states)
        right_hand_side[-1] = 1.0

        solved = np.clip(np.linalg.solve(system, right_hand_side), 0.0, None)

        return StateDistribution(table.states, solved / np.sum(solved))

    def log_probability_of(self, sequence: Sequence[str]) -> float:
        """The log probability of every step of ``sequence``, given its first state.

        Parameters
        ----------
        sequence:
            State names the fit saw, in order.

        Returns
        -------
        float
            The sum of the log probability of each step. Zero for a single
            state, which has no step to score, and minus infinity if any step
            has probability zero.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``sequence`` is a single string or names a state the fit never
            saw.
        EmptyValuesError
            If ``sequence`` is empty.
        """
        table = self.transitions
        states = checked_sequence(sequence, "the sequence to score")
        step_probabilities = np.array(
            [
                table.probability_of(source, target)
                for source, target in zip(states, states[1:], strict=False)
            ]
            or [1.0]
        )

        with np.errstate(divide="ignore"):
            return float(np.sum(np.log(step_probabilities)))

    def sample(
        self, start: str, length: int, random_seed: int | None = None
    ) -> tuple[str, ...]:
        """Draw a walk of ``length`` states beginning at ``start``.

        Each next state is drawn from the current state's row. A step the table
        gives probability zero is never taken, since the draw finds the first
        state whose running total exceeds a uniform number scaled to the row's
        own total, and a state of probability zero adds nothing to that total.

        Parameters
        ----------
        start:
            A state the fit saw. It is the first state of the walk.
        length:
            How many states the walk holds, the start included, at least one.
        random_seed:
            Seeds the draw, so the same seed draws the same walk.

        Returns
        -------
        tuple[str, ...]
            The walk, as state names.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If ``length`` is not a whole number of at least one, or ``start``
            was not seen during ``fit``.
        """
        table = self.transitions
        n_states_drawn = whole_number_of_at_least(length, 1, "a walk's length")
        current = int(
            np.flatnonzero(
                np.asarray(StateDistribution.concentrated_on(table.states, start))
            )[0]
        )

        running_totals = np.cumsum(table.values, axis=1)
        generator = np.random.default_rng(random_seed)

        walk = [current]
        for _ in range(n_states_drawn - 1):
            row = running_totals[current]
            current = int(
                np.searchsorted(row, generator.random() * row[-1], side="right")
            )
            walk.append(current)

        return tuple(table.states[position] for position in walk)

    @staticmethod
    def _starting_distribution(
        start: str | StateDistribution, states: tuple[str, ...]
    ) -> StateDistribution:
        """A walk's start as a distribution in the chain's own state order.

        Raises
        ------
        InvalidValuesError
            If ``start`` is neither a known state name nor a distribution over
            exactly these states.
        """
        if isinstance(start, StateDistribution):
            return start.in_order_of(states)
        return StateDistribution.concentrated_on(states, start)

    def __repr__(self) -> str:
        if self._counts is None:
            return f"MarkovChain(smoothing={self.smoothing!r}, unfitted)"
        return (
            f"MarkovChain(smoothing={self.smoothing!r}, "
            f"n_states={self._counts.n_states!r})"
        )
