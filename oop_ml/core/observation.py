"""Two routes through every calculation: one that answers, one that is watchable.

The problem
-----------
Fitting throws away almost everything it computes. A split search evaluates
tens of thousands of candidates and keeps one; a gradient walk takes four
hundred steps and keeps the last; a neighbour query measures a distance to
every remembered row and keeps five. Discarding is right for the calculation
and wrong for any caller that needs to see what happened -- a convergence plot,
a diagnostic, an audit of why one split was chosen over another, a step-by-step
rendering. Those callers want the intermediates, and the intermediates are
precisely what the answer is not.

Why not one route with a flag
-----------------------------
Because retaining the record is not free at fitting scale. One node of a
5000-row, 20-feature problem considers about 100000 candidates, and testing a
flag on every iteration puts a branch in the hottest loop in the library to
serve a caller who is usually not there. Nor is it only speed: the efficient
routes are written to allocate nothing per step, and a recording variant of the
same code would have to abandon that.

So the two concerns get two methods:

**The efficient route** is the existing one. It keeps running state, allocates
close to nothing, and returns the answer. Nothing about it changes, and no
observation feature is allowed to slow it down.

**The observed route** returns an :class:`Observation` -- every intermediate,
in the order it was produced, with the answer available on it. It is free to be
slower and to allocate freely, because it is called deliberately, on data sized
for looking at.

They are two readings of one definition, so each pair is covered by a test
asserting they agree. That is the same arrangement the neighbour models already
use for their threaded and serial searches, where the fast path is justified
only by a test proving it returns what the plain one does. A fast path and a
slow path with nothing between them are two implementations, not one
calculation observed two ways.

The convention
--------------
For every calculation with intermediates worth seeing, a component offers:

======================  ======================================================
efficient               ``_best_split(...) -> Split | None``
observed                ``split_search(...) -> SplitSearch``
======================  ======================================================

The observed method is named for the object it returns, and that object is
named for the thing it records -- a search, a path, a query. Neither is named
for what a caller might do with it.

Every observation is iterable over its steps and exposes ``result``, so a
caller can walk any of them without knowing which calculation produced it. Each
also carries a domain-named alias -- ``best``, ``final_weights`` -- because
that is what someone holding one specific observation will reach for.

Objects, not tuples
-------------------
Every intermediate is a value object, so a caller asks a step what it is rather
than unpacking it by position. They use ``__slots__`` rather than pydantic:
validation belongs on hyperparameters arriving once at construction, and paying
it per candidate measured 6.7x slower for nothing.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

ResultT = TypeVar("ResultT", covariant=True)


class Stage:
    """One named intermediate of a calculation that is a fixed sequence.

    Some calculations are loops -- a walk, a search -- and their observation is
    a list of like steps. Others are a short derivation whose stages are not
    alike at all: a design matrix, then a moment matrix, then a solution. This
    is what those yield when iterated, so a caller can walk any observation
    the same way without the stages having to pretend to be interchangeable.

    The name is a label for display, not an input to anything, which is why it
    is a plain string where an argument would have to be a closed enum.
    """

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: object) -> None:
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        """What this stage is called."""
        return self._name

    @property
    def value(self) -> object:
        """What the calculation had at this point."""
        return self._value

    def __repr__(self) -> str:
        return f"Stage({self._name!r})"


@runtime_checkable
class Observation(Protocol[ResultT]):
    """A record of the intermediates a calculation passed through.

    Structural rather than inherited, so an observation is free to be whatever
    shape its calculation calls for -- a scoreboard, a walk, a ranking -- and
    still be recognised by anything that only needs the steps and the answer.

    Implementations are iterable over their steps, in the order those steps
    happened, and expose the answer as ``result``.
    """

    @property
    def result(self) -> ResultT:
        """What the efficient route would have returned.

        The point of the pairing: an observation is not an approximation of
        the calculation, it is the calculation with its intermediates kept,
        and this is where that claim becomes checkable.
        """
        ...

    def __len__(self) -> int:
        """How many steps were recorded."""
        ...
