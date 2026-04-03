"""A quantity that shrinks as a walk proceeds, and why some walks require one.

Where this became necessary
----------------------------
Every iterative model here so far has held its learning rate fixed. Gradient
descent, gradient ascent and the coordinate sweeps all take the same size of
step on the last pass as on the first, and that is a defensible simplification
because they are descending a surface whose gradient shrinks on its own as they
approach the answer. The step size need not shrink when the thing it multiplies
already does.

Competitive learning has no such courtesy. A self-organising map moves its
winning unit a fixed *fraction* of the way toward the row it just saw,
regardless of how well placed that unit already was, so the update does not
decay by itself. Hold the fraction constant and the map never settles: it
chases whichever row arrived last, forever, and the final answer is a fact
about presentation order rather than about the data. Decaying the fraction is
not tuning, it is what makes the process converge at all.

Oja's rule is in the same position for a different reason, and both are
instances of the Robbins-Monro conditions on a stochastic approximation. A
sequence of rates converges if the rates sum to infinity, so the walk can still
reach anywhere from where it started, and their squares sum to something
finite, so the noise in the updates eventually stops mattering. A constant rate
fails the second condition and a rate decaying too fast fails the first.

Why this is a quantity and not a learning rate
-----------------------------------------------
Naming the class ``LearningSchedule`` was the first attempt and it was a lie
within one model. A self-organising map decays *two* things on the same shape
of curve: how far the winner moves, and how wide the neighbourhood of units
that move with it. The second is not a learning rate by any reading, so the
abstraction is a value that changes over the course of a walk, and a learning
rate is one thing a caller may use it for.

Why the pass count is a parameter rather than remembered
---------------------------------------------------------
A schedule is asked ``value_at(pass_number, total_passes)`` rather than being
told the total once at construction. That keeps a schedule a pure value object
with no fitted state, so the same instance can be handed to two models with
different pass limits and to a search that varies the limit. It also means the
decay is expressed as a fraction of the walk rather than as a rate per pass,
which is what a caller usually means: "start at 0.5 and finish near 0.01" is a
statement about the whole run, and it should not silently change meaning when
the cap moves from 100 passes to 1000.

The walk is counted from one, matching
:class:`~oop_ml.core.base.iterative_solver.IterativeSolver`, so the first pass
gets the starting value exactly and the last gets the ending value exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from oop_ml.core.exceptions import InvalidValuesError


class Schedule(BaseModel, ABC):
    """A value that changes over the course of a walk.

    A pydantic model rather than a plain class, for the reason
    :class:`~oop_ml.core.kernel.functions.Kernel` gives: a schedule carries
    parameters, and every hyperparameter in this library is validated at
    construction rather than at the first fit that happens to use it.
    """

    model_config = ConfigDict(extra="forbid")

    @abstractmethod
    def value_at(self, pass_number: int, total_passes: int) -> float:
        """The value this schedule holds on a given pass.

        Parameters
        ----------
        pass_number:
            Which pass, counted from 1, so that the first pass reports the
            starting value exactly.
        total_passes:
            How many passes the walk is allowed in total, which is what makes
            the decay a fraction of the run rather than a rate per pass.

        Returns
        -------
        float
            The value for that pass.

        Raises
        ------
        InvalidValuesError
            If ``total_passes`` is below one, or ``pass_number`` is outside
            ``1 .. total_passes``. Both are caller mistakes rather than data
            mistakes, and a schedule asked about pass zero would silently
            extrapolate past its own starting value.
        """

    @staticmethod
    def _elapsed_fraction(pass_number: int, total_passes: int) -> float:
        """How far through the walk this pass is, from 0.0 to 1.0.

        The shared guard and the shared arithmetic, so that a subclass writes
        only its curve.

        A single-pass walk is the case worth naming. There is no interval to be
        a fraction of, and the honest answer is 0.0, which hands back the
        starting value. Dividing by ``total_passes - 1`` without that check is a
        division by zero on a configuration nothing else refuses.
        """
        if total_passes < 1:
            raise InvalidValuesError(
                f"a walk runs for at least one pass, got {total_passes}"
            )
        if not 1 <= pass_number <= total_passes:
            raise InvalidValuesError(
                f"pass {pass_number} is outside a walk of {total_passes} passes"
            )
        if total_passes == 1:
            return 0.0
        return (pass_number - 1) / (total_passes - 1)


class ConstantSchedule(Schedule):
    """The same value on every pass.

    The control, and the honest way to say that a model does not decay
    anything. Handing this to a self-organising map is a legitimate thing to do
    and produces a map that does not settle, which is worth being able to
    demonstrate rather than being prevented from expressing.

    Parameters
    ----------
    value:
        What every pass gets. Must be finite and non-negative, since every
        caller here multiplies an update by it.
    """

    value: float = Field(ge=0.0)

    def value_at(self, pass_number: int, total_passes: int) -> float:
        """``value``, whatever pass it is asked about.

        The bounds are still checked, so that a caller who has miscounted their
        own walk is told here rather than by whichever schedule they swap in
        later.
        """
        self._elapsed_fraction(pass_number, total_passes)
        return self.value


class LinearDecaySchedule(Schedule):
    """A straight line from ``start`` down to ``end`` across the walk.

    The first pass reports ``start`` exactly and the last reports ``end``
    exactly, which is what makes this predictable to reason about. A
    neighbourhood radius shrinking linearly from 5 units to 1 over 200 passes
    is at 3 on pass 100, and that is worth being able to say without working
    out an exponent.

    Parameters
    ----------
    start:
        The value on the first pass. Finite and non-negative.
    end:
        The value on the last pass. Finite and non-negative. May exceed
        ``start``, which grows rather than decays; nothing here forbids it,
        because nothing in the mathematics does, and a growing neighbourhood is
        a legitimate thing to want to demonstrate failing.
    """

    start: float = Field(ge=0.0)
    end: float = Field(default=0.0, ge=0.0)

    def value_at(self, pass_number: int, total_passes: int) -> float:
        """``start`` plus the elapsed fraction of the way to ``end``."""
        elapsed = self._elapsed_fraction(pass_number, total_passes)
        return self.start + (self.end - self.start) * elapsed


class ExponentialDecaySchedule(Schedule):
    """A geometric fall from ``start`` to ``end`` across the walk.

    The usual choice for a learning rate, because what matters about a rate is
    its order of magnitude rather than its absolute size. Halving repeatedly
    spends most of the walk at small values, where a linear decay spends half
    of it above the midpoint.

    Worth seeing the difference in numbers. From 0.5 to 0.005 over 100 passes,
    at pass 50 a linear decay is at 0.2525 and this is at 0.0502, five times
    smaller. That is the whole reason to prefer it for a rate and to prefer a
    linear decay for a radius measured in grid units, where the midpoint really
    should be the middle.

    Parameters
    ----------
    start:
        The value on the first pass. Strictly positive.
    end:
        The value on the last pass. Strictly positive.

    Raises
    ------
    pydantic.ValidationError
        If either bound is zero or negative, by way of ``Field(gt=0.0)``. A
        geometric fall multiplies by a constant ratio each pass and so can
        approach zero without ever arriving, which means zero is not a value it
        can be asked to reach, and a negative bound would make the ratio
        negative and the value alternate in sign. This is the one place a
        schedule refuses what a linear decay happily accepts, and the refusal is
        the mathematics rather than a policy.

        A hand-written validator here would be unreachable, since the field
        constraint fires first, and pydantic is where every other
        hyperparameter in this library is refused.
    """

    start: float = Field(gt=0.0)
    end: float = Field(gt=0.0)

    def value_at(self, pass_number: int, total_passes: int) -> float:
        """``start`` times the ratio to ``end``, raised to the elapsed fraction.

        Written as ``start * (end / start) ** elapsed`` rather than as a
        per-pass ratio raised to the pass number, so that the last pass lands on
        ``end`` exactly rather than near it.
        """
        elapsed = self._elapsed_fraction(pass_number, total_passes)
        return self.start * (self.end / self.start) ** elapsed
