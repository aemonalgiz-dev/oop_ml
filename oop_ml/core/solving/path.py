"""The walk a solver took, not just where it stopped.

``_solve`` returns the coefficients and discards the four hundred sets it held
on the way there. For fitting that is right; for inspection it throws away the
entire subject. The interesting thing about gradient ascent is not its answer
-- Newton reaches the same one -- it is that it needed 394 passes to get there
where Newton needed 8, and that the shape of the two walks differs so much that
one is a straight line on a log plot and the other bends.

So the walk is modelled. Every pass records the weights it started from, the
step it took, and the largest coefficient movement in that step, which is the
number the convergence test actually reads. A tool showing how a solver behaves
has the whole trajectory rather than its endpoint.

One pairing, four models. ``GradientDescentRegression``,
``LogisticRegression``, ``NewtonLogisticRegression`` and
``MultinomialLogisticRegression`` all inherit
:class:`~oop_ml.core.base.iterative_solver.IterativeSolver` and differ only in
``_step``, so recording the walk once records it for all of them -- and makes
the comparison between them a matter of putting two paths side by side.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum

import numpy as np

from oop_ml.core.types import FloatArray


class SolverStop(StrEnum):
    """Which of the two exits a walk took.

    Both are recorded because they mean opposite things about the answer. A
    walk that converged has coefficients worth reading; a walk that ran out of
    passes has coefficients that were still moving when the counter stopped
    it, and reporting those without saying so is how a diverging fit looks
    like a successful one.

    Attributes
    ----------
    CONVERGED:
        Every coefficient moved less than ``tolerance`` on the final pass.
    PASS_LIMIT_REACHED:
        The pass counter ran out first. The coefficients are wherever the walk
        happened to be.
    """

    CONVERGED = "converged"
    PASS_LIMIT_REACHED = "pass limit reached"


class SolverStep:
    """One pass of a walk: where it started, what it did, where it landed.

    Parameters
    ----------
    pass_number:
        Counting from 1, matching what ``passes_run`` reports.
    weights_before:
        The coefficients this pass started from. The first pass starts at
        zero, which is worth seeing rather than assuming.
    step:
        What was added to them. Its sign and size are the whole of what a
        solver does differently from another solver.
    """

    __slots__ = ("_pass_number", "_step", "_weights_before")

    def __init__(
        self, pass_number: int, weights_before: FloatArray, step: FloatArray
    ) -> None:
        self._pass_number = pass_number
        self._weights_before = weights_before
        self._step = step

    @property
    def pass_number(self) -> int:
        """Which pass this was, counting from 1."""
        return self._pass_number

    @property
    def weights_before(self) -> FloatArray:
        """The coefficients before this pass."""
        return self._weights_before

    @property
    def step(self) -> FloatArray:
        """What this pass added to them."""
        return self._step

    @property
    def weights_after(self) -> FloatArray:
        """The coefficients after this pass."""
        return self._weights_before + self._step

    @property
    def largest_movement(self) -> float:
        """The biggest absolute change any one coefficient made.

        The number the convergence test reads. Measured on the step rather
        than on the objective, because near an optimum the objective is flat
        and its improvement reaches zero in floating point while the
        coefficients are still visibly moving.
        """
        return float(np.max(np.abs(self._step)))

    def __repr__(self) -> str:
        return (
            f"SolverStep({self._pass_number}, "
            f"largest movement {self.largest_movement:.3e})"
        )


class SolverPath:
    """Every pass a solver took, and where it ended up.

    Iterate it to walk the passes in order, which is what a convergence plot
    or a step-by-step explanation wants.

    Parameters
    ----------
    steps:
        The passes, in order.
    final_weights:
        Where the walk ended. Held separately rather than derived from the
        last step, so that a walk of zero passes still has an answer.
    stopped_because:
        Which exit was taken.
    """

    __slots__ = ("_final_weights", "_steps", "_stopped_because")

    def __init__(
        self,
        steps: Sequence[SolverStep],
        final_weights: FloatArray,
        stopped_because: SolverStop,
    ) -> None:
        self._steps = tuple(steps)
        self._final_weights = final_weights
        self._stopped_because = stopped_because

    @property
    def result(self) -> FloatArray:
        """What the efficient path would have returned: the final weights."""
        return self._final_weights

    @property
    def final_weights(self) -> FloatArray:
        """Where the walk ended."""
        return self._final_weights

    @property
    def stopped_because(self) -> SolverStop:
        """Which exit the walk took."""
        return self._stopped_because

    @property
    def converged(self) -> bool:
        """Whether the steps became smaller than the tolerance."""
        return self._stopped_because is SolverStop.CONVERGED

    @property
    def passes_run(self) -> int:
        """How many passes were taken."""
        return len(self._steps)

    @property
    def movements(self) -> FloatArray:
        """The largest coefficient movement on each pass, in order.

        The series a convergence plot draws. On a first-order walk it falls
        roughly geometrically; on a second-order one the exponents roughly
        double each pass, which is the difference the two solvers exist to
        show.
        """
        return np.array(
            [step.largest_movement for step in self._steps], dtype=np.float64
        )

    def __iter__(self) -> Iterator[SolverStep]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"SolverPath({len(self._steps)} passes, {self._stopped_because.value})"
