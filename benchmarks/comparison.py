"""The objects a benchmark run produces.

A timing on its own is not worth much, since a library that is fast and wrong is
not fast at all. What matters is the pair: how long each library took, and
whether the two of them landed in the same place. Those belong together in one
object rather than in two lists that a caller has to zip back up, which is the
same reasoning behind ``RegressionEvaluation`` holding the predictions beside
the truth.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from enum import Enum
from typing import Any


class Agreement(Enum):
    """Whether the two libraries produced the same answer.

    An enum rather than a boolean or a string, because there is a genuine third
    case. Iterative solvers with different stopping rules are not expected to
    match coefficient for coefficient, and reporting that as a failure would be
    misleading, while reporting it as a success would be a lie.
    """

    MATCHES = "matches"
    DIFFERS = "differs"
    NOT_COMPARED = "not compared"


class Timing:
    """The best wall-clock time observed over several runs of one callable.

    Best rather than mean, because the thing being measured is how long the work
    takes, and everything that pushes a particular run above that floor is the
    operating system rather than the code.
    """

    __slots__ = ("_result", "_seconds")

    def __init__(self, seconds: float, result: Any) -> None:
        self._seconds = seconds
        self._result = result

    @classmethod
    def of(cls, work: Callable[[], Any], repeats: int = 3) -> Timing:
        """Run ``work`` ``repeats`` times and keep the fastest."""
        best_seconds = float("inf")
        result = None

        for _ in range(repeats):
            started_at = time.perf_counter()
            result = work()
            best_seconds = min(best_seconds, time.perf_counter() - started_at)

        return cls(best_seconds, result)

    @property
    def seconds(self) -> float:
        """How long the fastest run took."""
        return self._seconds

    @property
    def result(self) -> Any:
        """Whatever the callable returned, so agreement can be checked."""
        return self._result

    def __repr__(self) -> str:
        return f"Timing({self._seconds:.4f}s)"


class Comparison:
    """One task, timed in both libraries, and whether they agreed.

    Parameters
    ----------
    task:
        What was measured, such as ``"Ridge"``.
    size:
        The shape it was measured on, such as ``"20000x50"``.
    ours:
        The timing for ``oop_ml``.
    theirs:
        The timing for scikit-learn.
    agreement:
        Whether the two results matched, and whether that was even checked.
    """

    __slots__ = ("_agreement", "_ours", "_size", "_task", "_theirs")

    def __init__(
        self,
        task: str,
        size: str,
        ours: Timing,
        theirs: Timing,
        agreement: Agreement,
    ) -> None:
        self._task = task
        self._size = size
        self._ours = ours
        self._theirs = theirs
        self._agreement = agreement

    @property
    def task(self) -> str:
        """What was measured."""
        return self._task

    @property
    def size(self) -> str:
        """The shape of the data it ran against."""
        return self._size

    @property
    def ours(self) -> Timing:
        """The ``oop_ml`` timing."""
        return self._ours

    @property
    def theirs(self) -> Timing:
        """The scikit-learn timing."""
        return self._theirs

    @property
    def agreement(self) -> Agreement:
        """Whether the two libraries produced the same answer."""
        return self._agreement

    @property
    def ratio(self) -> float:
        """How long we took relative to scikit-learn.

        Below 1.0 means ``oop_ml`` was faster on this task.
        """
        return self._ours.seconds / self._theirs.seconds

    def as_row(self) -> list[str]:
        """The line this comparison contributes to the reported table."""
        return [
            self._task,
            self._size,
            f"{self._ours.seconds:.4f}",
            f"{self._theirs.seconds:.4f}",
            f"{self.ratio:.1f}x",
            self._agreement.value,
        ]

    def __repr__(self) -> str:
        return f"Comparison({self._task!r}, {self._size!r}, ratio={self.ratio:.2f})"


class Comparisons:
    """Every comparison a run produced, in the order it produced them."""

    COLUMN_NAMES = ("task", "size", "oop_ml (s)", "sklearn (s)", "ratio", "answers")

    __slots__ = ("_comparisons",)

    def __init__(self, comparisons: Sequence[Comparison]) -> None:
        self._comparisons = tuple(comparisons)

    @property
    def rows(self) -> list[list[str]]:
        """Every comparison as a table row."""
        return [comparison.as_row() for comparison in self._comparisons]

    @property
    def disagreements(self) -> list[Comparison]:
        """Any comparison where the two libraries did not land in the same place.

        This is the part of a benchmark that actually matters. A timing table
        with a disagreement hiding in it is measuring two different programs.
        """
        return [
            comparison
            for comparison in self._comparisons
            if comparison.agreement is Agreement.DIFFERS
        ]

    def __iter__(self) -> Iterator[Comparison]:
        return iter(self._comparisons)

    def __len__(self) -> int:
        return len(self._comparisons)

    def __repr__(self) -> str:
        return f"Comparisons({len(self._comparisons)} tasks)"
