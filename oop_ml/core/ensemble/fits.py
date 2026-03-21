"""What an ensemble is made of, kept rather than collapsed into an average.

An ensemble's answer is one number and its interest is entirely in the spread
behind that number. Whether the members agreed or barely agreed, whether the
average moved after the twentieth of them, which rows each member never saw --
none of it survives into the prediction, and all of it is the reason the model
works.

Two records, because the two families differ in what a member *is*.

An averaging ensemble's members are peers, fitted independently and interchangeable;
what matters per member is which rows it drew. A boosting ensemble's members are
a sequence, each fitted on what the ones before it got wrong, and what matters
per round is how much error was left when it started.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.ensemble.bootstrap import BootstrapSample
from oop_ml.core.types import FloatArray


class MemberFit:
    """One member of an averaging ensemble, and the sample it was fitted on.

    Parameters
    ----------
    position:
        Which member this is, counting from 0, in the order they were fitted.
        Order carries no meaning here -- members are peers -- but a stable
        position is what lets a caller line two records up.
    sample:
        The bootstrap sample this member saw.
    model:
        The fitted member itself. Ask it for its own observation to go a level
        deeper: a tree will answer ``split_search``.
    """

    __slots__ = ("_model", "_position", "_sample")

    def __init__(self, position: int, sample: BootstrapSample, model: object) -> None:
        self._position = position
        self._sample = sample
        self._model = model

    @property
    def position(self) -> int:
        """Which member this is, counting from 0."""
        return self._position

    @property
    def sample(self) -> BootstrapSample:
        """The rows this member drew, and the ones it missed."""
        return self._sample

    @property
    def model(self) -> object:
        """The fitted member."""
        return self._model

    def __repr__(self) -> str:
        return f"MemberFit({self._position}, {self._sample!r})"


class EnsembleFits:
    """Every member of an averaging ensemble, with what each one saw.

    Parameters
    ----------
    members:
        The fitted members, in the order they were fitted.
    member_predictions:
        ``(n_members, n_rows)`` -- what each member said about the training
        rows, before averaging. The spread across a column is the disagreement
        that averaging is exploiting, and is invisible in the answer.
    """

    __slots__ = ("_member_predictions", "_members")

    def __init__(
        self,
        members: Sequence[MemberFit],
        member_predictions: FloatArray,
    ) -> None:
        self._members = tuple(members)
        self._member_predictions = member_predictions

    @property
    def result(self) -> FloatArray:
        """What the ensemble predicts: the members combined."""
        return self._member_predictions.mean(axis=0)

    @property
    def members(self) -> tuple[MemberFit, ...]:
        """The fitted members, in order."""
        return self._members

    @property
    def member_predictions(self) -> FloatArray:
        """``(n_members, n_rows)`` before any combining."""
        return self._member_predictions

    @property
    def disagreement(self) -> FloatArray:
        """``(n_rows,)`` -- how much the members differ on each row.

        The standard deviation across members. Near zero means they all say the
        same thing and the ensemble bought nothing there; large means the
        answer rests on a genuine average rather than a consensus. It is also
        the closest thing a bagged model has to reporting its own uncertainty.
        """
        return self._member_predictions.std(axis=0)

    def running_average(self) -> FloatArray:
        """``(n_members, n_rows)`` -- the answer after 1, 2, ... members.

        What a "how many members do I need" plot draws. It flattens onto the
        floor that the members' correlation sets, and where it flattens is the
        only honest answer to that question.
        """
        totals = np.cumsum(self._member_predictions, axis=0)
        counts = np.arange(1, len(self._members) + 1, dtype=np.float64)

        return totals / counts[:, None]

    def __iter__(self) -> Iterator[MemberFit]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __repr__(self) -> str:
        return f"EnsembleFits({len(self._members)} members)"


class BoostingRound:
    """One round of a boosting fit: what was still wrong, and what was added.

    Parameters
    ----------
    number:
        Which round this is, counting from 1.
    residuals:
        ``(n_rows,)`` -- what the ensemble so far had failed to explain when
        this round began. This is what the round's member was fitted on, and
        watching it shrink is watching boosting work.
    model:
        The member fitted to those residuals.
    learning_rate:
        The fraction of this member's prediction that was actually added.
    training_error:
        The ensemble's error after this round, so a caller can see the point
        where more rounds stop helping and start overfitting.
    """

    __slots__ = (
        "_learning_rate",
        "_model",
        "_number",
        "_residuals",
        "_training_error",
    )

    def __init__(
        self,
        number: int,
        residuals: FloatArray,
        model: object,
        learning_rate: float,
        training_error: float,
    ) -> None:
        self._number = number
        self._residuals = residuals
        self._model = model
        self._learning_rate = learning_rate
        self._training_error = training_error

    @property
    def number(self) -> int:
        """Which round this is, counting from 1."""
        return self._number

    @property
    def residuals(self) -> FloatArray:
        """What was left unexplained when this round began."""
        return self._residuals

    @property
    def model(self) -> object:
        """The member fitted to those residuals."""
        return self._model

    @property
    def learning_rate(self) -> float:
        """The share of this member's prediction that was added."""
        return self._learning_rate

    @property
    def training_error(self) -> float:
        """The ensemble's error after this round."""
        return self._training_error

    def __repr__(self) -> str:
        return f"BoostingRound({self._number}, error {self._training_error:.6f})"


class BoostingRounds:
    """Every round of a boosting fit, in the order they happened.

    Order is the whole structure here, unlike an averaging ensemble where the
    members are peers. Round seven exists because of what rounds one to six
    failed to explain, and reading them out of order says nothing.

    Parameters
    ----------
    rounds:
        The rounds, in order.
    predictions:
        ``(n_rows,)`` -- the ensemble's final answer on the training rows.
    """

    __slots__ = ("_predictions", "_rounds")

    def __init__(
        self, rounds: Sequence[BoostingRound], predictions: FloatArray
    ) -> None:
        self._rounds = tuple(rounds)
        self._predictions = predictions

    @property
    def result(self) -> FloatArray:
        """What the ensemble predicts on the training rows."""
        return self._predictions

    @property
    def rounds(self) -> tuple[BoostingRound, ...]:
        """The rounds, in order."""
        return self._rounds

    @property
    def training_errors(self) -> FloatArray:
        """The error after each round, in order.

        Falls monotonically on the training set almost by construction, which
        is exactly why it must not be read as evidence. The round where a
        *held-out* error stops falling is the one worth finding, and this
        series will keep going down long after that.
        """
        return np.array([one.training_error for one in self._rounds], dtype=np.float64)

    def __iter__(self) -> Iterator[BoostingRound]:
        return iter(self._rounds)

    def __len__(self) -> int:
        return len(self._rounds)

    def __repr__(self) -> str:
        return f"BoostingRounds({len(self._rounds)} rounds)"
