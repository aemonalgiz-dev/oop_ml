"""Try every combination of hyperparameters, and be honest about the winner.

``CrossValidation`` scores one configuration. Nothing in this library searched a
space of them, so choosing ``penalty``, ``n_neighbours`` or ``gamma`` meant
writing the loop by hand every time -- and writing it by hand is where the two
mistakes below get made.

The mistake the type system can prevent
----------------------------------------
A search has to build a model per candidate, and the obvious way is pydantic's
``model_copy(update=...)``. **That silently accepts a name the model does not
have.** Measured::

    RidgeRegression(penalty=1.0).model_copy(update={"pentalty": 5.0})
    -> fit_intercept=True penalty=1.0

No error, and the penalty is still 1.0. A grid over ``pentalty`` would then fit
the same default model at every point, report a flat curve, and pick the first
value by tie-break -- a search that ran, produced a plausible number, and never
varied anything.

This is the same failure ``extra="forbid"`` was added to every model to stop,
and ``model_copy`` walks straight around it. So this module does two things
instead: :class:`ParameterRange` checks the name against the model's declared
fields **at construction**, before any data is seen, and :meth:`Candidate.
applied_to` rebuilds through the ordinary validating constructor rather than
copying. A typo is a construction-time error and a bad value is a validation
error, both of them before a single fold is fitted.

The mistake nothing can prevent, only report
---------------------------------------------
**The winning score is optimistic, and the more candidates you try the more
optimistic it gets.** Every cross-validated score is an estimate with noise on
it. Taking the maximum over many of them selects partly for the configuration
and partly for the noise, so the number that won is biased upward by the very
act of it having won.

Measured on a target that is pure noise, where no configuration can genuinely
beat ``R^2 = 0``, over 25 values of ``n_neighbours``::

    mean candidate score   -0.3215
    best candidate (k=19)  -0.0980   <- what the search reports
    same k, fresh folds    -0.2202   <- what it is worth
    optimism               +0.1222

The winner looks 0.22 better than the average candidate and is worth nothing at
all. Nothing in the search is broken; the maximum of 25 noisy estimates is
simply higher than any of them deserves.

So :attr:`SearchResult.best_score` is documented as a selection score and not a
performance estimate, and :meth:`SearchResult.honest_score_on` exists to get the
number you can actually quote -- refit the winner and score it on rows the
search never touched. Holding a test set out before searching and scoring on it
once afterwards is the cheap version; nesting a second cross-validation loop
around the whole search is the thorough one.

Why grid and not something cleverer
------------------------------------
Grid search tries every combination, so its cost is the product of the ranges:
three parameters at five values each is 125 fits, and each of those is ``k``
model fits. It does not scale, and everyone knows it.

What is less obvious is *why* random search often beats it at equal budget. In
a grid, 125 points over three parameters test only five distinct values of each
-- the other 120 are repeats along that axis. When one parameter matters and
two do not, which is the common case, a grid spends its whole budget
re-measuring the same five values of the one that counts. Random sampling tests
125 distinct values of every parameter. That is a real result and random search
is not implemented here yet.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from itertools import product
from typing import Any

from pydantic import BaseModel, ConfigDict

from oop_ml.core.base.estimator import MultiClassClassifier, Regressor
from oop_ml.core.data.dataset import Dataset
from oop_ml.core.data.feature import Feature
from oop_ml.core.exceptions import EmptyValuesError, InvalidValuesError
from oop_ml.model_selection.cross_validation import (
    CrossValidation,
)
from oop_ml.model_selection.splitting import KFold


class ParameterRange:
    """One hyperparameter, and the values to try for it.

    Parameters
    ----------
    name:
        The field to vary. Checked against the model's declared fields at
        construction, so ``pentalty`` is an error here rather than a flat curve
        several hundred fits later.
    values:
        What to try. Not validated individually -- a value out of range is
        caught by the model's own constructor when the candidate is built,
        which is where that rule already lives and should stay.

    Raises
    ------
    InvalidValuesError
        If ``name`` is not a field of ``model_type``.
    EmptyValuesError
        If no values are supplied, since a range of nothing contributes nothing
        and would silently empty the whole grid.
    """

    __slots__ = ("_name", "_values")

    def __init__(
        self, model_type: type[BaseModel], name: str, values: Sequence[Any]
    ) -> None:
        if name not in model_type.model_fields:
            raise InvalidValuesError(
                f"{model_type.__name__} has no hyperparameter {name!r}; it has "
                f"{sorted(model_type.model_fields)}"
            )

        if not values:
            raise EmptyValuesError(
                f"the range for {name!r} holds no values, so the grid would be empty"
            )

        self._name = name
        self._values = tuple(values)

    @property
    def name(self) -> str:
        """The field this varies."""
        return self._name

    @property
    def values(self) -> tuple[Any, ...]:
        """The values to try, in the order given."""
        return self._values

    def __len__(self) -> int:
        return len(self._values)

    def __repr__(self) -> str:
        return f"ParameterRange({self._name!r}, {len(self._values)} values)"


class Candidate:
    """One complete assignment of hyperparameters, ready to be built.

    Parameters
    ----------
    assignments:
        Field name to value, one entry per searched parameter.

    Raises
    ------
    EmptyValuesError
        If no assignments are supplied.
    """

    __slots__ = ("_assignments",)

    def __init__(self, assignments: dict[str, Any]) -> None:
        if not assignments:
            raise EmptyValuesError("a candidate assigns at least one parameter")

        self._assignments = dict(assignments)

    @property
    def assignments(self) -> dict[str, Any]:
        """What this candidate sets, as a copy."""
        return dict(self._assignments)

    def value_for(self, name: str) -> Any:
        """What this candidate assigns to one parameter.

        Raises
        ------
        InvalidValuesError
            If this candidate does not assign that parameter.
        """
        if name not in self._assignments:
            raise InvalidValuesError(
                f"this candidate assigns {sorted(self._assignments)}, not {name!r}"
            )

        return self._assignments[name]

    def applied_to(self, prototype: BaseModel) -> Any:
        """A fresh model like ``prototype`` but with these values set.

        **Rebuilt through the constructor, never copied.** ``model_copy`` does
        not validate, so a bad name or a bad value would pass through it
        silently; going back through ``type(prototype)(...)`` puts every
        candidate through the same ``extra="forbid"`` and field validation an
        ordinary construction gets.

        A fresh model rather than a mutated one, too. The prototype is the
        caller's, it is reused for every candidate, and a search that left it
        modified would make the result depend on the order the grid ran in.

        The unset fields are read with ``getattr`` rather than taken from
        ``model_dump()``, and that is not a preference. ``model_dump``
        *recurses*: a field holding another pydantic model comes back as a
        plain dict, and rebuilding from it then tries to instantiate the
        field's declared type -- which for ``KernelRidgeRegression.kernel`` is
        the abstract ``Kernel``, and for ``RegressionPipeline.model`` is the
        abstract ``Regressor``. Both raise. ``getattr`` hands back the
        configured object itself, which is what the constructor wants.

        Raises
        ------
        ValidationError
            If a value is outside what the model accepts -- a negative penalty,
            a zero neighbour count. That rule belongs to the model and is left
            there rather than duplicated into the search.
        """
        configured = {
            name: getattr(prototype, name) for name in type(prototype).model_fields
        }

        return type(prototype)(**{**configured, **self._assignments})

    def __repr__(self) -> str:
        described = ", ".join(
            f"{name}={value!r}" for name, value in sorted(self._assignments.items())
        )

        return f"Candidate({described})"


class SearchSpace:
    """Every combination of the ranges, bound to the model type they belong to.

    Parameters
    ----------
    model_type:
        Whose fields the ranges name. Held so the names can be checked once,
        here, rather than discovered wrong during the search.
    ranges:
        One per parameter being varied.

    Raises
    ------
    EmptyValuesError
        If no ranges are supplied.
    InvalidValuesError
        If two ranges name the same parameter, which would make the grid
        ambiguous about which one wins.
    """

    __slots__ = ("_model_type", "_ranges")

    def __init__(
        self, model_type: type[BaseModel], ranges: Sequence[ParameterRange]
    ) -> None:
        if not ranges:
            raise EmptyValuesError("a search space varies at least one parameter")

        names = [one.name for one in ranges]

        if len(set(names)) != len(names):
            raise InvalidValuesError(f"a parameter can be varied once; got {names}")

        self._model_type = model_type
        self._ranges = tuple(ranges)

    @classmethod
    def over(cls, model_type: type[BaseModel], **ranges: Sequence[Any]) -> SearchSpace:
        """Build a space from keyword arguments, which is how it usually reads.

        ``SearchSpace.over(RidgeRegression, penalty=[0.1, 1.0, 10.0])`` rather
        than assembling ``ParameterRange`` objects by hand. The names are still
        checked against the model's fields, because that is
        ``ParameterRange``'s job and this only builds them.
        """
        return cls(
            model_type,
            [
                ParameterRange(model_type, name, values)
                for name, values in ranges.items()
            ],
        )

    @property
    def model_type(self) -> type[BaseModel]:
        """The model whose fields these ranges name."""
        return self._model_type

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Which parameters this varies, in the order given."""
        return tuple(one.name for one in self._ranges)

    @property
    def n_candidates(self) -> int:
        """How many models a full grid search would fit, before folds.

        The product of the ranges' lengths, and worth reading before starting:
        multiply it by ``n_folds`` for the number of fits the search will
        actually perform.
        """
        total = 1

        for one in self._ranges:
            total *= len(one)

        return total

    def candidates(self) -> list[Candidate]:
        """Every combination of one value from each range.

        The Cartesian product. With ``penalty`` over three values and
        ``fit_intercept`` over two, that is six candidates, each assigning both
        parameters -- not five, and not two separate one-parameter sweeps.

        Combinations rather than sweeps is the whole point of a grid, and it is
        also its cost: parameters interact, so the best penalty at one
        intercept setting need not be the best at the other, and only trying
        them together can find that out.

        ``itertools.product`` over the ranges' values gives the tuples; pair
        each tuple with :attr:`parameter_names` to make the assignment.

        Returns
        -------
        list[Candidate]
            One per combination, ``n_candidates`` of them.
        """
        names = self.parameter_names

        return [
            Candidate(dict(zip(names, combination, strict=True)))
            for combination in product(*(one.values for one in self._ranges))
        ]

    def __iter__(self) -> Iterator[ParameterRange]:
        return iter(self._ranges)

    def __len__(self) -> int:
        return len(self._ranges)

    def __repr__(self) -> str:
        return (
            f"SearchSpace({self._model_type.__name__}, "
            f"{list(self.parameter_names)}, {self.n_candidates} candidates)"
        )


class ScoredCandidate:
    """A candidate paired with what it scored, which is why it is an object.

    A search produces two things per candidate and they have to stay together;
    returning parallel lists would leave the caller matching them up by
    position.

    Parameters
    ----------
    candidate:
        What was tried.
    score:
        Its mean cross-validated score. Higher is better for both tasks --
        ``R^2`` for a regressor, pooled accuracy for a classifier -- so the
        comparison below does not need to know which it has.
    """

    __slots__ = ("_candidate", "_score")

    def __init__(self, candidate: Candidate, score: float) -> None:
        self._candidate = candidate
        self._score = float(score)

    @property
    def candidate(self) -> Candidate:
        """What was tried."""
        return self._candidate

    @property
    def score(self) -> float:
        """What it scored, cross-validated."""
        return self._score

    def beats(self, other: ScoredCandidate) -> bool:
        """Whether this scored better than ``other``.

        Strictly, so a tie goes to the incumbent -- the same rule
        :meth:`~oop_ml.core.tree.split.Split.beats` and
        ``InitialisationAttempt.beats`` follow. Two candidates that are
        genuinely equivalent come back differing in the last bits, and swapping
        on a tie would make the winner depend on the order the grid ran in.
        """
        return self._score > other._score

    def __repr__(self) -> str:
        return f"ScoredCandidate({self._candidate!r}, score={self._score:.4f})"


class SearchResult:
    """Every candidate and what it scored, with the winner named.

    Parameters
    ----------
    scored:
        One per candidate, in the order the search ran them.

    Raises
    ------
    EmptyValuesError
        If no candidates were scored.
    """

    __slots__ = ("_scored",)

    def __init__(self, scored: Sequence[ScoredCandidate]) -> None:
        if not scored:
            raise EmptyValuesError("a search scores at least one candidate")

        self._scored = tuple(scored)

    @property
    def best(self) -> ScoredCandidate:
        """The highest-scoring candidate, ties going to the earlier one."""
        winner = self._scored[0]

        for contender in self._scored[1:]:
            if contender.beats(winner):
                winner = contender

        return winner

    @property
    def best_score(self) -> float:
        """What the winner scored -- **a selection score, not a performance one**.

        This number is optimistic, and the more candidates were tried the more
        optimistic it is, because taking a maximum over noisy estimates selects
        for the noise as well as for the configuration. The module docstring
        has the measurement: on a target that is pure noise the winner of 25
        candidates reported -0.0980 and was worth -0.2202.

        Use it to compare candidates, which is what it is honest about. Do not
        quote it as what the chosen model achieves -- see
        :meth:`honest_score_on`.
        """
        return self.best.score

    @property
    def score_spread(self) -> float:
        """Best minus worst across the candidates.

        Read it beside the winner. A spread of 0.002 says the search found
        nothing to choose between and the winner is a coin toss; a spread of
        0.4 says the hyperparameter genuinely matters here.
        """
        scores = [one.score for one in self._scored]

        return max(scores) - min(scores)

    @property
    def n_candidates(self) -> int:
        """How many configurations were scored."""
        return len(self._scored)

    def ranked(self) -> list[ScoredCandidate]:
        """Every candidate, best first."""
        return sorted(self._scored, key=lambda one: one.score, reverse=True)

    def best_model(self, prototype: BaseModel) -> Any:
        """An unfitted model configured as the winner was.

        Unfitted on purpose. The models the search built were fitted to folds,
        and the one worth keeping is refitted on everything -- so this hands
        back the configuration and leaves the fitting to the caller, who is the
        only one who knows which rows to use.
        """
        return self.best.candidate.applied_to(prototype)

    def honest_score_on(
        self,
        prototype: Regressor[Sequence[Feature], Feature],
        holdout: Dataset,
    ) -> float:
        """Refit the winner on nothing, and score it on rows the search never saw.

        The number to quote. :attr:`best_score` was chosen *because* it was the
        largest, so it carries the selection's optimism; this one was not used
        to choose anything, so it does not.

        ``holdout`` must be data the search never touched -- held out before
        the search began, not carved off afterwards. Passing the rows the
        search cross-validated over gives a number with the same problem and no
        warning attached.

        Raises
        ------
        EmptyValuesError, NonEqualArrayLengthError
            From the underlying fit, if ``holdout`` is malformed.
        """
        model = self.best_model(prototype)
        model.fit(holdout.input_features, holdout.target_feature)

        return model.score(holdout.input_features, holdout.target_feature)

    def __iter__(self) -> Iterator[ScoredCandidate]:
        return iter(self._scored)

    def __len__(self) -> int:
        return self.n_candidates

    def __repr__(self) -> str:
        return (
            f"SearchResult({self.n_candidates} candidates, "
            f"best={self.best.candidate!r}, best_score={self.best_score:.4f})"
        )


class GridSearch(BaseModel):
    """Cross-validate every combination in a space and keep the best.

    Parameters
    ----------
    folds:
        The splitter each candidate is scored with. The *same* splitter for
        every candidate, which matters: comparing a score from one fold
        arrangement against a score from another compares the arrangements as
        much as the candidates.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    folds: KFold = KFold()

    def search(
        self,
        prototype: Regressor[Sequence[Feature], Feature],
        space: SearchSpace,
        dataset: Dataset,
    ) -> SearchResult:
        """Score every candidate as a regressor, by mean cross-validated R^2.

        Raises
        ------
        InvalidValuesError
            If ``space`` was built for a different model type than
            ``prototype``.
        TooFewValuesError
            If the dataset has fewer rows than folds.
        """
        self._check_space_matches(prototype, space)

        return SearchResult(
            self._scored_candidates(
                prototype,
                space,
                dataset,
                lambda result: result.mean_r2_score,
                CrossValidation(folds=self.folds).evaluate,
            )
        )

    def search_classifier(
        self,
        prototype: MultiClassClassifier[Sequence[Feature], Feature],
        space: SearchSpace,
        dataset: Dataset,
    ) -> SearchResult:
        """Score every candidate as a classifier, by pooled accuracy.

        Pooled rather than averaged, for the reason the cross-validation module
        records: a classifier's metrics are ratios over rows, and averaging
        ``k`` of them lets a small fold count as much as a large one.

        Two methods rather than one that inspects its argument, following the
        seam ``CrossValidation`` already has.

        Raises
        ------
        InvalidValuesError
            If ``space`` was built for a different model type than
            ``prototype``.
        """
        self._check_space_matches(prototype, space)

        return SearchResult(
            self._scored_candidates(
                prototype,
                space,
                dataset,
                lambda result: result.pooled_accuracy,
                CrossValidation(folds=self.folds).evaluate_classifier,
            )
        )

    def _scored_candidates(
        self,
        prototype: Any,
        space: SearchSpace,
        dataset: Dataset,
        read_score: Callable[[Any], float],
        cross_validate: Callable[[Any, Dataset], Any],
    ) -> list[ScoredCandidate]:
        """Build, cross-validate and score every candidate in the space.

        The loop both entry points share, and the only part of this class that
        does anything. For each candidate the space produces:

        1. Build a model from it with ``candidate.applied_to(prototype)``. A
           fresh one every time -- the prototype must come out of the search
           exactly as it went in, or the result depends on grid order.
        2. Cross-validate it with ``cross_validate``, which is already bound to
           this search's folds.
        3. Read one number off the result with ``read_score``, which is what
           differs between the two tasks and the only thing that does.
        4. Pair the two in a ``ScoredCandidate``.

        Note what is *not* here: no comparison, no tracking of a best-so-far.
        Scoring every candidate and letting ``SearchResult`` find the winner
        keeps the losing scores, and the losing scores are what
        ``score_spread`` reads to tell a caller whether the winner meant
        anything.

        Parameters
        ----------
        prototype:
            The model to vary. Never modified.
        space:
            The combinations to try.
        dataset:
            What every candidate is cross-validated over.
        read_score:
            Takes a cross-validation result, returns the one number to compare.
        cross_validate:
            Takes a model and the dataset, returns a cross-validation result.

        Returns
        -------
        list[ScoredCandidate]
            One per candidate, in the order the space produced them.
        """
        return [
            ScoredCandidate(
                candidate,
                read_score(cross_validate(candidate.applied_to(prototype), dataset)),
            )
            for candidate in space.candidates()
        ]

    @staticmethod
    def _check_space_matches(prototype: Any, space: SearchSpace) -> None:
        """Raise unless the space names fields of this prototype's own type.

        The names were checked against ``space.model_type`` when the space was
        built, so a space handed to the wrong model would put that check
        somewhere it does not apply -- and every candidate would then be
        rebuilt through a constructor that rejects the names, hundreds of fits
        after the mistake was made.
        """
        if type(prototype) is not space.model_type:
            raise InvalidValuesError(
                f"this space varies {space.model_type.__name__} parameters, but "
                f"the prototype is a {type(prototype).__name__}"
            )

    def __repr__(self) -> str:
        return f"GridSearch(folds={self.folds!r})"
