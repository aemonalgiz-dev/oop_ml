"""How much each feature mattered, bound to the feature it is talking about.

The same argument :mod:`~oop_ml.core.data.coefficients` makes: a bare array of
numbers that you index by remembering the column order is a collection of pairs,
and a pair of a name and a number is an object nobody has written yet. What is
different here is the invariant. A coefficient can be any sign and any size;
importances are non-negative and normalised to sum to one, so a reader can say
"this feature accounts for a fifth of it" without first working out the total.

Because the raw quantity is always a sum of positive contributions -- impurity
removed, or accuracy lost -- ``from_scores`` is the constructor that actually
gets used, and the plain one exists for shares that are already normalised.

The normalisation is also the sharpest limitation. Importances are *shares of
this model's explanation*, not measurements of the world. Two features carrying
the same information split that share between them and each look half as
important as either would alone, which is a fact about the fit rather than
about the features.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol, runtime_checkable

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonEqualArrayLengthError,
    NonUniqueFeaturesError,
)

# Shares come from summing floats, so they land near 1 rather than on it.
IMPORTANCE_SUM_TOLERANCE = 1e-9


class FeatureContribution:
    """One raw, unnormalised credit to a feature, from one place.

    Separate from :class:`FeatureImportance` because the invariants are
    opposite. An importance is a *share*: bounded by 1, and one per feature.
    A contribution is a *part*: unbounded, and a feature collects as many as it
    earned. A tree node contributes ``n_samples * gain`` and a feature winning
    six splits contributes six times; an ensemble member contributes its own
    share of the explanation and every member contributes once.

    Collapsing the two into one class would mean either dropping the bound that
    makes a share meaningful, or forbidding the repetition that makes a
    contribution useful.

    Parameters
    ----------
    name:
        The feature being credited.
    amount:
        How much, in whatever units the producer is working in. Non-negative,
        because both producers measure something removed or lost and neither
        can sensibly go backwards.

    Raises
    ------
    InvalidValuesError
        If the name is blank or the amount is negative.
    """

    __slots__ = ("_amount", "_name")

    def __init__(self, name: str, amount: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("Feature name must be a non-empty string")
        if float(amount) < 0.0:
            raise InvalidValuesError(
                f"Contribution to {name!r} must be non-negative, got {amount}"
            )

        self._name = name
        self._amount = float(amount)

    @property
    def name(self) -> str:
        """The feature being credited."""
        return self._name

    @property
    def amount(self) -> float:
        """How much, unnormalised."""
        return self._amount

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureContribution):
            return NotImplemented
        return self._name == other._name and self._amount == other._amount

    def __hash__(self) -> int:
        return hash((self._name, self._amount))

    def __repr__(self) -> str:
        return f"FeatureContribution({self._name!r}, {self._amount:.4f})"


@runtime_checkable
class Reports(Protocol):
    """Anything fitted that can say how much each feature mattered.

    Structural rather than a base class, so that
    :class:`~oop_ml.core.base.ensemble.AveragingEnsemble` can average its
    members' importances without the frame having to know or care that its
    members are trees. The module docstring there is explicit that neither
    frame cares what its members are, and importing ``TreeModel`` to narrow
    the type would quietly make that untrue.
    """

    @property
    def feature_importances(self) -> FeatureImportances:
        """Every feature's share of this model's explanation."""
        ...


class FeatureImportance:
    """One share of the explanation, together with whose share it is.

    Parameters
    ----------
    name:
        The feature this share belongs to.
    value:
        Its share, between 0 and 1 inclusive.

    Raises
    ------
    InvalidValuesError
        If the name is blank or the value falls outside ``[0, 1]``.
    """

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("Feature name must be a non-empty string")
        if not 0.0 <= float(value) <= 1.0:
            raise InvalidValuesError(
                f"Importance for {name!r} must lie in [0, 1], got {value}"
            )

        self._name = name
        self._value = float(value)

    @property
    def name(self) -> str:
        """The feature this share belongs to."""
        return self._name

    @property
    def value(self) -> float:
        """Its share of the explanation, between 0 and 1."""
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureImportance):
            return NotImplemented
        return self._name == other._name and self._value == other._value

    def __hash__(self) -> int:
        return hash((self._name, self._value))

    def __repr__(self) -> str:
        return f"FeatureImportance({self._name!r}, {self._value:.4f})"


class FeatureImportances:
    """Every feature's share, addressable by name and rankable.

    Parameters
    ----------
    importances:
        One entry per feature the model was fitted on, including the ones that
        earned nothing. A feature scoring zero is a finding, and dropping it
        would hide that finding.

    Raises
    ------
    EmptyValuesError
        If no importances are supplied.
    NonUniqueFeaturesError
        If two entries share a name.
    InvalidValuesError
        If the shares do not sum to 1.
    """

    __slots__ = ("_by_name", "_importances")

    def __init__(self, importances: Sequence[FeatureImportance]) -> None:
        if not importances:
            raise EmptyValuesError("At least one feature importance is required")

        by_name: dict[str, float] = {}
        for importance in importances:
            if importance.name in by_name:
                raise NonUniqueFeaturesError(
                    f"Duplicate feature name in importances: {importance.name!r}"
                )
            by_name[importance.name] = importance.value

        total = sum(by_name.values())
        if abs(total - 1.0) > IMPORTANCE_SUM_TOLERANCE:
            raise InvalidValuesError(f"Feature importances must sum to 1, got {total}")

        self._importances = tuple(importances)
        self._by_name = by_name

    @classmethod
    def from_scores(
        cls, names: Sequence[str], scores: Sequence[float]
    ) -> FeatureImportances:
        """Normalise raw per-feature scores into shares.

        The constructor every producer actually wants, because impurity removed
        and accuracy lost are both unbounded sums of positive contributions.

        Parameters
        ----------
        names:
            The feature names, in the order the fit saw them.
        scores:
            One non-negative raw score per name, same length and order.

        Raises
        ------
        EmptyValuesError
            If no names are supplied.
        NonEqualArrayLengthError
            If the two sequences differ in length.
        InvalidValuesError
            If any score is negative, or if they total zero. A model that used
            no feature at all -- a tree that never found a split worth making
            -- has no shares to report, and the caller needs to hear that
            rather than receive a vector of zeroes that looks like an answer.
        """
        if not names:
            raise EmptyValuesError("At least one feature name is required")
        if len(names) != len(scores):
            raise NonEqualArrayLengthError(
                f"Got {len(names)} names and {len(scores)} scores"
            )
        if any(score < 0.0 for score in scores):
            raise InvalidValuesError("Feature scores must be non-negative")

        total = float(sum(scores))
        if total <= 0.0:
            raise InvalidValuesError(
                "No feature earned anything, so there are no shares to report"
            )

        return cls(
            [
                FeatureImportance(name, float(score) / total)
                for name, score in zip(names, scores, strict=True)
            ]
        )

    @classmethod
    def from_contributions(
        cls,
        names: Sequence[str],
        contributions: Iterable[FeatureContribution],
    ) -> FeatureImportances:
        """Total the contributions by name, then normalise.

        The constructor for producers that credit a feature more than once. A
        tree walk appends one contribution per node it visits and never has to
        keep a running total of its own; an ensemble appends one per member per
        feature. Both leave the accumulating here rather than hand-rolling the
        same dictionary twice.

        Parameters
        ----------
        names:
            Every feature the model was fitted on, in the order it saw them.
            Supplied separately from the contributions because a feature that
            never earned anything contributes nothing and would otherwise be
            missing entirely, and a zero is a finding rather than an absence.
        contributions:
            Any number per feature, in any order.

        Raises
        ------
        EmptyValuesError
            If no names are supplied.
        InvalidValuesError
            If a contribution names a feature not in ``names``, or if they
            total zero.
        """
        if not names:
            raise EmptyValuesError("At least one feature name is required")

        totals = dict.fromkeys(names, 0.0)
        for contribution in contributions:
            if contribution.name not in totals:
                known = ", ".join(names)
                raise InvalidValuesError(
                    f"Contribution names unknown feature "
                    f"{contribution.name!r}. Known features: {known}"
                )
            totals[contribution.name] += contribution.amount

        return cls.from_scores(names, [totals[name] for name in names])

    @property
    def n_features(self) -> int:
        """How many features the model was fitted on."""
        return len(self._importances)

    @property
    def most_important(self) -> FeatureImportance:
        """The largest share. Ties go to whichever name the fit saw first."""
        return max(self._importances, key=lambda one: one.value)

    def ranked(self) -> tuple[FeatureImportance, ...]:
        """Every feature, largest share first.

        A method rather than the iteration order, because two callers want
        different things: a report wants this, and anything lining these up
        against another model's importances wants the order the fit saw.
        """
        return tuple(sorted(self._importances, key=lambda one: one.value, reverse=True))

    def value_for(self, name: str) -> float:
        """The share earned by ``name``.

        Raises
        ------
        InvalidValuesError
            If no feature of that name was fitted.
        """
        if name not in self._by_name:
            known = ", ".join(sorted(self._by_name))
            raise InvalidValuesError(
                f"No importance for feature {name!r}. Known features: {known}"
            )

        return self._by_name[name]

    def __getitem__(self, name: str) -> float:
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[FeatureImportance]:
        return iter(self._importances)

    def __len__(self) -> int:
        return self.n_features

    def __repr__(self) -> str:
        leading = ", ".join(f"{one.name}={one.value:.3f}" for one in self.ranked()[:3])
        return (
            f"FeatureImportances({leading}, ...)"
            if self.n_features > 3
            else (f"FeatureImportances({leading})")
        )
