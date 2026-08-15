"""Learned weights, bound to the features they were learned for.

A fitted model usually hands its weights back as a bare ``dict[str, float]``,
or worse as a positional array that you index by remembering the column order.
Both of those are collections of pairs, and a pair of a name and a number is an
object that somebody has not written yet.

:class:`Coefficient` is that object, which is to say one weight that knows which
feature it belongs to. :class:`Coefficients` is the group of them, addressable
by name and iterable as objects, so that nothing downstream ever has to unpack
a two-slot tuple or place its trust in positional order.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)


class Coefficient:
    """One learned weight, together with the feature name it was learned for.

    Parameters
    ----------
    name:
        The feature this weight belongs to.
    value:
        The learned weight: the target's expected change per unit of this
        feature, with every other feature held fixed.
    """

    __slots__ = ("_name", "_value")

    def __init__(self, name: str, value: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("Coefficient name must be a non-empty string")

        self._name = name.strip()
        self._value = float(value)

    @property
    def name(self) -> str:
        """The feature this weight belongs to."""
        return self._name

    @property
    def value(self) -> float:
        """The learned weight."""
        return self._value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coefficient):
            return NotImplemented
        return self._name == other._name and self._value == other._value

    def __hash__(self) -> int:
        return hash((self._name, self._value))

    def __repr__(self) -> str:
        return f"Coefficient(name={self._name!r}, value={self._value!r})"


class Coefficients:
    """The learned weights of a fitted model, addressable by feature name.

    Parameters
    ----------
    coefficients:
        One :class:`Coefficient` per feature. Names have to be unique; a repeat
        would mean one feature's weight silently replacing another's.

    Raises
    ------
    EmptyValuesError
        If no coefficients are supplied.
    NonUniqueFeaturesError
        If two coefficients carry the same name.
    """

    __slots__ = ("_coefficients_by_name",)

    def __init__(self, coefficients: Sequence[Coefficient]) -> None:
        if not coefficients:
            raise EmptyValuesError("at least one coefficient is required")

        coefficients_by_name: dict[str, Coefficient] = {}
        for coefficient in coefficients:
            if coefficient.name in coefficients_by_name:
                raise NonUniqueFeaturesError(
                    f"duplicate coefficient for feature {coefficient.name!r}"
                )
            coefficients_by_name[coefficient.name] = coefficient

        self._coefficients_by_name = coefficients_by_name

    @property
    def n_coefficients(self) -> int:
        """How many weights were learned."""
        return len(self._coefficients_by_name)

    def value_for(self, name: str) -> float:
        """Return the weight learned for feature ``name``.

        Raises
        ------
        InvalidValuesError
            If no coefficient was learned for that feature.
        """
        coefficient = self._coefficients_by_name.get(name)
        if coefficient is None:
            raise InvalidValuesError(
                f"no coefficient for feature {name!r}; this model learned "
                f"{', '.join(sorted(self._coefficients_by_name))}"
            )

        return coefficient.value

    def __getitem__(self, name: str) -> float:
        """The weight for ``name``, so that ``coefficients_["age"]`` reads well."""
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return name in self._coefficients_by_name

    def __iter__(self) -> Iterator[Coefficient]:
        """Iterate the coefficients themselves, not their names or pairs."""
        return iter(self._coefficients_by_name.values())

    def __len__(self) -> int:
        return self.n_coefficients

    def __repr__(self) -> str:
        described = ", ".join(
            f"{coefficient.name}={coefficient.value}" for coefficient in self
        )
        return f"Coefficients({described})"
