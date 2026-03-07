"""What a standardizer learns: one column's centre and spread, bound to its name.

A fitted standardizer holds two numbers per feature. Kept as a bare mapping of
name to a pair, those numbers would be a collection of tuples the caller has to
unpack in the right order, which is the same shape of mistake that
``dict[str, float]`` was for the coefficients.

:class:`FeatureScaling` is the pair as an object, and it owns the arithmetic
that uses it: ``standardize`` and ``restore`` are its behaviour, not a formula
repeated at every call site. :class:`FeatureScalings` is the group, addressable
by feature name.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from oop_ml.exceptions import (
    AllSameValuesError,
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.types import FloatArray


class FeatureScaling:
    """The centre and spread learned for one feature.

    Parameters
    ----------
    name:
        The feature these statistics were learned from.
    mean:
        The column's mean, which is what gets subtracted to centre it.
    standard_deviation:
        The column's spread, which is what it gets divided by. This has to be
        positive, since a constant column has nothing to rescale and would leave
        us dividing by zero.

    Raises
    ------
    InvalidValuesError
        If ``name`` is not a non-empty string.
    AllSameValuesError
        If ``standard_deviation`` is not positive.
    """

    __slots__ = ("_mean", "_name", "_standard_deviation")

    def __init__(self, name: str, mean: float, standard_deviation: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("FeatureScaling name must be a non-empty string")

        if standard_deviation <= 0.0:
            raise AllSameValuesError(
                f"{name} has no spread to standardize by "
                f"(standard deviation {standard_deviation})"
            )

        self._name = name.strip()
        self._mean = float(mean)
        self._standard_deviation = float(standard_deviation)

    @property
    def name(self) -> str:
        """The feature these statistics belong to."""
        return self._name

    @property
    def mean(self) -> float:
        """The learned centre."""
        return self._mean

    @property
    def standard_deviation(self) -> float:
        """The learned spread."""
        return self._standard_deviation

    def standardize(self, values: FloatArray) -> FloatArray:
        """Centre and rescale: ``(values - mean) / standard_deviation``."""
        return (values - self._mean) / self._standard_deviation

    def restore(self, standardized_values: FloatArray) -> FloatArray:
        """Undo :meth:`standardize`, returning values to their original units."""
        return standardized_values * self._standard_deviation + self._mean

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureScaling):
            return NotImplemented
        return (
            self._name == other._name
            and self._mean == other._mean
            and self._standard_deviation == other._standard_deviation
        )

    def __hash__(self) -> int:
        return hash((self._name, self._mean, self._standard_deviation))

    def __repr__(self) -> str:
        return (
            f"FeatureScaling(name={self._name!r}, mean={self._mean!r}, "
            f"standard_deviation={self._standard_deviation!r})"
        )


class FeatureScalings:
    """The statistics a standardizer learned, addressable by feature name.

    Parameters
    ----------
    scalings:
        One :class:`FeatureScaling` per feature, names unique.

    Raises
    ------
    EmptyValuesError
        If no scalings are supplied.
    NonUniqueFeaturesError
        If two scalings carry the same name.
    """

    __slots__ = ("_scalings_by_name",)

    def __init__(self, scalings: Sequence[FeatureScaling]) -> None:
        if not scalings:
            raise EmptyValuesError("at least one feature scaling is required")

        scalings_by_name: dict[str, FeatureScaling] = {}
        for scaling in scalings:
            if scaling.name in scalings_by_name:
                raise NonUniqueFeaturesError(
                    f"duplicate scaling for feature {scaling.name!r}"
                )
            scalings_by_name[scaling.name] = scaling

        self._scalings_by_name = scalings_by_name

    @property
    def n_features(self) -> int:
        """How many features were scaled."""
        return len(self._scalings_by_name)

    def scaling_for(self, name: str) -> FeatureScaling:
        """Return the scaling learned for feature ``name``.

        Raises
        ------
        InvalidValuesError
            If nothing was learned for that feature.
        """
        scaling = self._scalings_by_name.get(name)
        if scaling is None:
            raise InvalidValuesError(
                f"no scaling for feature {name!r}; this standardizer learned "
                f"{', '.join(sorted(self._scalings_by_name))}"
            )

        return scaling

    def __getitem__(self, name: str) -> FeatureScaling:
        return self.scaling_for(name)

    def __contains__(self, name: object) -> bool:
        return name in self._scalings_by_name

    def __iter__(self) -> Iterator[FeatureScaling]:
        return iter(self._scalings_by_name.values())

    def __len__(self) -> int:
        return self.n_features

    def __repr__(self) -> str:
        described = ", ".join(scaling.name for scaling in self)
        return f"FeatureScalings({described})"
