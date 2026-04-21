"""What an affine scaler learned, a centre and a spread per feature.

Every affine scaler answers the same way, ``(value - centre) / spread``, so
what a fitted one hands back is the same pair of numbers per column whichever
backend fitted it and whichever of the four rules chose them. That makes these
two shared vocabulary rather than one backend's own.

They are deliberately not the same type as
:class:`~oop_ml.core.preprocessing.feature_scalings.FeatureScaling`, which
names its pair ``mean`` and ``standard_deviation``. Two vocabularies for one
idea is a duplication worth keeping, because those names are written into
saved documents and a centre that is a median is not a mean.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.exceptions import (
    AllSameValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.core.types import FloatArray


class AffineScaling:
    """One column's centre and spread, bound to the column's name.

    Named separately from
    :class:`~oop_ml.core.preprocessing.feature_scalings.FeatureScaling`
    because that one is specifically a mean and a standard deviation, with those
    two words in its public API and in the saved-document format. This is the
    general pair, and the module docstring explains why the two coexist.

    Parameters
    ----------
    name:
        The feature this describes.
    centre:
        What is subtracted before dividing. Zero for the scalers that do not
        centre.
    spread:
        What the centred value is divided by. Strictly positive.

    Raises
    ------
    InvalidValuesError
        If either number is not finite.
    AllSameValuesError
        If the spread is zero or negative. A column with no spread cannot be
        rescaled, and substituting a one would silently answer a question nobody
        asked.
    """

    __slots__ = ("_centre", "_name", "_spread")

    def __init__(self, name: str, centre: float, spread: float) -> None:
        centre_value = float(centre)
        spread_value = float(spread)

        if not np.isfinite(centre_value) or not np.isfinite(spread_value):
            raise InvalidValuesError(
                f"the scaling for {name!r} must be finite, got centre "
                f"{centre_value} and spread {spread_value}"
            )
        if spread_value <= 0.0:
            raise AllSameValuesError(
                f"{name!r} has a spread of {spread_value}, so it cannot be "
                "rescaled; a column carrying no variation has nothing to divide by"
            )

        self._name = name
        self._centre = centre_value
        self._spread = spread_value

    @property
    def name(self) -> str:
        """The feature this scaling belongs to."""
        return self._name

    @property
    def centre(self) -> float:
        """What is subtracted before dividing."""
        return self._centre

    @property
    def spread(self) -> float:
        """What the centred value is divided by."""
        return self._spread

    def scale(self, values: FloatArray) -> FloatArray:
        """``(values - centre) / spread``."""
        return (values - self._centre) / self._spread

    def restore(self, scaled_values: FloatArray) -> FloatArray:
        """The inverse, so a caller can read an answer back in original units."""
        return scaled_values * self._spread + self._centre

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AffineScaling):
            return NotImplemented
        return (
            self._name == other._name
            and self._centre == other._centre
            and self._spread == other._spread
        )

    def __hash__(self) -> int:
        return hash((self._name, self._centre, self._spread))

    def __repr__(self) -> str:
        return (
            f"AffineScaling(name={self._name!r}, centre={self._centre!r}, "
            f"spread={self._spread!r})"
        )


class AffineScalings:
    """Every column's scaling, addressable by name.

    Iterable rather than handing out its container, so nothing outside can
    reorder the scalings and silently transpose which column each one describes.

    Raises
    ------
    NonUniqueFeaturesError
        If two scalings share a name.
    """

    __slots__ = ("_by_name", "_scalings")

    def __init__(self, scalings: Sequence[AffineScaling]) -> None:
        by_name: dict[str, AffineScaling] = {}
        for scaling in scalings:
            if scaling.name in by_name:
                raise NonUniqueFeaturesError(
                    f"duplicate feature name: {scaling.name!r}"
                )
            by_name[scaling.name] = scaling

        self._scalings = tuple(scalings)
        self._by_name = by_name

    @property
    def n_features(self) -> int:
        """How many columns were learned."""
        return len(self._scalings)

    @property
    def names(self) -> tuple[str, ...]:
        """The feature names, in the order they were learned."""
        return tuple(scaling.name for scaling in self._scalings)

    def scaling_for(self, name: str) -> AffineScaling:
        """The scaling belonging to one feature.

        Raises
        ------
        NotFittedError
            Never. See :meth:`FeatureScaler.transform` for the unknown-name case.
        KeyError
            If the name was not among the fitted features.
        """
        if name not in self._by_name:
            raise KeyError(name)
        return self._by_name[name]

    def __getitem__(self, name: str) -> AffineScaling:
        return self.scaling_for(name)

    def __contains__(self, name: object) -> bool:
        return name in self._by_name

    def __iter__(self) -> Iterator[AffineScaling]:
        return iter(self._scalings)

    def __len__(self) -> int:
        return len(self._scalings)

    def __repr__(self) -> str:
        return f"AffineScalings(n_features={self.n_features!r})"
