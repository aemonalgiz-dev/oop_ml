"""A direction found in a space that was never built.

Kernel principal component analysis cannot reuse the ordinary component
vocabulary, and the reason is worth stating rather than working around. An
ordinary principal component binds a weight to each feature name. Here the
directions live in whatever space the kernel implies, whose coordinates have
no names at all, so what a component has instead is one coefficient per
training row.

That also means these carry no orthogonality check, where
:class:`~oop_ml.core.decomposition.components.PrincipalComponents` does. The
coefficient vectors are orthogonal in the metric the kernel induces rather
than the ordinary one, so asserting that their dot products vanish would be
asserting something false.
"""

from __future__ import annotations

from collections.abc import (
    Iterator,
    Sequence,
)

import numpy as np

from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
)
from oop_ml.core.types import FloatArray

KERNEL_COMPONENT_NAME_PREFIX = "kernel_component"
"""How kernel components are named, distinctly from ordinary ones.

Not ``component_1``. A caller holding both a ``PrincipalComponentAnalysis`` and
one of these should not be able to feed the output of either into the other by
accident, and identical column names are exactly what would let them.
"""

MINIMUM_COMPONENT_VARIANCE = 1e-10
"""Below this, a component is numerical noise rather than a direction.

A centred Gram matrix is singular by construction -- centring removes one degree
of freedom, so at least one eigenvalue is zero -- and the corresponding
eigenvector is whatever the solver happened to return in the null space.
Normalising it divides by nearly zero and produces a direction of pure rounding.
"""


class KernelComponent:
    """One direction in the implied space, named by the rows it is built from.

    Parameters
    ----------
    name:
        What this direction is called, by position.
    row_coefficients:
        One weight per training row. The direction is the combination of the
        (centred, implied) training points these weights describe. There is no
        per-feature reading of this, and that absence is the point -- see the
        module docstring.
    variance:
        The variance of the implied data along this direction.

    Raises
    ------
    InvalidValuesError
        If the name is blank, the coefficients are not a finite vector, or the
        variance is negative.
    """

    __slots__ = ("_name", "_row_coefficients", "_variance")

    def __init__(
        self, name: str, row_coefficients: FloatArray, variance: float
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError("KernelComponent name must be non-empty")

        as_array = np.asarray(row_coefficients, dtype=np.float64)

        if as_array.ndim != 1 or not np.all(np.isfinite(as_array)):
            raise InvalidValuesError(
                f"{name} must hold one finite coefficient per training row; got "
                f"shape {as_array.shape}"
            )

        if variance < 0.0:
            raise InvalidValuesError(
                f"{name} has variance {variance}, which cannot be negative"
            )

        self._name = name.strip()
        self._row_coefficients = as_array
        self._variance = float(variance)

    @property
    def name(self) -> str:
        """What this direction is called."""
        return self._name

    @property
    def row_coefficients(self) -> FloatArray:
        """How much each training row contributes to this direction."""
        return self._row_coefficients.copy()

    @property
    def variance(self) -> float:
        """The variance of the implied data along this direction."""
        return self._variance

    @property
    def n_training_rows(self) -> int:
        """How many rows this direction is built out of."""
        return int(self._row_coefficients.size)

    def __repr__(self) -> str:
        return (
            f"KernelComponent({self._name!r}, variance={self._variance:.4f}, "
            f"n_training_rows={self.n_training_rows})"
        )


class KernelComponents:
    """The kept directions, ordered by the variance along them.

    The ordering invariant is here for the same reason it is on
    :class:`~oop_ml.core.decomposition.components.PrincipalComponents`: ``eigh``
    returns ascending, and an unreversed sort produces a model that transforms
    perfectly well while reporting its worst direction as its best.

    There is no orthogonality check, and that absence is not an oversight. These
    coefficient vectors are orthogonal in the metric the kernel induces, not in
    the ordinary one -- ``u_i . u_j`` is not zero for them, and asserting that it
    should be would be asserting something false. What is true is that the
    implied directions are orthogonal, and that cannot be checked without the
    space nobody built.

    Raises
    ------
    EmptyValuesError
        If no components are supplied.
    InvalidValuesError
        If the variances increase, if the components are built from different
        numbers of rows, or if ``total_variance`` is not positive and at least
        their sum.
    """

    __slots__ = ("_components", "_total_variance")

    def __init__(
        self, components: Sequence[KernelComponent], total_variance: float
    ) -> None:
        if not components:
            raise EmptyValuesError("a decomposition needs at least one component")

        self._components = tuple(components)
        self._total_variance = float(total_variance)

        expected = self._components[0].n_training_rows
        for component in self._components[1:]:
            if component.n_training_rows != expected:
                raise InvalidValuesError(
                    f"{component.name} is built from {component.n_training_rows} "
                    f"rows against {expected}"
                )

        for earlier, later in zip(self._components, self._components[1:], strict=False):
            if later.variance > earlier.variance + 1e-12:
                raise InvalidValuesError(
                    f"components must be ordered by decreasing variance; "
                    f"{later.name} explains {later.variance} against "
                    f"{earlier.name}'s {earlier.variance}"
                )

        if self._total_variance <= 0.0:
            raise InvalidValuesError(
                f"total variance must be positive; got {self._total_variance}"
            )

        if self.kept_variance > self._total_variance * (1.0 + 1e-09):
            raise InvalidValuesError(
                f"components explain {self.kept_variance}, more than the stated "
                f"total of {self._total_variance}"
            )

    @property
    def n_components(self) -> int:
        """How many directions were kept."""
        return len(self._components)

    @property
    def n_training_rows(self) -> int:
        """How many rows every direction is built from."""
        return self._components[0].n_training_rows

    @property
    def total_variance(self) -> float:
        """The variance over every direction, kept and discarded."""
        return self._total_variance

    @property
    def kept_variance(self) -> float:
        """The variance along the directions held here."""
        return float(sum(component.variance for component in self._components))

    @property
    def coefficients(self) -> FloatArray:
        """The directions as a matrix, one component per row.

        Shape ``(n_components, n_training_rows)``, which is the orientation the
        projection wants.
        """
        return np.array(
            [component.row_coefficients for component in self._components],
            dtype=np.float64,
        )

    @property
    def variance_shares(self) -> tuple[float, ...]:
        """Each direction's variance over the total across all of them."""
        return tuple(
            component.variance / self._total_variance for component in self._components
        )

    @property
    def cumulative_shares(self) -> tuple[float, ...]:
        """The running total of :attr:`variance_shares`."""
        return tuple(float(total) for total in np.cumsum(self.variance_shares))

    def value_for(self, name: str) -> KernelComponent:
        """The component called ``name``.

        Raises
        ------
        InvalidValuesError
            If no component has that name.
        """
        for component in self._components:
            if component.name == name:
                return component

        raise InvalidValuesError(
            f"unknown component {name!r}; this holds "
            f"{[component.name for component in self._components]}"
        )

    def __getitem__(self, name: str) -> KernelComponent:
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return any(component.name == name for component in self._components)

    def __iter__(self) -> Iterator[KernelComponent]:
        return iter(self._components)

    def __len__(self) -> int:
        return self.n_components

    def __repr__(self) -> str:
        return f"KernelComponents(n_components={self.n_components})"
