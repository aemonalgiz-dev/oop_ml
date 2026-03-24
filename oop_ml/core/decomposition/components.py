"""What a decomposition learns: directions through the data, ranked by spread.

A fitted PCA holds a set of directions and, for each, how much of the data's
variance lies along it. Kept as a bare ``(n_components, n_features)`` array and
a parallel vector of eigenvalues, those are two things a caller has to keep in
step by hand, in an order nothing enforces, addressed by a position nothing
names. That is the same shape of mistake ``dict[str, float]`` was for the
coefficients.

:class:`PrincipalComponent` is one direction as an object: its loadings bound to
the feature names they weight, and the variance along it. :class:`Principal
Components` is the ordered group, and it owns the three rules that make a set of
directions a *decomposition* rather than a pile of vectors.

Two of those rules exist to make a class of bug unwriteable.

**Ordering.** ``numpy.linalg.eigh`` returns eigenvalues in *ascending* order,
which is the reverse of what PCA means by "the first component". Forgetting to
reverse it produces a fitted model that runs, transforms, and reconstructs --
and hands back the least informative direction as the most important one. Every
number is finite and plausible, so nothing downstream can notice. The
constructor refuses a non-decreasing run instead.

**Orthogonality.** The components of a real decomposition are perpendicular:
that is what makes the transformed columns uncorrelated, and it is what makes
``inverse_transform`` a transpose rather than a matrix inversion. A set that has
lost it will still project and still reconstruct, approximately, and the error
will look like ordinary numerical noise.

The third rule is about a denominator rather than a bug.

**Total variance is supplied, not summed.** A share of explained variance is
that component's variance over the variance of *the whole data*, and once a
caller keeps two components out of five, those are different numbers. Summing
what the object happens to hold would make every truncated decomposition claim
to explain 100% of everything, which is exactly the claim a caller is asking
about when they ask for the share at all.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from oop_ml.core.data.coefficients import Coefficients
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    NonUniqueFeaturesError,
)
from oop_ml.core.types import FloatArray

UNIT_LENGTH_TOLERANCE = 1e-08
"""How far a direction's length may sit from 1 before it is refused.

A direction is a unit vector by definition. It arrives here from an
eigendecomposition, so it is unit to within the solver's own rounding and no
further; anything appreciably off is a rescaled or a truncated vector rather
than a rounded one.
"""

ORTHOGONALITY_TOLERANCE = 1e-08
"""How far two directions' dot product may sit from 0 before they are refused.

Same argument as :data:`UNIT_LENGTH_TOLERANCE`, and the same order of magnitude,
because both quantities come out of the same decomposition.
"""

VARIANCE_ORDER_TOLERANCE = 1e-12
"""Slack allowed when checking that variances do not increase.

Two components explaining genuinely equal variance -- a perfectly symmetric
dataset, or two directions through pure noise -- come back differing in the last
bits, and the order between them is arbitrary rather than wrong. Comparing
strictly would refuse a decomposition that is correct.
"""


class PrincipalComponent:
    """One direction through the data, and the variance lying along it.

    Parameters
    ----------
    name:
        What this component is called. The transformer names them by position
        (``component_1`` and so on), because a direction through several
        features has no name of its own the way a feature does.
    loadings:
        The direction, as one weight per original feature bound to that
        feature's name. Reusing
        :class:`~oop_ml.core.data.coefficients.Coefficients` here is not a
        convenience: a loading really is a weight bound to a name, addressed
        the same way and read for the same reason -- ``loadings["floor_area"]``
        answers "how much does this component lean on floor area".
    variance:
        The variance of the data along this direction. Non-negative, since it
        is a variance.

    Raises
    ------
    InvalidValuesError
        If ``name`` is not a non-empty string, if ``variance`` is negative, or
        if the loadings do not form a unit vector.
    """

    __slots__ = ("_loadings", "_name", "_variance")

    def __init__(self, name: str, loadings: Coefficients, variance: float) -> None:
        if not isinstance(name, str) or not name.strip():
            raise InvalidValuesError(
                "PrincipalComponent name must be a non-empty string"
            )

        if variance < 0.0:
            raise InvalidValuesError(
                f"{name} has variance {variance}, and a variance cannot be negative"
            )

        self._name = name.strip()
        self._loadings = loadings
        self._variance = float(variance)
        self._check_is_a_direction()

    def _check_is_a_direction(self) -> None:
        """Raise unless the loadings have length 1.

        A component says *which way*, not *how far*. Scaling the vector would
        scale every transformed coordinate with it while leaving the reported
        variance untouched, so the two would stop describing the same thing.
        """
        length = float(np.linalg.norm(self.direction))

        if abs(length - 1.0) > UNIT_LENGTH_TOLERANCE:
            raise InvalidValuesError(
                f"{self._name} has length {length}, and a component must be a "
                f"unit vector"
            )

    @property
    def name(self) -> str:
        """What this component is called."""
        return self._name

    @property
    def loadings(self) -> Coefficients:
        """The direction, one weight per feature name."""
        return self._loadings

    @property
    def variance(self) -> float:
        """The variance of the data along this direction."""
        return self._variance

    @property
    def direction(self) -> FloatArray:
        """The loadings as a plain vector, in the order the features were given."""
        return np.array(
            [coefficient.value for coefficient in self._loadings], dtype=np.float64
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The features this component weights, in order."""
        return tuple(coefficient.name for coefficient in self._loadings)

    def loading_for(self, name: str) -> float:
        """How much this component leans on one feature.

        Raises
        ------
        InvalidValuesError
            If ``name`` is not one of the features this component weights.
        """
        return self._loadings[name]

    def __repr__(self) -> str:
        return (
            f"PrincipalComponent({self._name!r}, variance={self._variance:.4f}, "
            f"n_features={len(self._loadings)})"
        )


class PrincipalComponents:
    """An ordered, orthogonal set of directions, with the total they came from.

    Parameters
    ----------
    components:
        The directions, most variance first. All must weight the same features,
        in the same order.
    total_variance:
        The variance of the whole dataset the decomposition was fitted to --
        the sum over *every* direction, including those not kept. This is the
        denominator of every share reported here, and it is supplied rather
        than summed for the reason the module docstring gives.

    Raises
    ------
    EmptyValuesError
        If no components are supplied.
    NonUniqueFeaturesError
        If two components share a name.
    InvalidValuesError
        If the components weight different features, if their variances
        increase at any point, if two of them are not orthogonal, or if
        ``total_variance`` is not positive and at least their sum.
    """

    __slots__ = ("_components", "_total_variance")

    def __init__(
        self, components: Sequence[PrincipalComponent], total_variance: float
    ) -> None:
        if not components:
            raise EmptyValuesError("a decomposition needs at least one component")

        self._components = tuple(components)
        self._total_variance = float(total_variance)

        self._check_names_are_unique()
        self._check_components_weight_the_same_features()
        self._check_variances_do_not_increase()
        self._check_components_are_orthogonal()
        self._check_total_variance_covers_them()

    def _check_names_are_unique(self) -> None:
        names = [component.name for component in self._components]

        if len(set(names)) != len(names):
            raise NonUniqueFeaturesError(f"component names must be unique; got {names}")

    def _check_components_weight_the_same_features(self) -> None:
        """Raise unless every component spans the same features in one order.

        Position ``i`` of one direction has to mean the same feature as position
        ``i`` of the next, or the projection is summing across mismatched
        columns.
        """
        expected = self._components[0].feature_names

        for component in self._components[1:]:
            if component.feature_names != expected:
                raise InvalidValuesError(
                    f"{component.name} weights {component.feature_names}, but "
                    f"{self._components[0].name} weights {expected}"
                )

    def _check_variances_do_not_increase(self) -> None:
        """Raise unless the components are ordered by decreasing variance.

        The check that catches an unreversed ``eigh``. See the module docstring
        for why that failure is otherwise silent.
        """
        for earlier, later in zip(self._components, self._components[1:], strict=False):
            if later.variance > earlier.variance + VARIANCE_ORDER_TOLERANCE:
                raise InvalidValuesError(
                    f"components must be ordered by decreasing variance; "
                    f"{later.name} explains {later.variance} against "
                    f"{earlier.name}'s {earlier.variance}"
                )

    def _check_components_are_orthogonal(self) -> None:
        """Raise unless every pair of directions is perpendicular."""
        directions = self.directions
        products = directions @ directions.T
        np.fill_diagonal(products, 0.0)

        if np.any(np.abs(products) > ORTHOGONALITY_TOLERANCE):
            worst = int(np.argmax(np.abs(products)))
            first, second = divmod(worst, self.n_components)
            raise InvalidValuesError(
                f"components must be orthogonal; "
                f"{self._components[first].name} and "
                f"{self._components[second].name} have dot product "
                f"{products[first, second]}"
            )

    def _check_total_variance_covers_them(self) -> None:
        """Raise unless the stated total is positive and at least what is kept.

        Kept variance above the total would make a share exceed 1, which is not
        a rounding problem but a denominator taken from the wrong data.
        """
        if self._total_variance <= 0.0:
            raise InvalidValuesError(
                f"total variance must be positive; got {self._total_variance}"
            )

        if self.kept_variance > self._total_variance * (1.0 + 1e-09):
            raise InvalidValuesError(
                f"components explain {self.kept_variance}, which is more than "
                f"the stated total of {self._total_variance}"
            )

    @property
    def n_components(self) -> int:
        """How many directions were kept."""
        return len(self._components)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The original features every component weights, in order."""
        return self._components[0].feature_names

    @property
    def n_features(self) -> int:
        """How many features the decomposition started from."""
        return len(self.feature_names)

    @property
    def total_variance(self) -> float:
        """The variance of the whole dataset, kept and discarded together."""
        return self._total_variance

    @property
    def kept_variance(self) -> float:
        """The variance along the directions actually held here."""
        return float(sum(component.variance for component in self._components))

    @property
    def directions(self) -> FloatArray:
        """The directions as a matrix, one component per row.

        Shape ``(n_components, n_features)``, which is the orientation a
        projection wants: ``centred_rows @ directions.T`` gives one transformed
        column per component.
        """
        return np.array(
            [component.direction for component in self._components], dtype=np.float64
        )

    @property
    def variance_shares(self) -> tuple[float, ...]:
        """Each component's variance over the whole dataset's variance.

        These sum to 1 only when every component was kept. That is the point:
        a two-of-five decomposition reporting 0.83 is telling you what the
        other three were worth.
        """
        return tuple(
            component.variance / self._total_variance for component in self._components
        )

    @property
    def cumulative_shares(self) -> tuple[float, ...]:
        """The running total of :attr:`variance_shares`.

        Read as "the first ``k`` components explain this much", which is the
        form the choice of ``k`` is actually made in.
        """
        return tuple(float(total) for total in np.cumsum(self.variance_shares))

    def n_components_for(self, share: float) -> int:
        """How many components are needed to explain at least ``share``.

        The usual way of choosing ``k``: fit everything once, then ask how many
        directions reach 95% of the variance.

        Raises
        ------
        InvalidValuesError
            If ``share`` is outside ``(0, 1]``, or if the components held here
            never reach it -- which they cannot when some were discarded.
        """
        if not 0.0 < share <= 1.0:
            raise InvalidValuesError(f"a share must fall in (0, 1]; got {share}")

        for position, reached in enumerate(self.cumulative_shares, start=1):
            if reached >= share - 1e-12:
                return position

        raise InvalidValuesError(
            f"these components explain {self.cumulative_shares[-1]:.4f} in "
            f"total, which never reaches {share}"
        )

    def value_for(self, name: str) -> PrincipalComponent:
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
            f"unknown component {name!r}; this decomposition holds "
            f"{[component.name for component in self._components]}"
        )

    def __getitem__(self, name: str) -> PrincipalComponent:
        return self.value_for(name)

    def __contains__(self, name: object) -> bool:
        return any(component.name == name for component in self._components)

    def __iter__(self) -> Iterator[PrincipalComponent]:
        return iter(self._components)

    def __len__(self) -> int:
        return self.n_components

    def __repr__(self) -> str:
        return (
            f"PrincipalComponents(n_components={self.n_components}, "
            f"explained={self.cumulative_shares[-1]:.4f})"
        )
