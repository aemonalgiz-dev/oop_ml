"""PCA in a space whose coordinates nobody can write down.

Theory
------
Ordinary PCA eigendecomposes the covariance matrix ``C = X'X / (n - 1)``, which
is ``(p, p)``. In an expanded space ``p`` is enormous or infinite, so that
matrix cannot be built and this looks like the end of it.

It is not, because of the same rearrangement kernel ridge uses. ``X'X`` and
``XX'`` share their non-zero eigenvalues, and their eigenvectors are related by
a multiplication::

    if  XX' u = lambda u      then    X'X (X'u) = lambda (X'u)

So eigendecomposing the ``(n, n)`` Gram matrix gives you the same spectrum as
the ``(p, p)`` covariance, and each eigenvector ``u`` of the Gram matrix
*names* a direction in the expanded space -- as the combination ``X'u`` of the
training rows. The direction is never formed; the coefficients are the model.

Two consequences that make this different from ordinary PCA
------------------------------------------------------------
**There are no loadings.** ``PrincipalComponent`` binds one weight to each
feature name, because in the original space a direction has a coordinate per
feature and "this component leans on floor area by 0.71" is a sentence. Here a
direction lives in a space whose coordinates have no names, and often no finite
count. What a component has instead is one coefficient per **training row**:
"this direction is built mostly out of rows 3, 17 and 40". That is a different
kind of statement about a different kind of object, which is why this cannot
reuse the PCA vocabulary and does not try.

**There is no inverse transform.** Ordinary PCA reconstructs by
``coordinates @ directions``, landing back among the original features. Here
that lands in the *implied* space, and getting from there back to the original
features means inverting the feature map -- which for an RBF kernel does not
exist, since the map is into infinitely many dimensions and most points there
are not the image of anything. This is called the pre-image problem, it is
solved approximately by iterative methods, and it is genuinely absent rather
than merely unimplemented.

Centring, which is the step that is easy to skip
-------------------------------------------------
Variance is measured about the mean, so PCA centres first. Here the points that
need centring are in the implied space and cannot be touched. The way through is
the identity on
:meth:`~oop_ml.core.kernel.matrix.KernelMatrix.centred`: the Gram matrix of the
centred features can be computed from the Gram matrix of the uncentred ones.
Skip it and the first component points at the implied mean rather than
describing any spread, exactly as in ordinary PCA -- and unlike ordinary PCA,
there is no picture to notice it in.

Query rows are centred against the *training* mean by
``centred_against``, for the reason every transformer here re-uses fitted
statistics rather than relearning them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
from pydantic import ConfigDict, Field, PrivateAttr

from oop_ml.core.base.estimator import Transformer
from oop_ml.core.data.feature import Feature
from oop_ml.core.data.feature_set import FeatureSet
from oop_ml.core.data.row_block import RowBlock, rows_of
from oop_ml.core.exceptions import (
    EmptyValuesError,
    InvalidValuesError,
    TooFewValuesError,
)
from oop_ml.core.kernel.functions import Kernel, LinearKernel
from oop_ml.core.kernel.matrix import KernelMatrix
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


class KernelPrincipalComponentAnalysis(Transformer[Sequence[Feature]]):
    """PCA through the Gram matrix, in whatever space the kernel implies.

    Parameters
    ----------
    kernel:
        Which space to decompose in. With
        :class:`~oop_ml.core.kernel.functions.LinearKernel` this recovers
        ordinary PCA's coordinates up to a sign, which is the control that says
        the machinery is right.
    n_components:
        How many directions to keep. At most the number of training rows, since
        that is how many the Gram matrix has -- a limit ordinary PCA does not
        have, where the ceiling is the feature count.

    Raises
    ------
    pydantic.ValidationError
        If ``n_components`` is below 1. Field bounds are pydantic's to
        enforce, so the error is pydantic's too.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    kernel: Kernel = LinearKernel()
    n_components: int | None = Field(default=None, ge=1)

    _components: KernelComponents | None = PrivateAttr(default=None)
    _training_rows: RowBlock | None = PrivateAttr(default=None)
    _training_matrix: KernelMatrix | None = PrivateAttr(default=None)

    @property
    def components(self) -> KernelComponents:
        """The directions this fit found, most variance first.

        Raises
        ------
        NotFittedError
            If read before ``fit``.
        """
        self._check_fitted()
        assert self._components is not None
        return self._components

    def fit(self, input_values: Sequence[Feature]) -> Self:
        """Decompose the centred Gram matrix of ``input_values``.

        Builds the Gram matrix, centres it by the identity on ``KernelMatrix``,
        hands it to :meth:`_solve`, and keeps both the rows and the uncentred
        matrix -- ``transform`` needs the rows to build a query matrix and the
        matrix to centre it against.

        Raises
        ------
        EmptyValuesError
            If no features are supplied.
        TooFewValuesError
            If there are fewer than two rows.
        InvalidValuesError
            If ``n_components`` exceeds the number of rows.
        """
        feature_set = FeatureSet(input_values)

        if feature_set.n_samples < 2:
            raise TooFewValuesError(
                f"a decomposition needs at least two rows; got {feature_set.n_samples}"
            )

        if self.n_components is not None and self.n_components > feature_set.n_samples:
            raise InvalidValuesError(
                f"cannot keep {self.n_components} components from "
                f"{feature_set.n_samples} rows"
            )

        names = tuple(feature.name for feature in feature_set)
        rows = self._as_rows(feature_set, names)
        matrix = self.kernel.between(rows, rows)

        self._training_rows = rows
        self._training_matrix = matrix
        self._components = self._solve(matrix.centred())
        self._mark_fitted()

        return self

    def transform(self, input_values: Sequence[Feature]) -> list[Feature]:
        """Project rows onto the learned directions.

        Build the kernel matrix pairing the query rows against the *training*
        rows, centre it against the training matrix, and multiply by the
        components' coefficients transposed. One output feature per kept
        component, named ``kernel_component_1`` upward.

        The centring is the step to be careful about. Query rows must be
        shifted by the mean the fit learned, which is what
        ``centred_against`` does; centring them against themselves is the same
        leak ``PrincipalComponentAnalysis`` guards against, and here nothing
        about the output would look wrong.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        centred = self.query_matrix(input_values)
        coordinates = centred.values @ self.components.coefficients.T

        return [
            Feature(component.name, coordinates[:, position])
            for position, component in enumerate(self.components)
        ]

    def _solve(self, centred: KernelMatrix) -> KernelComponents:
        """Eigendecompose the centred Gram matrix.

        The concept. Given the already-centred ``(n, n)`` matrix:

        1. ``numpy.linalg.eigh`` it. Ascending eigenvalues, eigenvectors as
           columns, exactly as in ordinary PCA.
        2. Reverse both into descending order, and clamp the eigenvalues at
           zero. A centred Gram matrix is singular by construction -- centring
           removes a degree of freedom, so at least one eigenvalue is
           genuinely zero and comes back as a small negative.
        3. **An eigenvalue is used for two different things here, at two
           different scales, and this is the step that catches people out.**
           The Gram matrix is ``X_c X_c'`` where the covariance is
           ``X_c' X_c / (n - 1)``, so a raw eigenvalue is ``n - 1`` times the
           variance it corresponds to. On the ``ROTATED_ELLIPSE`` fixture,
           whose known variances are 4.0 and 1.0, the raw eigenvalues come back
           as 16.0 and 4.0 across five rows.

           So: **report** ``eigenvalue / (n - 1)`` as the variance, which is
           what makes it the same quantity ordinary PCA reports. But
           **normalise** by the square root of the *raw* eigenvalue in step 6,
           because that is the length of the direction the eigenvector names.
           Using the scaled value in both places leaves every coordinate too
           large by ``sqrt(n - 1)``, and using the raw value in both leaves
           every reported variance too large by ``n - 1``. Neither shows up in
           the shares, which are ratios and cancel the factor either way.
        4. The total variance is the sum of **all** the scaled eigenvalues,
           taken before any truncation, for the reason ordinary PCA takes it
           there.
        5. Keep the first ``n_components``, or all of them.
        6. **Normalise each kept eigenvector by ``sqrt(eigenvalue)``.** This is
           the step with no counterpart in ordinary PCA and the one that is
           easy to miss. ``eigh`` returns unit-length ``u``, but the direction
           it names in the implied space is ``X'u``, whose length is
           ``sqrt(lambda)`` rather than 1. Dividing by ``sqrt(lambda)`` makes
           the *implied* direction a unit vector, which is what makes the
           projected coordinates come out with variance ``lambda`` instead of
           ``lambda^2``. Skip it and the transform still runs and still
           separates the data -- every coordinate is simply scaled wrong, by a
           different factor per component.
        7. Drop any component whose eigenvalue is below
           ``MINIMUM_COMPONENT_VARIANCE``, rather than dividing by its square
           root. Those are the null-space directions centring created, and
           normalising them divides by nearly zero.
        8. Build one ``KernelComponent`` per survivor, named by
           :meth:`name_for`, and hand the lot to ``KernelComponents`` with the
           full total.

        Parameters
        ----------
        centred:
            The centred training Gram matrix, square.

        Returns
        -------
        KernelComponents
            The kept directions. Do not set ``_fitted`` here.
        """
        centred.check_square()
        eigenvalues, eigenvectors = np.linalg.eigh(centred.values)

        raw = np.maximum(eigenvalues[::-1], 0.0)
        directions = eigenvectors[:, ::-1]

        variances = raw / (centred.n_left - 1)
        total_variance = float(np.sum(variances))
        kept = self.n_components or centred.n_left

        components = []

        for position in range(kept):
            if raw[position] < MINIMUM_COMPONENT_VARIANCE:
                break

            components.append(
                KernelComponent(
                    self.name_for(position),
                    directions[:, position] / np.sqrt(raw[position]),
                    float(variances[position]),
                )
            )

        return KernelComponents(components, total_variance)

    def _as_rows(self, feature_set: FeatureSet, names: tuple[str, ...]) -> RowBlock:
        """The features as a row block, in the given order."""
        ordered = FeatureSet.matching(names, list(feature_set))

        return rows_of(
            np.column_stack([ordered.column(name).values for name in names]), names
        )

    def query_matrix(self, input_values: Sequence[Feature]) -> KernelMatrix:
        """The centred kernel matrix pairing query rows against training rows.

        The plumbing ``transform`` needs: checks the features, lays them out in
        the fitted order, builds the ``(n_queries, n_training)`` matrix, and
        centres it against the training one.

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        InvalidValuesError
            If the supplied features are not exactly the fitted ones.
        """
        self._check_fitted()
        assert self._training_rows is not None
        assert self._training_matrix is not None

        fitted = self._training_rows.feature_names
        supplied = {feature.name for feature in input_values}

        if supplied != set(fitted):
            raise InvalidValuesError(
                f"expected exactly the fitted features {sorted(fitted)}; "
                f"got {sorted(supplied)}"
            )

        rows = self._as_rows(FeatureSet(input_values), fitted)

        return self.kernel.between(rows, self._training_rows).centred_against(
            self._training_matrix
        )

    @staticmethod
    def name_for(position: int) -> str:
        """The name of the component at ``position``, counting from zero."""
        return f"{KERNEL_COMPONENT_NAME_PREFIX}_{position + 1}"

    def __repr__(self) -> str:
        if not self.is_fitted:
            return f"KernelPrincipalComponentAnalysis({self.kernel!r}, unfitted)"

        return (
            f"KernelPrincipalComponentAnalysis({self.kernel!r}, "
            f"n_components={self.components.n_components})"
        )
